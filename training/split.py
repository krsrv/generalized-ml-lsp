"""
Tools to split given NPZ files into training, test, validation (and potentially holdout)
"""

import os
import numpy as np
from pathlib import Path
from typing import Any
from datetime import datetime

def dprint(*args: Any, **kwargs: Any) -> None:
    """
    Print like built-in print(), but prefix the line with current local datetime:
    "YYYY-MM-DD HH:MM:SS.SSS: "
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # milliseconds
    print(f"{ts}: ", *args, **kwargs)

class Splitter:
    """
    Split the given dataset into splits according to specified ratios.

    Input .npz is expected to contain:
      * "{n}/{g}/{name}" keys (n,g are ints): parallel arrays that must be split
        together (sampled without replacement) *within each (n,g) group*.
      * All other keys are copied as-is into every output split file.
      * "split_seed" and "split_name" keys are added to each output split file.

    Output: one .npz per split, saved as: f"{output_prefix}{split_name}.npz"

    Example:
        splitter = Splitter(
            input="training-data/compiled/2-28.npz",
            output_prefix="training-data/compiled/2-28-split-",
            seed=1,
            splits={"test": 0.15, "validation": 0.15, "train": 0.7},
        )
        out_sizes = splitter.save_splits()

    Notes:
        * Groups with 0 examples are skipped.
        * Examples from each split are sampled from the set of all examples
            (across all (n,g) groups) without replacement. That may lead to
            some groups being under- or over-represented in some splits,
            especially for small groups.
    """

    def __init__(
        self,
        input: str | os.PathLike[str],
        output_prefix: str | os.PathLike[str],
        *,
        seed: int = 0,
        splits: dict[str, float] | None = None,
        compress: bool = False,
    ) -> None:
        if splits is None or len(splits) == 0:
            raise ValueError("splits must be a non-empty dict like {'train':0.8,'test':0.2}.")
        self.input = Path(input)
        self.output_prefix = str(Path(output_prefix))
        self.seed = int(seed)
        self.splits = dict(splits)  # preserves insertion order
        self.compress = bool(compress)
        self._validate_splits()

    def _validate_splits(self) -> None:
        total = float(sum(self.splits.values()))
        for k, v in self.splits.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"Invalid split name: {k!r}")
            if v < 0:
                raise ValueError(f"Split ratio must be non-negative: {k}={v}")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total}).")

    def get_existing_split_files(self) -> dict[str, Path]:
        """
        Returns: dict split_name -> Path of existing split files matching
        glob f"{self.output_prefix}*.npz"
        (regardless of whether the split with that name is requested).
        """
        existing: dict[str, Path] = {}
        prefix_path = Path(self.output_prefix)
        parent = prefix_path.parent
        stem = prefix_path.name
        for path in parent.glob(f"{stem}*.npz"):
            split_name = path.stem[len(stem):]
            existing[split_name] = path
        return existing

    @staticmethod
    def _try_parse_ng_key(key: str) -> tuple[str, str] | None:
        key_parts = key.split("/")
        if len(key_parts) < 3 or not key_parts[0].isdigit() or not key_parts[1].isdigit():
            return None
        n = int(key_parts[0])
        g = int(key_parts[1])
        if key_parts[0] != str(n):
            raise ValueError(f"Invalid n format: {key!r}")
        if key_parts[1] != str(g):
            raise ValueError(f"Invalid g format: {key!r}")
        return f"{n}/{g}", "/".join(key_parts[2:])

    @staticmethod
    def _largest_remainder_counts(total: int, ratios: list[float]) -> list[int]:
        """
        Allocate integer counts summing to `total`, approximately matching ratios,
        using the largest remainder method (deterministic with stable tie-breaks).
        """
        if total <= 0:
            return [0] * len(ratios)

        raw = [r * total for r in ratios]
        floors = [int(np.floor(x)) for x in raw]
        leftover = total - sum(floors)

        fracs = [x - f for x, f in zip(raw, floors)]
        # tie-break: earlier split wins:
        order = sorted(range(len(ratios)), key=lambda i: (-fracs[i], i))

        counts = floors[:]
        for i in range(leftover):
            counts[order[i]] += 1
        return counts

    def save_splits(self, verbose=False, overwrite=True) -> dict[str, int]:
        """
        Saves one .npz per split.

        Args:
            verbose: if True, print progress messages.
            overwrite: if False, perform a best-effort existence check before
                writing any files, and raise FileExistsError if any split files
                already exist (the check is not immune to race conditions).

        Returns: dict split_name -> total #examples across all (n,g) groups in that split.
        """
        if not overwrite:
            existing_splits = self.get_existing_split_files()
            if existing_splits:
                raise FileExistsError(
                    f"Split files already exist for {self.output_prefix}: "
                    f"{', '.join(str(p) for p in existing_splits.values())}"
                )
        rng = np.random.default_rng(self.seed)
        split_names = list(self.splits.keys())
        split_ratios = [float(self.splits[k]) for k in split_names]
        def vdprint(*args: Any, **kwargs: Any) -> None:
            if verbose:
                dprint(*args, **kwargs)

        with np.load(self.input) as z:
            vdprint(f"Reading {self.input}...")
            # Partition keys into per-(n,g) groups vs "other" (copied as-is)
            groups: dict[str, dict[str, np.ndarray]] = {}
            other: dict[str, np.ndarray] = {}

            for key in z.files:
                arr = z[key]
                parsed = self._try_parse_ng_key(key)
                if parsed is None:
                    if key in ('split_seed', 'split_name'):
                        raise ValueError(
                            f"Input data cannot contain reserved key: {key!r}")
                    other[key] = arr
                    continue
                ng, name = parsed
                groups.setdefault(ng, {})[name] = arr

            vdprint(f"Computing splits...")
            # Validate group arrays and get per-group lengths
            group_names = list(groups.keys())
            group_lengths = []
            for ng in group_names:
                d = groups[ng]
                lens = [int(np.shape(v)[0]) for v in d.values()]
                lens_set = set(lens)
                if len(lens_set) != 1:
                    detail = ", ".join(str(l) for l in lens)
                    raise ValueError(
                        f"Group {ng} has inconsistent first-dimension lengths: {detail}")
                group_lengths.append(lens[0])

            group_lengths = np.array(group_lengths)
            group_lengths_sum = np.pad(np.cumsum(group_lengths), (1, 0))
            grps = np.repeat(np.arange(len(group_lengths)), group_lengths)
            idxs = rng.permutation(group_lengths_sum[-1])
            grps = grps[idxs]
            split_sizes = self._largest_remainder_counts(
                group_lengths_sum[-1], split_ratios)
            offset = 0
            for split_name, split_size in zip(split_names, split_sizes):
                vdprint(f"  Computing split {split_name} with {split_size} examples...")
                split_idxs = idxs[offset:offset + split_size]
                split_grps = grps[offset:offset + split_size]
                offset += split_size
                ii = np.argsort(split_grps)
                split_idxs = split_idxs[ii]
                split_grps = split_grps[ii]
                split_grp_uniq, split_grp_cnt = np.unique(
                    split_grps, return_counts=True)
                out: dict[str, Any] = dict(other)  # copy-as-is into every split
                out['split_seed'] = self.seed
                out['split_name'] = split_name
                grp_offset = 0
                for grp_i, grp_c in zip(split_grp_uniq, split_grp_cnt):
                    ng = group_names[grp_i]
                    d = groups[ng]
                    cur_idx = split_idxs[grp_offset:grp_offset + grp_c] - group_lengths_sum[grp_i]
                    grp_offset += grp_c
                    for name, arr in d.items():
                        out[f"{ng}/{name}"] = arr[cur_idx]
                out_path = Path(f"{self.output_prefix}{split_name}.npz")
                if self.compress:
                    vdprint(f"  Saving compressed split to {out_path}...")
                    np.savez_compressed(out_path, **out)
                else:
                    vdprint(f"  Saving uncompressed split to {out_path}...")
                    np.savez(out_path, **out)

        vdprint("Done.")
        return dict(zip(split_names, split_sizes))


if __name__ == "__main__":
    splitter = Splitter(
        input="training-data/compiled/2-28.npz",
        output_prefix="training-data/compiled/2-28-split-",
        seed=1,
        splits={"test": 0.15, "validation": 0.15, "train": 0.7},
    )
    existing_splits = splitter.get_existing_split_files()
    assert not existing_splits, (
        f"Splits for {splitter.output_prefix} already exist: {existing_splits}")
    out_sizes = splitter.save_splits(verbose=True)
    print(
        f"Saved sizes: " + ", ".join(f"{k}: {v}" for k, v in out_sizes.items())
    )
