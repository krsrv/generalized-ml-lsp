import os
import sys
import random
import shutil
import string
from ctypes import sizeof
from enum import Enum

import numpy as np
from tabulate import tabulate


class Topology(Enum):
    FullyConnected = 0
    Random = 1
    Cubic = 2
    Square = 3
    Hex = 4
    HeavyHex = 5
    Linear = 6


class GateSet(Enum):
    Superconducting = 0
    IonTrap = 1
    Random = 2


class UnprepDataCompiler:
    """
    Given npz files with keys in {unprep_gate, layout, observation, topology, gate_set_type,
    depth, gates, gate_qubits}, combine them into npz fragments such that the resulting set
    of datapoints:
    * have # datapoints = `self.total_datapoints`
    * has probability distribution over dataset defined by `self.intra_topology_sampling_ratio`
      and `self.inter_topology_sampling_ratio`.

    Input: list of filenames to be combined
    Output via create_npz_file(filename): npz files with name "{filename}-f{fragment number}.npz".
    Note that `load_data` must be called before `create_npz_file`.
    """

    def __init__(self, files: list[str], seed: int=1) -> None:
        self.files = files
        # Desired output distribution:
        #   0. Total number of points
        self.total_datapoints = 30_000_000
        #   1. Within each topology, 70% data should be the correct gateset, and 15% mismatch and
        #     random each
        # Issue #3: fractions do not sum to 1 => ratios are not as printed.
        self.intra_topology_sampling_ratio = {
            Topology.FullyConnected: (0.15, 0.7, 0.15),
            Topology.Random: (0.15, 0.15, 0.7),
            Topology.Cubic: (0.7, 0.15, 0.15),
            Topology.Square: (0.7, 0.15, 0.15),
            Topology.Hex: (0.7, 0.15, 0.15),
            Topology.HeavyHex: (0.7, 0.15, 0.15),
            Topology.Linear: (0.7, 0.15, 0.15),
        }

        #   2. Overall distribution between topologies
        self.inter_topology_sampling_ratio = {
            Topology.FullyConnected: 0.17,
            Topology.Random: 0.20,
            Topology.Cubic: 0.05,
            Topology.Square: 0.17,
            Topology.Hex: 0.155,
            Topology.HeavyHex: 0.10,
            Topology.Linear: 0.155,
        }
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def load_data(self):
        self.data_list = {}
        self.total_size = 0
        for file in self.files:
            try:
                self.data_list[file] = np.load(file)
                self.total_size += self.data_list[file]["depth"].shape[0]
            except:
                print(f"Error in {file}")
                self.data_list.pop(file, None)

    def get_total_size(self):
        return self.total_size

    def _pick_samples(
        self, arrays: list[np.ndarray], files: list[str], num_samples: int,
        output_dict: dict
    ) -> int:
        sizes = np.array([x["depth"].shape[-1] for x in arrays])
        cum_sizes = np.pad(
                np.cumsum(sizes), (1, 0), mode="constant", constant_values=0)
        assert cum_sizes[-1] >= num_samples
        if len(cum_sizes) == 0 or cum_sizes[-1] == 0:
            return 0
        sample_indices = self.rng.choice(
            cum_sizes[-1], size=num_samples, replace=False, shuffle=False)
        sample_indices.sort()
        file_indices = cum_sizes.searchsorted(sample_indices, "right") - 1
        file_idxs, file_num_samples = np.unique(
            file_indices, return_counts=True)
        assert sum(file_num_samples) == num_samples
        offset = 0
        total_samples = 0
        for file_idx, file_cnt in zip(file_idxs, file_num_samples):
            indices = sample_indices[offset : offset + file_cnt] - (
                cum_sizes[file_idx])
            key = files[file_idx]
            assert key not in output_dict
            output_dict[key] = indices
            total_samples += len(indices)
            offset += file_cnt
        return total_samples

    def _print_tabular_top_gs_data(self, data: dict[int, int]) -> None:
        """
        Pretty print data which stores the number of datapoints corresponding to
        each (topology, gateset) int tuple.
        """
        # Build header row
        headers = [""] + [gs.name for gs in GateSet]

        # Build table data
        table = []
        for top in Topology:
            row = [top.name]
            for gs in GateSet:
                row.append(data.get((top.value, gs.value), 0))
            table.append(row)

        print(tabulate(table, headers=headers, tablefmt="grid"))

    def create_npz_file(self, filename: str):
        """
        Main function to create npz file outputs. `filename` is the prefix (eg - "output/2-28").
        The function chunks the data into ~10 GB sizes and dumps them as fragments
        (eg - "output/2-28-f0.npz").
        Note: run `load_data` before calling this function.
        """
        topo_gs_idxs = {}
        total_size = {}
        for k, data in self.data_list.items():
            topo = data["topology"][0]
            gs = data["gate_set_type"][0]
            if (topo, gs) not in total_size:
                total_size[(topo, gs)] = 0
                topo_gs_idxs[(topo, gs)] = []
            total_size[(topo, gs)] += data["depth"].shape[0]
            topo_gs_idxs[(topo, gs)].append(k)

        print("Raw dataset distribution")
        self._print_tabular_top_gs_data(total_size)

        sample_idxs = dict()
        for topo in Topology:
            for gs in GateSet:
                proposed_final_sample_count = int(
                    self.total_datapoints
                    * self.inter_topology_sampling_ratio[topo]
                    * self.intra_topology_sampling_ratio[topo][gs.value]
                )
                available_sample_count = total_size.get((topo.value, gs.value), 0)
                if proposed_final_sample_count > available_sample_count:
                    print(
                        f"Total number of datapoints not consistent with current "
                        f"dataset for {topo}, {gs}. "
                        f"Expected {proposed_final_sample_count} "
                        f"vs available {available_sample_count}", file=sys.stderr)
                else:
                    print(topo, gs, "consistent with sampling requirements")

                in_filenames = topo_gs_idxs.get((topo.value, gs.value), [])
                sample_count = self._pick_samples(
                    [self.data_list[k] for k in in_filenames],
                    in_filenames,
                    proposed_final_sample_count,
                    output_dict=sample_idxs
                )
                total_size[(topo.value, gs.value)] = sample_count

        print("After filtering for each (topology, gateset)")
        self._print_tabular_top_gs_data(total_size)

        ## Reorganize into (n, g) pairs and dump:
        n_g = {}
        for file, idxs in sample_idxs.items():
            n, g = self.data_list[file]["n"][0], self.data_list[file]["g"][0]
            if (n, g) not in n_g:
                n_g[(n, g)] = []
            n_g[(n, g)].append((file, idxs))

        dump_data = {}
        global_n_data = {}
        global_g_data = {}
        counter = 0
        curr_size = 0
        for k, vs in n_g.items():
            n, g = k
            if n in global_n_data:
                cur_n_data = global_n_data[n]
            else:
                global_n_data[n] = cur_n_data = {"layout": [], "topology": []}
            if g in global_g_data:
                cur_g_data = global_g_data[g]
            else:
                global_g_data[g] = cur_g_data = {
                    "gates": [], "gate_qubits": [], "gate_set_type": []}
            for i, v in enumerate(vs):
                file, idxs = v
                curr_data = self.data_list[file]
                original_bs = curr_data["depth"].shape[0]
                assert curr_data["depth"].shape == (original_bs,)
                bs = len(idxs)
                observations = curr_data["observation"].reshape(original_bs, n)[idxs, :]
                assert observations.dtype == np.uint64
                out_layouts = cur_n_data["layout"]
                global_n_idx = len(out_layouts)
                out_layouts.append(curr_data["layout"].reshape(n, n).astype(np.bool_))
                cur_n_data["topology"].append(curr_data["topology"].item())
                out_gates = cur_g_data["gates"]
                global_g_idx = len(out_gates)
                out_gates.append(curr_data["gates"].reshape(g))
                cur_g_data["gate_qubits"].append(
                    curr_data["gate_qubits"].reshape(g, 2))
                cur_g_data["gate_set_type"].append(curr_data["gate_set_type"].item())

                # Verify-and-convert: if one of the assertions below fail,
                # that means that data ranges changed and the type has to be adjusted.
                unprep_gate = curr_data["unprep_gate"][idxs]
                assert np.all(0 <= unprep_gate) and np.all(unprep_gate < 2**16)
                unprep_gate = unprep_gate.astype(np.uint16)
                depth = curr_data["depth"][idxs]
                assert np.all(0 <= depth) and np.all(depth < 2**16)
                # Conservative: max(depth) = 91 at the time of writing.
                depth = depth.astype(np.uint16)
                assert global_n_idx < 256
                assert global_g_idx < 256
                formatted_data = {
                    "unprep_gate": unprep_gate,
                    "depth": depth,
                    "observation": observations,
                    "global_n_idx": np.full((bs,), global_n_idx, dtype=np.uint8),
                    "global_g_idx": np.full((bs,), global_g_idx, dtype=np.uint8),
                }
                for key in formatted_data:
                    curr_size += formatted_data[key].nbytes
                    if i == 0:
                        dump_data[f"{n}/{g}/{key}"] = formatted_data[key]
                    else:
                        dump_data[f"{n}/{g}/{key}"] = np.concatenate(
                            (dump_data[f"{n}/{g}/{key}"], formatted_data[key]), axis=0
                        )
            counter += 1
            print(f"Completed dataset sampling for {n}/{g} ({counter} out of {len(n_g)})")
        # Add global n, g data
        for n, n_data in global_n_data.items():
            for k, v in n_data.items():
                dump_data[f"global_n/{n}/{k}"] = np.array(v)
        for g, g_data in global_g_data.items():
            for k, v in g_data.items():
                dump_data[f"global_g/{g}/{k}"] = np.array(v)
        dump_data["seed"] = self.seed
        print(f"Saving data")
        np.savez(f"{filename}.npz", **dump_data)


if __name__ == "__main__":
    import argparse
    import re
    import time

    parser = argparse.ArgumentParser(
        description="Convert C++ data output to ML training format")
    parser.add_argument(
        "--filename", type=str, required=True,
        help="Output filename, without npz extension")
    parser.add_argument("--folder", type=str, required=True, help="Input folder")
    args = parser.parse_args()

    assert not args.filename.endswith(".npz"), (
        "Output filename must not end with '.npz'")
    base_dir = os.path.dirname(args.filename)
    base_filename = os.path.basename(args.filename)
    existing_fragments = [
        fname
        for fname in os.listdir(base_dir)
        if re.match(rf"{re.escape(base_filename)}-f\d+\.npz$", fname)
        or fname == f"{base_filename}.npz"
    ]
    assert (
        not existing_fragments
    ), f"Fragments for {args.filename} already exist: {existing_fragments}"

    tic = time.time()
    folder = args.folder
    in_filenames = sorted(os.listdir(folder))
    files = [os.path.join(folder, x) for x in in_filenames]
    compiler = UnprepDataCompiler(files)
    compiler.load_data()
    print("Total size:", compiler.get_total_size())
    compiler.create_npz_file(args.filename)
    toc = time.time()
    print(f"Converted {folder} -> {args.filename} ({toc-tic} sec)")
