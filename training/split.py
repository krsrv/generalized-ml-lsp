"""
Tools to split given NPZ files into training, test, validation (and potentially holdout)
"""

import os
import re
import time
from functools import wraps

import numpy as np


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


class Splitter:
    """
    Module to generate a train, test, validation split from a set of given files. Sample usage:
    ```
    splitter = Splitter(
        [
            "training-data/compiled/2-28-f0.npz",
            "training-data/compiled/2-28-f1.npz",
            "training-data/compiled/2-10_20000.npz",
        ]
    )
    splitter.set_batch_size(64)
    test_size, validation_size, train_size = splitter.generate_split(
        "training-data/compiled", "sample"
    )
    ```
    """
    keywords = [
        "unprep_gate",
        "depth",
        "observation",
        "global_n_idx",
        "global_g_idx",
    ]
    keyword_dtypes = [
        np.int32,
        np.int32,
        np.uint64,
        np.uint8,
        np.uint8,
    ]

    def __init__(self, files: list[str], seed: int = 1) -> None:
        self.data = []
        total_file_size = sum([os.path.getsize(file) for file in files])
        mem_limit = 15 * (1 << 30)  # 15 GB
        if total_file_size < mem_limit:
            print("Loading files in memory")
        else:
            print("Not loading files in memory")
        for file in files:
            if not file.endswith(".npz"):
                continue
            # Note that if npz is in a compressed format, np.load
            # will return a lazy loader.
            if total_file_size < mem_limit:
                with np.load(file, allow_pickle=False) as npzfile:
                    data = {key: npzfile[key] for key in npzfile}
            else:
                data = np.load(file, allow_pickle=False)
            self.data.append(data)

        self.batch_size = None
        self.test_split = 0.15
        self.validation_split = 0.15
        self.generate_metadata()
        self.hash_set = set()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_metadata(self) -> None:
        self.aggregate_metadata = {}  # (n, g) -> size
        self.per_file_metadata = {}  # (file_index, n, g) -> size
        self.reverse_map_file = {}  # (n, g) -> [file_indices]
        self.total_size = 0
        self.global_data = {}

        for i, data in enumerate(self.data):
            for j, key in enumerate(data):
                key_split = key.split("/")
                if key_split[0] in ["global_g", "global_n", "seed"]:
                    self.global_data[key] = data[key]
                    continue

                # The key should be of the form n/g/{data}
                assert len(key_split) == 3
                n, g, keyword = key_split
                if (int(n), int(g)) not in self.aggregate_metadata:
                    self.aggregate_metadata[(int(n), int(g))] = 0
                    self.reverse_map_file[(int(n), int(g))] = []

                if keyword == "depth":
                    size = data[key].shape[0]
                    self.total_size += size
                    self.per_file_metadata[(i, int(n), int(g))] = size
                    self.reverse_map_file[(int(n), int(g))].append(i)
                    self.aggregate_metadata[(int(n), int(g))] += size

    def set_batch_size(self, batch_size) -> None:
        self.batch_size = batch_size
        self._calculate_batch_metadata()

    def _calculate_batch_metadata(self) -> None:
        # Calculate number of batches in data for each (n, g) tuple.
        self.aggregate_num_batches = np.array(
            [int(v // self.batch_size) for k, v in self.aggregate_metadata.items()]
        )
        # self.aggregate_num_batches = self.aggregate_num_batches[
        #     np.nonzero(self.aggregate_num_batches)
        # ]
        # Get total number of batches in dataset.
        self.total_batches = np.sum(self.aggregate_num_batches)

    def _sample_batch_indices(self, size) -> tuple[np.ndarray, np.ndarray]:
        """
        Among all (n, g) tuples, calculate the indices of (n, g) tuples to sample along with the
        number of corresponding samples.
        """
        assert self.batch_size is not None
        self.test_size = int(
            self.total_batches * self.test_split * self.batch_size
        )  # 10 * self.batch_size
        # Pick the batch indices corresponding for test dataset (from 0, ..., TB-1).
        buff_idxs = np.sort(
            self.rng.choice(
                np.arange(self.total_batches),
                size // self.batch_size,
                replace=False,
            )
        )
        # Find the (n, g) tuple index corresponding to each batch index picked.
        search_arr = np.cumsum(self.aggregate_num_batches)
        idxs = search_arr.searchsorted(buff_idxs, "right")
        test_idxs, test_idx_counts = np.unique(idxs, return_counts=True)
        return test_idxs, test_idx_counts

    @timeit
    def generate_split(self, folder: str, file_prefix: str) -> None:
        """
        Member function to generate the actual split. The function works by splitting
        each set of datapoints for (n, g) into a train, test, validation npz file,
        storing them in `{folder}/_tmp`, and then finally merging the splits into one
        big file.
        The data is not actually manipulated until the very end stage where we actually
        dump contents to each file. We instead first sample and split indices into the
        train, test, validation buckets.
        """
        assert self.batch_size is not None
        ###########################################################################
        # Among all (n, g) tuples, calculate the indices of (n, g) tuples
        # to sample along with the number of corresponding samples.
        ###########################################################################
        # Calculate the size of each split.
        self.test_size = int(
            self.total_batches * self.test_split * self.batch_size
        )  # 10 * self.batch_size
        self.validation_size = int(
            self.total_batches * self.validation_split * self.batch_size
        )  # 10 * self.batch_size
        self.train_size = self.total_size - self.test_size - self.validation_size

        # Sample batch indices corresponding to each split. The sampling is
        # from (0, 1, ..., self.total_batches-1)
        shuffled_indices = np.arange(self.total_batches)
        self.rng.shuffle(shuffled_indices)
        test_batch_indices = np.sort(
            shuffled_indices[: (self.test_size // self.batch_size)]
        )
        validation_batch_indices = np.sort(
            shuffled_indices[
                (self.test_size // self.batch_size) : (
                    (self.test_size + self.validation_size) // self.batch_size
                )
            ]
        )
        train_batch_indices = np.sort(
            shuffled_indices[
                (self.test_size + self.validation_size) // self.batch_size :
            ]
        )

        # Find the actual (n, g) tuple index corresponding to each batch index picked.
        search_arr = np.cumsum(self.aggregate_num_batches)
        test_ng_idxs, test_ng_num_batches = np.unique(
            search_arr.searchsorted(test_batch_indices, "right"), return_counts=True
        )
        validation_ng_idxs, validation_ng_num_batches = np.unique(
            search_arr.searchsorted(validation_batch_indices, "right"),
            return_counts=True,
        )
        train_ng_idxs, train_ng_num_batches = np.unique(
            search_arr.searchsorted(train_batch_indices, "right"), return_counts=True
        )
        print("Generated samples for split.")

        ###########################################################################
        # test_ng_idxs contains the list of (n, g) indices to sample from.
        # test_ng_num_batches contains the corresponding number of samples
        # we need to draw. This data needs to be mapped to the actual data
        # spread across files.
        # First, for a given index and number of batches to sample, randomly
        # sample indices to create the data. Then map the indices to actual
        # file and within-file-offsets. Finally dump the data to file using
        # the collected info.
        ###########################################################################
        print(f"Total number of keys to iterate over: {len(self.aggregate_metadata)}")
        total_test_size, total_train_size, total_validation_size = 0, 0, 0
        for idx, key in enumerate(self.aggregate_metadata.keys()):
            n, g = key
            if self.aggregate_metadata[key] < self.batch_size:
                print(
                    f"Skipping {n}/{g}: #examples = {self.aggregate_metadata[key]} < batch size {self.batch_size}"
                )
                # If 0-value elements in self.aggregate_metadata are filtered out, we need to
                # keep idx in track with the the filtered array indexing.
                # idx = idx - 1
                continue
            print(f"Running {n}/{g} ({idx})")

            shuffled_indices = np.arange(self.aggregate_metadata[key])
            self.rng.shuffle(shuffled_indices)

            # Create test data.
            test_size = self._get_sample_size(test_ng_idxs, test_ng_num_batches, idx)
            file_idxs, offset_idxs = self.retrieve_offsets_from_indices(
                n, g, np.sort(shuffled_indices[:test_size])
            )
            dump_size = self.dump_to_file(
                n,
                g,
                file_idxs,
                offset_idxs,
                f"{folder}/_tmp/{file_prefix}-{n}-{g}-test.npz",
            )
            total_test_size += dump_size

            # Create validation data.
            validation_size = self._get_sample_size(
                validation_ng_idxs, validation_ng_num_batches, idx
            )
            file_idxs, offset_idxs = self.retrieve_offsets_from_indices(
                n, g, np.sort(shuffled_indices[test_size : test_size + validation_size])
            )
            dump_size = self.dump_to_file(
                n,
                g,
                file_idxs,
                offset_idxs,
                f"{folder}/_tmp/{file_prefix}-{n}-{g}-validation.npz",
            )
            total_validation_size += dump_size

            # Create train data.
            _train_size = self._get_sample_size(
                train_ng_idxs, train_ng_num_batches, idx
            )
            file_idxs, offset_idxs = self.retrieve_offsets_from_indices(
                n, g, np.sort(shuffled_indices[test_size + validation_size :])
            )
            dump_size = self.dump_to_file(
                n,
                g,
                file_idxs,
                offset_idxs,
                f"{folder}/_tmp/{file_prefix}-{n}-{g}-train.npz",
            )
            total_train_size += dump_size

        self.coalesce_files(folder, "_tmp", file_prefix)
        self.delete_temp_files(folder)
        return total_test_size, total_validation_size, total_train_size

    def retrieve_offsets_from_indices(self, n, g, idxs):
        # Find the corresponding indices for file and entry within file
        size_arr = np.array(
            [self.per_file_metadata[(i, n, g)] for i in self.reverse_map_file[(n, g)]]
        )
        search_arr = np.cumsum(size_arr)
        search_arr_idxs = search_arr.searchsorted(idxs, "right")
        list_idxs = idxs - search_arr[search_arr_idxs]
        file_idxs = np.array(self.reverse_map_file[(n, g)])[search_arr_idxs]
        return file_idxs, list_idxs

    def dump_to_file(
        self,
        n: int,
        g: int,
        file_idxs: np.ndarray,
        offset_idxs: np.ndarray,
        filename: str,
    ) -> None:
        """
        Helper function to dump each (n, g) dataset (either train, test, or validation).
        The function retrieves the actual data using the file_idxs and offset_idxs arguments.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Condense the file_idxs array into a smaller one without repititions.
        file_idxs, counts = np.unique(file_idxs, return_counts=True)
        if len(file_idxs) == 0:
            return 0
        counts = np.concatenate(([0], np.cumsum(counts)))
        offset_idxs = [
            offset_idxs[counts[i] : counts[i + 1]] for i in range(file_idxs.shape[0])
        ]
        output = {}
        for file_idx, offset_idx in zip(file_idxs, offset_idxs):
            data = self.data[file_idx]
            for keyword, dtype in zip(self.keywords, self.keyword_dtypes):
                keyword_data = data[f"{n}/{g}/{keyword}"]
                if keyword in output:
                    output[keyword] = np.concatenate(
                        (
                            output[keyword],
                            keyword_data[offset_idx].astype(dtype),
                        )
                    )
                else:
                    output[keyword] = keyword_data[offset_idx].astype(dtype)
        np.savez_compressed(filename, **output)
        return output["depth"].shape[0]

    @timeit
    def coalesce_files(self, folder: str, tmp_dir: str, file_prefix: str) -> None:
        """
        Helper function to merge all splits in `_tmp` folder into a a single big file.
        """
        test_files, train_files, validation_files = [], [], []
        for file in os.listdir(f"{folder}/{tmp_dir}"):
            if file.startswith(file_prefix):
                if file.endswith("train.npz"):
                    train_files.append(file)
                elif file.endswith("validation.npz"):
                    validation_files.append(file)
                elif file.endswith("test.npz"):
                    test_files.append(file)

        for dataset, file_list in zip(
            ["test", "train", "validation"], [test_files, train_files, validation_files]
        ):
            print(f"Coalescing {dataset} split")
            output = {}
            for file in file_list:
                data = np.load(f"{folder}/{tmp_dir}/{file}")
                n, g = file.split("/")[-1].split("-")[-3:-1]
                for keyword, dtype in zip(self.keywords, self.keyword_dtypes):
                    key = f"{n}/{g}/{keyword}"
                    if key not in output:
                        output[key] = data[keyword]
                    else:
                        output[f"{n}/{g}/{keyword}"] = np.concatenate(
                            (
                                output[f"{n}/{g}/{keyword}"],
                                data[keyword],
                            )
                        )
            for key in self.global_data:
                output[key] = self.global_data[key]
            np.savez_compressed(f"{folder}/{file_prefix}-{dataset}.npz", **output)

    def delete_temp_files(self, folder: str) -> None:
        print("Deleting temp folder")
        tmp_folder = f"{folder}/_tmp"
        if os.path.exists(tmp_folder) and os.path.isdir(tmp_folder):
            for root, dirs, files in os.walk(tmp_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(tmp_folder)

    def add_hash_from_file(self, file) -> None:
        with np.load(file) as data:
            for key in data.keys():
                if not key.endswith("observation"):
                    continue
                gate_key = f"{key.split('/')[0]}/{key.split('/')[1]}/gate"
                num_samples = data[gate_key].shape[0]
                self.hash_set.add(data[key].reshape(-1, num_samples))

    def _get_sample_size(
        self, ng_idxs: np.ndarray, num_batches: np.ndarray, q: int
    ) -> int:
        idx = ng_idxs.searchsorted(q)
        if idx >= len(ng_idxs) or ng_idxs[idx] != q:
            return 0
        return num_batches[idx] * self.batch_size


def calculate_hash(input):
    # For arrays stored in HDF5, casting to np.array is wasteful. Directly hash with the string.
    return hash(str(input))


if __name__ == "__main__":
    import time

    splitter = Splitter(
        [
            "training-data/big_data.npz",
        ]
    )
    print(f"Total size: {splitter.total_size}")
    splitter.set_batch_size(64)
    folder = "training-data/"
    prefix = "data"
    existing_splits = [
        fname
        for fname in os.listdir(folder)
        if fname in [f"{prefix}-train.npz", f"{prefix}-test.npz", f"{prefix}-validation.npz"]
    ]
    assert not existing_splits, f"Splits for {folder}/{prefix} already exist: {existing_splits}"
    test_size, validation_size, train_size = splitter.generate_split(folder, prefix)
    print(
        f"(Test, Validation, Train) size: ({test_size}, {validation_size}, {train_size})"
    )
