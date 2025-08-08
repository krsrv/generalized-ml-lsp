"""
Tools to split a given HDF5 file into training, test, validation (and potentially holdout)
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
    keywords = ["gate", "depth", "observation", "layout", "gate_oh", "gate_qubit_oh"]
    keyword_dtypes = [np.int16, np.int16, np.bool_, np.bool_, np.int8, np.int8]

    def __init__(self, files) -> None:
        self.data = []
        for file in files:
            if not file.endswith(".npz"):
                continue
            data = np.load(file)
            self.data.append(data)

        self.batch_size = None
        self.test_split = 0.10
        self.validation_split = 0.10
        self.generate_metadata()
        self.hash_set = set()

    def generate_metadata(self) -> None:
        self.aggregate_metadata = {}  # (n, g) -> size
        self.per_file_metadata = {}  # (file_index, n, g) -> size
        self.reverse_map_file = {}  # (n, g) -> [file_indices]
        self.total_size = 0
        for i, data in enumerate(self.data):
            for j, key in enumerate(data.keys()):
                # The key should be of the form n/g/{data}
                assert len(key.split("/")) == 3
                n, g, keyword = key.split("/")

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
        self.aggregate_num_batches = self.aggregate_num_batches[
            np.nonzero(self.aggregate_num_batches)
        ]
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
            np.random.choice(
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
        np.random.shuffle(shuffled_indices)
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
                print(f"Skipping {n}/{g}: #examples < batch size")
                continue
            print(f"Running {n}/{g} ({idx})")

            shuffled_indices = np.arange(self.aggregate_metadata[key])
            np.random.shuffle(shuffled_indices)

            # Create test data.
            test_size = self._get_sample_size(test_ng_idxs, test_ng_num_batches, idx)
            file_idxs, offset_idxs = self.retrieve_offsets_from_indices(
                n, g, np.sort(shuffled_indices[:test_size])
            )
            self.dump_to_file(
                n,
                g,
                file_idxs,
                offset_idxs,
                f"{folder}/tmp/{file_prefix}-{n}-{g}-test.npz",
            )
            total_test_size += test_size

            # Create validation data.
            validation_size = self._get_sample_size(
                validation_ng_idxs, validation_ng_num_batches, idx
            )
            file_idxs, offset_idxs = self.retrieve_offsets_from_indices(
                n, g, np.sort(shuffled_indices[test_size : test_size + validation_size])
            )
            self.dump_to_file(
                n,
                g,
                file_idxs,
                offset_idxs,
                f"{folder}/tmp/{file_prefix}-{n}-{g}-validation.npz",
            )
            total_validation_size += validation_size

            # Create train data.
            train_size = self._get_sample_size(train_ng_idxs, train_ng_num_batches, idx)
            file_idxs, offset_idxs = self.retrieve_offsets_from_indices(
                n, g, np.sort(shuffled_indices[test_size + validation_size :])
            )
            self.dump_to_file(
                n,
                g,
                file_idxs,
                offset_idxs,
                f"{folder}/tmp/{file_prefix}-{n}-{g}-train.npz",
            )
            total_train_size += train_size

        self.coalesce_files(folder, "tmp", file_prefix)
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
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        output = {
            keyword: np.array([], dtype=dtype)
            for keyword, dtype in zip(self.keywords, self.keyword_dtypes)
        }

        # Condense the file_idxs array into a smaller one without repitions.
        file_idxs, counts = np.unique(file_idxs, return_counts=True)
        counts = np.concatenate(([0], np.cumsum(counts)))
        offset_idxs = [
            offset_idxs[counts[i] : counts[i + 1]] for i in range(file_idxs.shape[0])
        ]
        for file_idx, offset_idx in zip(file_idxs, offset_idxs):
            data = self.data[file_idx]
            for keyword, dtype in zip(self.keywords, self.keyword_dtypes):
                keyword_data = data[f"{n}/{g}/{keyword}"]
                size = data[f"{n}/{g}/gate"].shape[0]
                keyword_data = keyword_data.reshape((size, -1))
                output[keyword] = np.concatenate(
                    (
                        output[keyword],
                        keyword_data[offset_idx, :].reshape(-1).astype(dtype),
                    )
                )
        np.savez_compressed(filename, **output)

    @timeit
    def coalesce_files(self, folder: str, tmp_dir: str, file_prefix: str) -> None:
        test_files, train_files, validation_files = [], [], []
        for file in os.listdir(f"{folder}/{tmp_dir}"):
            if file.startswith(file_prefix) and len(file.split("-")) == 4:
                if file.endswith("train.npz"):
                    train_files.append(file)
                elif file.endswith("validation.npz"):
                    validation_files.append(file)
                elif file.endswith("test.npz"):
                    test_files.append(file)
        for dataset, file_list in zip(
            ["test", "train", "validation"], [test_files, train_files, validation_files]
        ):
            output = {}
            for file in file_list:
                data = np.load(f"{folder}/{tmp_dir}/{file}")
                n, g = file.split("/")[-1].split("-")[1:3]
                for keyword, dtype in zip(self.keywords, self.keyword_dtypes):
                    key = f"{n}/{g}/{keyword}"
                    if key not in output:
                        output[key] = np.array([], dtype=dtype)
                    output[f"{n}/{g}/{keyword}"] = np.concatenate(
                        (
                            output[f"{n}/{g}/{keyword}"],
                            data[keyword],
                        )
                    )
            np.savez_compressed(f"{folder}/{file_prefix}-{dataset}.npz", **output)

    def delete_temp_files(self, folder: str) -> None:
        tmp_folder = f"{folder}/tmp"
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

    np.random.seed(1)

    splitter = Splitter(
        [
            "training-data/compiled/2-10_20000.npz",
            "training-data/compiled/11-14_20000.npz",
            "training-data/compiled/15-18_20000.npz",
            "training-data/compiled/19-20_20000.npz",
        ]
    )
    print(f"Total size: {splitter.total_size}")
    splitter.set_batch_size(64)
    prefix = "new-sample"
    test_size, validation_size, train_size = splitter.generate_split(
        "training-data/compiled", prefix
    )
    print(
        f"(Test, Validation, Train) size: ({test_size}, {validation_size}, {train_size})"
    )
