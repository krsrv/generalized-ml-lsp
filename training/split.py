"""
Tools to split a given HDF5 file into training, test, validation (and potentially holdout)
"""

import os

import h5py
import numpy as np

from training.utils import prepare_hdf5_dataset, write_to_file


class Splitter:
    def __init__(self, files) -> None:
        self.files = []
        for file in files:
            if not file.endswith(".hdf5"):
                continue
            file = h5py.File(file, "r")
            self.files.append(file)

        self.batch_size = None
        self.test_split = 0.10
        self.validation_split = 0.10
        self.generate_metadata()
        self.hash_set = set()

    def generate_metadata(self) -> None:
        self.per_file_metadata = {}
        self.aggregate_metadata = {}
        self.reverse_map = {}
        total = 0
        for i, file in enumerate(self.files):
            for n in file.keys():
                for g in file[n].keys():
                    size = file[n][g]["n"].shape[0]
                    total += size

                    self.per_file_metadata[(i, int(n), int(g))] = size

                    if (int(n), int(g)) not in self.aggregate_metadata:
                        self.aggregate_metadata[(int(n), int(g))] = 0
                        self.reverse_map[(int(n), int(g))] = []
                    self.aggregate_metadata[(int(n), int(g))] += size
                    self.reverse_map[(int(n), int(g))].append(i)

                    # if total > 100000:
                    #     return

    def set_batch_size(self, batch_size) -> None:
        self.batch_size = batch_size

    def add_hash_from_hdf5(self, files) -> None:
        for file in files:
            with h5py.File(file, "r") as f:
                for n in f.keys():
                    for g in f.keys():
                        for o in f[n][g]["observation"]:
                            self.hash_set.add(calculate_hash(o))

    def generate_test_indices(self) -> int:
        """
        Among all (n, g) tuples, calculate the indices of (n, g) tuples to sample along with the
        number of corresponding samples.
        """
        assert self.batch_size is not None
        size_arr = np.array(
            [int(v // self.batch_size) for k, v in self.aggregate_metadata.items()]
        )
        total_batches = np.sum(size_arr)
        self.test_size = int(
            total_batches * self.test_split * self.batch_size
        )  # 10 * self.batch_size
        buff_idxs = np.sort(
            np.random.choice(
                np.arange(total_batches),
                self.test_size // self.batch_size,
                replace=False,
            )
        )
        search_arr = np.cumsum(size_arr)
        idxs = search_arr.searchsorted(buff_idxs, "right")
        self.test_idxs, self.test_idx_counts = np.unique(idxs, return_counts=True)
        return self.test_size

    def generate_test_split(self, folder: str, file_prefix: str):
        """
        Given the indices of (n, g) tuples to sample along with the number of corresponding samples
        (generated using `generate_test_indices`), calculate the exact physical indices
        corresponding to the HDF5 file and index within the (n, g) group. Once these are calculated,
        dump the contents to a new HDF5 file.
        """
        assert self.batch_size is not None
        output_file = f"{folder}/{file_prefix}-test.hdf5"
        self.create_output_file(output_file)
        keys = list(self.aggregate_metadata.keys())

        for idx, count in zip(self.test_idxs, self.test_idx_counts):
            n, g = keys[idx]
            total_samples = self.aggregate_metadata[(n, g)]
            # Generate the raw indices for sampling
            buff_idxs = np.sort(
                np.random.choice(
                    np.arange(total_samples), count * self.batch_size, replace=False
                )
            )

            # Find the corresponding indices for file and entry within file
            size_arr = np.array(
                [self.per_file_metadata[(i, n, g)] for i in self.reverse_map[(n, g)]]
            )
            search_arr = np.cumsum(size_arr)
            file_idxs = search_arr.searchsorted(buff_idxs, "right")
            list_idxs = buff_idxs - search_arr[file_idxs]

            # Dump the contents to file:
            self.dump_to_file(n, g, file_idxs, list_idxs, output_file, add_to_hash=True)

    def generate_train_validation_split(
        self, folder: str, file_prefix: str
    ) -> tuple[int, int]:
        """
        Given a list of examples to avoid (via hashes), dump the remaining data into train and
        validation HDF5 files probabilistically.
        """
        assert self.batch_size is not None
        train_file = f"{folder}/{file_prefix}-train.hdf5"
        validation_file = f"{folder}/{file_prefix}-validation.hdf5"
        self.create_output_file(train_file)
        self.create_output_file(validation_file)

        keys = list(self.aggregate_metadata.keys())
        train_count, validation_count = 0, 0

        for key in keys:
            n, g = key
            per_key_file_idxs = []
            per_key_list_idxs = []
            for i in self.reverse_map[(n, g)]:
                file = self.files[i][f"{n}/{g}"]
                all_idxs = np.arange(self.per_file_metadata[(i, n, g)])
                # Find all the indices per file which do not collide with the hash set.
                hashes = np.array(
                    [calculate_hash(x) for x in file["observation"][all_idxs]]
                )
                mask = ~np.isin(hashes, np.array(self.hash_set))
                indices = np.where(mask)[0]
                per_key_list_idxs.append(indices)
                per_key_file_idxs.append(np.ones_like(indices) * i)

            # Create the entire file and list idx
            per_key_file_idxs = np.concat(per_key_file_idxs)
            per_key_list_idxs = np.concat(per_key_list_idxs)

            # Randomly sample train and validation indices.
            per_key_count = per_key_file_idxs.shape[0]
            per_key_validation_size = int(per_key_count * self.validation_split)
            per_key_train_size = per_key_count - per_key_validation_size
            validation_idx = np.random.choice(
                np.arange(per_key_count),
                size=(per_key_validation_size,),
                replace=False,
            )

            # Dump contents to respective files
            self.dump_to_file(
                n,
                g,
                per_key_file_idxs[validation_idx],
                per_key_list_idxs[validation_idx],
                validation_file,
            )
            self.dump_to_file(
                n,
                g,
                per_key_file_idxs[~validation_idx],
                per_key_list_idxs[~validation_idx],
                train_file,
            )
            train_count += per_key_train_size
            validation_count += per_key_validation_size

        return train_count, validation_count

    def dump_to_file(
        self, n, g, file_idxs, list_idxs, output_file: str, add_to_hash: bool = False
    ) -> None:
        """
        Given a list of indices for files and entries, write the corresponding entries to the output
        file. The function assumes that the (n, g) dataset has not been created in the file.
        """
        prepare_hdf5_dataset(output_file, n, g)
        key = f"{n}/{g}"
        for file_idx, list_idx in zip(file_idxs, list_idxs):
            file = self.files[file_idx]
            dset: h5py.Group = file[key]
            dict_obj = {k: [dset[k][list_idx]] for k in dset.keys()}
            # print()
            if add_to_hash:
                self.hash_set.add(calculate_hash(dset["observation"][list_idx]))
            write_to_file(dict_obj, output_file, key)

    def create_output_file(self, file_name: str) -> None:
        if not os.path.exists(file_name):
            f = h5py.File(file_name, "w")
            f.close()
        else:
            print(f"Warning!!!!!!! File {file_name} already exists")
            os.remove(file_name)
            f = h5py.File(file_name, "w")
            f.close()


def calculate_hash(input):
    # For arrays stored in HDF5, casting to np.array is wasteful. Directly hash with the string.
    return hash(str(input))


def random_split(data_folder: str):
    splitter = Splitter(data_folder)
    splitter.generate_test_indices()
    splitter.generate_test_split()
    splitter.generate_train_validation_split()

    idxs = _generate_test_indices(metadata)
    hashes = _generate_test_split(idxs, files, output_file)
    """Create a list of indices which will correspond to the test set and create a test set based on it."""

    _generate_train_validation_split(hashes, files)
    """Simply iterate over the entire file list and randomly place each sample in train or validation."""
