"""
Custom dataloader class for HDF5 files (where keys are `n/g/{key}`).
"""

import h5py
import numpy as np


def _transform_graph(adjacency_matrix):
    """
    Transform an adjacency matrix to a laplacian matrix and return the eigenvalues, eigenvectors
    """
    n = adjacency_matrix.shape[2]
    laplacian = np.array(adjacency_matrix, dtype=np.int32)
    diagonals = -np.sum(laplacian, axis=2)
    laplacian = laplacian + diagonals[:, None, :] * np.eye(n)[None, :, :]
    return np.linalg.eigh(laplacian)


class UnprepHdf5Dataloader:
    """
    Class to handle HDF5 files (where keys are `n/g/{key}`). The data is a collection of circuit
    layout, gate sets, input state and unpreparation unitary (just 1 gate). For now, the class
    supports handling only 1 file as input.
    TODO: Increase support to multiple files.
    """

    def __init__(self, file: str):
        super().__init__()
        self.load_files(file)
        self.construct_metadata()

    def load_files(self, file):
        """
        Use only 1 file as input. Register the file as a member of the class.
        """
        self.file = h5py.File(file, "r")

    def construct_metadata(self):
        """
        Store metadata about the data. Metadata includes:
        1. aggregate_metadata - reverse map of (n, g) -> size of dataset
        2. size_list - list of sizes of each (n, g) dataset
        3. ng_list - list of (n, g). Same order as size_list.
        """
        # To support multiple files, look at the module training.split for inspiration.
        self.aggregate_metadata = {}
        self.reverse_map = {}
        total = 0
        file = self.file
        for n in file.keys():
            for g in file[n].keys():
                size = file[n][g]["n"].shape[0]
                total += size
                self.aggregate_metadata[(int(n), int(g))] = size
        self.size_list = np.array([v for v in self.aggregate_metadata.values()])
        self.p = self.size_list / np.sum(self.size_list)

        self.ng_list = list(self.aggregate_metadata.keys())

    def random_sample_ng(self, batch_size=64):
        """
        Randomly sample (n, g) pair. If the `batch_size` is greater than the number of data points
        available, resample (n, g).
        The sampling is done such that each data point is picked uniformly at random at the lowest
        data point level, and not at the (n, g) level.
        """
        size = 0
        while size < batch_size:
            idx = np.random.choice(np.arange(len(self.aggregate_metadata)), p=self.p)
            n, g = self.ng_list[idx]
            size = self.file[f"{n}/{g}"]["layout"].shape[0]
        return self.ng_list[idx]

    def random_sample_data(self, n, g, batch_size=64):
        """
        Given (n, g), randomly sample the datapoints under (n, g) according to the batch_size.
        """
        data: h5py.Group = self.file[f"{n}/{g}"]
        n_samples = data["layout"].shape[0]
        assert (
            n_samples >= batch_size
        ), f"Number of training samples ({n_samples}) is less than the batch size ({batch_size})"
        idxs = np.sort(
            np.random.choice(np.arange(0, n_samples), batch_size, replace=False)
        )
        eval, evec = _transform_graph(data["layout"][idxs, :, :])
        return {
            "eigval": eval,
            "eigvec": evec,
            "gate_oh": data["gate_oh"][idxs, :, :],
            "gate_qubit_oh": data["gate_qubit_oh"][idxs, :, :],
            "observation": data["observation"][idxs, :],
            "gate": data["gate"][idxs],
            "depth": data["depth"][idxs],
        }

    def __iter__(self):
        self.ng_iter_idx = 0
        self.batch_idx = 0
        self.batch_size = 64
        return self

    def set_ng_iter_idx(self, ng_idx):
        """
        Manually set the (n, g) index in the iter.
        """
        self.ng_iter_idx = ng_idx

    def set_batch_size(self, batch_size):
        """
        Manually set the `batch_size` in the iter.
        """
        self.batch_size = batch_size

    def __next__(self):
        """
        Return the next element in iter. The returned batch might not have `batch_size` elements, if
        there are fewer than `batch_size` elements remaining in the (n, g) dataset.
        """
        if self.ng_iter_idx >= len(self.ng_list):
            raise StopIteration

        # Extract the current (n, g) key
        n, g = self.ng_list[self.ng_iter_idx]
        max_size = self.aggregate_metadata[(n, g)]
        # Avoid iterating over empty datasets.
        while max_size == 0:
            self.ng_iter_idx += 1
            if self.ng_iter_idx >= len(self.ng_list):
                raise StopIteration
            n, g = self.ng_list[self.ng_iter_idx]
            max_size = self.aggregate_metadata[(n, g)]

        # Set the start and end indices
        if self.batch_idx + self.batch_size >= max_size:
            start_idx, end_idx = self.batch_idx, max_size
            self.batch_idx = 0
            self.ng_iter_idx += 1
        else:
            start_idx, end_idx = self.batch_idx, self.batch_idx + self.batch_size
            self.batch_idx = end_idx

        # Return the data
        data = self.file[f"{n}/{g}"]
        eval, evec = _transform_graph(data["layout"][start_idx:end_idx, :, :])
        object = {
            "eigval": eval,
            "eigvec": evec,
            "gate_oh": data["gate_oh"][start_idx:end_idx, :, :],
            "gate_qubit_oh": data["gate_qubit_oh"][start_idx:end_idx, :, :],
            "observation": data["observation"][start_idx:end_idx, :],
            "gate": data["gate"][start_idx:end_idx],
            "depth": data["depth"][start_idx:end_idx],
        }
        return object

    def get_total_size(self):
        return np.sum(self.size_list)

    def __len__(self):
        return len(self.ng_list)

    def __getitem__(self, index):
        """
        Assumes `index` is of the form (n, g, offset)
        """
        n, g, index = index
        data = self.files[f"{n}/{g}"]
        return data["layout"][index, :, :]
