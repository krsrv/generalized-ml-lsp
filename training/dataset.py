"""
Custom dataloader class for HDF5 files (where keys are `n/g/{key}`).
"""

import time
from collections import defaultdict
from functools import wraps

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


class UnprepNpzDataloader:
    """
    Class to handle HDF5 files (where keys are `n/g/{key}`). The data is a collection of circuit
    layout, gate sets, input state and unpreparation unitary (just 1 gate). For now, the class
    supports handling only 1 file as input.
    TODO: Increase support to multiple files.
    """

    def __init__(self, file: str, shuffle: bool = True):
        super().__init__()
        self.load_file(file)
        self.construct_metadata()
        self.shuffle = shuffle
        self.batch_size = 64

    @timeit
    def load_file(self, file):
        """
        Use only 1 file as input. Register the file as a member of the class.
        """
        self.data = np.load(file, "r")
        self.cache = defaultdict(dict)

        for key in self.data:
            parts = key.split("/")
            if len(parts) == 3:
                n, g, d = parts
                group_key = f"{n}/{g}"
                self.cache[group_key][d] = self.data[key]  # loads into memory

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
        data = self.data
        for key in data.keys():
            if not key.endswith("gate"):
                continue
            size = data[key].shape[0]
            total += size
            n, g, _ = key.split("/")
            self.aggregate_metadata[(int(n), int(g))] = size
        self.size_list = np.array([v for v in self.aggregate_metadata.values()])
        self.p = self.size_list / np.sum(self.size_list)

        self.ng_list = list(self.aggregate_metadata.keys())

    def set_batch_size(self, batch_size):
        """
        Manually set the `batch_size` in the iter.
        """
        self.batch_size = batch_size

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

    def __iter__(self):
        self.iter_idx = 0
        # Set the iteration order
        self.iter_order = []
        for k, v in self.aggregate_metadata.items():
            n, g = k
            num_batches = int(v // self.batch_size)
            if num_batches == 0:
                continue
            idxs = np.arange(v)
            if self.shuffle:
                np.random.shuffle(idxs)
            for i in range(num_batches):
                self.iter_order.append(
                    (k, np.sort(idxs[i * self.batch_size : (i + 1) * self.batch_size]))
                )
        if self.shuffle:
            np.random.shuffle(self.iter_order)
        # print(self.iter_order[:5])
        return self

    # @timeit
    def __next__(self):
        """
        Return the next element in iter. The returned batch might not have `batch_size` elements, if
        there are fewer than `batch_size` elements remaining in the (n, g) dataset.

        Description of the data (for num_samples = 1):
        layout: (n, n) = Adjacency matrix
        eigval: (n) = Eigenvalues of the Laplacian matrix
        eigvec: (n, n) = Eigenvectors of the Laplacian matrix
        gate_oh: (g) = gate instances
        gate_qubit_oh: (2*g) = qubits involved in the gate instances
        observation: (2 * n * n + n) = Observation of the target state
        gate: (1) = Gate index
        depth: (1) = Depth of the circuit

        Example for n = 3, layout = fully connected, gate set = {H=0, CNOT=1, CZ=2}
        layout: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        gate_oh: [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
            Explanation: 3 Hadamard, 6 CNOT, 3 CZ gate instances
        gate_qubit_oh: [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2, 1, 2, 1, 3, 2, 3, 2, 1, 3, 1, 3, 2]
            Explanation: Each gate has 2 qubits for its gate instance. The second qubit is 0 if
                it's a single qubit gate. The first 6 correspond to Hadamard, next 12 to CNOT and
                last 6 to CZ.
        observation: [[1,0,0,0,0,0,0], [0,0,0,0,1,0,0], [0,0,1,0,0,1,1]]
            Explanation: Each row is [X, Z, sign], in reverse ordering of qubits, i.e., right most
            column is the first qubit.
        gate: 10
        depth: 3
        """
        if self.iter_idx >= len(self.iter_order):
            raise StopIteration

        # Extract the current (n, g) key
        k, idxs = self.iter_order[self.iter_idx]
        n, g = k

        # Return the data
        data = self.cache[f"{n}/{g}"]
        num_samples = data[f"gate"].shape[0]
        eval, evec = _transform_graph(
            data[f"layout"].reshape((num_samples, n, n))[idxs, :, :]
        )
        object = {
            "layout": data[f"layout"].reshape((num_samples, n, n))[idxs, :, :],
            "eigval": eval,
            "eigvec": evec,
            "gate_oh": data[f"gate_oh"].reshape((num_samples, -1))[idxs, :],
            "gate_qubit_oh": data[f"gate_qubit_oh"].reshape((num_samples, -1))[idxs, :],
            "observation": data[f"observation"].reshape((num_samples, -1))[idxs, :],
            "gate": data[f"gate"][idxs],
            "depth": data[f"depth"][idxs],
        }
        self.iter_idx += 1
        return object

    def get_total_size(self):
        return np.sum(self.size_list)
