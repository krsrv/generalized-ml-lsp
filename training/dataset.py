"""
Custom dataloader class for HDF5 files (where keys are `n/g/{key}`).
"""

import time
from collections import defaultdict
from functools import wraps

import numpy as np


def transform_graph(adjacency_matrix):
    """
    Transform an adjacency matrix to a laplacian matrix and return the eigenvalues, eigenvectors.
    Adjacency matrix should be of size (bs, n, n) where each (n, n) submatrix represents a
    different layout.
    """
    dim = len(adjacency_matrix.shape)
    assert dim == 2 or dim == 3
    if dim == 3:
        n = adjacency_matrix.shape[2]
        laplacian = np.array(adjacency_matrix, dtype=np.int32)
        diagonals = -np.sum(laplacian, axis=2)
        laplacian = laplacian + diagonals[:, None, :] * np.eye(n)[None, :, :]
        return np.linalg.eigh(laplacian)
    else:
        n = adjacency_matrix.shape[1]
        laplacian = np.array(adjacency_matrix, dtype=np.int32)
        diagonals = -np.diag(np.sum(laplacian, axis=1))
        laplacian = laplacian + diagonals
        return np.linalg.eigh(laplacian)


def _convert_to_bool(data: np.ndarray, n: int, old: bool = False):
    if old:
        return data
    """
    Convert int64 format of stabilizer to a bool format. The int64 format is defined in
    `tableau_xz31.hpp`. It stores a [X1 ... Xn Z1 ... Zn sign] bool vector as a uint64 `v`
    where (copied verbatim from `tableau_xz31.hpp`):
    # 1. Pauli is written as (-1)^{s} i^{a · b} X^{a} Z^{b}, where a,b are vectors of
    # length 31 with entries in {0,1}, s in {0, 1}.
    # 2.1. ((v >> 63) & 1) == s,
    # 2.2. ((v >> 31) & 1) == 0,
    # 2.3. ((v >> j) & 1) == b[j] for j in [0, 30],
    # 2.4. ((v >> (j + 32)) & 1) == a[j] for j in [0, 30].
    """
    bs, _ = data.shape
    new_data = np.zeros((bs, n, 2 * n + 1), dtype=np.bool_)
    for i in range(n):
        # Z stabilizer
        new_data[:, :, n + i] = (data >> i) & 1
        # X stabilizer
        new_data[:, :, i] = (data >> (32 + i)) & 1
    new_data[:, :, -1] = (data >> 63) & 1
    return new_data.reshape(bs, -1)


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


class ShardReader:
    def __init__(self, files: list[str]):
        super().__init__()
        self.files = files
        self.load_data()
        self.construct_metadata()
        self.batch_size = 64

    def load_data(self):
        self.data = {}
        for file in self.files:
            self.data[file] = np.load(file)

    def construct_metadata(self):
        """
        Store metadata about the data. Metadata includes:
        1. per_shard_aggregate_metadata - ith element stores a dictionary of (n, g) -> size
        2. size_list - list of sizes in each shard
        """
        # To support multiple files, look at the module training.split for inspiration.
        import os

        self.per_shard_aggregate_metadata = [None for _ in self.files]
        total = 0
        for i, file in enumerate(self.files):
            data = np.load(file)
            self.per_shard_aggregate_metadata[i] = {}
            for key in data:
                if not key.endswith("unprep_gate"):
                    continue
                size = data[key].shape[0]
                total += size
                n, g, _ = key.split("/")
                self.per_shard_aggregate_metadata[i][(int(n), int(g))] = size
            del data
        self.size_list = np.array(
            [w for v in self.per_shard_aggregate_metadata for w in v.values()]
        )

    def set_batch_size(self, batch_size):
        """
        Manually set the `batch_size` in the iter.
        """
        self.batch_size = batch_size

    def _init_shard_iter(self, shard_idx):
        """
        Load data in memory and reset iteration indices for shard corresponding to `shard_idx`
        """
        if hasattr(self, "loaded_data"):
            del self.loaded_data
        # Load the data in memory
        self.loaded_data = {}
        data = np.load(self.files[shard_idx])
        for key in data:
            self.loaded_data[key] = data[key]

        # Set the iteration elements
        # Set up bookmarks for:
        #  1. iteration order of (n, g) to go over for this shard
        #  2. iteration index
        #  3. starting offset for each (n, g)
        self.iter_idx = 0
        self.iter_ng_counter = {}
        self.iter_order = []
        for k, v in self.per_shard_aggregate_metadata[shard_idx].items():
            num_batches = int(v // self.batch_size)
            self.iter_ng_counter[k] = 0
            for i in range(num_batches):
                self.iter_order.append(k)
        np.random.shuffle(self.iter_order)

    def __iter__(self):
        self.shard_idx = 0
        self._init_shard_iter(self.shard_idx)
        return self

    def __next__(self):
        if self.iter_idx >= len(self.iter_order):
            self.shard_idx += 1
            # If we reach the end of all shards, we have finished iterating
            if self.shard_idx >= len(self.files):
                raise StopIteration
            # Else, move to next shard
            print("Reading new shard")
            self._init_shard_iter(self.shard_idx)

        ######################################################################
        # For iteration, we pick the current (n, g) value to sample, and pick
        # `batch_size` elements starting from the current offset for (n, g).
        # We are guaranteed that the starting offset won't exceed the number
        # of datapoints that exist for (n, g).
        ######################################################################
        # Extract the current (n, g) key
        k = self.iter_order[self.iter_idx]
        n, g = k

        # Return the data
        start_idx = self.iter_ng_counter[(n, g)]
        idxs = np.arange(start_idx, start_idx + self.batch_size)
        layout = self.loaded_data[f"{n}/{g}/layout"][idxs]
        eval, evec = transform_graph(layout)
        gate_qubits = self.loaded_data[f"{n}/{g}/gate_qubits"][idxs].reshape(layout.shape[0], -1)
        # Construct the input
        object = {
            "layout": layout,
            "eigval": eval,
            "eigvec": evec,
            "gates": self.loaded_data[f"{n}/{g}/gates"][idxs],
            "gate_qubits": gate_qubits,
            "observation": _convert_to_bool(self.loaded_data[f"{n}/{g}/observation"][idxs], n),
            "unprep_gate": self.loaded_data[f"{n}/{g}/unprep_gate"][idxs],
            "depth": self.loaded_data[f"{n}/{g}/depth"][idxs],
        }
        self.iter_ng_counter[(n, g)] += self.batch_size
        self.iter_idx += 1
        return object

    def get_total_size(self):
        return np.sum(self.size_list)


class UnprepNpzDataloader:
    """
    Class to handle npz files (where keys are `n/g/{key}`). The data is a collection of circuit
    layout, gate sets, input state and unpreparation unitary (just 1 gate). For now, the class
    supports handling only 1 file as input.
    TODO: Increase support to multiple files.
    """

    def __init__(self, file: str, shuffle: bool = True, old: bool = False, mload: bool = True):
        super().__init__()
        self.load_file(file, mload)
        self.old = old
        self.construct_metadata()
        self.shuffle = shuffle
        self.rng = np.random.default_rng()
        self.batch_size = 64

    @timeit
    def load_file(self, file: str, mload: bool = True):
        """
        Use only 1 file as input. Register the file as a member of the class.
        """
        data = np.load(file, "r")
        if mload:
            self.data = {}
            # Load in memory
            for key in data:
                self.data[key] = data[key]
        else:
            self.data = data

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
            if self.old:
                if not key.endswith("gate"):
                    continue
            else:
                if not key.endswith("unprep_gate"):
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
        gates: (g) = gate instances
        gate_qubits: (2*g) = qubits involved in the gate instances
        observation: (2 * n * n + n) = Observation of the target state
        unprep_gate: (1) = Gate index
        depth: (1) = Depth of the circuit

        Example for n = 3, layout = fully connected, gate set = {H=0, CNOT=1, CZ=2}
        layout: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        gates: [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
            Explanation: 3 Hadamard, 6 CNOT, 3 CZ gate instances
        gate_qubits: [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2, 1, 2, 1, 3, 2, 3, 2, 1, 3, 1, 3, 2]
            Explanation: Each gate has 2 qubits for its gate instance. The second qubit is 0 if
                it's a single qubit gate. The first 6 correspond to Hadamard, next 12 to CNOT and
                last 6 to CZ.
        observation: [[1,0,0,0,0,0,0], [0,0,0,0,1,0,0], [0,0,1,0,0,1,1]]
            Explanation: Each row is [X1, ..., Xn, Z1, ..., Zn, sign], with the leftmost column
            being the first qubit.
        gate: 10
        depth: 3
        """
        if self.iter_idx >= len(self.iter_order):
            raise StopIteration

        # Extract the current (n, g) key
        k, idxs = self.iter_order[self.iter_idx]
        # size = len(idxs)
        n, g = k

        # Return the data
        num_samples = (
            self.data[f"{n}/{g}/gate"].shape[0]
            if self.old
            else self.data[f"{n}/{g}/unprep_gate"].shape[0]
        )
        layout = self.data[f"{n}/{g}/layout"].reshape((num_samples, n, n))[idxs].astype(np.bool_)
        eval, evec = transform_graph(layout)
        if self.old:
            gates = self.data[f"{n}/{g}/gate_oh"].reshape((num_samples, -1))[idxs]
            gate_qubits = self.data[f"{n}/{g}/gate_qubit_oh"].reshape((num_samples, -1))[idxs]
        else:
            gates = self.data[f"{n}/{g}/gates"][idxs]
            gate_qubits = self.data[f"{n}/{g}/gate_qubits"][idxs].reshape(layout.shape[0], -1)
        # Construct the input
        object = {
            "layout": layout,
            "eigval": eval,
            "eigvec": evec,
            "gates": gates,
            "gate_qubits": gate_qubits,
            "observation": _convert_to_bool(
                self.data[f"{n}/{g}/observation"].reshape((num_samples, -1))[idxs], n, self.old
            ),
            # self.data[f"{n}/{g}/observation"],
            "unprep_gate": (
                self.data[f"{n}/{g}/gate"][idxs]
                if self.old
                else self.data[f"{n}/{g}/unprep_gate"][idxs]
            ),
            "depth": self.data[f"{n}/{g}/depth"][idxs],
        }
        self.iter_idx += 1
        return object

    def get_total_size(self):
        return np.sum(self.size_list)


class UnprepNpyDataloader:
    """
    Class to handle npy files (where file names are `n-g-{key}`). The data is a collection of circuit
    layout, gate sets, input state and unpreparation unitary (just 1 gate). For now, the class
    supports handling only 1 file as input.
    TODO: Increase support to multiple files.
    """

    def __init__(self, folder: str, shuffle: bool = True, old: bool = False):
        super().__init__()
        self.folder = folder
        self.old = old
        self.construct_metadata()
        self.shuffle = shuffle
        self.rng = np.random.default_rng()
        self.batch_size = 64

    def construct_metadata(self):
        """
        Store metadata about the data. Metadata includes:
        1. aggregate_metadata - reverse map of (n, g) -> size of dataset
        2. size_list - list of sizes of each (n, g) dataset
        3. ng_list - list of (n, g). Same order as size_list.
        """
        # To support multiple files, look at the module training.split for inspiration.
        import os

        self.aggregate_metadata = {}
        total = 0
        for file in os.listdir(self.folder):
            data = np.load(self.folder + "/" + file)
            if not file.endswith("unprep_gate.npy"):
                continue
            size = data.shape[0]
            total += size
            n, g, _ = file.split("-")
            self.aggregate_metadata[(int(n), int(g))] = size
            del data
        self.size_list = np.array([v for v in self.aggregate_metadata.values()])

    def set_batch_size(self, batch_size):
        """
        Manually set the `batch_size` in the iter.
        """
        self.batch_size = batch_size

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
        gates: (g) = gate instances
        gate_qubits: (2*g) = qubits involved in the gate instances
        observation: (2 * n * n + n) = Observation of the target state
        unprep_gate: (1) = Gate index
        depth: (1) = Depth of the circuit

        Example for n = 3, layout = fully connected, gate set = {H=0, CNOT=1, CZ=2}
        layout: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        gates: [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]
            Explanation: 3 Hadamard, 6 CNOT, 3 CZ gate instances
        gate_qubits: [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2, 1, 2, 1, 3, 2, 3, 2, 1, 3, 1, 3, 2]
            Explanation: Each gate has 2 qubits for its gate instance. The second qubit is 0 if
                it's a single qubit gate. The first 6 correspond to Hadamard, next 12 to CNOT and
                last 6 to CZ.
        observation: [[1,0,0,0,0,0,0], [0,0,0,0,1,0,0], [0,0,1,0,0,1,1]]
            Explanation: Each row is [X1, ..., Xn, Z1, ..., Zn, sign], with the leftmost column
            being the first qubit.
        gate: 10
        depth: 3
        """
        if self.iter_idx >= len(self.iter_order):
            raise StopIteration

        # Extract the current (n, g) key
        k, idxs = self.iter_order[self.iter_idx]
        # size = len(idxs)
        n, g = k

        # Return the data
        # unprep_gate = self.data[
        #     f"{n}/{g}/unprep-gate.npy"
        # ]
        unprep_gate = np.load(f"{self.folder}/{n}-{g}-unprep_gate.npy")
        num_samples = unprep_gate.shape[0]
        layout = (
            np.load(f"{self.folder}/{n}-{g}-layout.npy")
            .reshape((num_samples, n, n))[idxs]
            .astype(np.bool_)
        )
        eval, evec = transform_graph(layout)
        if self.old:
            gates = self.data[f"{n}/{g}/gate_oh"].reshape((num_samples, -1))[idxs]
            gate_qubits = self.data[f"{n}/{g}/gate_qubit_oh"].reshape((num_samples, -1))[idxs]
        else:
            gates = np.load(f"{self.folder}/{n}-{g}-gates.npy").reshape((num_samples, -1))[idxs]
            gate_qubits = np.load(f"{self.folder}/{n}-{g}-gate_qubits.npy").reshape(
                (num_samples, -1)
            )[idxs]
        # Construct the input
        object = {
            "layout": layout,
            "eigval": eval,
            "eigvec": evec,
            "gates": gates,
            "gate_qubits": gate_qubits,
            "observation": np.load(f"{self.folder}/{n}-{g}-observation.npy"),  # _convert_to_bool(
            #     self.data[f"{n}/{g}/observation"].reshape((num_samples, -1))[idxs, :], n, self.old
            # ),
            "unprep_gate": unprep_gate[idxs],
            "depth": np.load(f"{self.folder}/{n}-{g}-depth.npy")[idxs],
        }
        self.iter_idx += 1
        return object

    def get_total_size(self):
        return np.sum(self.size_list)
