"""
Given ordered data generated through the training.generate module, transform into a batchable
input where each file has random circuits parameterized by the number of qubits and the number of
gates applicable in the circuit.
TODO: Change batching dataset input file to pass the gates appropriately given the new input.
"""

import os
import pickle
from collections import Counter

import h5py
import numpy as np
from torch.utils.data.dataset import Dataset

from models.input import GT_1Q, GT_2Q
from training.dataset import TrainingInstance

"""
Utility functions for creating gate embeddings
"""


def _construct_gate_embeddings(
    gt_1q: np.ndarray, gt_2q: np.ndarray, adjacency_matrix: np.ndarray
):
    num_classes = len(GT_1Q) + len(GT_2Q)
    gt_1q_oh = np.eye(num_classes, dtype=np.bool_)[gt_1q - 1]  # Shape: (G1, G)
    n = adjacency_matrix.shape[-1]
    gt_1q_oh = np.repeat(gt_1q_oh, n, axis=-2)  # Shape: (G1 * n, G)

    gt_2q_oh = np.eye(num_classes, dtype=np.bool_)[
        gt_2q - 1 + len(GT_1Q)
    ]  # Shape: (G2, G)
    edges = np.count_nonzero(adjacency_matrix)
    gt_2q_oh = np.repeat(gt_2q_oh, edges, axis=-2)  # Shape: (G2 * E, G)

    return np.concat((gt_1q_oh, gt_2q_oh), axis=-2)


def _construct_gate_qubit_embeddings(
    gt_1q: np.ndarray, gt_2q: np.ndarray, adjacency_matrix: np.ndarray
):
    n = adjacency_matrix.shape[-1]
    qbit_gt_1q = np.pad(
        np.diag(np.ones(n, dtype=np.bool_)), ((0, 0), (0, n))
    )  # Shape: (n, 2n)
    qbit_gt_1q = np.tile(qbit_gt_1q, (gt_1q.shape[-1], 1))

    ctrl, tgt = np.nonzero(adjacency_matrix)
    qbit_gt_2q = np.concat(
        (np.eye(n, dtype=np.bool_)[ctrl], np.eye(n, dtype=np.bool_)[tgt]), axis=-1
    )
    qbit_gt_2q = np.tile(qbit_gt_2q, (gt_2q.shape[-1], 1))
    return np.concat((qbit_gt_1q, qbit_gt_2q), axis=-2)


def new_dump_object() -> dict:
    return {
        "n": [],
        "layout": [],
        "gate_oh": [],
        "gate_qubit_oh": [],
        "depth": [],
        "gate": [],
        "observation": [],
    }


"""
Class definitions
"""


class RawData(Dataset):
    def __init__(self, file_index_list):
        super().__init__()
        self.file_index_list = file_index_list

    def __len__(self):
        return len(self.file_index_list)

    def __getitem__(self, index) -> TrainingInstance:
        file, idx = self.file_index_list[index]
        with open(f"{file}", "rb") as f:
            data_list: list[TrainingInstance] = pickle.load(f)
        data: TrainingInstance = data_list[idx]
        return data


"""
HDF5 file handling functions
"""


def prepare_hdf5_dataset(output_file, n, g, size):
    key = f"{n}/{g}"
    gate_oh_size = len(GT_1Q) + len(GT_2Q)
    with h5py.File(output_file, "a") as f:
        f.create_dataset(
            f"{key}/n", shape=(0,), maxshape=(size,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/layout",
            shape=(0, n, n),
            maxshape=(size, n, n),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/gate_oh",
            shape=(0, g, gate_oh_size),
            maxshape=(size, g, gate_oh_size),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/gate_qubit_oh",
            shape=(0, g, 2 * n),
            maxshape=(size, g, 2 * n),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/depth", shape=(0,), maxshape=(size,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/gate", shape=(0,), maxshape=(size,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/observation",
            shape=(0, 2 * n * n + n),
            maxshape=(size, 2 * n * n + n),
            dtype="bool",
            chunks=True,
        )


def write_to_file(dict_obj, output_file, key, size):
    with h5py.File(output_file, "a") as f:
        for k, v in dict_obj.items():
            dset: h5py.Dataset = f[f"{key}/{k}"]
            old_size = dset.shape[0]
            if type(v) != np.ndarray:
                v = np.array(v)
            new_size = old_size + v.shape[0]
            # dset.resize((new_size, *dset.shape[1:]))
            try:
                if len(dset.shape) == 3:
                    dset[old_size:new_size, :, :] = v
                elif len(dset.shape) == 2:
                    dset[old_size:new_size, :] = v
            except Exception as e:
                print(k, v)
                raise e


"""
Main functions
"""


def read_raw_data(folders):
    """
    Read .pkl files which store detailed dump information of randomly created circuits
    """
    n_g_training_map = dict()
    # n_g_testing_map = dict()
    for folder in folders:
        folder = f"{folder}"
        for i, file in enumerate(os.listdir(folder)):
            try:
                with open(f"{folder}/{file}", "rb") as f:
                    data_list: list[TrainingInstance] = pickle.load(f)
            except Exception as e:
                print(f"ERROR processing {folder}/{file}: Exception {e}")
            for j, data in enumerate(data_list):
                n_gates = len(data.gate_set_2q) * len(
                    data.layout.adjacency_list
                ) + data.n * len(data.gate_set_1q)
                if (data.n, n_gates) not in n_g_training_map.keys():
                    n_g_training_map[(data.n, n_gates)] = []
                n_g_training_map[(data.n, n_gates)].append((f"{folder}/{file}", j))
            if i % 100 == 0:
                print(f"Read {i} files in {folder}")
    return n_g_training_map


def create_h5_file(map_data, folder, file):
    output_file = f"{folder}/{file}"
    if not os.path.exists(output_file):
        f = h5py.File(output_file, "w")
        f.close()
    else:
        os.remove(output_file)
        f = h5py.File(output_file, "w")
        f.close()

    for k, v in map_data.items():
        n, g = k
        ng_dataset = RawData(v)
        size = len(ng_dataset)
        print(f"(n, g) = {k}, {size} entries")

        dump_object = new_dump_object()
        key = f"{n}/{g}"

        prepare_hdf5_dataset(output_file, n, g, size)
        for i in range(size):
            data = ng_dataset[i]
            dump_object["n"].append(data.n)

            laplacian = data.layout.graph.numpy()
            adjacency = laplacian - np.diag(np.diag(laplacian))
            adjacency = np.array(adjacency, dtype=np.bool_)
            dump_object["layout"].append(adjacency)

            gt_1q = np.array([x.value for x in data.gate_set_1q], dtype=np.int32)
            gt_2q = np.array([x.value for x in data.gate_set_2q], dtype=np.int32)
            dump_object["gate_oh"].append(
                _construct_gate_embeddings(gt_1q, gt_2q, adjacency)
            )
            dump_object["gate_qubit_oh"].append(
                _construct_gate_qubit_embeddings(gt_1q, gt_2q, adjacency)
            )

            dump_object["depth"].append(data.circuit_depth)
            dump_object["gate"].append(np.array(data.gates[-1:], dtype=np.int64))

            observation = data.observation
            observation = np.array(observation, dtype=np.bool_)
            dump_object["observation"].append(observation)

            if i % 100 == 0:
                write_to_file(dump_object, output_file, key, 100)
                dump_object = new_dump_object()

        write_to_file(dump_object, output_file, key, size % 100)


if __name__ == "__main__":
    import multiprocessing

    def worker(data, output_folder, file):
        create_h5_file(data, output_folder, file)

    folders = [
        "/Users/tport/Desktop/USC/Semesters/Projects/LSP/sllsp/training-data/main-25-07-07-2",
        # "/scratch1/sauravk/lsp-raw-training-data"
    ]
    ng_data = read_raw_data(folders)

    max_processes = 1
    num_processes = min(multiprocessing.cpu_count(), max_processes)
    output_folder = (
        "/Users/tport/Desktop/USC/Semesters/Projects/LSP/sllsp/training-data/compiled"
        # "/scratch1/sauravk/lsp-training-data"
    )
    processes = []
    item_list = list(ng_data.items())
    if num_processes > 1:
        for i in range(num_processes):
            start = i * int(len(ng_data) // num_processes)
            end = (
                (i + 1) * int(len(ng_data) // num_processes)
                if i < num_processes - 1
                else len(ng_data)
            )
            temp_dict = {x[0]: x[1] for x in item_list[start:end]}
            p = multiprocessing.Process(
                target=worker, args=(temp_dict, output_folder, f"foo-{i}.hdf5")
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        import time

        tic = time.time()
        worker(ng_data, output_folder, "25-07-07-2.hdf5")
        print("Time elapsed:", time.time() - tic)
