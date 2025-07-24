import h5py
import numpy as np

from models.input import GT_1Q, GT_2Q

"""
Utility functions for creating gate embeddings
"""


def construct_gate_embeddings(
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


def construct_gate_qubit_embeddings(
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


"""
HDF5 file handling functions
"""


def prepare_hdf5_dataset(output_file: str, n: int, g: int) -> None:
    """
    Sets up the HDF5 file for dumping contents. Pre-specifying expected feature dimensions makes the
    module much faster.

    * Assumes that the HDF5 file already exists, and there are no conflicting dataset names.
    * The keys are {key}/{n, layout, gate_oh, gate_qubit_oh, depth, observation}
    * The `maxshape` argument is set to None, which means that the file can be extended infinitely.
    * The `chunk` argument is set to True, which means that the contents will be written in chunks
    ideally.

    Args:
        output_file: the full path to the file, including the ".hdf5" extension
        n, g: number of qubits and gate instances
    """
    key = f"{n}/{g}"
    gate_oh_size = len(GT_1Q) + len(GT_2Q)
    with h5py.File(output_file, "a") as f:
        f.create_dataset(
            f"{key}/n", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/layout",
            shape=(0, n, n),
            maxshape=(None, n, n),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/gate_oh",
            shape=(0, g, gate_oh_size),
            maxshape=(None, g, gate_oh_size),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/gate_qubit_oh",
            shape=(0, g, 2 * n),
            maxshape=(None, g, 2 * n),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/depth", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/gate", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/observation",
            shape=(0, 2 * n * n + n),
            maxshape=(None, 2 * n * n + n),
            dtype="bool",
            chunks=True,
        )


def write_to_file(dict_obj: dict, output_file: str, key: str) -> None:
    """
    Write given dictionary object to HDF5 file, with corresponding keys given by "{key}/{dict key}".
    The keys should have ideally have been initialized using the `prepare_hdf5_dataset` function.
    This function is meant for writing to the HDF5 file in chunks.
    """
    with h5py.File(output_file, "a") as f:
        for k, v in dict_obj.items():
            try:
                dset: h5py.Dataset = f[f"{key}/{k}"]
                old_size = dset.shape[0]
                if type(v) != np.ndarray:
                    v = np.array(v)
                new_size = old_size + v.shape[0]
                # Necessary step, because the
                dset.resize((new_size, *dset.shape[1:]))
                if len(dset.shape) == 3:
                    dset[old_size:new_size, :, :] = v
                elif len(dset.shape) == 2:
                    dset[old_size:new_size, :] = v
                elif len(dset.shape) == 1:
                    if len(v.shape) == 1:
                        dset[old_size:new_size] = v
                    elif len(v.shape) == 2:
                        dset[old_size:new_size] = v[:, 0]
            except Exception as e:
                print(f"Error occurred at (n, g)/(key, value) = {key}/({k}, {v})")
                raise e
