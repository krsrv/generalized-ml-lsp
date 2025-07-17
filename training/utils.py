"""
HDF5 file handling functions
"""

import h5py
import numpy as np

from models.input import GT_1Q, GT_2Q


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
            dset: h5py.Dataset = f[f"{key}/{k}"]
            old_size = dset.shape[0]
            if type(v) != np.ndarray:
                v = np.array(v)
            new_size = old_size + v.shape[0]
            # Necessary step, because the
            dset.resize((new_size, *dset.shape[1:]))
            try:
                if len(dset.shape) == 3:
                    dset[old_size:new_size, :, :] = v
                elif len(dset.shape) == 2:
                    dset[old_size:new_size, :] = v
            except Exception as e:
                print(f"Error occurred at (key, value) = ({k}, {v})")
                raise e
