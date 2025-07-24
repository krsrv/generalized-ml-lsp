import os

import h5py
import numpy as np

from training.utils import prepare_hdf5_dataset, write_to_file


def new_dump_object() -> dict:
    """
    Generate the leaf object to dump in the HDF5 file.
    """
    return {
        "n": [],
        "layout": [],
        "gate_oh": [],
        "gate_qubit_oh": [],
        "depth": [],
        "gate": [],
        "observation": [],
    }


class UnprepDataExtractor:
    """
    Given an HDF5 file with keys `n/g/d/{key}`, which corresponds to a (gate sequence, state)
    data, pull out the last gate to create an unpreparation dataset (gate sequence[-1], state).
    """

    def __init__(self, file: str) -> None:
        self.file = h5py.File(file, "r")

    def extract_to(self, output_file: str):
        if os.path.exists(output_file):
            os.remove(output_file)
        # Generate the h5py file
        file = h5py.File(output_file, "w")
        file.close()

        for n in self.file:
            for g in self.file[n]:
                key = f"{n}/{g}"
                prepare_hdf5_dataset(output_file, int(n), int(g))
                for d in self.file[n][g]:
                    dump_object = new_dump_object()
                    data = self.file[n][g][d]
                    for datakey in data:
                        if datakey == "gates":
                            dump_object["gate"] = data[datakey][:, -1]
                        else:
                            dump_object[datakey] = data[datakey]
                    write_to_file(dump_object, output_file, key)


if __name__ == "__main__":
    import time

    input_files = [
        "training-data/overfit_4.hdf5",
        # "training-data/tired-1.hdf5",
        # "training-data/tired-2.hdf5",
        # "training-data/tired-3.hdf5",
    ]
    output_files = [
        "training-data/compiled/overfit_train.hdf5",
        # "training-data/compiled/extracted-1.hdf5",
        # "training-data/compiled/extracted-2.hdf5",
        # "training-data/compiled/extracted-3.hdf5",
    ]
    for input_file, output_file in zip(input_files, output_files):
        tic = time.time()
        extractor = UnprepDataExtractor(input_file)
        extractor.extract_to(output_file)
        toc = time.time()
        print(f"Converted {input_file} -> {output_file} ({toc-tic} sec)")
