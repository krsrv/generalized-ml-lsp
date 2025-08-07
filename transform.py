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
        self.data = np.load(file)

    def extract_to(self, output_file: str):
        # Ensure the output directory exists before writing the file
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        print(output_dir)
        if os.path.exists(output_file):
            os.remove(output_file)

        keys = list(self.data.keys())
        ng_set = set(x.split("/")[1] + "/" + x.split("/")[2] for x in keys)
        ng_dict = {}
        for key in ng_set:
            ng_dict[key + "/gate"] = np.array([], dtype=np.int_)
            ng_dict[key + "/depth"] = []
            ng_dict[key + "/observation"] = []
            ng_dict[key + "/layout"] = []
            ng_dict[key + "/gate_oh"] = []
            ng_dict[key + "/gate_qubit_oh"] = []
        for key in keys:
            ng = key.split("/")[1] + "/" + key.split("/")[2]
            case_key = key.split("/")[4]
            if case_key == "gate":
                ng_dict[ng + "/gate"] = np.concatenate(
                    (ng_dict[ng + "/gate"], np.array(self.data[key]))
                )
            elif case_key == "depth":
                ng_dict[ng + "/depth"] = np.concatenate(
                    (ng_dict[ng + "/depth"], np.array(self.data[key]))
                )
            elif case_key == "observation":
                ng_dict[ng + "/observation"] = np.concatenate(
                    (ng_dict[ng + "/observation"], np.array(self.data[key]))
                )
            elif case_key == "layout":
                ng_dict[ng + "/layout"] = np.concatenate(
                    (ng_dict[ng + "/layout"], np.array(self.data[key]))
                )
            elif case_key == "gate_oh":
                ng_dict[ng + "/gate_oh"] = np.concatenate(
                    (ng_dict[ng + "/gate_oh"], np.array(self.data[key]))
                )
            elif case_key == "gate_qubit_oh":
                ng_dict[ng + "/gate_qubit_oh"] = np.concatenate(
                    (ng_dict[ng + "/gate_qubit_oh"], np.array(self.data[key]))
                )
        for key in ng_dict.keys():
            try:
                ng_dict[key] = np.array(ng_dict[key])
            except Exception as e:
                print(f"Error processing key '{key}': {e}")
                for data in ng_dict[key]:
                    print(len(data))
                    print(data)
                raise e
        np.savez_compressed(output_file, **ng_dict)


if __name__ == "__main__":
    import time

    np.savez_compressed("training-data/2-14", a=np.array([1, 2, 3, 4]))
    input_files = [
        # "training-data/2-14_20000.npz",
        "training-data/15-19_20000.npz",
    ]
    output_files = [
        # "training-data/compiled/2-14_20000",
        "training-data/compiled/15-19_20000",
    ]
    for input_file, output_file in zip(input_files, output_files):
        tic = time.time()
        extractor = UnprepDataExtractor(input_file)
        extractor.extract_to(output_file)
        toc = time.time()
        print(f"Converted {input_file} -> {output_file} ({toc-tic} sec)")
