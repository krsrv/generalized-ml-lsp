import os

import numpy as np


class UnprepDataExtractor:
    """
    Given an npz file with keys `/n/g/hash/{key}`, collapse the hash level to create an
    unpreparation dataset with keys `/n/g/{key}`.
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
        # Original keys are "/n/g/hash/{keyword}"
        ng_set = set(x.split("/")[1] + "/" + x.split("/")[2] for x in keys)
        # Prepare the output arrays.
        ng_dict = {}
        for key in ng_set:
            ng_dict[key + "/gate"] = np.array([], dtype=np.int_)
            ng_dict[key + "/depth"] = np.array([], dtype=np.int_)
            ng_dict[key + "/observation"] = np.array([], dtype=np.bool_)
            ng_dict[key + "/layout"] = np.array([], dtype=np.bool_)
            ng_dict[key + "/gate_oh"] = np.array([], dtype=np.int_)
            ng_dict[key + "/gate_qubit_oh"] = np.array([], dtype=np.int_)

        # Collect data into the new output arrays.
        for key in keys:
            ng = key.split("/")[1] + "/" + key.split("/")[2]
            keyword = key.split("/")[4]
            if keyword == "n":
                continue
            ng_dict[ng + "/" + keyword] = np.concatenate(
                (ng_dict[ng + "/" + keyword], np.array(self.data[key]))
            )

        # Convert to np arrays
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

    input_files = [
        # List of input files, with .npz extension. Example:
        # "training-data/2-5_20000.npz",
    ]
    output_files = [
        # List of output files, without .npz extension. Example:
        # "training-data/compiled/2-5_20000"
    ]
    for input_file, output_file in zip(input_files, output_files):
        tic = time.time()
        extractor = UnprepDataExtractor(input_file)
        extractor.extract_to(output_file)
        toc = time.time()
        print(f"Converted {input_file} -> {output_file} ({toc-tic} sec)")
