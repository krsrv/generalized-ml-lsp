import os
import random
import shutil
import string
from enum import Enum

import numpy as np
from tabulate import tabulate


class Topology(Enum):
    FullyConnected = 0
    Random = 1
    Cubic = 2
    Square = 3
    Hex = 4
    HeavyHex = 5
    Linear = 6


class GateSet(Enum):
    Superconducting = 0
    IonTrap = 1
    Random = 2


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


class UnprepDataCompiler:
    """
    Given an npz file with keys `/n/g/hash/{key}`, collapse the hash level to create an
    unpreparation dataset with keys `/n/g/{key}`.
    """

    def __init__(self, files: list[str]) -> None:
        self.files = files

    def load_data(self):
        self.data_list = {}
        self.total_size = 0
        for file in self.files:
            try:
                self.data_list[file] = np.load(file)
                self.total_size += self.data_list[file]["depth"].shape[0]
            except:
                print(f"Error in {file}")
                self.data_list.pop(file, None)

    def extract_depth_distribution(self):
        depth = {}
        for data in self.data_list.values():
            n = data["n"][0]
            if n not in depth:
                depth[n] = data["depth"]
            else:
                depth[n] = np.concatenate((depth[n], data["depth"]))
        return depth

    def get_total_size(self):
        return self.total_size

    def _create_temp_folder(self):
        self.temp_folder = None
        while self.temp_folder is None or os.path.exists(self.temp_folder):
            self.temp_folder = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        os.makedirs(self.temp_folder)

    def _pick_samples(
        self, arrays: list[np.ndarray], files: list[str], num_samples: int, output_dict: dict
    ) -> int:
        # Sample indices which will correspond to the picked datapoints
        sizes = [x["depth"].shape[-1] for x in arrays]
        cum_sizes = np.cumsum(sizes)
        if len(cum_sizes) == 0 or cum_sizes[-1] == 0:
            return 0
        sample_indices = np.arange(cum_sizes[-1])
        np.random.shuffle(sample_indices)
        sample_indices = np.sort(sample_indices[:num_samples])
        # Map sampled indices to indices within original array
        ng_idxs, ng_num_samples = np.unique(
            cum_sizes.searchsorted(sample_indices, "right"), return_counts=True
        )
        assert sum(ng_num_samples) == num_samples

        offset = 0
        total_samples = 0
        for i in range(len(ng_num_samples)):
            batch_idx = ng_idxs[i]
            indices = sample_indices[offset : offset + ng_num_samples[i]] - (
                cum_sizes[batch_idx - 1] if i > 0 else 0
            )
            output_dict[files[batch_idx]] = indices
            total_samples += len(indices)
            offset += ng_num_samples[i]
        return total_samples

    def _print_tabular_top_gs_data(self, data):
        # Build header row
        headers = [""] + [gs.name for gs in GateSet]

        # Build table data
        table = []
        for top in Topology:
            row = [top.name]
            for gs in GateSet:
                row.append(data.get((top.value, gs.value), 0))
            table.append(row)

        print(tabulate(table, headers=headers, tablefmt="grid"))

    def _parse_tableau_data(self, data: np.ndarray, n: int):
        bs, _ = data.shape
        new_data = np.zeros((bs, n, 2 * n + 1))
        for i in range(n):
            for j in range(n):
                # Z stabilizer
                new_data[:, :, 2 * n - (i + 1)] = (data >> i) & 1
                # X stabilizer
                new_data[:, :, n - (i + 1)] = (data >> (32 + i)) & 1
        new_data[:, :, -1] = (data >> 63) & 1
        return new_data.reshape(bs, -1)

    def create_npz_file(self, filename: str):
        topo_gs_idxs = {}
        total_size = {}
        for k, data in self.data_list.items():
            # n, g = x["n"], x["g"]
            topo = data["topology"][0]
            gs = data["gate_set_type"][0]
            if (topo, gs) not in total_size:
                total_size[(topo, gs)] = 0
                topo_gs_idxs[(topo, gs)] = []
            total_size[(topo, gs)] += data["depth"].shape[0]
            topo_gs_idxs[(topo, gs)].append(k)

        ### Print output
        print("Raw dataset distribution")
        self._print_tabular_top_gs_data(total_size)

        # Desired probability distributions:
        #   * Within each topology, 70% data should be the correct gateset, and 15% mismatch and
        #     random each
        intra_topology_sampling_ratio = {
            Topology.FullyConnected: (0.15, 0.7, 0.15 * 4.2),
            Topology.Random: (0.15, 0.15, 0.7),
            Topology.Cubic: (0.7, 0.15, 0.15),
            Topology.Square: (0.7, 0.15, 0.15),
            Topology.Hex: (0.7, 0.15, 0.15),
            Topology.HeavyHex: (0.7, 0.15, 0.15),
            Topology.Linear: (0.7, 0.15, 0.15),
        }

        #   * Overall distribution between topologies
        inter_topology_sampling_ratio = {
            Topology.FullyConnected: 0.17,
            Topology.Random: 0.20,
            Topology.Cubic: 0.05,
            Topology.Square: 0.17,
            Topology.Hex: 0.155,
            Topology.HeavyHex: 0.10,
            Topology.Linear: 0.155,
        }

        total_datapoints = 30_000_000
        sample_idxs = dict()
        for topo in Topology:
            for gs in GateSet:
                proposed_final_sample_count = int(
                    total_datapoints
                    * inter_topology_sampling_ratio[topo]
                    * intra_topology_sampling_ratio[topo][gs.value]
                )
                available_sample_count = total_size.get((topo.value, gs.value), 0)
                if proposed_final_sample_count > available_sample_count:
                    print(
                        f"Total number of datapoints not consistent with current dataset for {topo}, {gs}. Expected {proposed_final_sample_count} vs available {available_sample_count}"
                    )
                else:
                    print(topo, gs, "consistent with sampling requirements")

                sample_count = self._pick_samples(
                    [self.data_list[i] for i in topo_gs_idxs.get((topo.value, gs.value), [])],
                    [i for i in topo_gs_idxs.get((topo.value, gs.value), [])],
                    proposed_final_sample_count,
                    sample_idxs,
                )
                total_size[(topo.value, gs.value)] = sample_count

        ### Print output
        print("After filtering for each (topology, gateset)")
        self._print_tabular_top_gs_data(total_size)

        ## Reorganize into (n, g) pairs and dump:
        n_g = {}
        for file, idxs in sample_idxs.items():
            n, g = self.data_list[file]["n"][0], self.data_list[file]["g"][0]
            if (n, g) not in n_g:
                n_g[(n, g)] = []
            n_g[(n, g)].append((file, idxs))

        dump_data = {}
        for k, vs in n_g.items():
            n, g = k
            for i, v in enumerate(vs):
                file, idxs = v
                curr_data = self.data_list[file]
                original_bs = curr_data["depth"].shape[0]
                bs = len(idxs)
                observations = (
                    curr_data["observation"].reshape(original_bs, -1)[idxs, :].reshape(-1)
                )
                formatted_data = {
                    "unprep_gate": curr_data["unprep_gate"][idxs],
                    "depth": curr_data["depth"][idxs],
                    "observation": observations,
                    # Copy for each
                    "layout": np.stack([curr_data["layout"]] * bs, axis=0),
                    "gates": np.stack([curr_data["gates"]] * bs, axis=0),
                    "gate_qubits": np.stack([curr_data["gate_qubits"]] * bs, axis=0),
                    "topology": np.concatenate([curr_data["topology"]] * bs),
                    "gate_set_type": np.concatenate([curr_data["gate_set_type"]] * bs),
                }
                for key in formatted_data:
                    if i == 0:
                        dump_data[f"{n}/{g}/{key}"] = formatted_data[key]
                    else:
                        dump_data[f"{n}/{g}/{key}"] = np.concatenate(
                            (dump_data[f"{n}/{g}/{key}"], formatted_data[key]), axis=0
                        )

            print(f"Completed dataset sampling for {n}/{g}")

        print("Saving to npz")
        np.savez_compressed(filename, **dump_data)


if __name__ == "__main__":
    import argparse
    import time

    # input_files = [
    #     # List of input files, with .npz extension. Example:
    #     # "training-data/2-5_20000.npz",
    # ]
    # output_files = [
    #     # List of output files, without .npz extension. Example:
    #     # "training-data/compiled/2-5_20000"
    # ]
    # for input_file, output_file in zip(input_files, output_files):
    #     tic = time.time()
    #     extractor = UnprepDataExtractor(input_file)
    #     extractor.extract_to(output_file)
    #     toc = time.time()
    #     print(f"Converted {input_file} -> {output_file} ({toc-tic} sec)")

    parser = argparse.ArgumentParser(description="Trainer hyperparameters")
    parser.add_argument("--filename", type=str, required=True, help="Output filename")
    args = parser.parse_args()

    assert args.filename.endswith(".npz"), "Output filename must end with '.npz'"
    assert not os.path.exists(args.filename), f"{args.filename} already exists"

    tic = time.time()
    folder = "../lsp_nonn/output/2_10_data/"
    files = [folder + x for x in list(os.listdir(folder))]
    compiler = UnprepDataCompiler(files)
    compiler.load_data()
    print("Total size:", compiler.get_total_size())
    compiler.create_npz_file(args.filename)
    toc = time.time()
    print(f"Converted {folder} -> {args.filename} ({toc-tic} sec)")
