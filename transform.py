"""
Given ordered data generated through the training.generate module, transform into a batchable
input where each file has random circuits parameterized by the number of qubits and the number of
gates applicable in the circuit.
TODO: Change batching dataset input file to pass the gates appropriately given the new input.
"""

import os
import pickle
from collections import Counter

import numpy as np
from torch.utils.data.dataset import Dataset

from training.dataset import TrainingInstance


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


folders = ["/scratch1/sauravk/lsp-raw-training-data"]
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
            if data.circuit_depth == 0:
                print()
            if (data.n, n_gates) not in n_g_training_map.keys():
                n_g_training_map[(data.n, n_gates)] = []
            n_g_training_map[(data.n, n_gates)].append((f"{folder}/{file}", j))
        if i % 100 == 0:
            print(f"Read {i} files in {folder}")

output_folder = "/scratch1/sauravk/lsp-training-data"
counter = Counter()
output_file_size = 512
for k, v in n_g_training_map.items():
    ng_dataset = RawData(v)
    size = len(ng_dataset)
    idx_list = np.random.permutation(range(size))
    print(k)
    for i in range(1 + int(size // output_file_size)):
        dump_object = []
        for j in idx_list[
            i * output_file_size : np.min(((i + 1) * output_file_size, size - 1))
        ]:
            data = ng_dataset[j]
            # if has_only_diagonal():
            #     continue
            data = TrainingInstance(
                data.n,
                data.layout,
                data.gate_set_1q,
                data.gate_set_2q,
                data.circuit_depth,
                np.array(data.gates[-1:], dtype=np.long),
                data.observation,
                None,
            )
            dump_object.append(data)
        with open(f"{output_folder}/n{k[0]}-g{k[1]}-{i}.pkl", "wb") as f:
            pickle.dump(dump_object, f)
