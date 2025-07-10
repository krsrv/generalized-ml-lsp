import json
import os
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.masked import masked_tensor
from torch.utils.data.dataset import Dataset

from models.input import GT_1Q, GT_2Q
from models.utils import create_oh_vectors_from_enum
from training.instance import TrainingInstance


def construct_metadata(folder: str) -> dict:
    print(f"Constructing metadata for data in {folder}")
    metadata = {"size": 0, "files": []}
    for file in os.listdir(folder):
        if not file.endswith(".pkl"):
            continue

        with open(f"{folder}/{file}", "rb") as f:
            data = pickle.load(f)
        data: list[TrainingInstance] = data
        metadata["files"].append((file, len(data)))
        metadata["size"] += len(data)
    print(f"Metadata constructed successfully")
    return metadata


def dump_metadata(metadata: dict, filename: str) -> None:
    """Dump the metadata dictionary to a given file as JSON."""
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata dumped to {filename}")


def fill(tensor: torch.Tensor, fill_shape: torch.Size, fill_type: str):

    if len(tensor.shape) == 2:
        padding = (
            0,
            fill_shape[1] - tensor.shape[1],
            0,
            fill_shape[0] - tensor.shape[0],
        )
        mask = torch.zeros(fill_shape, dtype=torch.bool)
        mask[: tensor.shape[0], : tensor.shape[1]] = True
    elif len(tensor.shape) == 1:
        padding = (0, fill_shape[0] - tensor.shape[0])
        mask = torch.zeros(fill_shape, dtype=torch.bool)
        mask[: tensor.shape[0]] = True
    tensor = F.pad(tensor, padding, "constant", 0)
    if fill_type == "mask":
        return masked_tensor(tensor, mask)
    return tensor


class UnpreparationDataset(Dataset):
    def __init__(self, folder, force_metadata_overwrite: bool = False):
        super().__init__()
        self.folder = folder
        metadata_file = "metadata.json"
        if not os.path.exists(f"{folder}/{metadata_file}") or force_metadata_overwrite:
            self.metadata = construct_metadata(folder)
        else:
            self.metadata = json.load(open(f"{folder}/{metadata_file}", "r"))
        self.construct_index_table()

    def construct_index_table(self) -> None:
        self.sizes = np.array([x[1] for x in self.metadata["files"]])
        self.sizes = np.cumsum(self.sizes)

    def get_access_index(self, idx: int) -> tuple[int, int]:
        file_idx = self.sizes.searchsorted(idx, side="left")
        list_idx = idx - self.sizes[file_idx]
        return file_idx, list_idx

    def write_metadata(self) -> None:
        dump_metadata(self.metadata, f"{self.folder}/metadata.json")

    def __len__(self):
        return self.metadata["size"]

    def __getitem__(self, index) -> Any:
        file_idx, list_idx = self.get_access_index(index)
        filename = self.metadata["files"][file_idx][0]
        with open(f"{self.folder}/{filename}", "rb") as f:
            data_list = pickle.load(f)
        data_list: list[TrainingInstance] = data_list
        data: TrainingInstance = data_list[list_idx]
        nq = data.n
        graph_eigval, graph_eigvec = torch.linalg.eigh(data.layout.graph.float())
        observation = torch.tensor(data.observation, dtype=torch.long)
        paulis = torch.narrow(observation, -1, 0, nq * nq) + 2 * torch.narrow(
            observation, -1, nq * nq, nq * nq
        )
        signs = torch.narrow(observation, -1, 2 * nq * nq, nq)

        n_edges = len(data.layout.adjacency_list)
        n_gates = n_edges * len(data.gate_set_2q) + nq * len(data.gate_set_1q)
        N = 20
        MAX_GATES = N * N * len(GT_2Q) + N * len(GT_1Q)
        return {
            "eigval": fill(graph_eigval, (N,), "pad"),
            "eigvec": fill(graph_eigvec, (N, N), "pad"),
            "ctrl": fill(data.layout.adjacency_oh_matrices[0], (N * N, N), "pad"),
            "tgt": fill(data.layout.adjacency_oh_matrices[1], (N * N, N), "pad"),
            "gt_1q": fill(
                create_oh_vectors_from_enum(data.gate_set_1q, GT_1Q),
                (len(GT_1Q), len(GT_1Q)),
                "pad",
            ),
            "gt_2q": fill(
                create_oh_vectors_from_enum(data.gate_set_2q, GT_2Q),
                (len(GT_2Q), len(GT_2Q)),
                "pad",
            ),
            "paulis": fill(paulis, (N * N,), "pad"),
            "signs": fill(signs, (N,), "pad"),
            "n_mask": fill(torch.ones((nq,)), (N,), "pad"),
            "gate_mask": fill(torch.ones((n_gates,)), (MAX_GATES,), "pad"),
            "last_gate": torch.from_dlpack(data.gates[-1]).long(),
            "depth": torch.tensor(data.circuit_depth),
        }
