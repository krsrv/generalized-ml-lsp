from enum import EnumType
import torch
import torch.nn.functional as F
from .input import Layout, GT_1Q, GT_2Q

def create_oh_vectors_from_enum(gate_set, enum: EnumType) -> torch.Tensor:
    # (-1) to ensure indexing starts from 0
    gate_idxs = (
        torch.tensor([gate.value for gate in gate_set], dtype=torch.int64) - 1
    )
    gates_oh = F.one_hot(gate_idxs, num_classes=len(enum)).float()
    return gates_oh