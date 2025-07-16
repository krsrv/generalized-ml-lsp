import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .embeddings import (
    GateEmbedding,
    PositionalEncoding,
    SignEmbedding,
    TableauCellEmbedding,
)
from .input import GT_1Q, GT_2Q, Layout
from .tokens import TokenProperties


def _pad_last_dim(tensor: Tensor, pad_size: int) -> Tensor:
    return F.pad(tensor, (0, pad_size), "constant", 0)


class Token_A_Embedding(nn.Module):
    """Global token representing the graph. It is the (truncated) eigenvalue list of the graph
    Laplacian.
    """

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.truncation_dim = token_dims.A_eigval_trunc_dim
        self.dim = token_dims.dA

    def forward(self, x: Tensor) -> Tensor:
        # Cut down x to `truncation_dim` first and then pad with 0s.
        if x.shape[-1] < self.truncation_dim:
            x = _pad_last_dim(x, self.truncation_dim - x.shape[-1])
        else:
            x = torch.narrow(x, -1, 0, self.truncation_dim)
        x = _pad_last_dim(x, self.dim - self.truncation_dim)
        return torch.unsqueeze(x, -2)


class Token_B_Embedding(nn.Module):
    """Token representing the qubits. It is formed by concatenating (truncated) eigenvectors
    of the graph Laplacian with the positional encodings.
    """

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.truncation_dim = token_dims.B_eigvec_trunc_dim
        self.positional_enc_layer = PositionalEncoding(
            embedding_dim=token_dims.B_positional_embed_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.truncation_dim:
            x = _pad_last_dim(x, self.truncation_dim - x.shape[-1])
        else:
            x = torch.narrow(x, -1, 0, self.truncation_dim)
        pos = self.positional_enc_layer(x)
        if len(x.shape) == 3 and len(pos.shape) == 2:
            pos = pos.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((x, pos), dim=-1)


class Token_C_Embedding(nn.Module):
    """Tokens representing the gates available in the circuit. Each gate is formed by
    adding the gate type embedding and the qubit tensor together. For 2 qubit gates,
    the layout is required as well, and the 2 qubit tensors corresponding to target and
    control are concatenated before addition with the gate type embedding.
    """

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.gate_embedding_layer = GateEmbedding(token_dims.C_gt_1q_dim)

    def forward(self, gates_oh: Tensor, gate_qubits_oh: Tensor, qubits: Tensor):
        return self.gate_embedding_layer(gates_oh, gate_qubits_oh, qubits)


class Token_D_Embedding(nn.Module):
    """Token representing the check matrix (stabilizer) signs."""

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.stabilizer_sign_embedding_layer = SignEmbedding(token_dims.D_stab_sign_dim)

    def forward(self, observation: Tensor):
        return self.stabilizer_sign_embedding_layer(observation)


class Token_E_Embedding(nn.Module):
    """Token representing each cell in the stabilizer matrix."""

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.tableau_cell_embedding = TableauCellEmbedding(token_dims.E_pauli_dim)

    def forward(self, qubits: Tensor, observation: Tensor):
        return self.tableau_cell_embedding(observation, qubits)
