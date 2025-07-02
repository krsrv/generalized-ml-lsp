import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .input import Layout, GT_1Q, GT_2Q
from .tokens import TokenProperties
from .embeddings import (
    PositionalEncoding,
    Gate1QEmbedding,
    Gate2QEmbedding,
    SignEmbedding,
    TableauCellEmbedding,
)


class Token_A_Embedding(nn.Module):
    """Global token representing the graph. It is the (truncated) eigenvalue list of the graph
    Laplacian.
    """

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.truncation_dim = token_dims.dA

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.truncation_dim:
            x = F.pad(x, (0, self.truncation_dim - x.shape[-1]), "constant", 0)
        else:
            x = torch.narrow(x, -1, 0, self.truncation_dim)
        return torch.unsqueeze(x, -2)


class Token_B_Embedding(nn.Module):
    """Token representing the qubits. It is formed by concatenating (truncated) eigenvectors
    of the graph Laplacian with the positional encodings.
    """

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.truncation_dim = token_dims.dB // 2
        self.positional_enc_layer = PositionalEncoding(
            embedding_dim=self.truncation_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.truncation_dim:
            x = F.pad(x, (0, self.truncation_dim - x.shape[-1]), "constant", 0)
        else:
            x = torch.narrow(x, -1, 0, self.truncation_dim)
        return torch.cat((x, self.positional_enc_layer(x)), dim=-1)


class Token_C_Embedding(nn.Module):
    """Tokens representing the gates available in the circuit. Each gate is formed by
    adding the gate type embedding and the qubit tensor together. For 2 qubit gates,
    the layout is required as well, and the 2 qubit tensors corresponding to target and
    control are concatenated before addition with the gate type embedding.
    """

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.gate_1q_embedding_layer = Gate1QEmbedding(token_dims.dC)
        self.gate_2q_embedding_layer = Gate2QEmbedding(token_dims.dC)

    def forward(
        self, gset_1q: list[GT_1Q], gset_2q: list[GT_2Q], qubits: Tensor, layout: Layout
    ):
        return torch.cat(
            (
                self.gate_1q_embedding_layer(gset_1q, qubits),
                self.gate_2q_embedding_layer(gset_2q, qubits, layout),
            ),
            dim=-2,
        )


class Token_D_Embedding(nn.Module):
    """Token representing the check matrix (stabilizer) signs."""

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.stabilizer_row_embedding_layer = SignEmbedding(token_dims.dD)

    def forward(self, nq: int, observation: Tensor):
        return self.stabilizer_row_embedding_layer(nq, observation)


class Token_E_Embedding(nn.Module):
    """Token representing each cell in the stabilizer matrix."""

    def __init__(self, token_dims: TokenProperties) -> None:
        super().__init__()
        self.tableau_cell_embedding = TableauCellEmbedding(token_dims.dE)

    def forward(self, qubits: Tensor, observation: Tensor):
        return self.tableau_cell_embedding(qubits, observation)
