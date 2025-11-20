import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .input import GT_1Q, GT_2Q, Layout
from .tokens import TokenProperties, Tokens
from .utils import create_oh_vectors_from_enum

NUM_GATE_TYPES = 15


def _pad_last_dim(tensor: Tensor, pad_size: int) -> Tensor:
    return F.pad(tensor, (0, pad_size), "constant", 0)


def _cartesian_add(x: Tensor, y: Tensor) -> Tensor:
    """Consider x, y to be 2D tensors of dim (nx, dx) and (ny, dy) where dx = dy.
    Then, the result will be a 2D tensor of dim (nx*ny, dx) with the following elements:
    [x[0] + y[0], x[0] + y[1], ..., x[0] + y[ny-1], x[1] + y[0], ..., x[nx-1] + y[ny-1]]
    """
    nx, dx = x.shape[-2:]
    ny, dy = y.shape[-2:]
    assert dx == dy, f"Dimensions of x and y should match. Got {dx} and {dy}"

    x = x.unsqueeze(-2)  # Shape: (nx, 1, d)
    x = x.expand(*x.shape[:-3], nx, ny, dx)  # Shape: (nx, ny, d)
    x = x.reshape(*x.shape[:-3], -1, dx)

    y = y.unsqueeze(-3)  # Shape: (1, ny, d)
    y = y.expand(*y.shape[:-3], nx, ny, dx)  # Shape: (nx, ny, d)
    y = y.reshape(*y.shape[:-3], -1, dx)
    return x + y


def _generate_qubit_pair(qubits: Tensor, ctrl: Tensor, tgt: Tensor) -> Tensor:
    """Given a layout of qubits with specified connectivity, and the qubit embeddings [q[0], ..., q[n-1]]
    return a tensor of concat(q[i], q[j]) where (i, j) forms an edge in the graph layout.
    """
    return torch.concat(
        (torch.matmul(ctrl.float(), qubits), torch.matmul(tgt.float(), qubits)), dim=-1
    )


class PositionalEncoding(nn.Module):
    """Positional encodings for transformers. We use sinusoidal positional embeddings, but
    in a form different than usual positional encoding for transformers. Normally, we have a N tokens
    as input (where N is fixed), and we construct N positional vectors while truncating the embedding
    dimension to match the dimension of each token vector. This behavior also means the encodings can
    be constructed at initialization. Instead, here N is not fixed. So we will construct new positional
    encodings for each input.

    Given an input tensor of size (..., m, n) the output will be a tensor of size (m, embedding_dim)

    Also a nice explaination on why sinusoidal encodings:
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """

    def __init__(self, embedding_dim: int = 50, use_batch: bool = False) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        # Interpret the last 2 dimensions of x as matrix dimensions and derive the positional
        # encodings based on the shape. In row vector format, (n_rows, n_columns) refers to
        # `n_rows` number of vectors to work with, and `n_columns` is the dimension of each vector.
        n, _d = x.shape[-2:]
        numtr = torch.arange(n).unsqueeze(1)  # k
        dentr = torch.exp(
            torch.arange(0, self.embedding_dim, 2) * (-np.log(10000))
        ).unsqueeze(
            0
        )  # 1 / 10000 ^ (2i/d)

        output = torch.zeros((n, self.embedding_dim)).to(x.device)
        # PE(k, 2i) = sin(k / 10000^(2i / d))
        output[:, 0::2] = torch.sin(numtr * dentr)
        # PE(k, 2i+1) = cos(k / 10000^(2i / d))
        output[:, 1::2] = torch.cos(numtr * dentr)
        return output


class GateEmbedding(nn.Module):
    """Learnt embeddings for each gate type. It implicitly makes use of the properties of
    the Enum GT_1Q, which means the model would need to be retrained if it is tweaked.

    Given an input gate set of size G and n qubits (tensor size (n, q_embedding_dim)) the output will
    be a tensor of size (G*n, embedding_dim)
    """

    def __init__(self, embedding_dim: int = 60) -> None:
        super().__init__()
        # Use nn.Linear for proper initialization.
        self.embedding_dim = embedding_dim
        self.num_classes = NUM_GATE_TYPES
        self.layer = nn.Linear(self.num_classes, embedding_dim, bias=False)

    def forward(
        self, gates_oh: Tensor, gate_qubit_oh: Tensor, qubits: Tensor
    ) -> Tensor:
        """Generate embeddings of each pair (Gate, Qubit) where qubit index moves first."""
        device = qubits.device
        gate_embeddings = self.layer(
            F.one_hot(gates_oh, num_classes=NUM_GATE_TYPES).to(device).float()
        )

        n = qubits.shape[-2]
        # Note that `gate_qubit_encodings` is 1-indexed, with 0 representing no qubit (for single
        # qubit gates).
        gate_qubit_oh = F.one_hot(gate_qubit_oh, num_classes=n + 1).to(device).float()
        zeros = torch.zeros(
            *qubits.shape[:-2], 1, qubits.shape[-1], device=device, dtype=qubits.dtype
        )
        qubits = torch.cat([zeros, qubits], dim=-2)
        qubit_embeddings = torch.matmul(gate_qubit_oh, qubits)  # shape (..., 2c, d)
        qubit_embeddings = qubit_embeddings.reshape(
            *qubit_embeddings.shape[:-2], gate_embeddings.shape[-2], -1
        )  # shape (..., c, 2d)
        return gate_embeddings + qubit_embeddings


class SignEmbedding(nn.Module):
    """Learnt embeddings for each stabilizer sign.

    Given an input tensor of size 2*n*n+n the output will a tensor of size (n, embedding_dim).
    """

    def __init__(self, embedding_dim: int = 60) -> None:
        super().__init__()
        # Use nn.Linear for proper initialization.
        self.embedding_dim = embedding_dim
        self.n_signs = 2
        # input dim = 2 because there are only 2 signs
        self.layer = nn.Linear(self.n_signs, embedding_dim, bias=False)
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def forward(self, signs: Tensor) -> Tensor:
        """
        Args:
            nq: number of qubits
            observation: output of (bool) check matrix. The last dimension should have the format
                (X1 X2 ... Xnq Z1 Z2 .. Znq)_1 ... (X1 X2 ... Xnq Z1 Z2 .. Znq)_nq (S_1 ... S_nq)
                where ()_i represents the ith stabilizer, S_i represents the sign of the ith
                stabilizer
        """
        device = signs.device
        signs_oh = F.one_hot(signs, num_classes=self.n_signs).to(device).float()
        sign_embeddings = self.layer(signs_oh)
        return sign_embeddings + self.positional_encoding(sign_embeddings)


class TableauCellEmbedding(nn.Module):
    """I, X, Z, Y"""

    def __init__(self, embedding_dim: int = 60) -> None:
        super().__init__()
        # Use nn.Linear for proper initialization.
        self.embedding_dim = embedding_dim
        self.n_paulis = 4
        # input dim = 4 because there are only 4 Paulis
        self.layer = nn.Linear(self.n_paulis, embedding_dim, bias=False)
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def forward(self, paulis: Tensor, qubits: Tensor) -> Tensor:
        nq = qubits.shape[-2]
        # `paulis` is of shape (n, 2*n). The first n columns represent X stabilizers and
        # the next n represent Z stabilizers.
        # Map (I, X, Y, Z) to (0, 1, 3, 2)
        paulis = torch.narrow(paulis, -1, 0, nq) + 2 * torch.narrow(paulis, -1, nq, nq)
        paulis = paulis.reshape((*paulis.shape[:-2], -1))
        paulis_oh = F.one_hot(paulis, num_classes=self.n_paulis).to(qubits.device)
        pauli_embeddings = self.layer(paulis_oh.float())
        nq = qubits.shape[-2]
        # Expand qubits from (n, d) to (n*n, d)
        # `expand` copies data from the subtensor following the expansion dimension. We want
        # to repeat the entire qubit vector, so we expand at the 3rd last dimension.
        qubits = qubits.unsqueeze(-3)  # Shape: (1, nq, d)
        qubits = qubits.expand(*qubits.shape[:-3], nq, nq, -1)  # Shape: (nq, nq, d)
        qubits = qubits.reshape(*qubits.shape[:-3], nq * nq, -1, 1)
        qubits = qubits.squeeze(-1)
        return qubits + pauli_embeddings


class DepthProjectionLayer(nn.Module):
    def __init__(self, token_dim: TokenProperties) -> None:
        super().__init__()
        self.depth_prediction_layer = nn.Linear(token_dim.dA, 1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.squeeze(self.depth_prediction_layer(x), -1).squeeze(-1)


class GateProjectionLayer(nn.Module):
    def __init__(self, token_dim: TokenProperties) -> None:
        super().__init__()
        self.gate_prediction_layer = nn.Linear(token_dim.dC, 1)
        # self.softmax = nn.Softmax(-1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.squeeze(self.gate_prediction_layer(x), -1)


class ResidualLayer(nn.Module):

    def __init__(
        self, alpha: float = 0.20, norm: bool = False, token_dim: TokenProperties = None
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.normalize = norm
        if self.normalize:
            self.nA = nn.LayerNorm(token_dim.dA, elementwise_affine=False)
            self.nB = nn.LayerNorm(token_dim.dB, elementwise_affine=False)
            self.nC = nn.LayerNorm(token_dim.dC, elementwise_affine=False)
            self.nD = nn.LayerNorm(token_dim.dD, elementwise_affine=False)
            self.nE = nn.LayerNorm(token_dim.dE, elementwise_affine=False)
        else:
            self.nA = nn.Identity()
            self.nB = nn.Identity()
            self.nC = nn.Identity()
            self.nD = nn.Identity()
            self.nE = nn.Identity()

    def forward(self, x: Tensor | Tokens, f_x: Tensor | Tokens) -> Tensor | Tokens:
        assert type(x) == type(f_x), "Either both inputs should be Tensors or Tokens"
        if isinstance(x, Tokens) and isinstance(f_x, Tokens):
            return Tokens(
                self.nA(x.A + self.alpha * f_x.A),
                self.nB(x.B + self.alpha * f_x.B),
                self.nC(x.C + self.alpha * f_x.C),
                self.nD(x.D + self.alpha * f_x.D),
                self.nE(x.E + self.alpha * f_x.E),
            )
        else:
            return x + self.alpha * f_x
