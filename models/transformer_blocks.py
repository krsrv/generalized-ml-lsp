import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .embeddings import ResidualLayer
from .tokens import Tokens, TokenProperties


class HomogenousAttentionBlock(nn.Module):
    def __init__(self, token_dim: TokenProperties, n_head: int = 2) -> None:
        super().__init__()
        # Each native Transformer layer takes an input `d_model`, which is the expected input and
        # output dimension after the Q, K, V projections. There is no separate embedding_dim parameter
        # required as in the Heterogenous attention case. However, there are other parameters to tune:
        # size of the feedfoward network, number of heads etc.
        self.layer_B = nn.TransformerEncoderLayer(token_dim.dB, n_head)
        self.layer_C = nn.TransformerEncoderLayer(token_dim.dC, n_head)
        self.layer_D = nn.TransformerEncoderLayer(token_dim.dD, n_head)
        self.layer_E = nn.TransformerEncoderLayer(token_dim.dE, n_head)

    def forward(self, x: Tokens) -> Tokens:
        return Tokens(
            x.A,
            self.layer_B(x.B),
            self.layer_C(x.C),
            self.layer_D(x.D),
            self.layer_E(x.E),
        )


class HeterogenousAttentionBlock(nn.Module):
    def __init__(
        self, token_dims: TokenProperties, embedding_dim: int = 50, n_head: int = 4
    ) -> None:
        super().__init__()
        self.token_dims = token_dims
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.d_model = int(self.embedding_dim // self.n_head)
        assert (
            self.d_model * self.n_head == self.embedding_dim
        ), f"""d_model * n_head should equal embedding_dim.
        Received d_model -> {self.d_model}, n_head -> {self.n_head}, embedding_dim -> {self.embedding_dim}"""

        self.softmax = nn.Softmax(-1)
        self.residual_layer = ResidualLayer()
        self.init_attention_layers()
        self.init_feedforward_networks()

    def init_attention_layers(self) -> None:
        self.kwA = nn.Linear(self.token_dims.dA, self.embedding_dim)
        self.kwB = nn.Linear(self.token_dims.dB, self.embedding_dim)
        self.kwC = nn.Linear(self.token_dims.dC, self.embedding_dim)
        self.kwD = nn.Linear(self.token_dims.dD, self.embedding_dim)
        self.kwE = nn.Linear(self.token_dims.dE, self.embedding_dim)

        self.qwA = nn.Linear(self.token_dims.dA, self.embedding_dim)
        self.qwB = nn.Linear(self.token_dims.dB, self.embedding_dim)
        self.qwC = nn.Linear(self.token_dims.dC, self.embedding_dim)
        self.qwD = nn.Linear(self.token_dims.dD, self.embedding_dim)
        self.qwE = nn.Linear(self.token_dims.dE, self.embedding_dim)

        self.vwA = nn.Linear(self.token_dims.dA, self.embedding_dim)
        self.vwB = nn.Linear(self.token_dims.dB, self.embedding_dim)
        self.vwC = nn.Linear(self.token_dims.dC, self.embedding_dim)
        self.vwD = nn.Linear(self.token_dims.dD, self.embedding_dim)
        self.vwE = nn.Linear(self.token_dims.dE, self.embedding_dim)

    def init_feedforward_networks(self) -> None:
        self.fcn_A = nn.Linear(self.embedding_dim * 4, self.token_dims.dA)
        self.fcn_B = nn.Linear(self.embedding_dim * 4, self.token_dims.dB)
        self.fcn_C = nn.Linear(self.embedding_dim * 4, self.token_dims.dC)
        self.fcn_D = nn.Linear(self.embedding_dim * 4, self.token_dims.dD)
        self.fcn_E = nn.Linear(self.embedding_dim * 4, self.token_dims.dE)

    def get_attention_output(
        self, q: Tensor, kv_pairs: list[tuple[Tensor, Tensor]]
    ) -> Tensor:
        def single_attention_output(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            # Input tensors are (..., nh * d_model)
            q = torch.unsqueeze(q, -1)
            q = q.reshape(*q.shape[:-2], self.n_head, self.d_model).transpose(
                -3, -2
            )  # (nh, nq, d_model)
            k = torch.unsqueeze(k, -1)
            k = k.reshape(*k.shape[:-2], self.n_head, self.d_model).transpose(
                -3, -2
            )  # (nh, nk, d_model)
            v = torch.unsqueeze(v, -1)
            v = v.reshape(*v.shape[:-2], self.n_head, self.d_model).transpose(
                -3, -2
            )  # (nh, nv, d_model)
            weights = self.softmax(
                torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embedding_dim)
            )
            output = torch.matmul(weights, v)  # (nh, nq, d_model)
            output = output.transpose(-3, -2)
            output = output.reshape(*output.shape[:-2], -1, 1)  # (nq, nh * d_model, 1)
            output = torch.squeeze(output, -1)
            return output

        output = torch.concat(
            [single_attention_output(q, k, v) for k, v in kv_pairs], dim=-1
        )
        return output

    def forward(self, x: Tokens) -> Tokens:
        kA, qA, vA = self.kwA(x.A), self.qwA(x.A), self.vwA(x.A)
        kB, qB, vB = self.kwB(x.B), self.qwB(x.B), self.vwB(x.B)
        kC, qC, vC = self.kwC(x.C), self.qwC(x.C), self.vwC(x.C)
        kD, qD, vD = self.kwD(x.D), self.qwD(x.D), self.vwD(x.D)
        kE, qE, vE = self.kwE(x.E), self.qwE(x.E), self.vwE(x.E)

        attention_to_A = self.get_attention_output(
            qA, [(kB, vB), (kC, vC), (kD, vD), (kE, vE)]
        )
        attention_to_B = self.get_attention_output(
            qB, [(kC, vC), (kD, vD), (kE, vE), (kA, vA)]
        )
        attention_to_C = self.get_attention_output(
            qC, [(kD, vD), (kE, vE), (kA, vA), (kB, vB)]
        )
        attention_to_D = self.get_attention_output(
            qD, [(kE, vE), (kA, vA), (kB, vB), (kC, vC)]
        )
        attention_to_E = self.get_attention_output(
            qE, [(kA, vA), (kB, vB), (kC, vC), (kD, vD)]
        )

        additive_inp_A = self.fcn_A(attention_to_A)
        additive_inp_B = self.fcn_B(attention_to_B)
        additive_inp_C = self.fcn_C(attention_to_C)
        additive_inp_D = self.fcn_D(attention_to_D)
        additive_inp_E = self.fcn_E(attention_to_E)

        return Tokens(
            self.residual_layer(x.A, additive_inp_A),
            self.residual_layer(x.B, additive_inp_B),
            self.residual_layer(x.C, additive_inp_C),
            self.residual_layer(x.D, additive_inp_D),
            self.residual_layer(x.E, additive_inp_E),
        )
