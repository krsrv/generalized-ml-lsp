import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .embeddings import DepthProjectionLayer, GateProjectionLayer, ResidualLayer
from .input import GT_1Q, GT_2Q, Layout
from .token_embeddings import (
    Token_A_Embedding,
    Token_B_Embedding,
    Token_C_Embedding,
    Token_D_Embedding,
    Token_E_Embedding,
)
from .tokens import TokenProperties, Tokens
from .transformer_blocks import HeterogenousAttentionBlock, HomogenousAttentionBlock


class ModelV0(nn.Module):
    def __init__(
        self,
        dA: int,
        dB: int,
        dC: int,
        dD: int,
        dE: int,
        eigval_trunc_dim: int | None = None,
        eigvec_trunc_dim: int | None = None,
        gt_1q_dim: int | None = None,
        gt_2q_dim: int | None = None,
        stab_sign_dim: int | None = None,
        pauli_dim: int | None = None,
        num_transformer_blocks: int = 2,
        homo_attention_n_head: int = 4,
        hetero_attention_embed_dim: int = 50,
        hetero_attention_n_head: int = 4,
        positional_encoding_dim: int = 50,
    ) -> None:
        # TODO: initialize `embedding_dims` properly
        super().__init__()
        self.embedding_dims = TokenProperties(
            dA,
            dB,
            dC,
            dD,
            dE,
            eigval_trunc_dim,
            eigvec_trunc_dim,
            gt_1q_dim,
            gt_2q_dim,
            stab_sign_dim,
            pauli_dim,
        )

        self.token_A_embedding = Token_A_Embedding(self.embedding_dims)
        self.token_B_embedding = Token_B_Embedding(self.embedding_dims)
        self.token_C_embedding = Token_C_Embedding(self.embedding_dims)
        self.token_D_embedding = Token_D_Embedding(self.embedding_dims)
        self.token_E_embedding = Token_E_Embedding(self.embedding_dims)

        self.num_transformer_blocks = num_transformer_blocks
        self.heterogenous_attention_block = nn.ModuleList(
            [
                HeterogenousAttentionBlock(
                    self.embedding_dims,
                    embedding_dim=hetero_attention_embed_dim,
                    n_head=hetero_attention_n_head,
                )
                for _ in range(self.num_transformer_blocks)
            ]
        )
        self.computation_layer = nn.ModuleList(
            [
                HomogenousAttentionBlock(
                    self.embedding_dims, n_head=homo_attention_n_head
                )
                for _ in range(self.num_transformer_blocks)
            ]
        )
        self.gate_projection_layer = GateProjectionLayer(self.embedding_dims)
        self.depth_projection_layer = DepthProjectionLayer(self.embedding_dims)
        self.residual_layer = ResidualLayer()

    def forward(
        self,
        graph_eigvals: Tensor,
        graph_eigvecs: Tensor,
        gates_oh: Tensor,
        gate_qubit_oh: Tensor,
        observation: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        The first parameter defines the number of qubits. The next 4 define the layout. The next 2
        define the gate sets to be used. The next 2 are the stabilizer values.
        Args:
            n
            graph_eigvals
            graph_eigvecs
            ctrl_oh
            tgt_oh
            gate_set_1q_oh
            gate_set_2q_oh
            observation_paulis
            observation_signs
        """
        # Note:
        # 1. `graph_eigvec` -> each column represents an eigenvector
        # 2. eigh is specifically for Hermitian matrices. If the input is a real matrix,
        #    the eigenvectors and eigenvalues are both guaranteed to be reals as well.
        # graph_eigval, graph_eigvec = torch.linalg.eigh(graph.float())

        # Token A -> Global token representing the graph
        global_tensor = self.token_A_embedding(graph_eigvals)
        # Token B
        # Q: Is there no learned embedding for the qubits? It will be used in stabilizer and
        # tableau-cell tokens as well. There seems to be a very small amount of learnable
        # components then.
        qubit_tensors = self.token_B_embedding(graph_eigvecs)
        # Token C
        gate_tensors = self.token_C_embedding(gates_oh, gate_qubit_oh, qubit_tensors)
        # Token D
        n = graph_eigvals.shape[-1]
        observation_signs = torch.narrow(observation, -1, 2 * n * n, n).long()
        sign_tensors = self.token_D_embedding(observation_signs)
        # Token E
        observation_paulis = torch.narrow(observation, -1, 0, 2 * n * n).long()
        tableau_cell_tensors = self.token_E_embedding(qubit_tensors, observation_paulis)

        x = Tokens(
            global_tensor,
            qubit_tensors,
            gate_tensors,
            sign_tensors,
            tableau_cell_tensors,
        )
        for i in range(self.num_transformer_blocks):
            attention_output = self.heterogenous_attention_block[i](x)
            x = self.residual_layer(x, attention_output)
            computation_layer = self.computation_layer[i](x)
            x = self.residual_layer(x, computation_layer)
        return self.gate_projection_layer(x.C), self.depth_projection_layer(x.A)
