import torch
import torch.nn.functional as F
from enum import Enum


class GT_1Q(Enum):
    """Gate Type - 1 Qubit"""

    H = 1
    S = 2
    X = 3
    SQRT_X = 4


class GT_2Q(Enum):
    """Gate Type - 2 Qubit
    Model parameters, sampling procedure etc. will depend on the number of gates defined in
    this Enum. Any new gates added to the Enum can break the working functionality of a given
    model and set up.
    """

    CX = 1
    CZ = 2
    # SWAP = 3 # Unsupported
    # I_SWAP = 4 # Unsupported


class Layout:
    graph: torch.Tensor
    adjacency_list: list[tuple[int, int]]
    adjacency_oh_matrices: tuple[torch.Tensor, torch.Tensor]

    def __init__(self, graph: torch.Tensor) -> None:
        self.graph = graph.int()
        self.adjacency_list = Layout.construct_adjacency_list(self.graph)
        self.adjacency_oh_matrices = Layout.construct_adjacency_oh_matrices(self.graph)

    @staticmethod
    def construct_adjacency_list(graph: torch.Tensor) -> list[tuple[int, int]]:
        adj_list = []
        for q_idx, row in enumerate(graph):
            for t_idx, el in enumerate(row):
                # Assumes `graph` is an int tensor
                if q_idx > t_idx and (el == 1 or el == 1.0):
                    adj_list.append((q_idx, t_idx))
                    adj_list.append((t_idx, q_idx))
        return adj_list

    @staticmethod
    def construct_adjacency_oh_matrices(
        graph: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct matrices which one-hot encodings of the adjacency list representation.
        If (i, j) is a directed edge, then the first matrix will have one_hot(i) vector and the
        second will have one_hot(j) vector. The size of the matrices will be #edges * #nodes.
        """
        ctrl_list, tgt_list = [], []
        for q_idx, row in enumerate(graph):
            for t_idx, el in enumerate(row):
                if q_idx > t_idx and (el == 1 or el == 1.0):
                    ctrl_list.append(q_idx)
                    tgt_list.append(t_idx)

                    ctrl_list.append(t_idx)
                    tgt_list.append(q_idx)
        nq = graph.shape[-1]
        return F.one_hot(torch.tensor(ctrl_list), num_classes=nq), F.one_hot(
            torch.tensor(tgt_list), num_classes=nq
        )


def sample_layout(n: int, gen: torch.Generator | None = None) -> Layout:
    """Construct a random graph with n vertices.

    Args:
        n: number of vertices
        rng: Random number generator

    Returns:
        Graph laplacian
    """
    satisfactory = False
    while not satisfactory:
        graph = torch.empty(n, n).uniform_(0, 1, generator=gen)
        graph = torch.bernoulli(graph, generator=gen)
        graph = torch.triu(graph, diagonal=1)
        graph = graph + torch.transpose(graph, 0, 1)
        graph = graph - torch.diag(graph.sum(dim=0))
        graph = graph.int()
        satisfactory = torch.all(graph.diag(0) != 0)
    return Layout(graph)
