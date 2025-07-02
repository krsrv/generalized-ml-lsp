from torch import Tensor

class Tokens:
    def __init__(self, A: Tensor, B: Tensor, C: Tensor, D: Tensor, E: Tensor) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E

class TokenProperties:
    """Store information about the embedding dimensions of various components in the model.
    `TokenProperties` also calculates the embedding dimensions of the various token types
    based on the inputs, and can be accessed through the member variables `TokenProperties.dA`,
    `TokenProperties.dB` and so on.
    """
    def __init__(
        self, 
        eigval_trunc_dim: int, 
        eigvec_trunc_dim: int, 
        gt_1q_dim: int, 
        gt_2q_dim: int,
        stab_sign_dim: int,
        pauli_dim: int
    ):
        self.eigval_trunc_dim = eigval_trunc_dim
        self.eigvec_trunc_dim = eigvec_trunc_dim
        self.gt_1q_dim = gt_1q_dim
        self.gt_2q_dim = gt_2q_dim
        self.stab_sign_dim = stab_sign_dim
        self.pauli_dim = pauli_dim
    
        self.init_token_dims()

    def init_token_dims(self) -> None:
        self.dA = self.eigval_trunc_dim
        self.dB = 2 * self.eigvec_trunc_dim # Positional encoding will be of the same dim as the eigvec

        assert self.gt_1q_dim == self.gt_2q_dim, \
            f"""Embedding dim for 1 qubit GateType should equal embedding dim for 2 qubit GateType. Instead
            received 1 qubit GateType dim -> {self.gt_1q_dim} and 2 qubit GateType dim {self.gt_2q_dim} respectively."""
        assert self.gt_2q_dim == 2 * self.dB, \
            f"""2 qubit gate embedding dim must be 2 * Token B dim (by construction). Instead received
            2 qubit GateType dim -> {self.gt_2q_dim} and Token B dim -> {self.dB} respectively."""
        assert self.pauli_dim == self.dB, \
            f"""Pauli embedding dim must be the same as Token B dim (by construction). Instead received
            Pauli embed dim -> {self.pauli_dim} and Token B dim -> {self.dB} respectively."""

        self.dC = self.gt_2q_dim
        self.dD = self.stab_sign_dim
        self.dE = self.pauli_dim
        