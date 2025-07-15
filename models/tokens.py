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
        pauli_dim: int | None = None
    ):
        self.dA, self.dB, self.dC, self.dD, self.dE = dA, dB, dC, dD, dE

        self.set_token_A_dims(eigval_trunc_dim)
        self.set_token_B_dims(eigvec_trunc_dim)
        self.set_token_C_dims(gt_1q_dim, gt_2q_dim)
        self.set_token_C_dims(gt_1q_dim, gt_2q_dim)
        self.set_token_D_dims(stab_sign_dim)
        self.set_token_E_dims(pauli_dim)

    def set_token_A_dims(self, eigval_trunc_dim: int | None) -> None:
        self.A_eigval_trunc_dim : int = self.dA if eigval_trunc_dim is None else eigval_trunc_dim
        assert self.dA >= self.A_eigval_trunc_dim, \
            f"""Token A dim ({self.dA}) must be >= eigval_trunc_dim ({self.A_eigval_trunc_dim})"""
        
    def set_token_B_dims(self, eigvec_trunc_dim: int | None) -> None:
        if eigvec_trunc_dim is None:
            # Max 20 qubits, which can be stored in 6 bits
            self.B_eigvec_trunc_dim = self.dB - 6
            self.B_positional_embed_dim = 6
        else:
            self.B_eigvec_trunc_dim = eigvec_trunc_dim
            self.B_positional_embed_dim = self.dB - self.B_eigvec_trunc_dim
            assert self.B_positional_embed_dim > 0, \
                f"""eigvec_trunc_dim ({eigvec_trunc_dim}) cannot be greater than token B embedding 
                dim ({self.dB})"""
    
    def set_token_C_dims(self, gt_1q_dim : int | None, gt_2q_dim : int | None) -> None:
        assert self.dC >= 2 * self.dB, \
            f"""Token C dim ({self.dC}) must be >= twice Token B dim ({self.dB})"""
        
        self.C_gt_1q_dim : int = self.dC if gt_1q_dim is None else gt_1q_dim
        assert self.dC >= self.C_gt_1q_dim, \
            f"""Token C dim ({self.dC}) must be >= gt_1q_dim ({self.C_gt_1q_dim})"""
        self.C_pad_gt_1q_gate = max(self.dC - self.C_gt_1q_dim, 0)
        self.C_pad_gt_1q_qubit = self.dC - self.dB

        self.C_gt_2q_dim : int = self.dC if gt_2q_dim is None else gt_2q_dim
        assert self.dC >= self.C_gt_2q_dim, \
            f"""Token C dim ({self.dC}) must be >= gt_2q_dim ({self.C_gt_2q_dim})"""
        self.C_pad_gt_2q_gate = max(self.dC - self.C_gt_2q_dim, 0)
        self.C_pad_gt_2q_qubit = self.dC - 2 * self.dB
    
    def set_token_D_dims(self, stab_sign_dim: int | None) -> None:
        self.D_stab_sign_dim : int = self.dD if stab_sign_dim is None else stab_sign_dim
        assert self.dD >= self.D_stab_sign_dim, \
            f"""Token D dim ({self.dD}) must be >= stab_sign_dim ({self.D_stab_sign_dim})"""
        self.D_pad = self.dD - self.D_stab_sign_dim
    
    def set_token_E_dims(self, pauli_dim: int | None) -> None:
        self.E_pauli_dim : int = self.dE if pauli_dim is None else pauli_dim
        assert self.dE >= self.E_pauli_dim, \
            f"""Token E dim ({self.dE}) must be >= pauli_dim ({self.E_pauli_dim})"""
        self.E_pad_pauli = self.dE - self.E_pauli_dim
        assert self.dE >= self.dB, \
            f"""Token E dim ({self.dE}) must be >= Token B dim ({self.dB})"""
        self.E_pad_qubit = self.dE - self.dB
        