"""
Regression test for `Token_C_Embedding.forward`.

`ModelV0.forward` used to compute `torch.ones_like(gates_oh) * n` and pass it as an
unused leading `n` argument into `Token_C_Embedding.forward`, which just forwarded
its arguments to `GateEmbedding.forward` (which never reads `n`). Both the dead
compute and the dead parameter were removed; this test pins the resulting 3-arg
signature and checks the output is unchanged (identical to calling
`GateEmbedding` directly with the same three tensors).
"""

import torch

from models.embeddings import GateEmbedding
from models.token_embeddings import Token_C_Embedding


class _FakeTokenDims:
    C_gt_1q_dim = 8


def test_token_c_embedding_forward_matches_gate_embedding_directly():
    torch.manual_seed(0)
    bs, g, n, embed_dim = 2, 3, 4, _FakeTokenDims.C_gt_1q_dim
    gates_oh = torch.randint(0, 9, (bs, g))
    gate_qubits_oh = torch.randint(0, n + 1, (bs, 2 * g))
    qubits = torch.randn(bs, n, embed_dim // 2)

    token_c = Token_C_Embedding(_FakeTokenDims())
    out = token_c(gates_oh, gate_qubits_oh, qubits)

    direct = token_c.gate_embedding_layer(gates_oh, gate_qubits_oh, qubits)
    assert torch.equal(out, direct)
    assert out.shape == (bs, g, embed_dim)


def test_token_c_embedding_forward_rejects_old_four_arg_call():
    token_c = Token_C_Embedding(_FakeTokenDims())
    try:
        token_c(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))
    except TypeError:
        return
    raise AssertionError("expected TypeError: forward no longer accepts a leading `n` argument")


if __name__ == "__main__":
    test_token_c_embedding_forward_matches_gate_embedding_directly()
    test_token_c_embedding_forward_rejects_old_four_arg_call()
    print("All checks passed.")
