import time

import matplotlib.pyplot as plt
import numpy as np
import qiskit
import torch
import torch.nn as nn
from qiskit.circuit.library import UnitaryGate

from inference.infer import InferWrapper, format_observation, name_dict
from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader

"""
Gate set:
[
    lsp::GateType::H,       lsp::GateType::S,  lsp::GateType::Sdg,
    lsp::GateType::Z,       lsp::GateType::X,  lsp::GateType::sqrtX,
    lsp::GateType::sqrtXdg, lsp::GateType::CX, lsp::GateType::CZ
]
"""

# n = 3
# layout = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
# gate_set = np.array([0, 1, 3, 5, 7, 8])
# gate_set = np.random.permutation(gate_set)

# target = np.array(
#     [
#         [1, 0, 0, 1, 0, 0, 0],
#         [0, 1, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 1, 0],
#     ]
# )

_iswapdg = UnitaryGate(
    np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]]), label="iswapdg"
)

circuit_lambda_map = {
    0: lambda qc, x, y: qc.h(x),
    1: lambda qc, x, y: qc.s(x),
    2: lambda qc, x, y: qc.sdg(x),
    3: lambda qc, x, y: qc.z(x),
    4: lambda qc, x, y: qc.x(x),
    5: lambda qc, x, y: qc.sx(x),
    6: lambda qc, x, y: qc.sxdg(x),
    7: lambda qc, x, y: qc.cx(x, y),
    8: lambda qc, x, y: qc.cz(x, y),
    9: lambda qc, x, y: qc.iswap(x, y),
    10: lambda qc, x, y: qc.append(_iswapdg, [x, y]),
    11: lambda qc, x, y: qc.rzz(np.pi / 2, x, y),
    12: lambda qc, x, y: qc.rzz(-np.pi / 2, x, y),
    13: lambda qc, x, y: qc.rxx(np.pi / 2, x, y),
    14: lambda qc, x, y: qc.rxx(-np.pi / 2, x, y),
}


def get_qiskit_circuit(n: int, unitaries: np.ndarray, gates: np.ndarray, gate_qubits: np.ndarray):
    qc = qiskit.QuantumCircuit(n)
    # Refer to the input gates, gate_qubits array to extract details about the
    # gate under consideration (`unitaries`).
    gates = gates.reshape(-1)
    gate_qubits = gate_qubits.reshape(-1, 2)
    for u in unitaries:
        gate_idx = gates[u]
        q1, q2 = gate_qubits[u, 0], gate_qubits[u, 1]
        circuit_lambda_map[gate_idx](qc, q1 - 1, q2 - 1 if q2 > 0 else 0)
    return qc


def run_inference_train(
    model: nn.Module,
    dataset: "DataLoader",
    max_depth: int,
    beam_width: int,
    remove_duplicates: bool = False,
):
    wrapper = InferWrapper(model, None, max_depth)

    tic = time.time()
    count = 0
    for i, data in enumerate(iter(dataset)):
        assert data["layout"].shape[0] == 1, "Batch size should be 1."
        n = data["layout"].shape[-1]
        if count > 0:
            break
        count += 1

        layout = data["layout"][:, :, :]
        eigval = data["eigval"][:, :]
        eigvec = data["eigvec"][:, :, :]
        gate_oh = data["gates"][:, :]
        gate_qubit_oh = data["gate_qubits"][:, :]
        observation = data["observation"][:, :]

        output_paths = wrapper.infer_batch(
            layout,
            eigval,
            eigvec,
            gate_oh,
            gate_qubit_oh,
            observation,
            beam_width=beam_width,
            remove_duplicates=remove_duplicates,
        )
        print(f"Datapoint #{i} {sum([x.unprepped for x in output_paths])}\n")
        idx = 0
        path = output_paths[idx]
        path.gates = path.gates.numpy()
        gate_set = np.unique(gate_oh[idx])
        qcs: list[qiskit.QuantumCircuit] = []
        for gates in path.gates:
            qcs.append(get_qiskit_circuit(n, gates, gate_oh, gate_qubit_oh))

        print(f"Input state: ({n}) {format_observation(observation[idx], n)}")
        print(f"Allowed gates: {[name_dict[gate] for gate in gate_set]}")
        print("Unprepped successfully !!!!" if path.unprepped else "Unable to unprep :(")
        print(f"Proposed circuit:")
        for j, qc in enumerate(qcs):
            # fig = qc.draw(output="mpl")
            # plt.show()
            if not path.unprepped_list[j].item():
                continue
            print(qc)
            print(path.gates[j], path.unprepped_list[j])
            print("--------------------------------------------------------------")

        true_gate_idx = data["unprep_gate"][idx]
        gate_qubits = gate_qubit_oh.reshape(-1, 2)
        q1, q2 = gate_qubits[true_gate_idx, 0], gate_qubits[true_gate_idx, 1]
        print(f"True gate: {true_gate_idx}-{q1}-{q2}, {data["unprep_gate"][idx]}")
        depth_prediction = (path.depths[idx][0] + 2.2) * 2
        print(
            f"Depth: {depth_prediction} (predicted) vs {len(path.gates[0])} (inferred) vs {data["depth"][idx]} (actual)"
        )
        # print("\n\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--beam-width", type=int, default=5, required=True, help="Beam width")
    parser.add_argument("--max-depth", type=int, default=10, required=True, help="Max depth")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicates")
    parser.add_argument("--model-file", type=str, required=True, help="Model file")
    args = parser.parse_args()
    print(f"Args: {args}")

    dummy_wrapper = InferWrapper(
        ModelV0(
            128,
            32,
            64,
            32,
            32,
            hetero_attention_embed_dim=100,
        ),
        args.model_file,
        1,
    )

    seed = 1
    dataset = UnprepNpzDataloader(
        "training-data/split/2-10-validation.npz", shuffle=True, seed=seed
    )
    dataset.set_batch_size(1)

    run_inference_train(
        dummy_wrapper.model, dataset, args.max_depth, args.beam_width, args.remove_duplicates
    )
