import time

import matplotlib.pyplot as plt
import numpy as np
import qiskit
import torch
import torch.nn as nn

from inference.infer import InferWrapper, format_observation, get_gate_literals, name_dict
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


def get_qiskit_circuit(n: int, layout: np.ndarray, gate_set: np.ndarray, gates: np.ndarray):
    qc = qiskit.QuantumCircuit(n)
    gates = get_gate_literals(gates, layout, gate_set)
    for gate in gates:
        split_parts = gate.split("-")
        gate_type = split_parts[0]
        qubit_indices = [int(idx) for idx in split_parts[1:]]
        match gate_type:
            case "H":
                qc.h(qubit_indices[0])
            case "S":
                qc.s(qubit_indices[0])
            case "Sdg":
                qc.sdg(qubit_indices[0])
            case "Z":
                qc.z(qubit_indices[0])
            case "X":
                qc.x(qubit_indices[0])
            case "sqrtX":
                qc.sx(qubit_indices[0])
            case "sqrtXdg":
                qc.sxdg(qubit_indices[0])
            case "CX":
                qc.cx(qubit_indices[0], qubit_indices[1])
            case "CZ":
                qc.cz(qubit_indices[0], qubit_indices[1])
            case _:
                raise ValueError(f"Unknown gate type: {gate_type}")
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
    for i, data in enumerate(iter(dataset)):
        if i > 1:
            break
        assert data["layout"].shape[0] == 1, "Batch size should be 1."
        n = data["layout"].shape[-1]

        layout = data["layout"][:, :, :]
        eigval = data["eigval"][:, :]
        eigvec = data["eigvec"][:, :, :]
        gate_oh = data["gate_oh"][:, :]
        gate_qubit_oh = data["gate_qubit_oh"][:, :]
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
        print(
            "All paths populated? ",
            not any([x is None for x in output_paths]),
        )
        idx = 0
        path = output_paths[idx]
        path.gates = path.gates.numpy()
        gate_set = np.unique(gate_oh[idx])
        qcs: list[qiskit.QuantumCircuit] = []
        for gates in path.gates:
            qcs.append(get_qiskit_circuit(n, layout[idx], gate_set, gates))

        print(
            f"Input state: {format_observation(observation[idx], n)}, {','.join([str(x) for x in observation[idx].astype(np.int_)])}"
        )
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

        # print(f"Gate debug:")
        # for gates in path.gates:
        #     print(f"{get_gate_literals(gates, data['layout'][idx], gate_set)}")
        print(
            f"True gate: {get_gate_literals(data["gate"], data["layout"][idx], gate_set)[idx]}, {data["gate"][idx]}"
        )
        depth_prediction = (path.depths[idx][0] + 2.2) * 2
        print(
            f"Depth: {path.depths[0]} (predicted) vs {len(path.gates[0])} (inferred) vs {data["depth"][idx]} (actual)"
        )
        # print("\n\n\n")

        # Print gate losses over inference train


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--beam-width", type=int, default=5, required=True, help="Beam width")
    parser.add_argument("--max-depth", type=int, default=10, required=True, help="Max depth")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicates")
    args = parser.parse_args()
    print(f"Args: {args}")

    seed = 10
    np.random.seed(seed)
    model_file = "output/full_run_2_10-epochs=10-lr=0.001-beta=(0.9, 0.999)-iter-9/model-9-35357.pt"
    dummy_wrapper = InferWrapper(
        ModelV0(
            128,
            32,
            64,
            32,
            32,
            hetero_attention_embed_dim=100,
        ),
        model_file,
        1,
    )

    dataset = UnprepNpzDataloader("training-data/split/2-10-validation.npz", shuffle=True)
    dataset.set_batch_size(1)

    run_inference_train(
        dummy_wrapper.model, dataset, args.max_depth, args.beam_width, args.remove_duplicates
    )
