import numpy as np
import torch
import torch.nn as nn

from inference.infer import (
    InferWrapper,
    Simulator,
    format_observation,
    get_gate_literals,
    name_dict,
    print_gate_keys,
)
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

model = ModelV0(
    128,
    32,
    64,
    32,
    32,
    hetero_attention_embed_dim=100,
)

wrapper = InferWrapper(
    model,
    # "output/full_run-epochs=5-lr=0.001-beta=(0.9, 0.999)-iter-9/model-1-10000.pt",
    "output/full_run_2_10-epochs=5-lr=0.001-beta=(0.9, 0.999)-iter-2/model-3-15000.pt",
    5,
)

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

# # Check loss values for samples picked directly from the training set.
# np.random.seed(4)
train_data = UnprepNpzDataloader("training-data/compiled/2-10-validation.npz")
# for i, data in enumerate(iter(train_data)):
#     # data = next(iter(train_data))
#     if i > 5:
#         break
#     gate_prediction, depth_prediction = model.forward(
#         torch.tensor(data["eigval"], dtype=torch.float),
#         torch.tensor(data["eigvec"], dtype=torch.float),
#         torch.tensor(data["gate_oh"], dtype=torch.long),
#         torch.tensor(data["gate_qubit_oh"], dtype=torch.long),
#         torch.tensor(data["observation"], dtype=torch.bool),
#     )
#     # print(
#     #     f"""
#     # Input:
#     # Eigval -> {data["eigval"][0, :]}
#     # Eigvec -> {data["eigvec"][0, :, :]}
#     # Gates -> {data["gate_oh"][0, :]}
#     # Gate qubits -> {data["gate_qubit_oh"][0, :].reshape(-1, 2)}
#     # Observation -> {data["observation"][0,:].astype(int)}
#     # """
#     # )

#     gate_loss = nn.CrossEntropyLoss()(
#         gate_prediction, torch.tensor(data["gate"], dtype=torch.int64)
#     )
#     print(gate_prediction[0, :].detach().numpy())
#     print(f"Gate loss for {i} batch {gate_loss}")

# print()
# print()

# train_data.set_batch_size(1)
np.random.seed(1)
for i, data in enumerate(iter(train_data)):
    # data = next(iter(train_data))
    if i >= 4:
        break
    layout = np.squeeze(data["layout"][0, :, :])
    n = layout.shape[0]
    gate_set = np.unique(np.squeeze(data["gate_oh"][0, :])).astype(int)
    target = np.squeeze(data["observation"][0, :]).astype(int)

    # simulator = Simulator(layout, gate_set, None)
    # simulator.set_state(target.flatten())
    # simulator.step(0)
    # print(simulator.state)

    print(f"Num qubits: {n}")
    print(f"Available gates: {[name_dict[gate] for gate in gate_set]}")
    # print(f"Layout: {layout}")
    print(f"Starting state: {format_observation(target.flatten(), n)}")
    # print_gate_keys(layout, gate_set)

    curr_beam, success = wrapper.infer(layout, gate_set, target.flatten(), beam_width=5)
    print(f"Success? {'YESS!!!!!!!' if success else ':('}")
    # gate_list, observation_list, depth_list, gp_logit = wrapper.infer(
    #     layout, gate_set, target.flatten()
    # )
    # gates = wrapper.get_gate_literals(gate_list, layout, gate_set)
    # stabilizers = [format_observation(x, n) for x in observation_list]
    # for g, s in zip(gates, stabilizers[1:]):
    #     print(f"{g}: {s}")
    # print(depth_list)
    # print(
    #     nn.CrossEntropyLoss()(
    #         gp_logit, torch.tensor([data["gate"][0]], dtype=torch.long)
    #     )
    # )

    print()
    print(f"Actual depth: {np.squeeze(data['depth'][0])}")
    print(
        f"Actual gate: {np.squeeze(data['gate'][0])} = {get_gate_literals([data['gate'][0]], layout, gate_set)[0]}"
    )
    print()
    print()
