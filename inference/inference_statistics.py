import time

import numpy as np
import torch
import torch.nn as nn

from inference.infer import (
    InferWrapper,
    Path,
    format_observation,
    get_gate_literals,
    name_dict,
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


def elapsed_str(elapsed, curr_batch_idx, total_batches):
    avg_time = elapsed / (curr_batch_idx + 1) if curr_batch_idx > 0 else 0
    remaining_batches = total_batches - (curr_batch_idx + 1)
    est_remaining = avg_time * remaining_batches
    if est_remaining < 60:
        est_str = f"{est_remaining:.2f} seconds"
    elif est_remaining < 3600:
        est_str = f"{est_remaining/60:.2f} minutes"
    else:
        est_str = f"{est_remaining/3600:.2f} hours"
    return f"Iterated over {curr_batch_idx} batches ({elapsed} s)| Avg time: {avg_time:.7f} s/batch | Estimated time left for epoch: {est_str}"


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
    "output/full_run_2_10-epochs=20-lr=0.001-beta=(0.9, 0.999)-iter-6/model-13-10828.pt",
    20,
)

full_dataset = UnprepNpzDataloader("training-data/split/2-10-validation.npz")

# full_dataset.set_batch_size(1)
np.random.seed(1)

unprepped_successfully = []  # Unprepped correctly
unprepped_optimally = []  # Unprepped in the correct number of gates
depth_prediction = []  # Model's prediction of depth
depth_inference = []  # Max depth that model tried for before termination
actual_depth = []

tic = time.time()
for i, data in enumerate(iter(full_dataset)):
    if i >= 10:
        break
    layout = np.squeeze(data["layout"][0, :, :])
    n = layout.shape[0]
    gate_set = np.unique(np.squeeze(data["gate_oh"][0, :])).astype(int)
    target = np.squeeze(data["observation"][0, :]).astype(int)

    # curr_beam, success = wrapper.infer(layout, gate_set, target.flatten(), beam_width=5)
    curr_beam, success = wrapper.infer_batch(
        data["layout"],
        data["eigval"],
        data["eigvec"],
        data["gate_oh"],
        data["gate_qubit_oh"],
        data["observation"],
        beam_width=5,
    )

    # unprepped_successfully.append(success)
    # for path in curr_beam:
    #     if Path.is_successfully_unprepared([path], n):
    #         # All paths are guaranteed to be of same depth, so break out of loop
    #         # once depth condition has been checked.
    #         if data["depth"][0] == len(path.gates):
    #             unprepped_optimally.append(True)
    #         break

    # depth_prediction.append(curr_beam[0].depths[0])
    # if success:
    #     depth_inference.append(len(curr_beam[0].gates))
    # else:
    #     depth_inference.append(len(curr_beam[0].gates) + 1)

    # actual_depth.append(data["depth"][0])

    if i % 10 == 0:
        print(elapsed_str(time.time() - tic, i, full_dataset.get_total_size()))

# unprepped_successfully = np.array(unprepped_successfully)
# unprepped_optimally = np.array(unprepped_optimally)
# depth_inference = np.array(depth_inference)
# actual_depth = np.array(actual_depth)

# print(f"Total datapoints: {full_dataset.get_total_size()}")
# print(f"Number of correct inferences: {np.count_nonzero(unprepped_successfully)}")
# print(f"Number of optimal inferences: {np.count_nonzero(unprepped_optimally)}")
# print(f"Depth metric:")
# print(f"    Average: {np.mean(depth_inference)}")
# print(f"    Median: {np.median(depth_inference)}")
# print(f"    Stddev: {np.std(depth_inference)}")

# print(f"Actual depths:")
# print(f"    Average: {np.mean(actual_depth)}")
# print(f"    Median: {np.median(actual_depth)}")
# print(f"    Stddev: {np.std(actual_depth)}")

# difference = actual_depth - depth_inference
# print(f"Difference:")
# print(f"    Average: {np.mean(difference)}")
# print(f"    Median: {np.median(difference)}")
# print(f"    Stddev: {np.std(difference)}")
