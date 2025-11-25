import time

import numpy as np
import torch
import torch.nn as nn

from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader
from training.trainer import normalize_depth, unnormalize_depth


def elapsed_str(elapsed_bot, elapsed_bob, curr_batch_idx, total_batches):
    avg_time = elapsed_bot / (curr_batch_idx + 1) if curr_batch_idx > 0 else 0
    remaining_batches = total_batches - (curr_batch_idx + 1)
    est_remaining = avg_time * remaining_batches
    if est_remaining < 60:
        est_str = f"{est_remaining:.2f} seconds"
    elif est_remaining < 3600:
        est_str = f"{est_remaining/60:.2f} minutes"
    else:
        est_str = f"{est_remaining/3600:.2f} hours"
    return f"Iterated over {curr_batch_idx} epochs ({elapsed_bob} s)| Avg time: {avg_time:.7f} s/batch | Estimated time left for epoch: {est_str}"


def run_and_dump_output(model: nn.Module, dataset: UnprepNpzDataloader, output_file: str):
    n_data = torch.empty(0, dtype=torch.int)
    g_data = torch.empty(0, dtype=torch.int)
    true_depth_data = torch.empty(0, dtype=torch.int)
    depth_loss_data = torch.empty(0, dtype=torch.float)
    gate_loss_data = torch.empty(0, dtype=torch.float)

    num_batches = dataset.get_total_size() / dataset.batch_size
    device = next(model.parameters()).device
    tic = batch_tic = time.time()
    for i, data in enumerate(iter(dataset)):
        with torch.no_grad():
            gate_prediction, depth_prediction = model.forward(
                torch.tensor(data["eigval"], dtype=torch.float, device=device),
                torch.tensor(data["eigvec"], dtype=torch.float, device=device),
                torch.tensor(data["gates"], dtype=torch.long, device=device),
                torch.tensor(data["gate_qubits"], dtype=torch.long, device=device),
                torch.tensor(data["observation"], dtype=torch.bool, device=device),
            )
        gate_loss = nn.CrossEntropyLoss(reduction="none")(
            gate_prediction, torch.tensor(data["unprep_gate"], dtype=torch.int64, device=device)
        )
        gate_loss_data = torch.concat((gate_loss_data, gate_loss.to("cpu")))

        n = data["layout"].shape[-1]
        g = data["gates"].shape[-1]
        n_data = torch.concat((n_data, n * torch.ones(data["layout"].shape[0])))
        g_data = torch.concat((g_data, g * torch.ones(data["layout"].shape[0])))

        true_depth = torch.tensor(data["depth"], dtype=torch.float, device=device)
        depth_loss = nn.MSELoss(reduction="none")(
            depth_prediction,
            normalize_depth(true_depth, n),
        ).to("cpu")
        depth_loss_data = torch.concat((depth_loss_data, depth_loss))
        true_depth_data = torch.concat((true_depth_data, true_depth.to("cpu")))

        if i % 500 == 0:
            print(elapsed_str(time.time() - tic, time.time() - batch_tic, i, num_batches))
            batch_tic = time.time()

    print(f"Dumping to {output_file}")
    torch.save(
        {
            "n_data": n_data,
            "g_data": g_data,
            "true_depth_data": true_depth_data,
            "depth_loss_data": depth_loss_data,
            "gate_loss_data": gate_loss_data,
        },
        output_file,
    )


if __name__ == "__main__":
    import os

    model = ModelV0(
        128,
        32,
        64,
        32,
        32,
        hetero_attention_embed_dim=100,
    )
    file = "output/model.pt"
    saved_data = torch.load(file, weights_only=True)
    model.load_state_dict(saved_data["model_state_dict"])
    model.to("cuda")
    model.eval()

    # # Check loss values for samples picked directly from the training set.
    # np.random.seed(4)
    dataset = UnprepNpzDataloader("training-data/validation.npz")
    run_and_dump_output(model, dataset, f"{os.path.dirname(file)}/model_output_data.pt")
