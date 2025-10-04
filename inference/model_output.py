import numpy as np
import torch
import torch.nn as nn

from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader

device = "cuda"

# Initialize and load the model
model = ModelV0(
    128,
    32,
    64,
    32,
    32,
    hetero_attention_embed_dim=100,
)
model.to(device)
saved_data = torch.load(
    "output/full_run_2_10-epochs=20-lr=0.001-beta=(0.9, 0.999)-iter-6/model-19-10828.pt",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(saved_data["model_state_dict"])
model.eval()

# # Check loss values for samples picked directly from the training set.
# np.random.seed(4)
full_dataset = UnprepNpzDataloader("training-data/split/2-10-validation.npz")

n_data = torch.empty(0, dtype=torch.int)
g_data = torch.empty(0, dtype=torch.int)
true_depth_data = torch.empty(0, dtype=torch.int)
depth_loss_data = torch.empty(0, dtype=torch.float)
gate_loss_data = torch.empty(0, dtype=torch.float)

for i, data in enumerate(iter(full_dataset)):
    with torch.no_grad():
        gate_prediction, depth_prediction = model.forward(
            torch.tensor(data["eigval"], dtype=torch.float, device=device),
            torch.tensor(data["eigvec"], dtype=torch.float, device=device),
            torch.tensor(data["gate_oh"], dtype=torch.long, device=device),
            torch.tensor(data["gate_qubit_oh"], dtype=torch.long, device=device),
            torch.tensor(data["observation"], dtype=torch.bool, device=device),
        )
    gate_loss = nn.CrossEntropyLoss()(
        gate_prediction, torch.tensor(data["gate"], dtype=torch.int64, device=device)
    ).to("cpu")
    gate_loss_data = torch.concat((gate_loss_data, gate_loss.unsqueeze(0)))

    n = data["layout"].shape[-1]
    g = data["gate_oh"].shape[-1]
    n_data = torch.concat((n_data, n * torch.ones(data["layout"].shape[0])))
    g_data = torch.concat((g_data, g * torch.ones(data["layout"].shape[0])))

    true_depth = torch.tensor(data["depth"], dtype=torch.float, device=device)
    depth_loss = nn.MSELoss(reduction="none")(
        depth_prediction,
        true_depth / 2 - 2.2,
    ).to("cpu")
    # print(depth_loss.shape)
    depth_loss_data = torch.concat((depth_loss_data, depth_loss))
    true_depth_data = torch.concat((true_depth_data, true_depth.to("cpu")))

torch.save(
    {
        "n_data": n_data,
        "g_data": g_data,
        "true_depth_data": true_depth_data,
        "depth_loss_data": depth_loss_data,
        "gate_loss_data": gate_loss_data,
    },
    "output/model_output_data.pt",
)
