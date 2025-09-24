import torch
import torch.nn as nn
import torch.optim as optim

from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader

model = ModelV0(
    128,
    32,
    64,
    32,
    32,
    hetero_attention_embed_dim=100,
)
saved_data = torch.load(
    "output/overfit_2_10-epochs=5000-lr=0.001-beta=(0.9, 0.999)-iter-2/model-2000-0.pt",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(saved_data["model_state_dict"])
torch.set_grad_enabled(False)
model.eval()

train_data = UnprepNpzDataloader("training-data/compiled/2-10-train.npz", shuffle=False)
data = next(iter(train_data))

gate_prediction, depth_prediction = model.forward(
    torch.tensor(data["eigval"], dtype=torch.float),
    torch.tensor(data["eigvec"], dtype=torch.float),
    torch.tensor(data["gate_oh"], dtype=torch.long),
    torch.tensor(data["gate_qubit_oh"], dtype=torch.long),
    torch.tensor(data["observation"], dtype=torch.bool),
)
print(nn.Softmax(-1)(gate_prediction.detach()).argmax(-1).numpy())
print(data["gate"])

n = data["eigval"].shape[1]
print((depth_prediction.detach().numpy() + 1) * n * n / 4)
print(data["depth"])
