import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset

from models.model_v0 import ModelV0
from training.dataset import UnprepHdf5Dataloader

seed = time.time_ns()

# Torch RNG
gen = torch.Generator()
gen.manual_seed(seed)

np.random.seed(seed % 2**32)


class Trainer:
    def __init__(
        self, train_file, validation_file, test_file, checkpoint_folder
    ) -> None:
        hetero_attention_embed_dim = 100
        self.model = ModelV0(
            128,
            32,
            64,
            32,
            32,
            hetero_attention_embed_dim=hetero_attention_embed_dim,
        )
        self.optimizer = optim.Adam(self.model.parameters())

        self.train_data = UnprepHdf5Dataloader(train_file)
        self.validation_data = UnprepHdf5Dataloader(validation_file)
        self.test_data = UnprepHdf5Dataloader(test_file)

        self.batch_size = 64

        self.gate_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.MSELoss()
        self.alpha = 1

        self.checkpoint_folder = checkpoint_folder

    def compute_loss(self, gate_prediction, depth_prediction, true_gates, true_depth):
        return self.gate_loss(
            gate_prediction, true_gates
        ) + self.alpha * self.depth_loss(depth_prediction, true_depth.float())

    def run_model(self, data, use_grad=True):
        with torch.set_grad_enabled(use_grad):
            gate_prediction, depth_prediction = self.model.forward(
                torch.tensor(data["eigval"], dtype=torch.float).to(self.device),
                torch.tensor(data["eigvec"], dtype=torch.float).to(self.device),
                torch.tensor(data["gate_oh"], dtype=torch.bool).to(self.device),
                torch.tensor(data["gate_qubit_oh"], dtype=torch.bool).to(self.device),
                torch.tensor(data["observation"], dtype=torch.bool).to(self.device),
            )
        return gate_prediction, depth_prediction

    def set_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device is {self.device}")

    def train(self, epochs=1):
        self.set_device()
        self.model.to(self.device)
        validation_loss_history = []

        total_size = self.train_data.get_total_size()
        for epoch in range(epochs):
            n_iter = 10  # int(total_size // self.batch_size)
            for i in range(n_iter):
                n, g = self.train_data.random_sample_ng()
                train_data = self.train_data.random_sample_data(n, g, self.batch_size)

                self.optimizer.zero_grad()
                gate_prediction, depth_prediction = self.run_model(train_data)
                loss = self.compute_loss(
                    gate_prediction,
                    depth_prediction,
                    torch.tensor(train_data["gate"], dtype=torch.int64).to(self.device),
                    torch.tensor(train_data["depth"], dtype=torch.int64).to(
                        self.device
                    ),
                )
                loss.backward()

                if i % 500 == 0 and i > 0:
                    validation_loss_history.append(self.calculate_validation_score())
                    self.store_checkpoint(
                        epoch,
                        i,
                        loss.detach().cpu().item(),
                        validation_loss_history[-1],
                    )

            # Also store at the end of the model
            validation_loss_history.append(self.calculate_validation_score())
            self.store_checkpoint(
                epoch,
                i,
                loss.detach().cpu().item(),
                validation_loss_history[-1],
            )

    def calculate_validation_score(self):
        total_loss = 0.0
        total_samples = 0
        for data in iter(self.validation_data):
            gate_prediction, depth_prediction = self.run_model(data, use_grad=False)
            loss = self.compute_loss(
                gate_prediction,
                depth_prediction,
                torch.tensor(data["gate"], dtype=torch.int64).to(self.device),
                torch.tensor(data["gate"], dtype=torch.int64).to(self.device),
            )
            batch_size = data["gate"].shape[0]
            total_loss += loss.cpu().item() * batch_size
            total_samples += batch_size
        average_loss = total_loss / total_samples

    def store_checkpoint(self, epoch, iter_idx, train_loss, validation_loss):
        torch.save(
            {
                "epoch": epoch,
                "iter_idx": iter_idx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": train_loss,
                "validation_loss": validation_loss,
            },
            f"{self.checkpoint_folder}/model-{epoch}-{iter_idx}.pt",
        )

    def test(self):
        raise NotImplementedError


if __name__ == "__main__":
    device = "hpc"
    if device == "hpc":
        train_file = "/scratch1/sauravk/lsp-hdf5/sample-train.hdf5"
        validation_file = "/scratch1/sauravk/lsp-hdf5/sample-validation.hdf5"
        test_file = "/scratch1/sauravk/lsp-hdf5/sample-test.hdf5"
        model_output_folder = "/scratch1/sauravk/models"
    else:
        train_file = "training-data/compiled/hdf5/sample-train.hdf5"
        validation_file = "training-data/compiled/hdf5/sample-validation.hdf5"
        test_file = "training-data/compiled/hdf5/sample-test.hdf5"
        model_output_folder = "models/25-07-17"

    trainer = Trainer(
        train_file,
        validation_file,
        test_file,
        model_output_folder,
    )
    trainer.train()
