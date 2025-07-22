import os
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_v0 import ModelV0
from training.dataset import UnprepHdf5Dataloader

seed = time.time_ns()

# Torch RNG
gen = torch.Generator()
gen.manual_seed(seed)

np.random.seed(seed % 2**32)


class Trainer:
    def __init__(
        self,
        train_file,
        validation_file,
        test_file,
        checkpoint_folder,
        lr=0.001,
        betas=(0.9, 0.999),
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas)

        self.train_data = UnprepHdf5Dataloader(train_file)
        self.validation_data = (
            UnprepHdf5Dataloader(validation_file)
            if validation_file is not None
            else None
        )
        self.test_data = (
            UnprepHdf5Dataloader(test_file) if test_file is not None else None
        )
        print("Total sizes of datasets:")
        print(f"Train - {self.train_data.get_total_size()}")
        print(
            f"Validation - {self.validation_data.get_total_size() if self.validation_data is not None else 0}"
        )
        print(
            f"Test - {self.test_data.get_total_size() if self.test_data is not None else 0}"
        )

        self.batch_size = 64

        self.gate_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.MSELoss()
        self.alpha = 1

        self.checkpoint_folder = checkpoint_folder

    def compute_loss(self, gate_prediction, depth_prediction, true_gates, true_depth):
        return self.gate_loss(
            gate_prediction, true_gates
        ) + self.alpha * self.depth_loss(depth_prediction, true_depth.float())

    def run_model(self, data, use_grad=True, use_eval=False):
        with torch.set_grad_enabled(use_grad):
            if use_eval:
                self.model.eval()
            gate_prediction, depth_prediction = self.model.forward(
                torch.tensor(data["eigval"], dtype=torch.float).to(self.device),
                torch.tensor(data["eigvec"], dtype=torch.float).to(self.device),
                torch.tensor(data["gate_oh"], dtype=torch.bool).to(self.device),
                torch.tensor(data["gate_qubit_oh"], dtype=torch.bool).to(self.device),
                torch.tensor(data["observation"], dtype=torch.bool).to(self.device),
            )
            if use_eval:
                self.model.train()
        return gate_prediction, depth_prediction

    def set_device(self):
        if hasattr(self, "device") and self.device is not None:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model now on device={self.device}")

    def train(self, epochs=1):
        self.set_device()
        validation_loss_history = []
        training_loss_history = []

        self.train_data.set_ng_iter_idx(531)  # corresponding to (n, g) = (3, 7)
        self.train_data.set_batch_size(self.batch_size)
        train_data = next(iter(self.train_data))

        total_size = self.train_data.get_total_size()
        for epoch in range(epochs):
            n_iter = int(total_size // self.batch_size)  #
            for i in range(n_iter):
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
                training_loss_history.append(loss.detach().cpu().item())

                if i % 500 == 0:
                    self.dump_loss_history(training_loss_history)

            # Also store at the end of the model
            self.dump_loss_history(training_loss_history)

    def calculate_validation_score(self):
        self.set_device()
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
        return average_loss

    def dump_loss_history(self, loss_history):
        file = f"{self.checkpoint_folder}/training_loss.npy"
        np.save(file, np.array(loss_history))

    def store_checkpoint(self, epoch, iter_idx, validation_loss, train_loss):
        torch.save(
            {
                "epoch": epoch,
                "iter_idx": iter_idx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "validation_loss": validation_loss,
                "loss": train_loss,
            },
            f"{self.checkpoint_folder}/model-{epoch}-{iter_idx}.pt",
        )

    def test(self):
        raise NotImplementedError


def create_new_folder(prefix: str, args: Namespace):
    name = f"{args.expid}-lr={args.lr}-beta={args.beta1,args.beta2}-{args.name}"
    folder = f"{prefix}/{name}"
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


if __name__ == "__main__":
    # INSERT_YOUR_CODE
    import argparse

    parser = argparse.ArgumentParser(description="Trainer hyperparameters")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 for optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for"
    )
    parser.add_argument("--name", type=str, default="", help="Name suffix for folder")
    parser.add_argument("--expid", type=str, default="", help="Name suffix for folder")
    args = parser.parse_args()

    print(args)

    device = "hpc"
    if device == "hpc":
        train_file = "/scratch1/sauravk/lsp-hdf5/sample-test.hdf5"
        validation_file = None  # "/scratch1/sauravk/lsp-hdf5/sample-validation.hdf5"
        test_file = None  # "/scratch1/sauravk/lsp-hdf5/sample-test.hdf5"
        model_output_folder = create_new_folder("/scratch1/sauravk/models", args)
    else:
        train_file = "training-data/compiled/hdf5/sample-test.hdf5"
        validation_file = None  # "training-data/compiled/hdf5/sample-validation.hdf5"
        test_file = None  # "training-data/compiled/hdf5/sample-test.hdf5"
        model_output_folder = create_new_folder("output", args)
    print(f"Output folder: {model_output_folder}")

    trainer = Trainer(
        train_file,
        validation_file,
        test_file,
        model_output_folder,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    trainer.train(epochs=args.epochs)
    print("Training complete")
