import os
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader

seed = time.time_ns()

# Torch RNG
gen = torch.Generator()
gen.manual_seed(seed)

np.random.seed(seed % 2**32)


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


class Trainer:
    def __init__(
        self,
        train_file,
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

        # Initialize the dataloaders
        self.train_data = UnprepNpzDataloader(train_file, shuffle=False)

        print("Total sizes of datasets:")
        print(f"Train - {self.train_data.get_total_size()}")

        self.batch_size = 64

        self.gate_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.MSELoss()
        self.alpha = 1

        self.checkpoint_folder = checkpoint_folder

    def compute_loss(
        self, n, gate_prediction, depth_prediction, true_gates, true_depth
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate_loss(gate_prediction, true_gates), self.alpha * (
            self.depth_loss(depth_prediction, true_depth.float() * 4 / (n**2) - 1)
        )

    def run_model(self, data, use_grad=True, use_eval=False):
        with torch.set_grad_enabled(use_grad):
            if use_eval:
                self.model.eval()
            gate_prediction, depth_prediction = self.model.forward(
                torch.tensor(data["eigval"], dtype=torch.float).to(self.device),
                torch.tensor(data["eigvec"], dtype=torch.float).to(self.device),
                torch.tensor(data["gates"], dtype=torch.long).to(self.device),
                torch.tensor(data["gate_qubits"], dtype=torch.long).to(self.device),
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
        gate_loss_history, depth_loss_history = [], []
        training_loss_history = []

        train_data = next(iter(self.train_data))
        tic = batch_tic = time.time()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            gate_prediction, depth_prediction = self.run_model(train_data)
            gate_loss, depth_loss = self.compute_loss(
                train_data["eigval"].shape[1],  # n
                gate_prediction,
                depth_prediction,
                torch.tensor(train_data["unprep_gate"], dtype=torch.int64).to(self.device),
                torch.tensor(train_data["depth"], dtype=torch.int64).to(self.device),
            )
            loss = gate_loss + depth_loss
            loss.backward()
            self.optimizer.step()
            training_loss_history.append(loss.detach().cpu().item())
            gate_loss_history.append(gate_loss.detach().cpu().item())
            depth_loss_history.append(depth_loss.detach().cpu().item())

            if epoch % 500 == 0:
                self.dump_loss_history(
                    training_loss_history, gate_loss_history, depth_loss_history
                )
                print(elapsed_str(time.time() - tic, time.time() - batch_tic, epoch, epochs))
                batch_tic = time.time()
                self.store_checkpoint(epoch, 0, None, None)

        # Also store at the end of the model
        self.dump_loss_history(
            training_loss_history, gate_loss_history, depth_loss_history
        )
        self.store_checkpoint(epoch, 0, None, None)

    def dump_loss_history(
        self, loss_history, gate_loss_history=None, depth_loss_history=None
    ):
        file = f"{self.checkpoint_folder}/training_loss.npy"
        np.save(file, np.array(loss_history))
        if gate_loss_history is not None:
            np.save(
                f"{self.checkpoint_folder}/gate_loss.npy", np.array(gate_loss_history)
            )
        if depth_loss_history is not None:
            np.save(
                f"{self.checkpoint_folder}/depth_loss.npy", np.array(depth_loss_history)
            )

    def store_checkpoint(self, epoch, iter_idx, validation_loss, train_loss):
        torch.save(
            {
                "epoch": epoch,
                "iter_idx": iter_idx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": train_loss,
            },
            f"{self.checkpoint_folder}/model-{epoch}-{iter_idx}.pt",
        )

    def test(self):
        raise NotImplementedError


def create_new_folder(prefix: str, args: Namespace):
    name = f"{args.expid}-epochs={args.epochs}-lr={args.lr}-beta={args.beta1,args.beta2}-{args.name}"
    folder = f"{prefix}/{name}"
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


def update_metadata_with_args(args: Namespace, metadata: dict):
    metadata["args"] = {}
    for key, value in args._get_kwargs():
        metadata["args"][key] = value


def update_metadata_with_system(metadata: dict):
    metadata["cpu_count"] = [os.cpu_count()]
    if torch.cuda.is_available():
        metadata["gpu_count"] = torch.cuda.device_count()
        metadata["gpu"] = []
        for i in range(metadata["gpu_count"]):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_properties = torch.cuda.get_device_properties(i)
            metadata["gpu"].append(
                {
                    "name": gpu_name,
                    "memory": gpu_properties.total_memory / (1024**3),
                    "multi_processor_count": gpu_properties.multi_processor_count,
                }
            )


def dump_metadata(folder: str, metadata: dict):
    import json

    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f)


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
    parser.add_argument("--expid", type=str, default="", help="Name prefix for folder")
    parser.add_argument(
        "--device", type=str, default="hpc", help="Name suffix for folder"
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    if args.device == "hpc":
        args.train_file = "/scratch1/sauravk/lsp-hdf5/sample-train.hdf5"
        model_output_folder = create_new_folder("/scratch1/sauravk/models", args)
    elif args.device == "mac":
        args.train_file = "training-data/compiled/hdf5/sample-train.hdf5"
        model_output_folder = create_new_folder("output", args)
    elif args.device == "qserver":
        args.train_file = "training-data/split/2-10-train.npz"
        model_output_folder = create_new_folder("output", args)
    print(f"Output folder: {model_output_folder}")

    seed = 1
    np.random.seed(1)
    trainer = Trainer(
        args.train_file,
        model_output_folder,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    metadata = {}
    update_metadata_with_args(args, metadata)
    update_metadata_with_system(metadata)
    metadata["train_size"] = int(trainer.train_data.get_total_size())
    metadata["train_batches"] = int(
        trainer.train_data.get_total_size() // trainer.train_data.batch_size
    )
    dump_metadata(model_output_folder, metadata)

    tic = time.time()
    trainer.train(epochs=args.epochs)
    toc = time.time()
    print(f"Training complete ({toc-tic} s)")
    metadata["train_time"] = toc - tic

    dump_metadata(model_output_folder, metadata)
