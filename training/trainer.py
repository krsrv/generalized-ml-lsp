import os
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader


def elapsed_str(elapsed_bot, elapsed_bob, curr_batch_idx, total_batches, epoch_idx):
    avg_time = elapsed_bot / (curr_batch_idx + 1) if curr_batch_idx > 0 else 0
    remaining_batches = total_batches - (curr_batch_idx + 1)
    est_remaining = avg_time * remaining_batches
    if est_remaining < 60:
        est_str = f"{est_remaining:.2f} seconds"
    elif est_remaining < 3600:
        est_str = f"{est_remaining/60:.2f} minutes"
    else:
        est_str = f"{est_remaining/3600:.2f} hours"
    return f"Iterated over {curr_batch_idx} batchs, {epoch_idx} epochs ({elapsed_bob} s)| Avg time: {avg_time:.7f} s/batch | Estimated time left for epoch: {est_str}"


def normalize_depth(val, n, old: bool = False):
    # The depth distribution is uniform till d_max. Transform to a variable which
    # has mean 0 and variance 1.
    if old:
        return val / 2 - 2.2
    dmax = n * (n + 3) / 2 / np.log2(n)
    return (val - dmax / 2) / np.sqrt(dmax)


def unnormalize_depth(val, n, old: bool = False):
    if old:
        return (val + 2.2) * 2
    dmax = n * (n + 3) / 2 / np.log2(n)
    return val * np.sqrt(dmax) + dmax / 2


class HyperParameters:
    def __init__(self, params: dict):
        self.params = params
        self.load_defaults()

    def _update_if_not_set(self, key: str, default):
        if key not in self.params:
            self.params[key] = default

    def load_defaults(self):
        self._update_if_not_set("model/dA", 128)
        self._update_if_not_set("model/dB", 32)
        self._update_if_not_set("model/dC", 64)
        self._update_if_not_set("model/dD", 32)
        self._update_if_not_set("model/dE", 32)
        self._update_if_not_set("model/num_transformer_blocks", 2)
        self._update_if_not_set("model/homo_attention_n_head", 4)
        self._update_if_not_set("model/hetero_attention_embed_dim", 100)
        self._update_if_not_set("model/hetero_attention_n_head", 4)

        self._update_if_not_set("trainer/schedule", "naive")
        self._update_if_not_set("trainer/schedule/epochs", 1)
        # self._update_if_not_set("trainer/schedule/datapoints", 1)
        self._update_if_not_set("trainer/adam/lr", 0.001)
        self._update_if_not_set("trainer/adam/betas", (0.9, 0.999))

        assert "trainer/train_file" in self.params
        assert "trainer/validation_file" in self.params
        assert "trainer/test_file" in self.params

    def __getitem__(self, key):
        return self.params[key]


class ModelWrapper:
    def __init__(self, hyperparams: HyperParameters) -> None:
        self.model: nn.Module = ModelV0(
            hyperparams.params["model/dA"],
            hyperparams.params["model/dB"],
            hyperparams.params["model/dC"],
            hyperparams.params["model/dD"],
            hyperparams.params["model/dE"],
            num_transformer_blocks=hyperparams.params["model/num_transformer_blocks"],
            homo_attention_n_head=hyperparams.params["model/homo_attention_n_head"],
            hetero_attention_embed_dim=hyperparams.params["model/hetero_attention_embed_dim"],
            hetero_attention_n_head=hyperparams.params["model/hetero_attention_n_head"],
        )
        print("Model details:")
        print(f"# parameters = {self.count_parameters()}")

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class Trainer:
    def __init__(
        self, model_wrapper: ModelWrapper, hyperparams: HyperParameters, output_folder: str
    ) -> None:
        self.model = model_wrapper.model

        assert hyperparams.params["trainer/schedule"] == "naive"
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=hyperparams.params["trainer/adam/lr"],
            betas=hyperparams.params["trainer/adam/betas"],
        )
        print(f"Optimizer: {self.optimizer.__class__.__name__}")

        self.train_data = UnprepNpzDataloader(
            hyperparams.params["trainer/train_file"], shuffle=True
        )
        self.validation_data = UnprepNpzDataloader(
            hyperparams.params["trainer/validation_file"], shuffle=False
        )
        self.test_data = UnprepNpzDataloader(hyperparams.params["trainer/test_file"], shuffle=False)
        print("Total sizes of datasets:")
        print(f"Train - {self.train_data.get_total_size()}")
        print(f"Validation - {self.validation_data.get_total_size()}")
        print(f"Test - {self.test_data.get_total_size()}")

        self.batch_size = 64

        self.gate_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.MSELoss()
        self.alpha = 1

        self.checkpoint_folder = output_folder

    def compute_loss(
        self, n, gate_prediction, depth_prediction, true_gates, true_depth
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate_loss(gate_prediction, true_gates), self.alpha * (
            self.depth_loss(depth_prediction, normalize_depth(true_depth.float(), n))
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

    def train(self, params: HyperParameters):
        self.set_device()
        train_gate_loss_history, train_depth_loss_history = [], []
        validation_gate_loss_history, validation_depth_loss_history = [], []
        training_loss_history, validation_loss_history = [], []

        num_batches = self.train_data.get_total_size() / self.train_data.batch_size

        assert params["trainer/schedule"] == "naive", "Only naive training supported"
        for epoch in range(params["trainer/schedule/epochs"]):
            epoch_tic = time.time()
            batch_tic = time.time()
            for i, train_data in enumerate(iter(self.train_data)):
                n = train_data["eigval"].shape[1]
                self.optimizer.zero_grad()
                gate_prediction, depth_prediction = self.run_model(train_data)
                gate_loss, depth_loss = self.compute_loss(
                    n,
                    gate_prediction,
                    depth_prediction,
                    torch.tensor(train_data["unprep_gate"], dtype=torch.int64).to(self.device),
                    torch.tensor(train_data["depth"], dtype=torch.int64).to(self.device),
                )
                loss = gate_loss + depth_loss
                loss.backward()
                self.optimizer.step()

                training_loss_history.append(loss.detach().cpu().item())
                train_gate_loss_history.append(gate_loss.detach().cpu().item())
                train_depth_loss_history.append(depth_loss.detach().cpu().item())

                if i % 1000 == 0:
                    print(
                        elapsed_str(
                            time.time() - epoch_tic, time.time() - batch_tic, i, num_batches, epoch
                        )
                    )
                    self.dump_loss_history(
                        training_loss_history,
                        train_gate_loss_history,
                        train_depth_loss_history,
                    )
                    batch_tic = time.time()

                if i % 5000 == 0 and i > 0:
                    validation_gate_loss, validation_depth_loss = self.calculate_validation_score()
                    validation_gate_loss_history.append(validation_gate_loss)
                    validation_depth_loss_history.append(validation_depth_loss)
                    self.store_checkpoint(
                        epoch,
                        i,
                        {
                            "train_gate_loss": train_gate_loss_history[-1],
                            "train_depth_loss": train_depth_loss_history[-1],
                            "validation_gate_loss": validation_gate_loss_history[-1],
                            "validation_depth_loss": validation_depth_loss_history[-1],
                        },
                    )

                    batch_tic = time.time()

            # Also store at the end of the model
            validation_gate_loss, validation_depth_loss = self.calculate_validation_score()
            validation_gate_loss_history.append(validation_gate_loss)
            validation_depth_loss_history.append(validation_depth_loss)
            self.store_checkpoint(
                epoch,
                i,
                {
                    "train_gate_loss": train_gate_loss_history[-1],
                    "train_depth_loss": train_depth_loss_history[-1],
                    "validation_gate_loss": validation_gate_loss_history[-1],
                    "validation_depth_loss": validation_depth_loss_history[-1],
                },
            )
            self.dump_loss_history(
                training_loss_history, train_gate_loss_history, train_depth_loss_history
            )

    def evaluate_model(self, dataset) -> torch.Tensor:
        self.set_device()
        total_loss, total_gate_loss, total_depth_loss = 0.0, 0.0, 0.0
        total_samples = 0
        for i, data in enumerate(iter(dataset)):
            gate_prediction, depth_prediction = self.run_model(data, use_grad=False, use_eval=True)
            n = data["eigval"].shape[-1]
            gate_loss, depth_loss = self.compute_loss(
                n,
                gate_prediction,
                depth_prediction,
                torch.tensor(data["unprep_gate"], dtype=torch.int64).to(self.device),
                torch.tensor(data["depth"], dtype=torch.int64).to(self.device),
            )
            loss = gate_loss + depth_loss
            batch_size = data["unprep_gate"].shape[0]
            total_loss += loss.cpu().item() * batch_size
            total_gate_loss += gate_loss.cpu().item() * batch_size
            total_depth_loss += depth_loss.cpu().item() * batch_size
            total_samples += batch_size
        # average_loss = total_loss / total_samples
        return total_gate_loss / total_samples, total_depth_loss / total_samples

    def calculate_validation_score(self) -> torch.Tensor:
        return self.evaluate_model(self.validation_data)

    def calculate_test_score(self) -> torch.Tensor:
        return self.evaluate_model(self.test_data)

    def dump_loss_history(self, loss_history, gate_loss_history=None, depth_loss_history=None):
        file = f"{self.checkpoint_folder}/training_loss.npy"
        np.save(file, np.array(loss_history))
        if gate_loss_history is not None:
            np.save(f"{self.checkpoint_folder}/gate_loss.npy", np.array(gate_loss_history))
        if depth_loss_history is not None:
            np.save(f"{self.checkpoint_folder}/depth_loss.npy", np.array(depth_loss_history))

    def store_checkpoint(self, epoch, iter_idx, kwargs):
        data = {
            "epoch": epoch,
            "iter_idx": iter_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        data.update(kwargs)
        torch.save(
            data,
            f"{self.checkpoint_folder}/model-{epoch}-{iter_idx}.pt",
        )


def create_new_folder(prefix: str, args: Namespace):
    name = f"{args.prefix}-{args.suffix}"
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


def update_metadata_with_model(model: ModelWrapper, metadata: dict):
    metadata["model/parameter_count"] = model.count_parameters()


def update_metadata_with_params(hyperparams: HyperParameters, metadata: dict):
    metadata.update(hyperparams.params)


def dump_metadata(folder: str, metadata: dict):
    import json

    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    # INSERT_YOUR_CODE
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Trainer hyperparameters")
    parser.add_argument(
        "--param-file", type=str, required=True, help="json file for loading params"
    )
    parser.add_argument("--suffix", type=str, default="", help="Name suffix for folder")
    parser.add_argument("--prefix", type=str, default="", help="Name prefix for folder")
    parser.add_argument("--device", type=str, default="hpc", help="Name suffix for folder")
    args = parser.parse_args()
    print(f"Args: {args}")

    assert args.device == "qserver"
    with open(args.param_file, "r") as f:
        param_dict = json.load(f)
    params = HyperParameters(param_dict)
    args.train_file = params["trainer/train_file"]
    args.validation_file = params["trainer/validation_file"]
    args.test_file = params["trainer/test_file"]
    model_output_folder = create_new_folder("output", args)
    print(f"Output folder: {model_output_folder}")

    seed = 1  # time.time_ns()
    np.random.seed(seed)
    # Torch RNG
    gen = torch.Generator()
    gen.manual_seed(seed)

    model = ModelWrapper(params)
    trainer = Trainer(model, params, model_output_folder)

    metadata = {}
    update_metadata_with_args(args, metadata)
    update_metadata_with_system(metadata)
    update_metadata_with_model(model, metadata)
    update_metadata_with_params(params, metadata)

    metadata["validation_size"] = int(trainer.validation_data.get_total_size())
    metadata["train_size"] = int(trainer.train_data.get_total_size())
    metadata["train_batches"] = int(
        trainer.train_data.get_total_size() // trainer.train_data.batch_size
    )
    metadata["test_size"] = int(trainer.test_data.get_total_size())
    dump_metadata(model_output_folder, metadata)

    tic = time.time()
    loss = trainer.calculate_validation_score()
    toc = time.time()
    print(f"Initial validation loss = {loss} ({toc-tic} s)")
    metadata["validation_time"] = toc - tic

    tic = time.time()
    trainer.train(params)
    toc = time.time()
    print(f"Training complete ({toc-tic} s)")
    metadata["train_time"] = toc - tic

    # tic = time.time()
    # gate_loss, depth_loss = trainer.calculate_test_score()
    # loss = gate_loss + depth_loss
    # toc = time.time()
    # print(f"Test loss = {loss} ({toc-tic} s)")
    # metadata["test_time"] = toc - tic
    # metadata["test_loss"] = loss

    dump_metadata(model_output_folder, metadata)
