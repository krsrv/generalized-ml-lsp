import json
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_v0 import ModelV0
from training.dataset import ShardReader, UnprepNpzDataloader


def elapsed_str(elapsed_total, elapsed_last, curr_idx, total_batches, epoch_idx):
    avg_time = elapsed_total / (curr_idx + 1) if curr_idx > 0 else 0
    remaining_batches = total_batches - (curr_idx + 1)
    est_remaining = avg_time * remaining_batches
    if est_remaining < 60:
        est_str = f"{est_remaining:.2f} seconds"
    elif est_remaining < 3600:
        est_str = f"{est_remaining / 60:.2f} minutes"
    else:
        est_str = f"{est_remaining / 3600:.2f} hours"
    return (
        f"Iterated over {curr_idx} batches, {epoch_idx} epochs ({elapsed_last:.1f} s) | "
        f"Avg time: {avg_time:.7f} s/batch | Estimated time left for epoch: {est_str}"
    )


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
        self._set_defaults()

    def _set_defaults(self):
        defaults = {
            "model/dA": 128,
            "model/dB": 32,
            "model/dC": 64,
            "model/dD": 32,
            "model/dE": 32,
            "model/num_transformer_blocks": 2,
            "model/homo_attention_n_head": 4,
            "model/hetero_attention_embed_dim": 100,
            "model/hetero_attention_n_head": 4,
            "trainer/schedule": "naive",
            "trainer/schedule/epochs": 1,
            "trainer/batch_size": 32,
            "trainer/adam/lr": 0.001,
            "trainer/adam/betas": (0.9, 0.999),
        }
        for key, value in defaults.items():
            self.params.setdefault(key, value)

        required = ["trainer/train_file", "trainer/validation_file", "trainer/test_file"]
        for req in required:
            assert req in self.params, f"Missing required parameter: {req}"

    def __getitem__(self, key):
        return self.params[key]


class ModelWrapper:

    def __init__(self, hyperparams: HyperParameters):
        hp = hyperparams.params
        self.model = ModelV0(
            hp["model/dA"],
            hp["model/dB"],
            hp["model/dC"],
            hp["model/dD"],
            hp["model/dE"],
            num_transformer_blocks=hp["model/num_transformer_blocks"],
            homo_attention_n_head=hp["model/homo_attention_n_head"],
            hetero_attention_embed_dim=hp["model/hetero_attention_embed_dim"],
            hetero_attention_n_head=hp["model/hetero_attention_n_head"],
        )
        print("Model details:")
        print(f"# parameters = {self.count_parameters()}")

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class Trainer:

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        hyperparams: HyperParameters,
        output_folder: str,
        compile: bool = False,
    ):
        self.model = model_wrapper.model
        if compile:
            self.model.compile()
        hp = hyperparams.params

        assert hp["trainer/schedule"] == "naive"
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=hp["trainer/adam/lr"],
            betas=hp["trainer/adam/betas"],
        )
        print(f"Optimizer: {self.optimizer.__class__.__name__}")

        self.train_data = UnprepNpzDataloader(hp["trainer/train_file"], shuffle=True)
        self.validation_data = UnprepNpzDataloader(hp["trainer/validation_file"], shuffle=False)
        self.test_data = UnprepNpzDataloader(hp["trainer/test_file"], shuffle=False, mload=False)

        print("Total sizes of datasets:")
        print(f"Train - {self.train_data.get_total_size()}")
        print(f"Validation - {self.validation_data.get_total_size()}")
        print(f"Test - {self.test_data.get_total_size()}")

        batch_size = hp["trainer/batch_size"]
        for data_loader in [self.train_data, self.validation_data, self.test_data]:
            data_loader.set_batch_size(batch_size)

        self.gate_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.MSELoss()
        self.alpha = 1
        self.checkpoint_folder = output_folder
        self.device = None
        self.epoch_offset = 0

    def _get_checkpoint_mod(self, num_batches: int):
        # Checkpoint 2 times per epoch
        return int(np.ceil(num_batches / 2))

    def _get_history_mod(self, num_batches: int):
        # Checkpoint every 1000 batches
        return 1000

    def compute_loss(self, n, gate_pred, depth_pred, true_gates, true_depth):
        gate_loss = self.gate_loss(gate_pred, true_gates)
        depth_norm = normalize_depth(true_depth.float(), n)
        depth_loss = self.depth_loss(depth_pred, depth_norm) * self.alpha
        return gate_loss, depth_loss

    def run_model(self, data, use_grad=True, use_eval=False):
        with torch.set_grad_enabled(use_grad):
            if use_eval:
                self.model.eval()
            gate_pred, depth_pred = self.model(
                torch.tensor(data["eigval"], dtype=torch.float).to(self.device),
                torch.tensor(data["eigvec"], dtype=torch.float).to(self.device),
                torch.tensor(data["gates"], dtype=torch.long).to(self.device),
                torch.tensor(data["gate_qubits"], dtype=torch.long).to(self.device),
                torch.tensor(data["observation"], dtype=torch.bool).to(self.device),
            )
            if use_eval:
                self.model.train()
        return gate_pred, depth_pred

    def set_device(self):
        if self.device is not None:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model now on device={self.device}")

    def load_loss_history(self, params: HyperParameters):
        def load_default(key, dic, default, file):
            if key in dic:
                return dic[key]
            assert not os.path.exists(
                file
            ), f"Loss file {file} already exists. Rename before proceeding."
            return default

        base_dir = os.path.basename(params["model_file"]) if "model_file" in params.params else ""
        gate_hist = load_default(
            "train_gate_loss_history", params.params, np.array([]), f"{base_dir}/gate_loss.npy"
        )
        depth_hist = load_default(
            "train_depth_loss_history", params.params, np.array([]), f"{base_dir}/depth_loss.npy"
        )
        loss_hist = load_default(
            "training_loss_history", params.params, np.array([]), f"{base_dir}/training_loss.npy"
        )
        return gate_hist, depth_hist, loss_hist, [], []

    def load_checkpoint(self, params: HyperParameters):
        self.epoch_offset = 0
        if "model_file" in params.params:
            saved_data = torch.load(
                params.params["model_file"], map_location="cpu", weights_only=True
            )
            self.model.load_state_dict(saved_data["model_state_dict"])
            self.epoch_offset = saved_data["epoch"]

    def train(self, params: HyperParameters):
        train_gate_hist, train_depth_hist, loss_hist, val_gate_hist, val_depth_hist = (
            self.load_loss_history(params)
        )
        self.load_checkpoint(params)
        self.set_device()

        train_batches = self.train_data.get_total_size() / self.train_data.batch_size
        checkpoint_mod = self._get_checkpoint_mod(train_batches)
        loss_history_mod = self._get_history_mod(train_batches)
        print(
            f"Loss history and checkpoints will be stored every {loss_history_mod} and {checkpoint_mod} iterations respectively."
        )

        assert params["trainer/schedule"] == "naive", "Only naive training supported"
        for epoch in range(
            self.epoch_offset, self.epoch_offset + params["trainer/schedule/epochs"]
        ):
            epoch_tic = time.time()
            batch_tic = time.time()
            for i, data in enumerate(self.train_data):
                n = data["eigval"].shape[1]
                self.optimizer.zero_grad()
                gate_pred, depth_pred = self.run_model(data)
                gate_loss, depth_loss = self.compute_loss(
                    n,
                    gate_pred,
                    depth_pred,
                    torch.tensor(data["unprep_gate"], dtype=torch.int64).to(self.device),
                    torch.tensor(data["depth"], dtype=torch.int64).to(self.device),
                )
                loss = gate_loss + depth_loss
                loss.backward()
                self.optimizer.step()

                loss_hist = np.append(loss_hist, loss.detach().cpu().item())
                train_gate_hist = np.append(train_gate_hist, gate_loss.detach().cpu().item())
                train_depth_hist = np.append(train_depth_hist, depth_loss.detach().cpu().item())

                if i % 1000 == 0:
                    print(
                        elapsed_str(
                            time.time() - epoch_tic,
                            time.time() - batch_tic,
                            i,
                            train_batches,
                            epoch,
                        )
                    )
                    batch_tic = time.time()

                if i % loss_history_mod == 0:
                    self.dump_loss_history(loss_hist, train_gate_hist, train_depth_hist)

                if i % checkpoint_mod == 0 and i > 0:
                    val_gate_loss, val_depth_loss = self.calculate_validation_score()
                    val_gate_hist = np.append(val_gate_hist, val_gate_loss)
                    val_depth_hist = np.append(val_depth_hist, val_depth_loss)
                    self.store_checkpoint(
                        epoch,
                        i,
                        {
                            "train_gate_loss": train_gate_hist[-1],
                            "train_depth_loss": train_depth_hist[-1],
                            "validation_gate_loss": val_gate_hist[-1],
                            "validation_depth_loss": val_depth_hist[-1],
                        },
                    )

            # Store at end of epoch
            val_gate_loss, val_depth_loss = self.calculate_validation_score()
            val_gate_hist = np.append(val_gate_hist, val_gate_loss)
            val_depth_hist = np.append(val_depth_hist, val_depth_loss)
            self.store_checkpoint(
                epoch,
                i,
                {
                    "train_gate_loss": train_gate_hist[-1],
                    "train_depth_loss": train_depth_hist[-1],
                    "validation_gate_loss": val_gate_hist[-1],
                    "validation_depth_loss": val_depth_hist[-1],
                },
            )
            self.dump_loss_history(loss_hist, train_gate_hist, train_depth_hist)

    def evaluate_model(self, dataset):
        self.set_device()
        total_gate_loss = total_depth_loss = total_samples = 0.0
        for data in dataset:
            gate_pred, depth_pred = self.run_model(data, use_grad=False, use_eval=True)
            n = data["eigval"].shape[-1]
            gate_loss, depth_loss = self.compute_loss(
                n,
                gate_pred,
                depth_pred,
                torch.tensor(data["unprep_gate"], dtype=torch.int64).to(self.device),
                torch.tensor(data["depth"], dtype=torch.int64).to(self.device),
            )
            batch_size = data["unprep_gate"].shape[0]
            total_gate_loss += gate_loss.cpu().item() * batch_size
            total_depth_loss += depth_loss.cpu().item() * batch_size
            total_samples += batch_size
        return total_gate_loss / total_samples, total_depth_loss / total_samples

    def calculate_validation_score(self):
        return self.evaluate_model(self.validation_data)

    def calculate_test_score(self):
        return self.evaluate_model(self.test_data)

    def dump_loss_history(self, loss_history, gate_loss_history=None, depth_loss_history=None):
        np.save(f"{self.checkpoint_folder}/training_loss.npy", loss_history)
        if gate_loss_history is not None:
            np.save(f"{self.checkpoint_folder}/gate_loss.npy", gate_loss_history)
        if depth_loss_history is not None:
            np.save(f"{self.checkpoint_folder}/depth_loss.npy", depth_loss_history)

    def store_checkpoint(self, epoch, iter_idx, stats):
        data = {
            "epoch": epoch,
            "iter_idx": iter_idx,
            "model_state_dict": self.model.state_dict(),
            **stats,
        }
        torch.save(data, f"{self.checkpoint_folder}/model-{epoch}-{iter_idx}.pt")


def create_new_folder(prefix: str, args: Namespace):
    name = f"{args.prefix}-{args.expid}"
    folder = os.path.join(prefix, name)
    os.makedirs(folder, exist_ok=True)
    return folder


def update_metadata_with_args(args: Namespace, metadata: dict):
    metadata["args"] = dict(args._get_kwargs())


def update_metadata_with_system(metadata: dict):
    metadata["cpu_count"] = os.cpu_count()
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        metadata["gpu_count"] = gpu_count
        metadata["gpu"] = []
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            metadata["gpu"].append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory": props.total_memory / (1024**3),
                    "multi_processor_count": props.multi_processor_count,
                }
            )


def update_metadata_with_model(model: ModelWrapper, metadata: dict):
    metadata["model/parameter_count"] = model.count_parameters()


def update_metadata_with_params(hyperparams: HyperParameters, metadata: dict):
    metadata.update(hyperparams.params)


def dump_metadata(folder: str, metadata: dict):
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = ArgumentParser(description="Trainer hyperparameters")
    parser.add_argument(
        "--param-file", type=str, required=True, help="json file for loading params"
    )
    parser.add_argument("--prefix", type=str, default="", help="Name prefix for folder")
    parser.add_argument("--expid", type=str, default="", help="Name suffix for folder")
    parser.add_argument("--compile", action="store_true", help="Use torch compile")
    args = parser.parse_args()
    print(f"Args: {args}")

    with open(args.param_file, "r") as f:
        param_dict = json.load(f)
    params = HyperParameters(param_dict)

    # Attach dataset paths to args for folder naming
    args.train_file = params["trainer/train_file"]
    args.validation_file = params["trainer/validation_file"]
    args.test_file = params["trainer/test_file"]

    output_folder = create_new_folder("output", args)
    print(f"Output folder: {output_folder}")

    # Set global seeds for reproducibility
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = ModelWrapper(params)
    trainer = Trainer(model, params, output_folder, compile=args.compile)

    metadata = {}
    update_metadata_with_args(args, metadata)
    update_metadata_with_system(metadata)
    update_metadata_with_model(model, metadata)
    update_metadata_with_params(params, metadata)
    metadata.update(
        {
            "validation_size": int(trainer.validation_data.get_total_size()),
            "train_size": int(trainer.train_data.get_total_size()),
            "train_batches": int(
                trainer.train_data.get_total_size() // trainer.train_data.batch_size
            ),
            "test_size": int(trainer.test_data.get_total_size()),
        }
    )
    metadata["test_size"] = int(trainer.test_data.get_total_size())
    dump_metadata(output_folder, metadata)

    # tic = time.time()
    # loss = trainer.calculate_validation_score()
    # toc = time.time()
    # print(f"Initial validation loss = {loss} ({toc-tic} s)")
    # metadata["validation_time"] = toc - tic

    tic = time.time()
    trainer.train(params)
    toc = time.time()
    print(f"Training complete ({toc-tic:.1f} s)")
    metadata["train_time"] = toc - tic

    # Uncomment below to evaluate on test set after training and update metadata
    # tic = time.time()
    # gate_loss, depth_loss = trainer.calculate_test_score()
    # metadata["test_time"] = time.time() - tic
    # metadata["test_loss"] = gate_loss + depth_loss

    dump_metadata(output_folder, metadata)


if __name__ == "__main__":
    main()
