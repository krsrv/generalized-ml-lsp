import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser, Namespace
from pathlib import Path
from models.model_v0 import ModelV0
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from training.dataset import LSPDataLoader

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

def load_config(*, path: str | None = None, overrides: dict | None = None):
    config = OmegaConf.create("""
    model:
      dA: 128
      dB: 32
      dC: 64
      dD: 32
      dE: 32
      num_transformer_blocks: 2
      homo_attention_n_head: 4
      hetero_attention_embed_dim: 100
      hetero_attention_n_head: 4
    trainer:
      schedule:
        name: naive
        epochs: 1
      batch_size: 32
      adam:
        lr: 1e-3
        betas:
        - 0.9
        - 0.999
    changelog: "???"
    """)
    REQUIRED = ["trainer.train_file", "trainer.validation_file"]
    if path:
        config = OmegaConf.merge(config, OmegaConf.load(path))
    if overrides:
        config = OmegaConf.merge(config, overrides)
    missing = [k for k in REQUIRED if OmegaConf.select(config, k) is None]
    if missing:
        raise KeyError(f"Missing required parameter(s): {', '.join(missing)}")
    return config

def create_model(config: DictConfig, verbose=True):
    model = ModelV0(
        config.dA, config.dB, config.dC, config.dD, config.dE,
        num_transformer_blocks=config.num_transformer_blocks,
        homo_attention_n_head=config.homo_attention_n_head,
        hetero_attention_embed_dim=config.hetero_attention_embed_dim,
        hetero_attention_n_head=config.hetero_attention_n_head)
    parameter_count = sum(p.numel() for p in model.parameters())
    model.parameter_count = parameter_count
    if verbose:
        print(f"# model parameters = {parameter_count}")
    return model

class Metadata:
    def __init__(self):
        self.data = {}

    def add_args(self, args: Namespace):
        try:
            self.data["args"] = dict(args._get_kwargs())
        except AttributeError:
            self.data["args"] = args.__dict__
        return self

    def add_system_info(self):
        self.data["cpu_count"] = os.cpu_count()
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.data["gpu_count"] = gpu_count
            self.data["gpu"] = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                self.data["gpu"].append(
                    {
                        "name": torch.cuda.get_device_name(i),
                        "memory": props.total_memory / (1024**3),
                        "multi_processor_count": props.multi_processor_count,
                    }
                )
        return self

    def add_model_info(self, model: nn.Module):
        self.data["model/parameter_count"] = model.parameter_count
        return self

    def add_config(self, config: DictConfig):
        self.data["config"] = OmegaConf.to_container(config)
        return self

    def update(self, new_data: dict):
        self.data.update(new_data)
        return self

    def dump(self, folder: str):
        with open(os.path.join(folder, "metadata.json"), "w") as f:
            json.dump(self.data, f, indent=2)


class Trainer:

    def __init__(
        self, model: nn.Module, config: DictConfig, verbose: bool = True,
        compile: bool = False,
        seed: int = 1
    ):
        self.model = model
        if compile:
            self.model.compile()
        assert config.schedule.name == "naive"
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.adam.lr,
            betas=tuple(config.adam.betas)
        )
        if verbose:
            print(f"Optimizer: {self.optimizer.__class__.__name__}")

        self.train_data = LSPDataLoader(
            config.train_file, batch_size=config.batch_size, shuffle=True,
            seed=seed)
        self.validation_data = LSPDataLoader(
            config.validation_file, batch_size=config.batch_size, shuffle=False)
        self.test_data = None # Load only if needed (later)

        if verbose:
            print("Total #batches:")
            print(f"Train - {len(self.train_data)}")
            print(f"Validation - {len(self.validation_data)}")

        batch_size = config.batch_size
        for data_loader in [self.train_data, self.validation_data]:
            data_loader.set_batch_size(batch_size)

        self.gate_loss = nn.CrossEntropyLoss()
        self.depth_loss = nn.MSELoss()
        self.alpha = 1
        self.device = None
        self.epoch_offset = 0
        self.config = config

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

    def load_train_history(
        self, ok_exists: bool = True, ok_new: bool = True
    ):
        output_dir = Path(self.config.output_dir)
        hist_filename = output_dir / "train_history.npz"
        assert ok_exists or ok_new, "Either ok_exists or ok_new must be True."
        if hist_filename.exists():
            if not ok_exists:
                raise FileExistsError(f"File {hist_filename} already exists.")
            with np.load(hist_filename) as data:
                return (
                    list(data["train_gate_loss"]),
                    list(data["train_depth_loss"]),
                    list(data["train_loss"]),
                    list(data["val_gate"]),
                    list(data["val_depth"])
                )
        else:
            if not ok_new:
                raise FileNotFoundError(f"File {hist_filename} does not exist.")
            return [], [], [], [], []

    def save_train_history(
        self, train_gate_loss, train_depth_loss, train_loss, val_gate, val_depth
    ):
        output_dir = Path(self.config.output_dir)
        hist_filename = output_dir / "train_history.npz"
        np.savez(
            hist_filename,
            train_gate_loss=np.array(train_gate_loss),
            train_depth_loss=np.array(train_depth_loss),
            train_loss=np.array(train_loss),
            val_gate=np.array(val_gate),
            val_depth=np.array(val_depth)
        )

    def load_checkpoint(self) -> bool:
        """
        Loads model if the file exists.

        Returns: True if a checkpoint was loaded, False otherwise.
        """
        self.epoch_offset = 0
        model_filename = Path(self.config.output_dir) / "model_checkpoint.pt"
        if not model_filename.exists():
            return False
        saved_data = torch.load(
            model_filename, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(saved_data["model_state_dict"])
        self.epoch_offset = saved_data["epoch"]
        return True

    def save_checkpoint(self, epoch, iter_idx, stats):
        output_dir = Path(self.config.output_dir)
        model_filename = output_dir / "model_checkpoint.pt"
        data = {
            "epoch": epoch,
            "iter_idx": iter_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            **stats,
        }
        torch.save(data, model_filename)
        print(f"Stored checkpoint ({iter_idx} iteration, {epoch} epoch)", flush=True)

    def train(self):
        train_gate_hist, train_depth_hist, loss_hist, val_gate_hist, val_depth_hist = (
            self.load_train_history())
        self.load_checkpoint()
        self.set_device()

        train_batches = len(self.train_data)
        checkpoint_mod = self._get_checkpoint_mod(train_batches)
        loss_history_mod = self._get_history_mod(train_batches)
        print(
            f"Loss history and checkpoints will be stored every "
            f"{loss_history_mod} and {checkpoint_mod} iterations respectively.")

        for epoch in range(
            self.epoch_offset, self.epoch_offset + self.config.schedule.epochs
        ):
            epoch_tic = time.time()
            batch_tic = time.time()
            i = 0 # Avoid NameError for empty train set.
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

                loss_hist.append(loss.detach().cpu().item())
                train_gate_hist.append(gate_loss.detach().cpu().item())
                train_depth_hist.append(depth_loss.detach().cpu().item())

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
                    self.save_train_history(
                        loss_hist, train_gate_hist, train_depth_hist,
                        val_gate_hist, val_depth_hist)

                if i % checkpoint_mod == 0 and i > 0:
                    val_gate_loss, val_depth_loss = self.calculate_validation_score()
                    val_gate_hist.append(val_gate_loss)
                    val_depth_hist.append(val_depth_loss)
                    self.save_checkpoint(
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
            val_gate_hist.append(val_gate_loss)
            val_depth_hist.append(val_depth_loss)
            self.save_checkpoint(
                epoch,
                i,
                {
                    "train_gate_loss": train_gate_hist[-1],
                    "train_depth_loss": train_depth_hist[-1],
                    "validation_gate_loss": val_gate_hist[-1],
                    "validation_depth_loss": val_depth_hist[-1],
                },
            )
            self.save_train_history(
                loss_hist, train_gate_hist, train_depth_hist,
                val_gate_hist, val_depth_hist)

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
        if self.test_data is None:
            try:
                test_filename = self.config.test_file
            except AttributeError:
                raise ValueError(
                    "config.test_file is required for `calculate_test_score`.")
            self.test_data = LSPDataLoader(
                test_filename, batch_size=self.config.batch_size,
                shuffle=False)
        return self.evaluate_model(self.test_data)


def main():
    parser = ArgumentParser(description="Trainer hyperparameters")
    parser.add_argument(
        "--param-file", type=str, required=False, help="yaml with trainer parameters")
    parser.add_argument("--prefix", type=str, default=None, help="Name prefix for folder")
    parser.add_argument("--expid", type=str, default=None, help="Name suffix for folder")
    parser.add_argument("--compile", action="store_true", help="Use torch compile")
    parser.add_argument(
        "--info",
        type=str,
        default="",
        nargs="+",
        help="Additional information about experiment (logged in metadata.json). Use quotes to pass multi-word values.",
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    config_overrides = OmegaConf.create("""
    trainer:
      train_file: training-data/compiled/2-28-split-train.npz
      validation_file: training-data/compiled/2-28-split-validation.npz
      output_dir: output/bench0/
      schedule:
        epochs: 1
    changelog: "Benchmark data loader."
    seed: 1
    """)

    if args.param_file is not None:
        config = load_config(path=args.param_file)
    else:
        config = load_config(overrides=config_overrides)
    if args.prefix is not None and args.expid is not None:
        config.trainer.output_dir = os.path.join(
            "output", f"{args.prefix}-{args.expid}")
    os.makedirs(config.trainer.output_dir, exist_ok=True)
    print(f"Output folder: {config.trainer.output_dir}")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = create_model(config.model)
    trainer = Trainer(model, config.trainer, compile=args.compile, seed=config.seed)
    metadata = (Metadata()
        .add_args(args)
        .add_system_info()
        .add_model_info(model)
        .add_config(config)
        .update(
           {
               "validation_size": int(trainer.validation_data.num_samples),
               "train_size": int(trainer.train_data.num_samples),
               "train_batches": len(trainer.train_data),
           }))
    metadata.dump(config.trainer.output_dir)
    tic = time.time()
    trainer.train()
    toc = time.time()
    print(f"Training complete ({toc-tic:.1f} s)")
    metadata.data["train_time"] = toc - tic
    metadata.dump(config.trainer.output_dir)

if __name__ == "__main__":
    main()
