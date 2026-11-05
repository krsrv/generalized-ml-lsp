"""
Benchmark/profiling script for `InferWrapper.infer_batch`.

Measures end-to-end wall-clock time and a phase-level breakdown (model forward
pass, simulator call, top-k/truncation, beam filtering) without modifying
`inference/infer.py`. Phase timings are collected by monkeypatching bound
methods on the live objects for the duration of the benchmark run.

Usage:
    conda run -n torch-gpu python -m inference.benchmark_infer_pipeline
"""

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from inference.infer import BatchSimulator, InferWrapper, Path
from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader

DATASET_FILE = "training-data/split/compacted-2-10-validation.npz"
CHECKPOINT_GLOB = "output/overfit_2_10-epochs=5000-lr=0.001-beta=(0.9, 0.999)-iter-5/model-1000-0.pt"


@contextmanager
def timed(bucket: dict, name: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    bucket[name] += time.perf_counter() - start


def instrument(wrapper: InferWrapper, timings: dict):
    """Monkeypatch bound methods to record phase timings without touching infer.py."""
    orig_get_model_prediction = wrapper.get_model_prediction
    orig_run_simulation = BatchSimulator.run_simulation
    orig_filter_beam = Path.filter_beam
    orig_update_states = Path.update_states

    def get_model_prediction(*args, **kwargs):
        with timed(timings, "model_forward"):
            return orig_get_model_prediction(*args, **kwargs)

    def run_simulation(self, *args, **kwargs):
        with timed(timings, "simulator"):
            return orig_run_simulation(self, *args, **kwargs)

    def filter_beam(self, *args, **kwargs):
        with timed(timings, "filter_beam"):
            return orig_filter_beam(self, *args, **kwargs)

    def update_states(self, *args, **kwargs):
        with timed(timings, "update_states_total"):
            return orig_update_states(self, *args, **kwargs)

    wrapper.get_model_prediction = get_model_prediction
    BatchSimulator.run_simulation = run_simulation
    Path.filter_beam = filter_beam
    Path.update_states = update_states

    def restore():
        BatchSimulator.run_simulation = orig_run_simulation
        Path.filter_beam = orig_filter_beam
        Path.update_states = orig_update_states

    return restore


def build_wrapper(max_depth: int, compile_model: bool) -> InferWrapper:
    model = ModelV0(128, 32, 64, 32, 32, hetero_attention_embed_dim=100)
    return InferWrapper(model, CHECKPOINT_GLOB, max_depth, compile=compile_model)


def load_batch(batch_size: int):
    dataset = UnprepNpzDataloader(DATASET_FILE, shuffle=True, seed=1)
    dataset.set_batch_size(batch_size)
    data = next(iter(dataset))
    return data


def run_once(wrapper, data, beam_width, remove_duplicates, max_depth, timings=None):
    restore = instrument(wrapper, timings) if timings is not None else None
    try:
        t0 = time.perf_counter()
        output_paths = wrapper.infer_batch(
            data["layout"],
            data["eigval"],
            data["eigvec"],
            data["gates"],
            data["gate_qubits"],
            data["observation"],
            beam_width=beam_width,
            remove_duplicates=remove_duplicates,
            max_depth=max_depth,
        )
        total = time.perf_counter() - t0
    finally:
        if restore is not None:
            restore()
    return output_paths, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--remove-duplicates", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    data = load_batch(args.batch_size)
    wrapper = build_wrapper(args.max_depth, args.compile)

    # Warm up (pays one-time CUDA/JIT costs).
    run_once(wrapper, data, args.beam_width, args.remove_duplicates, args.max_depth)

    totals = []
    agg_timings = defaultdict(float)
    for _ in range(args.repeats):
        timings = defaultdict(float)
        _, total = run_once(
            wrapper, data, args.beam_width, args.remove_duplicates, args.max_depth, timings=timings
        )
        totals.append(total)
        for k, v in timings.items():
            agg_timings[k] += v

    totals = np.array(totals)
    print(f"\n=== infer_batch benchmark (bs={args.batch_size}, bw={args.beam_width}, "
          f"max_depth={args.max_depth}, repeats={args.repeats}) ===")
    print(f"Total wall time: mean={totals.mean():.4f}s std={totals.std():.4f}s "
          f"min={totals.min():.4f}s max={totals.max():.4f}s")
    print("\nPhase breakdown (summed over repeats, averaged per run):")
    grand_total = sum(agg_timings.values())
    for k, v in sorted(agg_timings.items(), key=lambda kv: -kv[1]):
        avg = v / args.repeats
        pct = 100 * v / grand_total if grand_total > 0 else 0
        print(f"  {k:25s} avg/run={avg:.4f}s  ({pct:5.1f}% of measured phases)")


if __name__ == "__main__":
    main()
