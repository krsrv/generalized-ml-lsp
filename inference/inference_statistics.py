import time

import numpy as np
import torch
import torch.nn as nn

from inference.infer import InferWrapper, Path, format_observation, get_gate_literals, name_dict
from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader

"""
Gate set:
[
    lsp::GateType::H,       lsp::GateType::S,  lsp::GateType::Sdg,
    lsp::GateType::Z,       lsp::GateType::X,  lsp::GateType::sqrtX,
    lsp::GateType::sqrtXdg, lsp::GateType::CX, lsp::GateType::CZ
]
"""


def elapsed_str(elapsed, curr_batch_idx, total_batches):
    avg_time = elapsed / (curr_batch_idx + 1) if curr_batch_idx > 0 else 0
    remaining_batches = total_batches - (curr_batch_idx + 1)
    est_remaining = avg_time * remaining_batches
    if est_remaining < 60:
        est_str = f"{est_remaining:.2f} seconds"
    elif est_remaining < 3600:
        est_str = f"{est_remaining/60:.2f} minutes"
    else:
        est_str = f"{est_remaining/3600:.2f} hours"
    return f"Iterated over {curr_batch_idx} batches ({elapsed} s)| Avg time: {avg_time:.7f} s/batch | Estimated time left for epoch: {est_str}"


def run_inference(
    model_file: str,
    dataset_file: str,
    output_file: str,
    max_depth: int,
    batch_size: int,
    beam_width: int,
):
    model = ModelV0(
        128,
        32,
        64,
        32,
        32,
        hetero_attention_embed_dim=100,
    )
    wrapper = InferWrapper(
        model,
        model_file,
        max_depth,
    )

    full_dataset = UnprepNpzDataloader(dataset_file, shuffle=False)

    full_dataset.set_batch_size(batch_size)

    unprepped_successfully = []  # Unprepped correctly
    unprepped_optimally = []  # Unprepped in the correct number of gates
    depth_prediction = []  # Model's prediction of depth
    depth_inference = []  # Max depth that model tried for before termination
    actual_depth = []

    tic = time.time()
    for i, data in enumerate(iter(full_dataset)):
        # print(f"Batch {i}: {data["layout"].shape[-1]}, {data["gate_oh"].shape[-1]}")
        # continue
        output_paths = wrapper.infer_batch(
            data["layout"],
            data["eigval"],
            data["eigvec"],
            data["gate_oh"],
            data["gate_qubit_oh"],
            data["observation"],
            beam_width=beam_width,
        )
        for i, path in enumerate(output_paths):
            # print("Depth shape", path.depths.shape)
            # print("Gate shape", path.gates.shape)
            if path.unprepped:
                # All paths are guaranteed to be of same depth
                if data["depth"][i] == len(path.gates):
                    unprepped_optimally.append(True)
                depth_inference.append(len(path.gates))
            else:
                depth_inference.append(max_depth + 1)
            depth_prediction.append(path.depths[0][0])
            unprepped_successfully.append(path.unprepped)

        actual_depth.append(data["depth"])

        if i % 1000 == 0:
            np.savez(
                output_file,
                depth_inference=depth_inference,
                depth_prediction=depth_prediction,
                actual_depth=actual_depth,
                unprepped_successfully=unprepped_successfully,
            )
            print(
                elapsed_str(
                    time.time() - tic,
                    i,
                    full_dataset.get_total_size() / full_dataset.batch_size,
                )
            )

    unprepped_successfully = np.array(unprepped_successfully)
    unprepped_optimally = np.array(unprepped_optimally)
    depth_inference = np.array(depth_inference)
    depth_prediction = np.array(depth_prediction)
    actual_depth = np.array(actual_depth)

    # print(f"Total datapoints: {full_dataset.get_total_size()}")
    # print(f"Number of correct inferences: {np.count_nonzero(unprepped_successfully)}")
    # # print(f"Number of optimal inferences: {np.count_nonzero(unprepped_optimally)}")
    # print(f"Depth metric:")
    # print(f"    Average: {np.mean(depth_inference)}")
    # print(f"    Median: {np.median(depth_inference)}")
    # print(f"    Stddev: {np.std(depth_inference)}")

    # print(f"Actual depths:")
    # print(f"    Average: {np.mean(actual_depth)}")
    # print(f"    Median: {np.median(actual_depth)}")
    # print(f"    Stddev: {np.std(actual_depth)}")

    # difference = actual_depth - depth_inference
    # print(f"Difference:")
    # print(f"    Average: {np.mean(difference)}")
    # print(f"    Median: {np.median(difference)}")
    # print(f"    Stddev: {np.std(difference)}")

    np.savez(
        output_file,
        depth_inference=depth_inference,
        depth_prediction=depth_prediction,
        actual_depth=actual_depth,
        unprepped_successfully=unprepped_successfully,
    )


if __name__ == "__main__":
    # INSERT_YOUR_CODE
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--beam-width", type=int, default=5, required=True, help="Beam width")
    parser.add_argument("--max-depth", type=int, default=10, required=True, help="Max depth")
    parser.add_argument("--batch-size", type=int, default=32, required=True, help="Batch size")
    args = parser.parse_args()
    print(f"Args: {args}")

    args.model_file = (
        "output/full_run_2_10-epochs=20-lr=0.001-beta=(0.9, 0.999)-iter-7/model-18-33698.pt"
    )
    args.dataset = "training-data/split/2-10-validation.npz"
    parent_dir = os.path.dirname(args.model_file)
    args.output_file = os.path.join(
        parent_dir,
        f"parallel-inference-bw-{args.beam_width}-md-{args.max_depth}-bs-{args.batch_size}.npz",
    )
    print(f"Output file: {args.output_file}")

    seed = 1
    np.random.seed(seed)
    run_inference(
        args.model_file,
        args.dataset,
        args.output_file,
        args.max_depth,
        args.batch_size,
        args.beam_width,
    )
