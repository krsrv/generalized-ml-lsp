import time

import numpy as np
import torch
import torch.nn as nn

from inference.infer import InferWrapper, is_1_qubit_gate
from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader


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
    return f"Iterated over {curr_batch_idx} batches ({elapsed_bob} s)| Avg time: {avg_time:.7f} s/batch | Estimated time left for epoch: {est_str}"


def run_inference(
    model_file: str,
    dataset_file: str,
    output_file: str,
    max_depth: int,
    batch_size: int,
    beam_width: int,
    remove_duplicates: bool = False,
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

    full_dataset = UnprepNpzDataloader(dataset_file, shuffle=True)

    full_dataset.set_batch_size(batch_size)

    unprepped_successfully = np.array([])  # Unprepped correctly
    unprepped_better = np.array([])  # Unprepped in the correct number of gates
    depth_prediction = np.array([])  # Model's prediction of depth
    depth_inference = np.array([])  # Max depth that model tried for before termination
    actual_depth = np.array([])
    has_2_qubit_gateset = np.array([])
    has_2_qubit_gate_truth = np.array([])

    tic = batch_tic = time.time()
    for batch_idx, data in enumerate(iter(full_dataset)):
        # print(f"Batch {batch_idx}: {data["layout"].shape[-1]}, {data["gates"].shape[-1]}")
        # continue
        output_paths = wrapper.infer_batch(
            data["layout"],
            data["eigval"],
            data["eigvec"],
            data["gates"],
            data["gate_qubits"],
            data["observation"],
            beam_width=beam_width,
            remove_duplicates=remove_duplicates,
        )
        for i, path in enumerate(output_paths):
            # print("Depth shape", path.depths.shape)
            # print("Gate shape", path.gates.shape)
            width_, inferred_depth = path.gates.shape
            if path.unprepped:
                # All paths are guaranteed to be of same depth
                depth_inference = np.append(depth_inference, inferred_depth)
            else:
                depth_inference = np.append(depth_inference, max_depth + 1)
            depth_prediction = np.append(depth_prediction, path.depths[0][0])
            unprepped_successfully = np.append(unprepped_successfully, path.unprepped)
            unprepped_better = np.append(unprepped_better, data["depth"][i] >= inferred_depth)
            has_2_qubit_gate = torch.any(
                torch.tensor([not is_1_qubit_gate(gate) for gate in data["gates"][path.identifier]])
            )
            has_2_qubit_gateset = np.append(has_2_qubit_gateset, has_2_qubit_gate)
            has_2_qubit_gate_truth = np.append(
                has_2_qubit_gate_truth, not is_1_qubit_gate(data["unprep_gate"][path.identifier])
            )

        actual_depth = np.append(actual_depth, data["depth"])

        if batch_idx % 500 == 0:
            print(
                elapsed_str(
                    time.time() - tic,
                    time.time() - batch_tic,
                    batch_idx,
                    full_dataset.get_total_size() / full_dataset.batch_size,
                )
            )
            batch_tic = time.time()
        if batch_idx % 1000 == 0:
            np.savez(
                output_file,
                depth_inference=depth_inference,
                depth_prediction=depth_prediction,
                actual_depth=actual_depth,
                unprepped_successfully=unprepped_successfully,
                unprepped_better=unprepped_better,
                has_2_qubit_gateset=has_2_qubit_gateset,
                has_2_qubit_gate_truth=has_2_qubit_gate_truth,
                seed=seed,
            )

    # print(f"Total datapoints: {full_dataset.get_total_size()}")
    # print(f"Number of correct inferences: {np.count_nonzero(unprepped_successfully)}")
    # print(f"Number of optimal inferences: {np.count_nonzero(unprepped_better)}")
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
        unprepped_better=unprepped_better,
        has_2_qubit_gateset=has_2_qubit_gateset,
        has_2_qubit_gate_truth=has_2_qubit_gate_truth,
        seed=seed,
    )


if __name__ == "__main__":
    # INSERT_YOUR_CODE
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--beam-width", type=int, default=5, required=True, help="Beam width")
    parser.add_argument("--max-depth", type=int, default=10, required=True, help="Max depth")
    parser.add_argument("--batch-size", type=int, default=32, required=True, help="Batch size")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicates")
    parser.add_argument("--model-file", type=str, required=True, help="Model file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset file")
    args = parser.parse_args()
    print(f"Args: {args}")

    parent_dir = os.path.dirname(args.model_file)
    args.output_file = os.path.join(
        parent_dir,
        f"parallel-inference-bw-{args.beam_width}-md-{args.max_depth}-bs-{args.batch_size}-{'wo' if args.remove_duplicates else 'w'}-duplicate.npz",
    )
    print(f"Output file: {args.output_file}")

    global seed
    seed = 1
    np.random.seed(seed)
    run_inference(
        args.model_file,
        args.dataset,
        args.output_file,
        args.max_depth,
        args.batch_size,
        args.beam_width,
        args.remove_duplicates,
    )
