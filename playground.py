import time

import numpy as np
import torch
import torch.nn as nn

from models.model_v0 import ModelV0
from training.dataset import UnprepNpzDataloader


def time_str(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f} seconds"
    elif sec < 3600:
        return f"{sec/60:.2f} minutes"
    else:
        return f"{sec/3600:.2f} hours"


def iteration_string(elapsed: int, i: int, num_batches: int) -> str:
    avg_time = elapsed / (i + 1) if i > 0 else 0
    remaining_batches = num_batches - (i + 1)
    est_remaining = avg_time * remaining_batches
    est_str = time_str(est_remaining)
    if est_remaining < 60:
        est_str = f"{est_remaining:.2f} seconds"
    elif est_remaining < 3600:
        est_str = f"{est_remaining/60:.2f} minutes"
    else:
        est_str = f"{est_remaining/3600:.2f} hours"
    return f"Iterated over {i} batches | Avg time: {avg_time:.7f} s/batch | Estimated time left: {est_str}"


def test_generate_matrix():
    print("TEST: time to loop over matrices")
    num = 100_000
    shape = (10, 100)
    tic = time.time()
    for i in range(num):
        torch.randn(shape)
        if i % 1_000 == 0:
            print(f"Iterated over {i} examples")
    torch.cuda.synchronize()
    toc = time.time()

    print(f"Time to generate {num} {shape} matrices: {toc-tic} s")


def test_load_matrix_to_gpu():
    print("TEST: time to load matrix to GPU")
    num = 100_000
    shape = (10, 100)
    tic = time.time()
    for i in range(num):
        torch.randn(shape).to("cuda")
        if i % 1_000 == 0:
            print(f"Iterated over {i} examples")
    torch.cuda.synchronize()
    toc = time.time()

    print(f"Time to load {num} {shape} matrices: {toc-tic} s")


def test_matrix_multiplication_on_gpu():
    print("TEST: time to multiply matrices on GPU")
    num = 10_000
    shape_a = (64, 1000)
    shape_b = (1000, 2000)
    tic = time.time()
    for i in range(num):
        a = torch.randn(shape_a, device="cuda")
        b = torch.randn(shape_b, device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        if i % 1_000 == 0:
            print(f"Iterated over {i} multiplications")
    toc = time.time()
    total_cost = num * shape_a[0] * shape_a[1] * shape_b[1]
    print(f"Time to multiply {num} pairs of {(shape_a, shape_b)}matrices: {toc-tic} s")
    print(
        f"Empirical FLOP/s: {total_cost / (toc-tic)} FLOP/s ~ {total_cost / (1e12 * (toc-tic))} TFLOP/s"
    )
    print("Note that the calculated value will change with the input matrix sizes")


def test_matrix_multiplication_from_file_on_gpu():
    import pickle

    print("TEST: time to multiply matrices of given shape on GPU")
    with open("global_mult_list.pkl", "rb") as f:
        mult_list = pickle.load(f)

    np.random.shuffle(mult_list)
    # mult_list = mult_list[:1_000_000]
    num_batches = len(mult_list)
    print(f"Number of multiplications: {num_batches}")

    print(f"Step 1: Generate random matrices of given size")
    tic = time.time()
    for i, shape in enumerate(mult_list):
        a = torch.randn((int(shape[0]), int(shape[1]), int(shape[2])), device="cuda")
        b = torch.randn((int(shape[0]), int(shape[2]), int(shape[3])), device="cuda")
        torch.cuda.synchronize()
        del a
        del b
        if i % 10_000 == 0:
            elapsed = time.time() - tic
            print(iteration_string(elapsed, i, num_batches))
    generate_time = time.time() - tic
    print(f"Time to generate given matrix shapes: {generate_time} s")

    print(f"Step 2: Multiply random matrices of given size")
    tic = time.time()
    for i, shape in enumerate(mult_list):
        a = torch.randn((int(shape[0]), int(shape[1]), int(shape[2])), device="cuda")
        b = torch.randn((int(shape[0]), int(shape[2]), int(shape[3])), device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        del a
        del b
        del c
        if i % 10_000 == 0:
            elapsed = time.time() - tic
            print(iteration_string(elapsed, i, num_batches))
    toc = time.time()
    multiplication_time = toc - tic
    print(f"Time to generate and multiply given matrix shapes: {multiplication_time} s")
    print(f"Difference: {multiplication_time - generate_time} s")


def test_load_train_data_to_cpu():
    print("TEST: time to loop over train data on cpu")
    # folder = "/scratch1/sauravk/lsp-npz"
    folder = "training-data/compiled"
    train_data = UnprepNpzDataloader(
        f"{folder}/new-sample-train-2-20.npz", shuffle=True
    )

    print(f"Train dataset: {train_data.get_total_size()}")

    batch_size = 64
    train_data.set_batch_size(batch_size)
    num_batches = train_data.get_total_size() / batch_size

    total_loaded = 0
    tic = time.time()
    for i, data in enumerate(iter(train_data)):
        # Start timing for this batch
        gate = torch.tensor(data["gate"], dtype=torch.int64)
        depth = torch.tensor(data["depth"], dtype=torch.int64)
        eigval = torch.tensor(data["eigval"], dtype=torch.float)
        eigvec = torch.tensor(data["eigvec"], dtype=torch.float)
        gate_oh = torch.tensor(data["gate_oh"], dtype=torch.long)
        gate_qubit_oh = torch.tensor(data["gate_qubit_oh"], dtype=torch.long)
        observation = torch.tensor(data["observation"], dtype=torch.bool)
        torch.cuda.synchronize()
        total_loaded += 1

        if i % 1_000 == 0:
            elapsed = time.time() - tic
            print(iteration_string(elapsed, i, num_batches))
    toc = time.time()
    print(
        f"Time to loop over all {total_loaded} batches of train dataset: {toc-tic:.4f} s"
    )


def test_load_train_data_to_gpu():
    print("TEST: time to load train data to GPU")
    # folder = "/scratch1/sauravk/lsp-npz"
    folder = "training-data/compiled"
    train_data = UnprepNpzDataloader(
        f"{folder}/new-sample-train-2-20.npz", shuffle=True
    )
    batch_size = 64
    train_data.set_batch_size(batch_size)
    num_batches = train_data.get_total_size() / batch_size
    print(
        f"Dataset size: {train_data.get_total_size()}, {num_batches} batches (each {batch_size} long)"
    )

    total_loaded = 0
    tic = time.time()
    for i, data in enumerate(iter(train_data)):
        # Start timing for this batch
        gate = torch.tensor(data["gate"], dtype=torch.int64, device="cuda")
        depth = torch.tensor(data["depth"], dtype=torch.int64, device="cuda")
        eigval = torch.tensor(data["eigval"], dtype=torch.float, device="cuda")
        eigvec = torch.tensor(data["eigvec"], dtype=torch.float, device="cuda")
        gate_oh = torch.tensor(data["gate_oh"], dtype=torch.long, device="cuda")
        gate_qubit_oh = torch.tensor(
            data["gate_qubit_oh"], dtype=torch.long, device="cuda"
        )
        observation = torch.tensor(data["observation"], dtype=torch.bool, device="cuda")
        torch.cuda.synchronize()
        total_loaded += 1

        if i % 1_000 == 0:
            elapsed = time.time() - tic
            print(iteration_string(elapsed, i, num_batches))
    toc = time.time()
    print(
        f"Time to load all {total_loaded} batches of train dataset to GPU: {toc-tic:.4f} s"
    )


def test_run_inference():
    print("TEST: time to run inference on GPU")
    # folder = "/scratch1/sauravk/lsp-npz"
    folder = "training-data/compiled"
    train_data = UnprepNpzDataloader(
        f"{folder}/new-sample-train-2-20.npz", shuffle=True
    )
    batch_size = 64
    train_data.set_batch_size(batch_size)
    num_batches = train_data.get_total_size() / batch_size
    print(
        f"Dataset size: {train_data.get_total_size()}, {num_batches} batches (each {batch_size} long)"
    )

    model = ModelV0(
        128,
        32,
        64,
        32,
        32,
        hetero_attention_embed_dim=100,
    )
    model.to("cuda")

    tic = time.time()
    for i, data in enumerate(iter(train_data)):
        gate_prediction, depth_prediction = model.forward(
            torch.tensor(data["eigval"], dtype=torch.float).to("cuda"),
            torch.tensor(data["eigvec"], dtype=torch.float).to("cuda"),
            torch.tensor(data["gate_oh"], dtype=torch.long).to("cuda"),
            torch.tensor(data["gate_qubit_oh"], dtype=torch.long).to("cuda"),
            torch.tensor(data["observation"], dtype=torch.bool).to("cuda"),
        )
        if i % 1_000 == 0:
            elapsed = time.time() - tic
            print(iteration_string(elapsed, i, num_batches))
        del gate_prediction
        del depth_prediction

    torch.cuda.synchronize()
    toc = time.time()

    print(f"Time to run inference over train dataset: {toc-tic} s")


def test_run_inference_and_loss():
    print("TEST: time to run inference on GPU")
    # folder = "/scratch1/sauravk/lsp-npz"
    folder = "training-data/compiled"
    train_data = UnprepNpzDataloader(
        f"{folder}/new-sample-train-2-20.npz", shuffle=True
    )
    batch_size = 64
    train_data.set_batch_size(batch_size)
    num_batches = train_data.get_total_size() / batch_size
    print(
        f"Dataset size: {train_data.get_total_size()}, {num_batches} batches (each {batch_size} long)"
    )

    model = ModelV0(
        128,
        32,
        64,
        32,
        32,
        hetero_attention_embed_dim=100,
    )

    gate_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.MSELoss()

    def compute_loss(n, gate_prediction, depth_prediction, true_gates, true_depth):
        return gate_loss_fn(gate_prediction, true_gates), (
            depth_loss_fn(depth_prediction, true_depth.float() * 4 / (n**2) - 1)
        )

    model.to("cuda")

    tic = time.time()
    for i, data in enumerate(iter(train_data)):
        gate_prediction, depth_prediction = model.forward(
            torch.tensor(data["eigval"], dtype=torch.float).to("cuda"),
            torch.tensor(data["eigvec"], dtype=torch.float).to("cuda"),
            torch.tensor(data["gate_oh"], dtype=torch.long).to("cuda"),
            torch.tensor(data["gate_qubit_oh"], dtype=torch.long).to("cuda"),
            torch.tensor(data["observation"], dtype=torch.bool).to("cuda"),
        )
        gate_loss, depth_loss = compute_loss(
            data["eigval"].shape[1],  # n
            gate_prediction,
            depth_prediction,
            torch.tensor(data["gate"], dtype=torch.int64).to("cuda"),
            torch.tensor(data["depth"], dtype=torch.int64).to("cuda"),
        )
        loss = gate_loss + depth_loss
        loss.to("cpu")
        if i % 1_000 == 0:
            elapsed = time.time() - tic
            print(iteration_string(elapsed, i, num_batches))

    torch.cuda.synchronize()
    toc = time.time()

    print(f"Time to run inference and loss over train dataset: {toc-tic} s")


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    # test_generate_matrix()
    # test_load_train_data_to_cpu()
    test_load_train_data_to_gpu()
    # test_matrix_multiplication_on_gpu()
    # test_run_inference()
    # test_run_inference_and_loss()
    # test_matrix_multiplication_from_file_on_gpu()
