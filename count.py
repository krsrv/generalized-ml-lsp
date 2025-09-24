# Count
import time

from models.embeddings import NUM_GATE_TYPES
from models.tokens import TokenProperties
from training.dataset import UnprepNpzDataloader

n = 10
g = 50
count = 0

global_mult_list = []
global_ratio_list = []

th_com = 3.9e12
th_mem = 64e9


class Data:
    def __init__(self, n, g):
        super().__init__()
        self.n = n
        self.g = g


class Cost:
    def __init__(self, compute: list | None = None, memory: list | None = None):
        super().__init__()
        self.compute = compute if compute is not None else 0
        self.memory = memory if memory is not None else 0
        self.max = max((self.compute / th_com, self.memory / th_mem))
        if self.memory != 0:
            self.ratio = self.compute / self.memory
            global_ratio_list.append(self.ratio)
        return
        if compute is None:
            self.compute = []
        elif isinstance(compute, int):
            self.compute = [compute]
        else:
            self.compute = compute

        if memory is None:
            self.memory = []
        elif isinstance(memory, int):
            self.memory = [memory]
        else:
            self.memory = memory

    def __add__(self, other):
        if not isinstance(other, Cost):
            return NotImplemented
        return Cost(
            compute=self.compute + other.compute,
            memory=self.memory + other.memory,
            # max=self.max + other.max,
        )


def time_str(sec):
    if sec < 60:
        return f"{sec:.2f} seconds"
    elif sec < 3600:
        return f"{sec/60:.2f} minutes"
    else:
        return f"{sec/3600:.2f} hours"


def multiplication_cost(
    a: (int, int), b: (int, int), batch: int = 1, repeat: int = 1
) -> Cost:
    assert a[1] == b[0], f"Matrix dimensions need to match for *: {a} vs {b}"
    cost = Cost()
    for _ in range(repeat):
        global_mult_list.append((batch, a[0], a[1], b[1]))
        cost = cost + Cost(
            compute=batch * a[0] * a[1] * b[1],
            memory=batch * (a[0] * a[1] + a[1] * b[1] + b[1] * a[0]),
        )
    return cost


def count_gate_embedding(embed_dim: int, p: Data, batch: int = 1) -> Cost:
    return multiplication_cost(
        (p.g, NUM_GATE_TYPES), (NUM_GATE_TYPES, embed_dim), batch=batch
    )


def count_token_A(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    return Cost()


def count_token_B(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    qubit_oh_dim = (p.n, 1)
    positional_matrix_dim = (1, t.B_positional_embed_dim)
    # Positional embedding
    return multiplication_cost(qubit_oh_dim, positional_matrix_dim, batch=batch)


def count_token_C(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    gate_tensor_dim = (p.g, NUM_GATE_TYPES)
    gate_embedding_matrix_dim = (NUM_GATE_TYPES, t.C_gt_1q_dim)
    gate_oh_cost = multiplication_cost(
        gate_tensor_dim, gate_embedding_matrix_dim, batch=batch
    )

    gate_qubit_tensor_dim = (p.g, 2 * p.n)
    gate_qubit_embedding_matrix_dim = (2 * p.n, t.dB)
    qubit_oh_cost = multiplication_cost(
        gate_qubit_tensor_dim, gate_qubit_embedding_matrix_dim, batch=batch
    )

    return gate_oh_cost + qubit_oh_cost


def count_token_D(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    sign_oh_dim = (p.n, 2)
    sign_embedding_matrix_dim = (2, t.D_stab_sign_dim)
    # 2 because of positional embedding and sign embedding
    return multiplication_cost(
        sign_oh_dim, sign_embedding_matrix_dim, repeat=2, batch=batch
    )


def count_token_E(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    cell_oh_dim = (p.n * p.n, 4)
    cell_embedding_matrix_dim = (4, t.E_pauli_dim)
    return multiplication_cost(cell_oh_dim, cell_embedding_matrix_dim, batch=batch)


def count_hetero_transformer(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    token_dims = [(1, t.dA), (p.n, t.dB), (p.g, t.dC), (p.n, t.dD), (p.n * p.n, t.dE)]
    n_head = 4
    embed_dim = 100
    d_model = embed_dim // n_head
    cost = Cost()
    for dims in token_dims:
        # Projecting to Q, K, V
        cost = cost + multiplication_cost(
            dims, (dims[1], embed_dim), repeat=3, batch=batch
        )
    for i, dim_A in enumerate(token_dims):
        for j, dim_B in enumerate(token_dims):
            if i == j:
                continue
            # Calculating QK^T
            cost = cost + multiplication_cost(
                (dim_A[0], d_model), (d_model, dim_B[0]), repeat=n_head, batch=batch
            )
            # Calculating (QK^T) V
            cost = cost + multiplication_cost(
                (dim_A[0], dim_B[0]), (dim_B[0], embed_dim), repeat=n_head, batch=batch
            )
        # Final projection layer
        cost = cost + multiplication_cost(
            (dim_A[0], 4 * embed_dim), (4 * embed_dim, dim_A[1]), batch=batch
        )
    return cost


def count_homo_transformer(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    token_dims = [(p.n, t.dB), (p.g, t.dC), (p.n, t.dD), (p.n * p.n, t.dE)]
    n_head = 4
    cost = Cost()
    for dims in token_dims:
        embed_dim = dims[1]
        d_model = embed_dim // n_head
        # Projecting to Q, K, V
        cost = cost + multiplication_cost(
            dims, (embed_dim, embed_dim), repeat=3, batch=batch
        )
        # Calculating QK^T
        cost = cost + multiplication_cost(
            (dims[0], d_model), (d_model, dims[0]), repeat=n_head, batch=batch
        )
        # Calculating (QK^T) V
        cost = cost + multiplication_cost(
            (dims[0], dims[0]), (dims[0], d_model), repeat=n_head, batch=batch
        )
        # Final projection layer
        cost = cost + multiplication_cost(dims, (dims[1], dims[1]), batch=batch)
    return cost


def count_depth_cost(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    return multiplication_cost((1, t.dA), (t.dA, 1), batch=batch)


def count_gate_cost(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    return multiplication_cost((p.g, t.dC), (t.dC, 1), batch=batch)


def single_iter_count(t: TokenProperties, p: Data, batch: int = 1) -> Cost:
    cost = Cost()
    cost = cost + count_token_A(t, p, batch=batch)
    cost = cost + count_token_B(t, p, batch=batch)
    cost = cost + count_token_C(t, p, batch=batch)
    cost = cost + count_token_D(t, p, batch=batch)
    cost = cost + count_token_E(t, p, batch=batch)
    for i in range(2):
        cost = (
            cost
            + count_hetero_transformer(t, p, batch=batch)
            + count_homo_transformer(t, p, batch=batch)
        )
    cost = (
        cost + count_depth_cost(t, p, batch=batch) + count_gate_cost(t, p, batch=batch)
    )
    return cost


def get_size(shape):
    if len(shape) == 1:
        return shape[0]
    return shape[0] * get_size(shape[1:])


# Analyze cost statistics
# folder = "/scratch1/sauravk/lsp-npz"
folder = "training-data/compiled"
train_data = UnprepNpzDataloader(f"{folder}/new-sample-train-2-20.npz", shuffle=False)
print(f"Dataset size: {train_data.get_total_size()}")

batch_size = 64
train_data.set_batch_size(batch_size)
num_batches = train_data.get_total_size() / batch_size

token_dims = TokenProperties(128, 32, 64, 32, 32)

cost = Cost()
total_num = 0
tic = time.time()
for i, data in enumerate(iter(train_data)):
    n, g = data["layout"].shape[1], data["gate_oh"].shape[1]
    batch = data["layout"].shape[0]
    cost = cost + single_iter_count(token_dims, Data(n, g), batch=batch)
    total_num += sum(get_size(x.shape) for x in data.values())

    if i % 1_000 == 0:
        elapsed = time.time() - tic
        avg_time = elapsed / (i + 1) if i > 0 else 0
        remaining_batches = num_batches - (i + 1)
        est_remaining = avg_time * remaining_batches
        print(
            f"Iterated over {i} batches | Avg time: {avg_time:.4f} s/batch | Estimated time left: {time_str(est_remaining)}"
        )
toc = time.time()
print(f"Total iteration over dataset completed in {toc - tic:.4f} seconds")
print(
    f"Total numbers: {total_num}. Average: {total_num / (i)} per batch, {total_num / (i * batch_size)} per datapoint"
)


total_memory_cost = cost.memory
total_compute_cost = cost.compute
total_cost = cost.max

print(
    f"Total memory cost: {2 * total_memory_cost:.2e} FLOPs = {time_str(total_memory_cost / th_mem)}"
)
print(
    f"Total compute cost: {2 * total_compute_cost:.2e} FLOPs = {time_str(total_compute_cost / th_com)}"
)
print(f"Total back of envelope estimate for cost: {time_str(total_cost)}")

import numpy as np

# Compute ratio of memory to compute for each item
# ratios = []
# for mem, comp in zip(cost.memory, cost.compute):
#     if comp != 0:
#         ratios.append(mem / comp)
#     else:
#         ratios.append(float("inf"))  # or np.nan if you prefer

ratios_np = np.array(global_ratio_list)

# Compute statistics
mean_ratio = np.mean(ratios_np)
median_ratio = np.median(ratios_np)
std_ratio = np.std(ratios_np)
min_ratio = np.min(ratios_np)
max_ratio = np.max(ratios_np)
percentiles = np.percentile(ratios_np, [25, 50, 75, 90, 95, 99])

print("Memory/Compute Ratio Statistics:")
print(f"  Mean: {mean_ratio:.4f}")
print(f"  Median: {median_ratio:.4f}")
print(f"  Std: {std_ratio:.4f}")
print(f"  Min: {min_ratio:.4f}")
print(f"  Max: {max_ratio:.4f}")
print("  Percentiles:")
for p, v in zip([25, 50, 75, 90, 95, 99], percentiles):
    print(f"    {p}th: {v:.4f}")


# Draw a text histogram of the distribution of memory/compute ratios
def draw_text_histogram(data, bins=10, width=50, char="#"):
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = hist.max()
    for i in range(len(hist)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        count = hist[i]
        bar_len = int(width * count / max_count) if max_count > 0 else 0
        bar = char * bar_len
        print(f"{left:8.4f} - {right:8.4f} | {bar} ({count})")


print("Histogram of Memory/Compute Ratio Distribution:")
draw_text_histogram(ratios_np, bins=20)


import pickle

with open("global_mult_list.pkl", "wb") as f:
    pickle.dump(global_mult_list, f)
print(f"global_mult_list dumped to global_mult_list.pkl")
