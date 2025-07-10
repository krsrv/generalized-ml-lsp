import os
import pickle
import random
import string
import time
from typing import Any

import jax  # Only for interacting with LSP environment
import numpy as np
import torch

from envs.logical_state_preparation_env import LogicalStatePreparationEnv
from models.input import GT_1Q, GT_2Q, Layout, sample_layout
from simulators.clifford_gates import CliffordGates
from training.instance import TrainingInstance

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
jax.config.update("jax_platform_name", "cpu")


def sample_gate_set(
    gen: torch.Generator | None = None,
) -> tuple[list[GT_1Q], list[GT_2Q]]:
    """Construct a sample gate set. There will be at least 1 gate in the resulting set.

    Args:
        gen: Torch Generator

    Returns:
        tuple[list[GT_1Q], list[GT_2Q]] representing the available gates
    """
    satisfactory = False
    while not satisfactory:
        gate_set_1q = [
            gate for gate in GT_1Q if torch.rand(1, generator=gen).item() < 0.5
        ]
        gate_set_2q = [
            gate for gate in GT_2Q if torch.rand(1, generator=gen).item() < 0.5
        ]
        satisfactory = (len(gate_set_1q) + len(gate_set_2q)) != 0
    return (gate_set_1q, gate_set_2q)


def construct_gate_instances(
    gate_set_1q: list[GT_1Q], gate_set_2q: list[GT_2Q], graph: torch.Tensor
) -> list[Any]:
    """Return the list of gates available in the layout. The elements are (gate, qubit index) and
    (gate, control qubit index, target qubit index) for 1 and 2 qubit gates respectively.

    Args:
        gate_set_1q: number of vertices
        gate_set_2q: Random number generator
        graph: Laplacian of the

    Returns:
        list of gate instances: [(gate, qubit index) | (gate, control qubit index, target qubit index)]
    """
    gate_list = []
    for q_idx, row in enumerate(graph):
        gate_list = gate_list + [(gate, q_idx) for gate in gate_set_1q]
        for t_idx, el in enumerate(row):
            if torch.eq(el, torch.Tensor(1)):
                for gate in gate_set_2q:
                    if gate == GT_2Q.CZ:
                        # Control and target are interchangeable
                        if q_idx > t_idx:
                            gate_list = gate_list + [
                                (gate, q_idx, t_idx) for gate in gate_set_2q
                            ]
                    else:
                        gate_list = gate_list + [
                            (gate, q_idx, t_idx) for gate in gate_set_2q
                        ]
    return gate_list


def create_lsp_env(
    layout: Layout, gate_set_1q: list[GT_1Q], gate_set_2q: list[GT_2Q], max_steps: int
) -> LogicalStatePreparationEnv:
    """Create LSP environment based on the chosen layout and gate sets."""
    n = layout.graph.shape[0]
    # Set target to be the all 0 state.
    identity_string = "".join(["I" for _ in range(n)])
    target = [
        "+" + identity_string[:i] + "Z" + identity_string[i + 1 :] for i in range(n)
    ]

    # Set the gates to be the ones we sampled.
    clifford_gates = CliffordGates(n)
    gate_list = []
    for gate in gate_set_1q:
        match gate:
            case GT_1Q.H:
                gate_list.append(clifford_gates.h)
            case GT_1Q.S:
                gate_list.append(clifford_gates.s)
            case GT_1Q.X:
                gate_list.append(clifford_gates.x)
            case GT_1Q.SQRT_X:
                gate_list.append(clifford_gates.sqrt_x)
    for gate in gate_set_2q:
        match gate:
            case GT_2Q.CX:
                gate_list.append(clifford_gates.cx)
            case GT_2Q.CZ:
                gate_list.append(clifford_gates.cz)
            # case GT_2Q.SWAP:
            #     raise NotImplementedError("SWAP gate not supported in LSP environment yet")
            # case GT_2Q.I_SWAP:
            #     raise NotImplementedError("iSWAP gate not supported in LSP environment yet")

    # Prep the graph input.
    return LogicalStatePreparationEnv(
        target,
        gates=gate_list,
        graph=layout.adjacency_list,
        distance_metric="jaccard",
        max_steps=max_steps,
        threshold=0.99,
        use_max_reward=False,
    )


def sample_parameters(
    n_min: int, n_max: int, gen: torch.Generator | None = None, use_max_depth=True
):
    """Sample the following:
    number of qubits
    layout
    1 qubit gate set
    2 qubit gate set
    circuit depth

    The sampled objects are returned wrapped in a TrainingInstance object.
    """
    # n = Number of qubits
    n = torch.randint(low=n_min, high=n_max, size=(1,), generator=gen)[0]
    n = n.int().item()

    # Layout
    layout = sample_layout(n, gen)
    # Gate set
    gate_set_1q, gate_set_2q = sample_gate_set(gen)

    if len(gate_set_1q) + len(gate_set_2q) < 2:
        d_max = 3
    else:
        d_max = np.max(
            (
                len(gate_set_1q) + len(gate_set_2q),
                np.floor(
                    n * n / np.log2(len(gate_set_1q) + len(gate_set_2q)),
                    casting="unsafe",
                    dtype=np.int32,
                ),
            )
        )
    if use_max_depth:
        depth = d_max
    else:
        depth = (
            torch.randint(low=1, high=d_max, size=(1,), generator=gen)[0]
            if d_max > 1
            else torch.tensor(1)
        )
        depth = depth.item()

    return TrainingInstance(
        n,
        layout,
        gate_set_1q,
        gate_set_2q,
        depth,
        None,
        None,
        None,
    )


def generate_training_data_for_given_params(
    base_structure: TrainingInstance,
    jax_rng_key: jax.Array,
    gen: torch.Generator | None = None,
    store_every_gate_output: bool = False,
) -> tuple[TrainingInstance | list[TrainingInstance], jax.Array]:
    """Given a partiall populated TrainingInstance `x` objects, generate a circuit of n = x.n qubits
    with topology sepcified by x.layout, gate set by x.gate_set_{1q,2q} and depth x.circuit_depth
    Args:
        TrainingInstance
        keys: Tuple of 4 RNGs for jax
        gen: torch.Generator
    """
    key, key_reset, key_act, key_step = jax.random.split(jax_rng_key, 4)

    lsp_env = create_lsp_env(
        base_structure.layout,
        base_structure.gate_set_1q,
        base_structure.gate_set_2q,
        base_structure.circuit_depth,
    )

    env_params = None
    _observation, env_state = lsp_env.reset_env(key_reset, env_params)

    n = base_structure.n
    assert (
        _observation.shape[-1] == 2 * n * n + n
    ), f"""Implementation (e.g. StabilizerEncoding) depends on the shape
    of the observation being 2 * n * n + n (all stabilizers, followed by n signs). Received {_observation.shape}
    instead"""

    gate_list = []
    training_instance_list = []
    for d in range(base_structure.circuit_depth):
        key_act, _rng = jax.random.split(key_act)
        gate_list.append(lsp_env.action_space(env_params).sample(key_act))

        key_step, _rng = jax.random.split(key_step)
        observation, env_state, _reward, _done, _info = lsp_env.step_env(
            key_step, env_state, gate_list[-1], env_params
        )
        training_instance_list.append(
            TrainingInstance(
                base_structure.n,
                base_structure.layout,
                base_structure.gate_set_1q,
                base_structure.gate_set_2q,
                d,
                gate_list.copy(),
                np.array(observation),
                lsp_env,
            )
        )
    if store_every_gate_output:
        return (training_instance_list, key)
    else:
        return (training_instance_list[-1], key)


def generate_training_data(
    N: int,
    jax_rng_key: jax.Array,
    gen: torch.Generator,
    folder: str,
    prefix: str = "",
    use_random: bool = False,
) -> None:
    def generate_file_name(folder, index, n_count, use_random):
        file_name = f"{folder}/"
        if prefix != "":
            file_name += f"{prefix}-"
        if use_random:
            file_name = (
                file_name
                + "".join(
                    random.choice(string.ascii_lowercase + string.digits)
                    for _ in range(14)
                )
                + f"-{n_count}"
                + ".pkl"
            )
            if os.path.exists(file_name):
                return generate_file_name(folder, index, n_count, use_random)
        else:
            file_name = file_name + f"{index}-{n_count}.pkl"
        return file_name

    batch: list[TrainingInstance] = []
    batch_size = 1_00
    batch_count = 0
    n_min, n_max = 2, 20
    # generated_set = set()
    for _i in range(N):
        try:
            # Sample random parameters
            base_structure = sample_parameters(n_min, n_max, gen, use_max_depth=True)
            # print(f"n: {base_structure.n}, gates: {base_structure.gate_set_1q}, {base_structure.gate_set_2q}, edges: {len(base_structure.layout.adjacency_list)}, max_depth: {base_structure.circuit_depth}")
            instances, jax_rng_key = generate_training_data_for_given_params(
                base_structure, jax_rng_key, gen, store_every_gate_output=True
            )
            # if hash(instance) in generated_set:
            #     continue
            # generated_set.add(hash(instance))
            # Handle both single instance and list of instances
            if isinstance(instances, list):
                batch = batch + instances
            else:
                batch.append(instances)

            # Force garbage collection to free memory
            # import gc
            # gc.collect()

        except Exception as e:
            print(f"Error generating instance {_i}: {e}")
            continue
        file_name = generate_file_name(folder, batch_count, _i, use_random)
        if len(batch) > batch_size:
            with open(file_name, "wb") as f:
                pickle.dump(batch, f)
                batch = []
                batch_count += 1
    file_name = generate_file_name(folder, batch_count, _i, use_random)
    if len(batch) > 0:
        with open(file_name, "wb") as f:
            pickle.dump(batch, f)
            batch = []
            batch_count += 1


def parallel_task(n_instances, folder, prefix, random_name):
    seed = time.time_ns()

    # JAX RNG
    key = jax.random.key(seed)

    # Torch RNG
    gen = torch.Generator()
    gen.manual_seed(seed)
    tic = time.time()
    generate_training_data(n_instances, key, gen, folder, prefix, random_name)
    toc = time.time()
    print(toc - tic)


if __name__ == "__main__":
    import argparse
    import multiprocessing

    def task(_):
        parallel_task(args.n, args.f, args.p, args.random_name)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", type=int, default=100, help="Number of training instances to generate"
    )
    parser.add_argument(
        "-f",
        type=str,
        default="training-data",
        help="Relative path to (existing) output folder",
    )
    parser.add_argument(
        "-p",
        type=str,
        default="",
        help="file name prefix",
    )
    parser.add_argument(
        "--random-name", action="store_true", help="Use random names for file outputs"
    )
    parser.add_argument("-t", type=int, default=8, help="Number of processes to spawn")
    args = parser.parse_args()
    with multiprocessing.Pool(processes=args.t) as pool:  # 8 CPUs available
        results = pool.map(task, range(args.t))
    # return results
    # parallelize_task()
