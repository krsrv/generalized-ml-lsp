"""
Generate raw data corresponding to gates in a random circuit. The module generates an HDF5 file
with keys of the form `n/g/d/{key}` where `n` = number of qubits, `g` = number of gate instances,
`d` = circuit depth, {key} = {n, layout, observation, depth, gates, gate_oh, gate_qubits_oh}.
"""

import os
import time
from typing import Any

import h5py
import jax  # Only for interacting with LSP environment
import numpy as np
import torch

from envs.logical_state_preparation_env import LogicalStatePreparationEnv
from models.input import GT_1Q, GT_2Q, Layout, sample_layout
from simulators.clifford_gates import CliffordGates
from training.utils import write_to_file
from transform import _construct_gate_embeddings, _construct_gate_qubit_embeddings

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
jax.config.update("jax_platform_name", "cpu")


class Params:
    """Class to store data about a Clifford circuit, with the topology, applied gates, available gate sets
    and a simulator.
    Members:
        n: int -> number of qubits
        layout: Layout -> graph layout
        gate_set_1q: list[GT_1Q] -> list of 1 qubit gate types
        gate_set_2q: list[GT_2Q] -> list of 2 qubit gate types
        circuit_depth: int -> number of gates applied
        gates: list[Any] -> gates applied to the circuit. The gates are represented as indices of the env.action_space()
            list.
        observation: torch.Tensor -> the stabilizer/check matrix after the circuit execution
        env: LogicalStatePreparationEnv -> simulator
    """

    def __init__(
        self,
        n: int,
        layout: Layout,
        gate_set_1q: list[GT_1Q],
        gate_set_2q: list[GT_2Q],
        circuit_depth: int,
    ):
        self.n = n
        self.layout = layout
        self.gate_set_1q = gate_set_1q
        self.gate_set_2q = gate_set_2q
        self.circuit_depth = circuit_depth
        self.g = self.n * len(self.gate_set_1q) + len(self.layout.adjacency_list) * len(
            self.gate_set_2q
        )

    def __hash__(self):
        return hash(
            (
                self.n,
                str(self.layout.graph.numpy()),
                str([x.value for x in self.gate_set_1q]),
                str([x.value for x in self.gate_set_2q]),
                self.circuit_depth,
                str(self.gates),
            )
        )


def prepare_hdf5_dataset(output_file: str, n: int, g: int, d: int) -> None:
    """
    Sets up the HDF5 file for dumping contents. Pre-specifying expected feature dimensions makes the
    module much faster.

    * Assumes that the HDF5 file already exists, and there are no conflicting dataset names.
    * The keys are {prefix}/{n, layout, gate_oh, gate_qubit_oh, depth, observation}
    * The key prefix is n/g/d, where d is the depth of the circuit
    * The `maxshape` argument is set to None, which means that the file can be extended infinitely.
    * The `chunk` argument is set to True, which means that the contents will be written in chunks
    ideally.

    Args:
        output_file: the full path to the file, including the ".hdf5" extension
        n, g: number of qubits and gate instances
        d: circuit depth
    """
    key = f"{n}/{g}/{d}"
    gate_oh_size = len(GT_1Q) + len(GT_2Q)
    with h5py.File(output_file, "a") as f:
        if key in f:
            return
        f.create_dataset(
            f"{key}/n", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/layout",
            shape=(0, n, n),
            maxshape=(None, n, n),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/gate_oh",
            shape=(0, g, gate_oh_size),
            maxshape=(None, g, gate_oh_size),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/gate_qubit_oh",
            shape=(0, g, 2 * n),
            maxshape=(None, g, 2 * n),
            dtype="bool",
            chunks=True,
        )
        f.create_dataset(
            f"{key}/depth", shape=(0,), maxshape=(None,), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/gates", shape=(0, d), maxshape=(None, d), dtype="int64", chunks=True
        )
        f.create_dataset(
            f"{key}/observation",
            shape=(0, 2 * n * n + n),
            maxshape=(None, 2 * n * n + n),
            dtype="bool",
            chunks=True,
        )


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

    The sampled objects are returned wrapped in a Params object.
    """
    # n = Number of qubits
    n = torch.randint(low=n_min, high=n_max, size=(1,), generator=gen)[0]
    n = n.int().item()

    # Layout
    layout = sample_layout(n, gen)
    # Gate set
    gate_set_1q, gate_set_2q = sample_gate_set(gen)
    depth = sample_depth(
        n,
        gate_set_1q,
        gate_set_2q,
        use_max_depth=use_max_depth,
        gen=gen,
    )
    return Params(n, layout, gate_set_1q, gate_set_2q, depth)


def sample_depth(
    n: int,
    gate_set_1q: list[GT_1Q],
    gate_set_2q: list[GT_2Q],
    use_max_depth=True,
    gen: torch.Generator | None = None,
):
    """
    Given n, 1 and 2 qubit gate sets, randomly sample the circuit depth. If `use_max_depth` is True,
    simply choose the max depth for which we expect the random circuit to be close to optimal.
    """
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

    return depth


def new_dump_object() -> dict:
    """
    Generate the leaf object to dump in the HDF5 file.
    """
    return {
        "n": [],
        "layout": [],
        "gate_oh": [],
        "gate_qubit_oh": [],
        "depth": [],
        "gates": [],
        "observation": [],
    }


def generate_data_for_params(
    params: Params,
    output_dict: dict,
    jax_rng_key: jax.Array,
    gen: torch.Generator | None = None,
) -> dict:
    """Given a Params `x` objects, generate a circuit of n = x.n qubits with topology sepcified by
    x.layout, gate set by x.gate_set_{1q,2q} and depth x.circuit_depth
    Args:
        Params
        keys: Tuple of 4 RNGs for jax
        gen: torch.Generator
    """
    key, key_reset, key_act, key_step = jax.random.split(jax_rng_key, 4)

    lsp_env = create_lsp_env(
        params.layout,
        params.gate_set_1q,
        params.gate_set_2q,
        params.circuit_depth,
    )

    env_params = None
    _observation, env_state = lsp_env.reset_env(key_reset, env_params)

    n = params.n
    assert (
        _observation.shape[-1] == 2 * n * n + n
    ), f"""Implementation (e.g. StabilizerEncoding) depends on the shape
    of the observation being 2 * n * n + n (all stabilizers, followed by n signs). Received {_observation.shape}
    instead"""

    gate_list = []
    for d in range(1, params.circuit_depth + 1):
        key_act, _rng = jax.random.split(key_act)
        gate_list.append(lsp_env.action_space(env_params).sample(key_act))

        key_step, _rng = jax.random.split(key_step)
        observation, env_state, _reward, _done, _info = lsp_env.step_env(
            key_step, env_state, gate_list[-1], env_params
        )
        output_dict[d]["n"].append(n)

        laplacian = params.layout.graph.numpy()
        adjacency = laplacian - np.diag(np.diag(laplacian))
        adjacency = np.array(adjacency, dtype=np.bool_)
        output_dict[d]["layout"].append(adjacency)

        gt_1q = np.array([x.value for x in params.gate_set_1q], dtype=np.int32)
        gt_2q = np.array([x.value for x in params.gate_set_2q], dtype=np.int32)
        output_dict[d]["gate_oh"].append(
            _construct_gate_embeddings(gt_1q, gt_2q, adjacency)
        )
        output_dict[d]["gate_qubit_oh"].append(
            _construct_gate_qubit_embeddings(gt_1q, gt_2q, adjacency)
        )

        output_dict[d]["depth"].append(d)
        output_dict[d]["gates"].append(np.array(gate_list, dtype=np.int64))

        observation = np.array(observation, dtype=np.bool_)
        output_dict[d]["observation"].append(observation.copy())

    return output_dict, key


def generate_and_write_training_data(
    ng_pair_count: int,
    file: str,
    jax_rng_key: jax.Array,
    gen: torch.Generator,
) -> None:
    """
    Given a number of (n, g) pairs to sample, and path to an HDF5 file, for each count:
    1. Sample a random circuit layout
    2. Generate 10 random circuits for the given random circuit
    3. Dump the generated circuit to the HDF5 file.
    """
    ## Remove the file if it already exists.
    # if os.path.exists(file):
    #     os.remove(file)

    # Create the h5py file
    file_handler = h5py.File(file, "w")
    file_handler.close()

    n_min, n_max = 2, 20
    repeat_count = 10
    for i in range(ng_pair_count):
        # Sample random qubit layout
        params = sample_parameters(n_min, n_max, gen, use_max_depth=True)
        # Create new dump object corresponding to each circuit depth
        depth_instance_map = {
            k: new_dump_object() for k in range(1, params.circuit_depth + 1)
        }
        # Populate `depth_instance_map` with `repeat_count` random circuits
        for _ in range(repeat_count):
            depth_instance_map, jax_rng_key = generate_data_for_params(
                params, depth_instance_map, jax_rng_key, gen
            )
        # For each depth, dump the contents to the HDF5 file.
        for depth, instance_dict in depth_instance_map.items():
            prepare_hdf5_dataset(file, params.n, params.g, depth)
            write_to_file(instance_dict, file, f"{params.n}/{params.g}/{depth}")
        print(f"Completed {i} pairs")


def parallel_task_wrapper(ng_pair_count, filename):
    seed = time.time_ns()
    # JAX RNG
    key = jax.random.key(seed)
    # Torch RNG
    gen = torch.Generator()
    gen.manual_seed(seed)

    tic = time.time()
    generate_and_write_training_data(ng_pair_count, filename, key, gen)
    toc = time.time()

    print(f"{filename} generated ({toc - tic} sec)")


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", type=int, default=100, help="Number of training instances to generate"
    )
    parser.add_argument(
        "-f",
        type=str,
        default="training-data/sd.hdf5",
        help="Relative path to (existing) output folder",
    )
    parser.add_argument("-t", type=int, default=8, help="Number of processes to spawn")
    parser.add_argument("--expid", type=str, default="", help="Experiment ID")
    args = parser.parse_args()

    if args.t > 1:
        print(f"Starting {args.t} processes")
        params = [(args.n, f"{args.f}/{args.expid}-{i}.hdf5") for i in range(args.t)]
        with multiprocessing.Pool(processes=args.t) as pool:  # 8 CPUs available
            results = pool.starmap(parallel_task_wrapper, params)
    else:
        print(f"Starting {args.t} process")
        parallel_task_wrapper(args.n, f"{args.f}/{args.expid}.hdf5")
    print("Data generation complete")
