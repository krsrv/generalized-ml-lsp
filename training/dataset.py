import jax  # Only for interacting with LSP environment
import numpy as np
import torch
from typing import Any
from models.input import GT_1Q, GT_2Q, Layout, sample_layout
from envs.logical_state_preparation_env import LogicalStatePreparationEnv
from simulators.clifford_gates import CliffordGates


class TrainingInstance:
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
        gates: list[Any],
        observation: torch.Tensor,
        env: LogicalStatePreparationEnv,
    ) -> None:
        self.n = n
        self.layout = layout
        self.gate_set_1q = gate_set_1q
        self.gate_set_2q = gate_set_2q
        self.circuit_depth = circuit_depth
        self.gates = gates
        self.observation = observation
        self.env = env
    
    def __hash__(self):
        return hash((
            self.n,
            str(self.layout.graph.numpy()),
            str([x.value for x in self.gate_set_1q]),
            str([x.value for x in self.gate_set_2q]),
            self.circuit_depth,
            str(self.gates)
        ))


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
    identity_string = f"{"".join(["I" for _ in range(n)])}"
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


def generate_training_data_for_given_params(
    n: int,
    jax_rng_key: jax.Array,
    gen: torch.Generator | None = None,
) -> tuple[TrainingInstance, jax.Array]:
    """Given `n` = # qubits, generate a circuit of n qubits with a random topology
    and random gate set, with depth = `depth`.
    Args:
        n
        keys: Tuple of 4 RNGs for jax
        gen: torch.Generator
    """
    key, key_reset, key_act, key_step = jax.random.split(jax_rng_key, 4)

    layout = sample_layout(n, gen)
    gate_set_1q, gate_set_2q = sample_gate_set(gen)

    if len(gate_set_1q) + len(gate_set_2q) < 2:
        d_max = 3
    else:
        d_max = np.floor(
            n * n / np.log2(len(gate_set_1q) + len(gate_set_2q)),
            casting="unsafe",
            dtype=np.int32,
        )
    print(f"GT 1Q: {gate_set_1q}, GT 2Q: {gate_set_2q}, n: {n}, max depth: {d_max}")
    depth = (
        torch.randint(low=1, high=d_max, size=(1,), generator=gen)[0]
        if d_max > 1
        else torch.tensor(1)
    )
    depth = depth.item()

    lsp_env = create_lsp_env(layout, gate_set_1q, gate_set_2q, depth)

    env_params = None
    _observation, env_state = lsp_env.reset_env(key_reset, env_params)
    assert (
        _observation.shape[-1] == 2 * n * n + n
    ), f"""Implementation (e.g. StabilizerEncoding) depends on the shape
    of the observation being 2 * n * n + n (all stabilizers, followed by n signs). Received {_observation.shape}
    instead"""

    gate_list = []
    for _ in range(depth):
        key_act, _rng = jax.random.split(key_act)
        gate_list.append(lsp_env.action_space(env_params).sample(key_act))

        key_step, _rng = jax.random.split(key_step)
        observation, env_state, _reward, _done, _info = lsp_env.step_env(
            key_step, env_state, gate_list[-1], env_params
        )
    observation = torch.from_dlpack(observation).long()
    return (
        TrainingInstance(
            n, layout, gate_set_1q, gate_set_2q, depth, gate_list, observation, lsp_env
        ),
        key,
    )
