from typing import Any

import torch

from envs.logical_state_preparation_env import LogicalStatePreparationEnv
from models.input import GT_1Q, GT_2Q, Layout


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
        observation: torch.Tensor | list[torch.Tensor],
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
