import subprocess

import numpy as np
import torch
import torch.nn as nn

name_dict = {
    0: "H",
    1: "S",
    2: "Sdg",
    3: "Z",
    4: "X",
    5: "sqrtX",
    6: "sqrtXdg",
    7: "CX",
    8: "CZ",
}


def _transform_graph(adjacency_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Transform an adjacency matrix to a laplacian matrix and return the eigenvalues, eigenvectors
    """
    n = adjacency_matrix.shape[1]
    laplacian = np.array(adjacency_matrix, dtype=np.int32)
    diagonals = np.diag(-np.sum(laplacian, axis=1))
    laplacian = laplacian + diagonals
    return np.linalg.eigh(laplacian)


def is_1_qubit_gate(gate):
    return gate < 7


def is_symmetric_gate(gate):
    return gate == 8


def get_gate_vectors(layout: np.ndarray, gate_set: list) -> (list, list, dict):
    # The implementation might be inefficient, but needs to match up with the
    # output from the CC functions Gate::getIndex() and Gate::fromIndex()
    gates, gate_qubits = [], []
    reverse_gate_dict = {}
    n = layout.shape[0]
    idx = 0
    for gate in gate_set:
        if is_1_qubit_gate(gate):
            for i in range(n):
                gates += [gate]
                gate_qubits += [i + 1, 0]
                reverse_gate_dict[idx] = f"{name_dict[gate]}-{i}"
                idx += 1
        else:
            for i in range(n):
                for j in range(n):
                    if j <= i:
                        continue
                    if layout[i, j]:
                        gates += [gate]
                        gate_qubits += [i + 1, j + 1]
                        reverse_gate_dict[idx] = f"{name_dict[gate]}-{i}-{j}"
                        idx += 1
            if not is_symmetric_gate(gate):
                for i in range(n):
                    for j in range(n):
                        if j <= i:
                            continue
                        if layout[i, j]:
                            gates += [gate]
                            gate_qubits += [j + 1, i + 1]
                            reverse_gate_dict[idx] = f"{name_dict[gate]}-{j}-{i}"
                            idx += 1
    return gates, gate_qubits, reverse_gate_dict


def format_observation(obs: np.ndarray, n: int):
    obs = obs.reshape(n, -1)
    pauli_map = {0: "I", 1: "X", 2: "Z", 3: "Y"}
    output = []
    for row in obs:
        pauli_value = (row[:n] + 2 * row[n : 2 * n])[::-1]
        pauli = [pauli_map[x] for x in pauli_value]
        sign = "+" if row[-1] == 0 else "-"
        output.append(sign + "".join(pauli))
    return ",".join(output)


def print_gate_keys(layout: np.ndarray, gate_set: np.ndarray):
    _, _, rgd = get_gate_vectors(layout, gate_set)
    for k, v in rgd.items():
        print(f"{k}: {v}")


def get_gate_literals(gate_array: np.ndarray, layout: np.ndarray, gate_set: np.ndarray):
    _, _, rgd = get_gate_vectors(layout, gate_set)
    return [rgd[gate] for gate in gate_array]


class Path:
    """
    Class for each path in beam search.
    """

    def __init__(self, observations, depths, gates, cost):
        self.observations = observations
        self.depths = depths
        self.gates = gates
        self.cost = cost

    @staticmethod
    def _is_all_Z_state(observation: np.ndarray, n: int):
        observation = observation.reshape(n, -1)
        return (
            np.all(observation[:, :n] == 0)  # No X
            and np.all(observation[:, -1] == 0)  # No '-' sign
            and np.count_nonzero(observation == 1) == n  # Exactly n 1s
        )

    @staticmethod
    def is_successfully_unprepared(beam: list["Path"], n: int):
        """
        Return true if any of the input paths is an all 0 state.
        """
        return np.any([Path._is_all_Z_state(path.observations[-1], n) for path in beam])


class Simulator:
    """
    Simulate stabilizer circuits
    """

    def __init__(self, layout: np.ndarray, gate_set: np.ndarray, rgd: dict):
        self.layout = layout
        self.n = self.layout.shape[0]
        self.gate_set = gate_set
        self.rgd = rgd
        self.construct_input_format()

    def set_state(self, state):
        self.state = state

    def construct_input_format(self):
        self.layout_input = [self.n * (self.n - 1) // 2]
        for i in range(self.n):
            for j in range(self.n):
                if j <= i:
                    continue
                if self.layout[i, j]:
                    self.layout_input += [i, j]

        self.gate_set_input = [self.gate_set.shape[0]]
        for i in self.gate_set:
            self.gate_set_input += [i]

    def step(self, gate):
        # Prepare the arguments as strings
        args = [
            "../lsp_nonn/src/simulator.out",
            str(self.n),
            *[str(x) for x in self.layout_input],
            *[str(x) for x in self.gate_set_input],
            "".join([str(x) for x in self.state]),
            str(gate),
        ]
        # print(" ".join(args))
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
        )
        # Parse the output from the external executable
        if result.returncode != 0:
            # Print stderr for debugging
            print("Simulator stderr:", result.stderr)
            raise RuntimeError(f"Simulator failed: {result.stderr}")
        new_state = np.array([int(c) for c in result.stdout.strip()])
        self.set_state(new_state)
        return new_state


class InferWrapper:
    """
    Wrapper for running inference using ML models on LSP task. The inference has a max
    depth setting, and utilizes beam search.
    """

    def __init__(self, model: nn.Module, file: str, max_depth=20):
        super().__init__()
        self.max_depth = max_depth
        self.beam_width = 1
        self.model = model
        # load model from file
        saved_data = torch.load(file, map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved_data["model_state_dict"])
        self.model.eval()

    def _should_terminate(self, beam: list[Path], n: int, depth: int) -> bool:
        """
        Check whether any of the paths in the beam have been successfully unprepared to the
        all 0 state.
        """
        if depth > self.max_depth:
            return True
        return Path.is_successfully_unprepared(beam, n)

    def _run_inference_on_beam(self, beam, eval, evec, gates, gate_qubits):
        width = len(beam)
        gate_prediction_logit, depth_prediction = self.model.forward(
            torch.unsqueeze(torch.tensor(eval, dtype=torch.float32), dim=0).expand(
                width, -1
            ),
            torch.unsqueeze(torch.tensor(evec, dtype=torch.float32), dim=0).expand(
                width, -1, -1
            ),
            torch.unsqueeze(torch.tensor(gates), dim=0).expand(width, -1),
            torch.unsqueeze(torch.tensor(gate_qubits), dim=0).expand(width, -1),
            torch.tensor(np.array([x.observations[-1] for x in beam])),
        )
        return gate_prediction_logit, depth_prediction

    def _explore_and_truncate_beam(
        self, beam, n, eval, evec, gates, gate_qubits, beam_width
    ):
        """
        Run model inference to get the depth estimates and truncate to `beam_width` elements
        """
        gate_prediction_logit, depth_prediction = self._run_inference_on_beam(
            beam, eval, evec, gates, gate_qubits
        )
        costs = (depth_prediction.detach().numpy() + 1) * n * n / 4
        k = min(beam_width, len(costs))
        top_indices = np.argpartition(costs, max(0, k - 1))[:k]
        # print(f"Costs: {costs}")
        # print(f"Top indices: {top_indices}")
        beam = [beam[i] for i in top_indices]
        return beam, gate_prediction_logit[top_indices], depth_prediction[top_indices]

    def infer(
        self,
        layout: np.ndarray,
        gate_set: np.ndarray,
        target: np.ndarray,
        beam_width: int = 1,
    ):
        """
        High level function to run inference on a given problem for unpreparing state.
        Args:
            layout: (np.ndarray) Boolean adjacency matrix of size (n, n)
            gate_set: (np.ndarray) list of int representations of GateTypes. Look at the dictionary
                `name_dict` for the mapping.
            target: (np.ndarray) Boolean starting stabilizer of size (2*n+1), in the form
                [X1 ... Xn Z1 ... Zn Sign]
            beam_width: (int)
        """
        eval, evec = _transform_graph(layout)
        n = layout.shape[0]
        gates, gate_qubits, rgd = get_gate_vectors(layout, gate_set)
        simulator = Simulator(layout, gate_set, rgd)
        depth = 0
        simulator.set_state(target)
        observation = torch.tensor(target)

        curr_beam = [Path([target], [], [], +torch.inf)]
        new_beam = []

        while not self._should_terminate(curr_beam, n, depth):
            # Calculate predicted cost and truncate beam to beam_width
            curr_beam, gate_prediction_logit, depth_prediction = (
                self._explore_and_truncate_beam(
                    curr_beam, n, eval, evec, gates, gate_qubits, beam_width
                )
            )

            # Expand beam with new elements.
            gate_predictions = nn.Softmax(-1)(gate_prediction_logit.detach())
            for i, gate_prediction in enumerate(gate_predictions):
                top4 = gate_prediction.squeeze().numpy().argsort()[-4:][::-1]
                for gate in top4:
                    simulator.set_state(curr_beam[i].observations[-1])
                    observation = simulator.step(gate)
                    new_beam.append(
                        Path(
                            curr_beam[i].observations + [observation],
                            curr_beam[i].depths + [depth_prediction.detach()[i]],
                            curr_beam[i].gates + [gate],
                            (depth_prediction.detach()[i] + 1) * n * n / 4,
                        )
                    )
            curr_beam = new_beam
            new_beam = []
            depth += 1

        curr_beam, _gp, _d = self._explore_and_truncate_beam(
            curr_beam, n, eval, evec, gates, gate_qubits, beam_width
        )

        for i, path in enumerate(curr_beam):
            print(f"Path {i}: {Path.is_successfully_unprepared([path], n)}")
            # print(
            #     f"Observations : {[format_observation(x, n) for x in path.observations]}"
            # )
            print(f"Gates : {get_gate_literals(path.gates, layout, gate_set)}")
            print(f"Depths : {[(x + 1) * n * n / 4 for x in path.depths]}")
        return curr_beam, Path.is_successfully_unprepared(curr_beam, n)
