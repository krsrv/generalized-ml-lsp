import subprocess

import numpy as np
import torch
import torch.nn as nn

from training.dataset import transform_graph

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


def is_1_qubit_gate(gate):
    return gate < 7


def is_symmetric_gate(gate):
    return gate == 8


def get_gate_vectors(layout: np.ndarray, gate_set: list) -> (list, list, dict):
    # The implementation might be inefficient, but needs to match up with the
    # output from the CC functions Gate::getIndex() and Gate::fromIndex()
    assert len(layout.shape) == 2, "Only batch size = 1 is supported."
    gates, gate_qubits = [], []
    reverse_gate_dict = {}
    n = layout.shape[-1]
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


def is_successfully_unprepared(beam: list["Path"]):
    return np.any([path.is_successfully_unprepared() for path in beam])


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
            print(f"Arguments: {args}")
            raise RuntimeError(f"Simulator failed: {result.stderr}")
        new_state = np.array([int(c) for c in result.stdout.strip()])
        self.set_state(new_state)
        return new_state


class Path:
    """
    Class for each path in beam search.
    Or class for an entire beam.
    """

    def __init__(
        self, n: int, observations: list, depths: list, gates: list, cost: int
    ):
        self.n = n
        self.observations = observations
        self.depths = depths
        self.gates = gates
        self.cost = cost

    def __str__(self, layout=None, gate_set=None):
        st = f"Success: {self.is_successfully_unprepared()}"
        st += "\n"
        # st += f"Observations : {[format_observation(x, self.n) for x in self.observations]}"
        # st += "\n"
        # st += f"Gates : {', '.join(get_gate_literals(self.gates, layout, gate_set))}"
        # st += "\n"
        st += f"Depths : {', '.join([f'{(x + 2.2).numpy() * 2:.2f}' for x in self.depths])}"
        st += "\n"
        return st

    def is_successfully_unprepared(self):
        """
        Return true if any of the input paths is an all 0 state.
        """
        return Path._is_all_Z_state(self.observations[-1], self.n)

    def filter_beam(self, top_indices: torch.Tensor):
        """
        Truncate beam to keep top `k` elements for each batch. The corresponding indices
        are tracked by the tensor `top_indices`
        """
        k, bs = top_indices.shape
        width = self.observations.shape[0] // bs
        # Modify observations
        self.observations = self.observations.reshape(width, bs, -1)
        cols = (
            torch.arange(bs, device=self.observations.device)
            .unsqueeze(0)
            .expand_as(top_indices)
        )
        self.observations = self.observations[top_indices, cols]
        self.observations = self.observations.reshape(k * bs, -1)
        # Modify depths
        cols.to(self.depths.device)
        self.depths = self.depths.reshape(width, bs, -1)
        self.depths = self.depths[top_indices, cols]
        self.depths = self.depths.reshape(k * bs, -1)
        # Modify gates
        self.gates = self.gates.reshape(width, bs, -1)
        self.gates = self.gates[top_indices, cols]
        self.gates = self.gates.reshape(k * bs, -1)

    def append_tensors(self, new_gates: torch.Tensor, depths: torch.Tensor):
        """
        Update beam to add new gate and depth tensors.
        The `new_gates` tensor corresponds to the next gate to be applied in the circuit. For
        each path in the beam, and for each element in the batch, there are `k` new gates to
        be added.
        The `depths` tensor corresponds to the current depth prediction, and should be added
        to each of the new copy generated.
        """
        # Add new gates
        bw, bs, k = new_gates.shape
        new_gates = new_gates.transpose(1, 2).transpose(0, 1).unsqueeze(-1)
        self.gates = self.gates.reshape(bw, bs, -1).unsqueeze(0).expand(k, -1, -1, -1)
        self.gates = torch.concat((self.gates, new_gates), dim=-1)
        self.gates = self.gates.reshape(k * bw * bs, -1)
        # Attach depth predictions
        self.depths = self.depths.reshape(bw, bs, -1)
        depths = depths.unsqueeze(-1)
        self.depths = torch.concat((self.depths, depths), dim=-1)
        self.depths = (
            self.depths.unsqueeze(0).expand(k, -1, -1, -1).reshape(k * bw * bs, -1)
        )

    def update_observations(self, simulators: list[Simulator], bs: int, k: int):
        self.observations = (
            self.observations.reshape(-1, bs, 2 * self.n * self.n + self.n)
            .unsqueeze(0)
            .expand(k, -1, -1, -1)
        )
        bw = self.observations.shape[1]
        self.gates = self.gates.reshape(k, bw, bs, -1)

        # Permute so that observations and gates are 2D, with the bs index referring to each row
        # Start a new job for each of the bs elements, and sequentially process all the elements
        # inside the job.
        # For 10 batches:
        # With simulation + no parallelization = 520s
        # With no simulation = 56s
        # With simulation for 1 bs = 68s
        # Expected runtime for parallelized simulation would be 80-90s.
        for j in range(1):
            for t in range(k):
                for i in range(bw):
                    observation = self.observations[t, i, j, :]
                    simulators[j].set_state(observation.cpu().numpy())
                    simulators[j].step(self.gates[t, i, j, -1].cpu().numpy())
                    self.observations[t, i, j, :] = torch.tensor(
                        simulators[j].step(self.gates[t, i, j, -1].cpu().numpy()),
                        device=self.observations.device,
                    )
        self.observations = self.observations.reshape(k * bw * bs, -1)
        self.gates = self.gates.reshape(k * bw * bs, -1)

    @staticmethod
    def _is_all_Z_state(observation: np.ndarray, n: int):
        observation = observation.reshape(n, -1)
        return (
            np.all(observation[:, :n] == 0)  # No X
            and np.all(observation[:, -1] == 0)  # No '-' sign
            and np.count_nonzero(observation == 1) == n  # Exactly n 1s
        )


class DataHolder:
    def __init__(
        self,
        layout=None,
        gate_set=None,
        evals=None,
        evecs=None,
        gates=None,
        gate_qubits=None,
    ):
        assert (layout is not None and gate_set is not None) or (
            evals is not None
            and evecs is not None
            and gates is not None
            and gate_qubits is not None
        ), "Either (layout, gate_set) or (evals, evecs, gates, gate_qubits) must be provided"
        if layout is not None and gate_set is not None:
            self.gate_set = gate_set
            self.evals, self.evecs = transform_graph(layout)
            self.gates, self.gate_qubits, _ = get_gate_vectors(layout, gate_set)
            self.n = layout.shape[0]
        else:
            assert len(evals.shape) == 2, "Input data needs to be in a batch format."
            self.evals = evals
            self.evecs = evecs
            self.gates = gates
            self.gate_qubits = gate_qubits
            self.n = evals.shape[-1]
            self.g = self.gates.shape[-1]
            self.gate_set = [np.unique(gates) for gates in self.gates]

        self.prepare_for_inference()

    def prepare_for_inference(self) -> None:
        if not torch.is_tensor(self.evals):
            self.evals = torch.tensor(self.evals, dtype=torch.float32)
        if not torch.is_tensor(self.evecs):
            self.evecs = torch.tensor(self.evecs, dtype=torch.float32)
        if not torch.is_tensor(self.gates):
            self.gates = torch.tensor(self.gates, dtype=torch.long)
        if not torch.is_tensor(self.gate_qubits):
            self.gate_qubits = torch.tensor(self.gate_qubits, dtype=torch.long)

    def replicate(
        self, rep
    ) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        evals = (
            torch.unsqueeze(self.evals, dim=0).expand(rep, -1, -1).reshape(-1, self.n)
        )
        evecs = (
            torch.unsqueeze(self.evecs, dim=0)
            .expand(rep, -1, -1, -1)
            .reshape(-1, self.n, self.n)
        )
        gates = (
            torch.unsqueeze(self.gates, dim=0).expand(rep, -1, -1).reshape(-1, self.g)
        )
        gate_qubits = (
            torch.unsqueeze(self.gate_qubits, dim=0)
            .expand(rep, -1, -1)
            .reshape(-1, 2 * self.g)
        )
        return evals, evecs, gates, gate_qubits


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

        self._set_device()

    def _should_terminate(self, beam: list[Path], n: int, depth: int) -> bool:
        """
        Check whether any of the paths in the beam have been successfully unprepared to the
        all 0 state.
        """
        if depth > self.max_depth:
            return True
        return is_successfully_unprepared(beam)

    def _set_device(self):
        if hasattr(self, "device") and self.device is not None:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model now on device={self.device}")

    def _run_inference_on_beam(self, beam, eval, evec, gates, gate_qubits):
        width = len(beam)
        gate_prediction_logit, depth_prediction = self.model.forward(
            torch.unsqueeze(eval, dim=0).expand(width, -1).to(self.device),
            torch.unsqueeze(evec, dim=0).expand(width, -1, -1).to(self.device),
            torch.unsqueeze(gates, dim=0).expand(width, -1).to(self.device),
            torch.unsqueeze(gate_qubits, dim=0).expand(width, -1).to(self.device),
            torch.tensor(np.array([x.observations[-1] for x in beam])).to(self.device),
        )
        return gate_prediction_logit.to("cpu"), depth_prediction.to("cpu")

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

    def _create_simulators(
        self, layouts: np.ndarray, gates: np.ndarray
    ) -> list[Simulator]:
        simulator_list = []
        for layout, gate in zip(layouts, gates):
            gate_set = np.unique(gate)
            simulator_list.append(Simulator(layout, gate_set, None))
        return simulator_list

    def infer(
        self,
        layout: np.ndarray,
        gate_set: np.ndarray,
        target: np.ndarray,
        beam_width: int = 1,
    ) -> (list[Path], bool):
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
        data = DataHolder(layout=layout, gate_set=gate_set)
        simulator = Simulator(layout, gate_set, None)
        depth = 0
        simulator.set_state(target)
        observation = torch.tensor(target)

        curr_beam = [Path(data.n, [target], [], [], +torch.inf)]
        new_beam = []

        while not self._should_terminate(curr_beam, data.n, depth):
            # Calculate predicted cost and truncate beam to beam_width
            curr_beam, gate_prediction_logit, depth_prediction = (
                self._explore_and_truncate_beam(
                    curr_beam,
                    data.n,
                    data.evals,
                    data.evecs,
                    data.gates,
                    data.gate_qubits,
                    beam_width,
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
                            data.n,
                            curr_beam[i].observations + [observation],
                            curr_beam[i].depths + [depth_prediction.detach()[i]],
                            curr_beam[i].gates + [gate],
                            (depth_prediction.detach()[i] + 1) * data.n * data.n / 4,
                        )
                    )
            curr_beam = new_beam
            new_beam = []
            depth += 1

        curr_beam, _gp, _d = self._explore_and_truncate_beam(
            curr_beam,
            data.n,
            data.evals,
            data.evecs,
            data.gates,
            data.gate_qubits,
            beam_width,
        )
        return curr_beam, is_successfully_unprepared(curr_beam)

    def infer_batch(
        self,
        layouts: np.ndarray,
        evals: np.ndarray,
        evecs: np.ndarray,
        gates: np.ndarray,
        gate_qubits: np.ndarray,
        targets: np.ndarray,
        beam_width: int = 1,
    ) -> (list[Path], bool):
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
        # Initialize variables
        batch_size = evals.shape[0]
        data = DataHolder(
            evals=evals, evecs=evecs, gates=gates, gate_qubits=gate_qubits
        )
        observation = torch.tensor(targets).to(self.device)  # (1, bs, 2n+1)
        curr_beam = Path(
            data.n,
            observation,
            torch.empty(batch_size, device=self.device),
            torch.empty(batch_size, device=self.device),
            None,
        )
        simulators = self._create_simulators(layouts, gates)
        depth = 0

        with torch.no_grad():
            while depth < self.max_depth:
                ############
                # Calculate predicted cost
                ############
                width = curr_beam.observations.shape[0] // batch_size
                print(f"Iteration: {depth}")
                evals, evecs, gates, gate_qubits = data.replicate(width)
                gate_prediction_logit, depth_prediction = self.model.forward(
                    evals.to(self.device),  # (width * bs, n)
                    evecs.to(self.device),  # (width * bs, n, n)
                    gates.to(self.device),  # (width * bs, g)
                    gate_qubits.to(self.device),  # (width * bs, 2*g)
                    curr_beam.observations,  # (width * bs, 2n^2+n)
                )

                ############
                # Truncate
                ############
                depth_prediction = depth_prediction.reshape(width, batch_size)
                k = np.minimum(beam_width, width)
                depth_prediction, top_indices = torch.topk(depth_prediction, k, dim=0)
                # top_indices: (beam_width, bs)
                cols = (
                    torch.arange(batch_size, device=self.device)
                    .unsqueeze(0)
                    .expand_as(top_indices)
                )  # (beam_width, bs)
                # Filter other variables
                gate_prediction_logit = gate_prediction_logit.reshape(
                    width, batch_size, -1
                )
                gate_prediction_logit = gate_prediction_logit[top_indices, cols]

                ############
                # Expand beam with new elements.
                ############
                gate_predictions = nn.Softmax(-1)(gate_prediction_logit)
                expansion_ratio = 4
                _, top_gates = torch.topk(gate_predictions, expansion_ratio, dim=-1)
                curr_beam.filter_beam(top_indices)
                curr_beam.append_tensors(top_gates, depth_prediction)
                curr_beam.update_observations(simulators, batch_size, expansion_ratio)

                depth += 1

            # curr_beam, _gp, _d = self._explore_and_truncate_beam(
            #     curr_beam, n, eval, evec, gates, gate_qubits, beam_width
            # )

        return curr_beam, True  # is_successfully_unprepared(curr_beam)
