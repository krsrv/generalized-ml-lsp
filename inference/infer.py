import ctypes
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
                        # gate_qubits += [i + 1, j + 1] # Correct
                        gate_qubits += [j + 1, i + 1]  # Incorrect
                        reverse_gate_dict[idx] = f"{name_dict[gate]}-{i}-{j}"
                        idx += 1
            if not is_symmetric_gate(gate):
                for i in range(n):
                    for j in range(n):
                        if j <= i:
                            continue
                        if layout[i, j]:
                            gates += [gate]
                            gate_qubits += [i + 1, j + 1]  # incorrect
                            # gate_qubits += [j + 1, i + 1] # correct
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


class BatchSimulator:
    # Define a ctypes Structure matching the C struct
    class Stabilizers(ctypes.Structure):
        _fields_ = [
            ("stabilizers", ctypes.POINTER(ctypes.c_int)),
            ("is_unprepped", ctypes.POINTER(ctypes.c_bool)),
            ("n", ctypes.c_int),
            ("k", ctypes.c_int),
        ]

    def __init__(self, layouts: np.ndarray, gate_set: np.ndarray):
        assert (
            len(layouts.shape) == 3 and layouts.dtype == np.bool_
        ), f"layouts must be a 3D np.ndarray of dtype np.bool_, got shape={layouts.shape}, dtype={layouts.dtype}"
        assert (
            len(gate_set.shape) == 2 and gate_set.dtype == np.int8
        ), f"gate_set must be a 2D np.ndarray of dtype np.int8, got shape={gate_set.shape}, dtype={gate_set.dtype}"
        self.layouts = layouts
        self.gate_set = gate_set.astype(np.int32)
        self._setup_simulator_ctype()

    def _setup_simulator_ctype(self):
        # Load shared library
        self.lib = ctypes.CDLL(
            "../lsp_nonn/src/libsim.so",
        )
        self.lib.run_simulator.argtypes = [
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_bool),  # layout
            ctypes.c_int,  # num_gate_types
            ctypes.POINTER(ctypes.c_int),  # gate_set
            ctypes.POINTER(ctypes.c_bool),  # tableau_arr
            ctypes.c_int,  # num_applied_gates
            ctypes.POINTER(ctypes.c_int),  # gates
        ]
        self.lib.run_simulator.restype = BatchSimulator.Stabilizers

        self.lib.free_stabilizers_struct.argtypes = [BatchSimulator.Stabilizers]
        self.lib.free_stabilizers_struct.restype = None

    def remove_batch(self, is_batch_unprepped: np.ndarray):
        self.layouts = self.layouts[~is_batch_unprepped]
        self.gate_set = self.gate_set[~is_batch_unprepped]

    def run_simulation(
        self,
        states: np.ndarray,  # (bs, bw, 2*n*n+n)
        gates: np.ndarray,  # (bs, bw, k)
    ):
        assert (
            len(states.shape) == 3 and states.dtype == np.bool_
        ), f"states must be a 3D np.ndarray of dtype np.bool_, got shape={states.shape}, dtype={states.dtype}"
        assert (
            len(gates.shape) == 3 and gates.dtype == np.int_
        ), f"gates must be a 3D np.ndarray of dtype np.int_, got shape={gates.shape}, dtype={gates.dtype}"

        n = self.layouts.shape[-1]
        k = gates.shape[-1]
        gates = gates.astype(np.int32)
        new_state = np.zeros(states.shape[:2] + (k,) + states.shape[2:], dtype=states.dtype)
        is_unprepped = np.zeros(states.shape[:2] + (k,), dtype=np.bool_)
        for i in range(states.shape[0]):
            for j in range(states.shape[1]):
                layout = self.layouts[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
                gate_set = np.ascontiguousarray(self.gate_set[i]).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int)
                )
                state = states[i, j].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
                sim_result = self.lib.run_simulator(
                    n,
                    layout,
                    len(self.gate_set[i, :]),
                    gate_set,
                    state,
                    k,
                    gates[i, j].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                )
                # Access the array
                if not sim_result.stabilizers:
                    raise MemoryError("C function failed to allocate memory")
                new_state[i, j] = np.ctypeslib.as_array(
                    sim_result.stabilizers, shape=(k * (n * n * 2 + n),)
                ).reshape(k, -1)
                is_unprepped[i, j] = np.ctypeslib.as_array(sim_result.is_unprepped, shape=(k,))
                self.lib.free_stabilizers_struct(sim_result)

        return new_state, is_unprepped


class Path:
    """
    Class for each path in beam search.
    Or class for an entire beam.

    `unprepped` only makes sense if self.bs = 1
    Each class variable `observations`, `depths`, `gates` has dimension (width, bs, *)
    where width is the current width (defined by the observations tensor), and bs is the
    current batch size. For instance:
    * width increases after each `update_state` call
    * bs might decrease after an `update_state` call. Remember to update any other object
      which takes into account the batch size
    * width decreases after `filter_beam` call.
    """

    def __init__(
        self,
        n: int,
        observations: list,
        depths: list,
        gates: list,
        width: int = 1,
        bs: int = 64,
        unprepped: bool = False,
    ):
        self.n = n
        self.observations = observations
        self.depths = depths
        self.gates = gates
        self.width = width
        self.bs = bs
        self.unprepped = unprepped

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

    def print_debug_stmt(self):
        print("Beam debug:")
        print(f"    observations shape: {self.observations.shape}")
        if self.depths is not None:
            print(f"    depths shape: {self.depths.shape}")
        if self.gates is not None:
            print(f"    gates shape: {self.gates.shape}")

    def is_successfully_unprepared(self):
        """
        Return true if any of the input paths is an all 0 state.
        """
        observations = self.observations.reshape(self.width, self.bs, -1)
        observations = observations.transpose(0, 1)

        def is_ground_state(subtensor):
            # Example: check if all elements in the (bs, n) subtensor are zero
            return (
                np.all(subtensor[:, :n] == 0)  # No X
                and np.all(subtensor[:, -1] == 0)  # No '-' sign
                and np.count_nonzero(subtensor == 1) == n  # Exactly n 1s
            )
            return torch.all(subtensor == 0)

        # There is no torch.apply or torch.map for tensors, so we use a list comprehension
        # return torch.stack([check_valid(obs) for obs in observations])
        return Path._is_all_Z_state(self.observations[-1], self.n)

    def filter_beam(self, top_indices: torch.Tensor):
        """
        Truncate beam to keep top `k` elements for each batch. The corresponding indices
        are tracked by the tensor `top_indices`
        """
        k, bs = top_indices.shape
        # Modify observations
        self.observations = self.observations.reshape(self.width, bs, -1)
        cols = (
            torch.arange(bs, device=self.observations.device)
            .unsqueeze(0)
            .expand_as(top_indices)
        )
        self.observations = self.observations[top_indices, cols]
        self.observations = self.observations.reshape(k * bs, -1)
        # Modify depths
        if self.depths is not None:
            cols.to(self.depths.device)
            self.depths = self.depths.reshape(self.width, bs, -1)
            self.depths = self.depths[top_indices, cols]
            self.depths = self.depths.reshape(k * bs, -1)
        # Modify gates
        if self.gates is not None:
            cols.to(self.gates.device)
            self.gates = self.gates.reshape(self.width, bs, -1)
            self.gates = self.gates[top_indices, cols]
            self.gates = self.gates.reshape(k * bs, -1)

        self.width = k

    def append_depth_tensor(self, depths: torch.Tensor):
        """
        depths: (width, bs)
        self.depths: (width * bs, depth)
        """
        if self.depths is None:
            self.depths = depths.reshape(-1).unsqueeze(-1)
        else:
            self.depths = self.depths.reshape(self.width, self.bs, -1)
            self.depths = torch.concat((self.depths, depths.unsqueeze(-1)), dim=-1)
            self.depths = self.depths.reshape(self.width * self.bs, -1)

    def append_to(self, path_list: list["Path"]):
        self.observations = self.observations.reshape(self.width, self.bs, -1).transpose(0, 1)
        self.depths = self.depths.reshape(self.width, self.bs, -1).transpose(0, 1)
        self.gates = self.gates.reshape(self.width, self.bs, -1).transpose(0, 1)

        for idx in range(self.observations.shape[0]):
            path_list.append(
                Path(
                    self.n,
                    self.observations[idx].cpu(),
                    self.depths[idx].cpu(),
                    self.gates[idx].cpu(),
                    self.width,
                    1,
                    False,
                )
            )
        self.observations = None
        self.depths = None
        self.gates = None

    def update_states(
        self, new_gates: torch.Tensor, unprepped_states: list["Path"], simulator: BatchSimulator
    ) -> (int, int, np.ndarray[bool]):
        """
        Update beam to add new gate and depth tensors.
        The `new_gates` tensor corresponds to the next gate to be applied in the circuit. For
        each path in the beam, and for each element in the batch, there are `k` new gates to
        be added. Size: (beam width, batch size, # gates to explore)
        The `depths` tensor corresponds to the current depth prediction, and should be added
        to each of the new copy generated. Size: (beam width, batch size).
        After the update, new sizes are:
        gates: (k, bw, bs, depth) -> (k * bw * bs, depth)
        depths: (k, bw, bs, depth) -> (k * bw * bs, depth)

        Args:
            new_gates : torch.Tensor(bw, bs, k)
            unprepped_states : list[Path]. List of currently unprepared states
            simulator : Batch Simulator with layout and gate set info already populated.
                simulator.layout[i] and simulator.gate_set[i] corresponding to ith batch in
                batch size

        Modifies:
            unprepped_states : Adds all the batches which have at least one path where state
                has been successfully unprepared

        Returns:
            new batch size, new width, unprepped batches (bool np.ndarray)
        """
        ###############
        # Add new gates
        ###############
        bw, bs, k = new_gates.shape
        assert self.bs == bs, f"Batch size mismatch: self.bs={self.bs}, bs={bs}"
        new_gates = new_gates.transpose(1, 2).transpose(0, 1).unsqueeze(-1)
        if self.gates is None:
            self.gates = new_gates
        else:
            self.gates = self.gates.reshape(bw, bs, -1).unsqueeze(0).expand(k, -1, -1, -1)
            self.gates = torch.concat((self.gates, new_gates), dim=-1)
        # self.gates -> (k, bw, bs, depth)

        ###############
        # Update depths
        ###############
        self.depths = self.depths.reshape(bw, bs, -1)
        self.depths = self.depths.unsqueeze(0).expand(k, -1, -1, -1)
        # self.depths -> (k, bw, bs, depth)

        ###############
        # Update observations
        ###############
        self.observations = self.observations.reshape(bw, bs, -1).permute(1, 0, 2)
        gates = self.gates.reshape(k, bw, bs, -1).permute(2, 1, 0, 3)
        self.observations, is_unprepped = simulator.run_simulation(
            self.observations.cpu().numpy(),
            gates[:, :, :, -1].cpu().numpy(),
        )
        # self.observations -> (bs, bw, k, 2n^2+n) (np.ndarray)
        # is_unprepped -> (bs, bw, k)

        ###############
        # Remove unprepped states
        ###############
        is_batch_unprepped = np.any(is_unprepped, axis=(-1, -2))
        self.gates = self.gates.transpose(0, 2)
        self.depths = self.depths.transpose(0, 2)
        # Add unprepped states to `unprepped_states`
        for idx in np.nonzero(is_batch_unprepped)[0]:
            unprepped_states.append(
                Path(
                    self.n,
                    self.observations[idx].transpose(1, 0, 2).reshape(bw * k, -1),  # np.ndarray
                    self.depths[idx].transpose(0, 1).reshape(bw * k, -1).cpu(),  # torch.tensor
                    self.gates[idx].transpose(0, 1).reshape(bw * k, -1).cpu(),  # torch.tensor
                    bw * k,
                    1,
                    True,
                )
            )
        # Filter out unprepped states from current state variables
        self.bs = bs = self.bs - np.count_nonzero(is_batch_unprepped)
        self.gates = self.gates[~is_batch_unprepped].transpose(0, 2).reshape(k * bw * bs, -1)
        self.depths = self.depths[~is_batch_unprepped].transpose(0, 2).reshape(k * bw * bs, -1)
        self.observations = (
            torch.tensor(self.observations[~is_batch_unprepped], device=self.gates.device)
            .permute(2, 1, 0, 3)
            .reshape(k * bw * bs, -1)
        )

        self.width = k * bw
        return self.bs, self.width, is_batch_unprepped

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
            # self.gate_set = gate_set
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

    def remove_batch(self, is_batch_unprepped: np.ndarray):
        self.evals = self.evals[~is_batch_unprepped]
        self.evecs = self.evecs[~is_batch_unprepped]
        self.gates = self.gates[~is_batch_unprepped]
        self.gate_qubits = self.gate_qubits[~is_batch_unprepped]

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

    def _set_device(self):
        if hasattr(self, "device") and self.device is not None:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model now on device={self.device}")

    def _create_simulators(self, layouts: np.ndarray, gates: np.ndarray) -> BatchSimulator:
        return BatchSimulator(layouts, gates)

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
            layout: (np.ndarray) (bs, n, n)
            evals: (np.ndarray) (bs, n)
            evecs: (np.ndarray) (bs, n, n)
            gates: (np.ndarray) (bs, g)
            gate_qubits: (np.ndarray) (bs, 2*g)
            targets: (np.ndarray) (bs, 2*n+n)
            beam_width: (int)
        """
        # Initialize variables
        batch_size = evals.shape[0]
        data = DataHolder(
            evals=evals, evecs=evecs, gates=gates, gate_qubits=gate_qubits
        )
        observation = torch.tensor(targets).to(self.device)  # (1, bs, 2n+1)

        simulator = self._create_simulators(layouts, np.unique(gates, axis=-1))
        output_paths = []
        depth = 0
        width = 1

        with torch.no_grad():
            # Model inference runs only `self.max_depth` times. But simulation for each gate
            # prediction in i-th iter happens at beginning of the (i+1)-th loop. The
            # `self.max_depth+1`-th iter only runs the simulation and then breaks out of the
            # loop.
            while depth < self.max_depth + 1:
                ############
                # Expand beam with new elements.
                ############
                if depth == 0:
                    curr_beam = Path(
                        data.n,
                        observation,
                        None,  # float
                        None,  # int32
                        width=1,
                        bs=batch_size,
                    )
                else:
                    gate_predictions = nn.Softmax(-1)(gate_prediction_logit)
                    expansion_ratio = 4
                    _, top_gates = torch.topk(gate_predictions, expansion_ratio, dim=-1)
                    batch_size, width, unprepped_batches = curr_beam.update_states(
                        top_gates, output_paths, simulator
                    )
                    simulator.remove_batch(unprepped_batches)
                    data.remove_batch(unprepped_batches)
                    # Break the loop after simulating the last set of gates.
                    if depth == self.max_depth:
                        break

                ############
                # Calculate predicted cost
                ############
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
                curr_beam.filter_beam(top_indices)
                curr_beam.append_depth_tensor(depth_prediction)

                depth += 1

        curr_beam.append_to(output_paths)
        return output_paths
