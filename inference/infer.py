import ctypes

import numpy as np
import torch
import torch.nn as nn

from training.dataset import transform_graph

# Map from `tableau_xz31.hpp`
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
    9: "iSWAP",
    10: "iSWAPdg",
    11: "ZZ",
    12: "ZZdg",
    13: "XX",
    14: "XXdg",
}


def is_1_qubit_gate(gate):
    return gate < 7


def is_symmetric_gate(gate):
    return gate == 8


def format_observation(obs: np.ndarray, n: int):
    obs = obs.reshape(n, -1)
    pauli_map = {0: "I", 1: "X", 2: "Z", 3: "Y"}
    output = []
    for row in obs:
        pauli_value = row[:n] + 2 * row[n : 2 * n]
        pauli = [pauli_map[x] for x in pauli_value]
        sign = "+" if row[-1] == 0 else "-"
        output.append(sign + "".join(pauli))
    return ",".join(output)


def is_successfully_unprepared(beam: list["Path"]):
    return np.any([path.is_successfully_unprepared() for path in beam])


class BatchSimulator:
    """
    Class to run simulations on a batch of inputs. The key functions are `remove_batch` and `run_simulations`.
    Metadata about the inputs are stored during initialization.
    """
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
            len(gate_set.shape) == 2
        ), f"gate_set must be a 2D np.ndarray, got shape={gate_set.shape}"
        self.layouts = layouts
        self.gate_set = gate_set.astype(np.int32)
        self._setup_simulator_ctype()

    def _setup_simulator_ctype(self):
        # Load shared library
        self.lib = ctypes.CDLL(
            "../lsp_nonn/output/libsim.so",
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
        """
        Remove metadata corresponding to input batches
        Input:
            is_batch_unprepped: bool array with the same length as self.layouts. If true, then remove
                the corresponding layout and gate set.
        """
        self.layouts = self.layouts[~is_batch_unprepped]
        self.gate_set = self.gate_set[~is_batch_unprepped]

    def run_simulation(
        self,
        states: np.ndarray,  # (bs, bw, 2*n*n+n)
        gates: np.ndarray,  # (bs, bw, k)
        filter_size: int,
        hash_sets: list[set] = None,
        torch_device: str = "cpu",
        remove_duplicates: bool = False,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Run simulation on input states and gates. Each states[i,j] will yield new `filter_size` states,
        using the `gates[i,j]` tensor.
        Input:
            states: bool torch.Tensor(bs, bw, 2*n*n+n)
            gates: int torch.Tensor(bs, bw, 2*n*n+n)
            filter_size: expansion ratio
            hash_sets: list of sets of hashes of seen states
            torch_device: device for the torch tensors created
            remove_duplicates: whether to remove duplicates

        Output:
            new_states: bool torch.Tensor(bs, bw, filter_size, 2*n*n+n)
            is_unprepped: bool torch.Tensor(bs, bw, filter_size)
                Indicates whether the resulting state corresponds to the ground state
            is_duplicate: bool torch.Tensor(bs, bw, filter_size)
                Indicates whether the resulting state has already been seen before
            filtered_idx: int torch.Tensor(bs, bw, filter_size)
                Indices of `gates` used for obtaining new states
        """
        assert (
            len(states.shape) == 3 and states.dtype == np.bool_
        ), f"states must be a 3D np.ndarray of dtype np.bool_, got shape={states.shape}, dtype={states.dtype}"
        assert (
            len(gates.shape) == 3 and gates.dtype == np.int_
        ), f"gates must be a 3D np.ndarray of dtype np.int_, got shape={gates.shape}, dtype={gates.dtype}"

        n = self.layouts.shape[-1]
        k = gates.shape[-1]
        assert (
            k >= filter_size
        ), f"Need at least {filter_size} candidate gates to expand each input state. Instead received {k}"

        gates = gates.astype(np.int32)  # Needed for ctypes
        bs, bw, ts = states.shape
        new_state = torch.zeros((bs, bw, filter_size, ts), dtype=torch.bool, device=torch_device)
        is_unprepped = torch.zeros((bs, bw, filter_size), dtype=torch.bool, device=torch_device)
        is_duplicate = torch.zeros((bs, bw, filter_size), dtype=torch.bool, device=torch_device)
        filtered_idxs = torch.zeros((bs, bw, filter_size), dtype=torch.int)
        # Run simulation for each element in the beam
        for i in range(bs):
            seen_hashes = hash_sets[i] if hash_sets is not None else set()
            for j in range(bw):
                # Prepare input
                layout = self.layouts[i].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
                gate_set = np.ascontiguousarray(self.gate_set[i]).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int)
                )
                state = states[i, j].reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
                # Run simulation
                sim_result = self.lib.run_simulator(
                    n,
                    layout,
                    len(self.gate_set[i, :]),  # num_gate_types
                    gate_set,  # gate_set
                    state,  # tableau_arr
                    k,  # num_applied_gates
                    gates[i, j].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),  # gates
                )
                # Access the result array
                if not sim_result.stabilizers:
                    raise MemoryError("C function failed to allocate memory")
                buffer_states = torch.tensor(
                    np.ctypeslib.as_array(sim_result.stabilizers, shape=(k * ts,)).reshape(k, -1),
                    # device=torch_device,
                ).to(torch.bool)
                buffer_is_unprepped = torch.tensor(
                    np.ctypeslib.as_array(sim_result.is_unprepped, shape=(k,)), device=torch_device
                )
                self.lib.free_stabilizers_struct(sim_result)

                # Remove duplicates
                duplicate_idxs = []
                unique_idxs = []
                for idx, candidate in enumerate(buffer_states):
                    hv = hash(bytes(candidate.numpy()))
                    if hv in seen_hashes and remove_duplicates:
                        duplicate_idxs.append(idx)
                    else:
                        unique_idxs.append(idx)
                        seen_hashes.add(hv)
                    if len(unique_idxs) == filter_size:
                        break
                unique_idxs = torch.tensor(unique_idxs)
                duplicate_idxs = torch.tensor(duplicate_idxs)

                is_duplicate[i, j] = torch.zeros(filter_size, dtype=torch.bool, device=torch_device)
                if len(unique_idxs) >= filter_size:
                    filtered_idxs[i, j] = unique_idxs[:filter_size]
                else:
                    duplicate_length = np.max([filter_size - len(unique_idxs), 0])
                    filtered_idxs[i, j] = torch.concat(
                        (unique_idxs[:filter_size], duplicate_idxs[:duplicate_length])
                    )
                    is_duplicate[i, j, len(unique_idxs) : len(unique_idxs) + duplicate_length] = 0
                new_state[i, j] = buffer_states[filtered_idxs[i, j]].to(torch_device)
                is_unprepped[i, j] = buffer_is_unprepped[filtered_idxs[i, j]]

        return new_state, is_unprepped, is_duplicate, filtered_idxs.to(torch_device)


class Path:
    """
    Class for each path in beam search.
    Or class for an entire beam.

    `unprepped` if set, refers to whether any state in the beam is the ground state.
    `unprepped_list[i]` stores whether the i-th path is unprepped. Note that this variable
    is set only when `batch_size = 1`.

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
        unprepped: bool = None,
        unprepped_list: list[bool] = None,
        identifier: int = None,
        remove_duplicates: bool = False,
    ):
        self.n = n
        self.observations = observations
        self.depths = depths
        self.gates = gates
        self.width = width
        self.bs = bs
        self.unprepped = unprepped
        self.unprepped_list = unprepped_list
        self.identifier = identifier
        self.duplicate_tracker = None
        self.remove_duplicates = remove_duplicates
        self.seen_sets = [set() for _ in range(bs)]
        if width == 1:
            for i in range(bs):
                self.seen_sets[i].add(hash(bytes(self.observations[i].cpu().numpy())))

    def __str__(self, layout=None, gate_set=None):
        st = f"Success: {self.is_successfully_unprepared()}"
        st += "\n"
        # st += f"Observations : {[format_observation(x, self.n) for x in self.observations]}"
        # st += "\n"
        st += f"Depths : {', '.join([f'{(x).numpy():.2f}' for x in self.depths])}"
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
        For each path (# = self.width * self.bs), return True iff the observation corresponds to
        a ground state. The returned tensor is a bool tensor of size (width * bs).
        Note that this function is not required for running batch inference, and is provided as a
        helper function. In batch inference, the simulator module already computes this function
        using the ctypes library.
        """
        observations = self.observations.reshape(self.bs, self.width, -1)
        observations = observations.reshape(self.width * self.bs, self.n, -1)

        def is_ground_state(subtensor: torch.Tensor):
            return (
                torch.all(subtensor[:, : self.n] == 0)  # No X
                and torch.all(subtensor[:, -1] == 0)  # No '-' sign
                and (
                    torch.count_nonzero(subtensor == 1, 1) == torch.ones(self.n, dtype=torch.int)
                )  # Exactly n 1s
            )

        # There is no torch.apply or torch.map for tensors, so we use a list comprehension
        return torch.stack([is_ground_state(obs) for obs in observations]).reshape(
            self.bs, self.width
        )

    def filter_beam(self, top_indices: torch.Tensor):
        """
        Truncate beam to keep top `k` elements for each batch. The corresponding indices
        are tracked by the tensor `top_indices`
        """
        bs, k = top_indices.shape
        # Modify identifiers
        if self.identifier is not None:
            self.identifier = self.identifier.reshape(bs, self.width)
            self.identifier = self.identifier.gather(1, top_indices)
            self.identifier = self.identifier.reshape(k * bs)
        top_indices = top_indices.unsqueeze(-1)
        # Modify observations
        self.observations = self.observations.reshape(bs, self.width, -1)
        self.observations = self.observations.gather(
            1, top_indices.expand(-1, -1, self.observations.shape[-1])
        )
        self.observations = self.observations.reshape(k * bs, -1)
        # Modify depths
        if self.depths is not None:
            self.depths = self.depths.reshape(bs, self.width, -1)
            self.depths = self.depths.gather(1, top_indices.expand(-1, -1, self.depths.shape[-1]))
            self.depths = self.depths.reshape(k * bs, -1)
        # Modify gates
        if self.gates is not None:
            self.gates = self.gates.reshape(bs, self.width, -1)
            self.gates = self.gates.gather(1, top_indices.expand(-1, -1, self.gates.shape[-1]))
            self.identifier = self.identifier.reshape(bs, k)
            self.identifier = self.identifier.reshape(-1)
            self.gates = self.gates.reshape(k * bs, -1)

        self.width = k

    def append_depth_tensor(self, depths: torch.Tensor):
        """
        depths: (bs, width)
        self.depths: (width * bs, depth)
        """
        if self.depths is None:
            self.depths = depths.reshape(-1).unsqueeze(-1)
        else:
            self.depths = self.depths.reshape(self.bs, self.width, -1)
            self.depths = torch.concat((self.depths, depths.unsqueeze(-1)), dim=-1)
            self.depths = self.depths.reshape(self.width * self.bs, -1)

    def append_to(self, path_list: list["Path"]):
        if self.bs == 0:
            self.observations = None
            self.depths = None
            self.gates = None
            return
        self.observations = self.observations.reshape(self.bs, self.width, -1)
        self.depths = self.depths.reshape(self.bs, self.width, -1)
        self.gates = self.gates.reshape(self.bs, self.width, -1)
        self.identifier = self.identifier.reshape(self.bs, self.width)

        for idx in range(self.observations.shape[0]):
            path_list[self.identifier[idx, 0].cpu()] = Path(
                self.n,
                self.observations[idx].cpu(),
                self.depths[idx].cpu(),
                self.gates[idx].cpu(),
                width=self.width,
                bs=1,
                unprepped=False,
                unprepped_list=np.zeros(self.observations[idx].shape[0], dtype=np.bool_),
                identifier=self.identifier[idx, 0].cpu(),
            )

        self.observations = None
        self.depths = None
        self.gates = None
        self.identifier = None

    def update_states(
        self,
        gate_predictions: torch.Tensor,
        unprepped_states: list["Path"],
        simulator: BatchSimulator,
        ke: int,
    ) -> (int, int, np.ndarray[bool]):
        """
        Update beam to add new gate and depth tensors. The function will expand the beam and
        add new states corresponding to the gates and update the width to `width * ke`.
        The `gate_predictions` tensor tracks the probabilistic estimate of which gate should
        be applied next in the circuit. In a given batch element, for each path in the beam,
        there are `ke` new gates to be added. The tensor corresponding to new_gates will be
        of size (bs, bw, ke).
        The `depths` tensor corresponds to the current depth prediction, and should be added
        to each of the new copy generated.
        After the update, new sizes are:
        gates: (bs, bw, depth) -> (bs, bw, ke, depth+1) ==> (bs * bw * ke, depth+1)
        depths: (bs, bw, depth) -> (bs, bw, ke, depth) ==> (bs * bw * ke, depth)

        Any batch element which has been successfully unprepared in even one of the paths
        will be copied to the `unprepped_states` list at the correct index and removed from
        the beam.

        The depths tensor will be updated using the `append_depth_tensor` member function.

        Args:
            gate_predictions : torch.Tensor(bs, bw, num of gates = g)
            unprepped_states : list[Path]. List of currently unprepared states
            simulator : Batch Simulator with layout and gate set info already populated.
                simulator.layout[i] and simulator.gate_set[i] corresponding to the batch
                self.observations[i]
            ke: Expansion ratio

        Modifies:
            unprepped_states : Adds all the batches which have at least one path where state
                has been successfully unprepared
            Almost all member variables.

        Returns:
            new batch size, new width, unprepped batches (bool np.ndarray)
        """
        buffer_expansion = 4
        _, new_gates = torch.topk(
            gate_predictions, np.min([buffer_expansion * ke, gate_predictions.shape[-1]]), dim=-1
        )
        bs, bw, k_ = new_gates.shape
        assert self.bs == bs, f"Batch size mismatch: self.bs={self.bs}, bs={bs}"

        ###############
        # Update observations
        ###############
        self.observations = self.observations.reshape(bs, bw, -1)
        self.observations, is_unprepped, is_duplicate, filtered_idxs = simulator.run_simulation(
            self.observations.cpu().numpy(),  # states
            new_gates.cpu().numpy(),  # gates
            ke,  # filter_size
            hash_sets=self.seen_sets,
            torch_device=self.observations.device,
            remove_duplicates=self.remove_duplicates,
        )
        # self.observations -> (bs, bw, k, 2n^2+n)
        # self.duplicate_tracker  -> (bs, bw, k)
        # is_unprepped -> (bs, bw, k)

        ###############
        # Add new gates
        ###############
        new_gates = new_gates.gather(-1, filtered_idxs).unsqueeze(-1)
        if self.gates is None:
            self.gates = new_gates
        else:
            self.gates = self.gates.reshape(bs, bw, -1).unsqueeze(2).expand(-1, -1, ke, -1)
            self.gates = torch.concat((self.gates, new_gates), dim=-1)
        # self.gates -> (bs, bw, k, depth+1)

        ###############
        # Update depths
        ###############
        self.depths = self.depths.reshape(bs, bw, -1)
        self.depths = self.depths.unsqueeze(2).expand(-1, -1, ke, -1)
        # self.depths -> (bs, bw, k, depth+1)

        ###############
        # Update identifiier
        ###############
        if self.identifier is not None:
            self.identifier = self.identifier.reshape(bs, bw)
            self.identifier = self.identifier.unsqueeze(-1).expand(-1, -1, ke)
        # self.identifier -> (bs, bw, k)

        ###############
        # Remove unprepped states
        ###############
        is_batch_unprepped = torch.any(is_unprepped, dim=(-1, -2)).cpu().numpy()
        # Add unprepped states to `unprepped_states`
        for idx in np.nonzero(is_batch_unprepped)[0]:
            unprepped_states[self.identifier[idx, 0, 0].cpu()] = Path(
                self.n,
                self.observations[idx].reshape(bw * ke, -1).cpu(),  # torch.tensor
                self.depths[idx].reshape(bw * ke, -1).cpu(),  # torch.tensor
                self.gates[idx].reshape(bw * ke, -1).cpu(),  # torch.tensor
                width=bw * ke,
                bs=1,
                unprepped=True,
                unprepped_list=is_unprepped[idx].reshape(-1),
                identifier=self.identifier[idx, 0, 0].cpu(),
            )
        # Filter out unprepped states from current state variables
        self.bs = bs = self.bs - np.count_nonzero(is_batch_unprepped)
        if self.bs != 0:
            self.gates = self.gates[~is_batch_unprepped].reshape(ke * bw * bs, -1)
            self.depths = self.depths[~is_batch_unprepped].transpose(0, 2).reshape(ke * bw * bs, -1)
            self.observations = self.observations[~is_batch_unprepped].reshape(ke * bw * bs, -1)
            self.identifier = self.identifier[~is_batch_unprepped].reshape(-1)
            self.seen_sets = [s for i, s in enumerate(self.seen_sets) if not is_batch_unprepped[i]]
        else:
            self.gates = None
            self.depths = None
            self.observations = None
            self.identifier = None
            self.seen_sets = None

        self.width = ke * bw
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
            raise NotImplementedError("Layout and gate_set format is not yet supported.")
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
        evals = torch.unsqueeze(self.evals, dim=1).expand(-1, rep, -1).reshape(-1, self.n)
        evecs = (
            torch.unsqueeze(self.evecs, dim=1).expand(-1, rep, -1, -1).reshape(-1, self.n, self.n)
        )
        gates = torch.unsqueeze(self.gates, dim=1).expand(-1, rep, -1).reshape(-1, self.g)
        gate_qubits = (
            torch.unsqueeze(self.gate_qubits, dim=1).expand(-1, rep, -1).reshape(-1, 2 * self.g)
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
        if file is not None:
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

    def get_model_prediction(
        self,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        gates: torch.Tensor,
        gate_qubits: torch.Tensor,
        observations: torch.Tensor,
    ):
        """
        Get model prediction for given inputs. Chunk into batches of `chunk_size` if the batch
        size is too large.

        Args:
            evals: torch.Tensor, shape (width * bs, n)
            evecs: torch.Tensor, shape (width * bs, n, n)
            gates: torch.Tensor, shape (width * bs, g)
            gate_qubits: torch.Tensor, shape (width * bs, 2*g)
            observations: torch.Tensor, typically shape (width * bs, 2n+1)
        Returns:
            Output of the model's forward method.
        """
        # Chunk data for model into batches of size 1024. Process all of bs in chunks.
        bs = evals.shape[0]
        chunk_size = 1024
        outputs = []
        for start in range(0, bs, chunk_size):
            end = min(start + chunk_size, bs)
            evals_chunk = evals[start:end]
            evecs_chunk = evecs[start:end]
            gates_chunk = gates[start:end]
            gate_qubits_chunk = gate_qubits[start:end]
            observations_chunk = observations[start:end]
            outputs.append(
                self.model(
                    evals_chunk, evecs_chunk, gates_chunk, gate_qubits_chunk, observations_chunk
                )
            )
        if len(outputs) == 1:
            return outputs[0]
        return tuple(torch.cat([out[i] for out in outputs], dim=0) for i in range(len(outputs[0])))

    def infer_batch(
        self,
        layouts: np.ndarray,
        evals: np.ndarray,
        evecs: np.ndarray,
        gates: np.ndarray,
        gate_qubits: np.ndarray,
        targets: np.ndarray,
        beam_width: int = 1,
        remove_duplicates: bool = False,
    ) -> list[Path]:
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

        Returns:
            output_paths: list[Path]
                The length of the list is the same as the batch size. The width of an element
                might be more than the beam width, owing to the expansion stage of the beam
                search. But it will not be more than `expansion_ratio * beam_width`.
                The order of the elements in `output_paths` will match the batch order in the
                input.
        """
        # Initialize variables
        batch_size = evals.shape[0]
        data = DataHolder(
            evals=evals, evecs=evecs, gates=gates, gate_qubits=gate_qubits
        )
        observation = torch.tensor(targets).to(self.device)  # (1, bs, 2n+1)

        simulator = self._create_simulators(layouts, np.unique(gates, axis=-1))
        output_paths = [None] * batch_size
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
                        unprepped=None,
                        unprepped_list=[],
                        identifier=torch.arange(0, batch_size).to(self.device),
                        remove_duplicates=remove_duplicates,
                    )
                else:
                    gate_predictions = nn.Softmax(-1)(gate_prediction_logit)
                    expansion_ratio = 4 if depth != self.max_depth else 1
                    batch_size, width, unprepped_batches = curr_beam.update_states(
                        gate_predictions, output_paths, simulator, expansion_ratio
                    )
                    simulator.remove_batch(unprepped_batches)
                    data.remove_batch(unprepped_batches)
                    # Break the loop after simulating the last set of gates.
                    if depth == self.max_depth or batch_size == 0:
                        break

                ############
                # Calculate predicted cost
                ############
                evals, evecs, gates, gate_qubits = data.replicate(width)
                gate_prediction_logit, depth_prediction = self.get_model_prediction(
                    evals.to(self.device),  # (width * bs, n)
                    evecs.to(self.device),  # (width * bs, n, n)
                    gates.to(self.device),  # (width * bs, g)
                    gate_qubits.to(self.device),  # (width * bs, 2*g)
                    curr_beam.observations,  # (width * bs, 2n^2+n)
                )

                ############
                # Truncate
                ############
                depth_prediction = depth_prediction.reshape(batch_size, width)
                k = np.minimum(beam_width, width)
                # Filter predictions corresponding to no duplicates and lowest depth predictions
                depth_prediction, top_indices = torch.topk(-depth_prediction, k, dim=1)
                depth_prediction = -depth_prediction  # (bs, k)
                # Filter other variables
                gate_prediction_logit = gate_prediction_logit.reshape(batch_size, width, -1)
                gate_prediction_logit = gate_prediction_logit.gather(
                    1, top_indices.unsqueeze(-1).expand(-1, -1, gate_prediction_logit.shape[-1])
                )
                curr_beam.filter_beam(top_indices)
                if depth < self.max_depth + 1:
                    curr_beam.append_depth_tensor(depth_prediction)

                depth += 1

        curr_beam.append_to(output_paths)
        return output_paths
