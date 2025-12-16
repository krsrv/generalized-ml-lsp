"""
Custom dataloader for LSP data.
"""

import warnings
import numpy as np
import numba

def transform_graph(adjacency_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform an adjacency matrix to a laplacian matrix and return the
    eigenvalues, eigenvectors.
    Adjacency matrix should be of size (bs, n, n) where each (n, n) submatrix
    represents a different layout.

    float64 is used for intermediate calculations, output is float32.
    """
    adj = adjacency_matrix.astype(np.float64)
    n = adj.shape[-1]
    if adj.shape[-2] != n:
        raise ValueError(
            f"Adjacency matrix should be square, not shape={adj.shape}.")
    degrees = np.sum(adj, axis=-1)
    laplacian = -adj
    ii = np.arange(n)
    laplacian[..., ii, ii] += degrees
    eigval, eigvec = np.linalg.eigh(laplacian)
    return eigval.astype(np.float32), eigvec.astype(np.float32)

@numba.njit
def _dilate32(x: np.uint64) -> np.uint64:
    """
    Assuming high 32 bits of x are 0,
    alternates the low 32 bits with 0s to fill np.uint64 with odd bits =0.
    """
    # In C++ we can use _pdep_u64, but I'm not sure if numba supports it.
    # Here is a bit-twiddling implementation:
    x = (x | (x << 16)) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x <<  8)) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x <<  4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x <<  2)) & np.uint64(0x3333333333333333)
    x = (x | (x <<  1)) & np.uint64(0x5555555555555555)
    return x

@numba.njit
def _zip_lohi_shift(x: np.uint64) -> np.uint64:
    return ((_dilate32(x >> 32) << 1)
            | _dilate32(x & np.uint64(0x00000000FFFFFFFF)))

@numba.njit
def _convert_obs_to_sxzxz(obs: np.ndarray):
    for i in range(obs.size):
        obs.flat[i] = _zip_lohi_shift(obs.flat[i])

def convert_obs_to_sxzxz(obs: np.ndarray):
    """
    Changes the bit layout from 0bSXX..XX0ZZ..ZZ to 0bS0XZXZ..XZXZ.
    """
    assert obs.dtype == np.uint64
    _convert_obs_to_sxzxz(obs)

def scale_depth(val, n):
    """Scaling to ensure that scaled depth has mean ~0 and std ~1:

    For the latest dataset this results in mean=0.260, std=2.189.
    """
    assert np.all(n >= 2)
    dmax = n * (n + 3) / 2 / np.log2(n)
    return ((val - dmax / 2) / np.sqrt(dmax)).astype(np.float32)

def unscale_depth(val, n):
    dmax = n * (n + 3) / 2 / np.log2(n)
    return np.sqrt(dmax) * val + dmax / 2

class LSPDataLoader:
    def __init__(
            self, train_filename, batch_size: int, shuffle: bool=True,
            seed: int=1, validate: bool=False):
        self.shuffle = shuffle
        if shuffle:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.eigval = {} # key = n
        self.eigvec = {} # key = n
        self.gate = {} # key = g
        self.gate_qubit = {} # key = g
        self.observation_sxzxz = {} # key = ng
        self.unprep_gate = {} # key = ng
        self.scaled_depth = {} # key = ng
        self.global_n_idx = {} # key = ng
        self.global_g_idx = {} # key = ng
        with np.load(train_filename) as f:
            # Eagerly loading the whole data seems to be faster
            # than iterating over NpzFile object.
            data = dict(f)
        for key, v in data.items():
            key_parts = key.split('/')
            assert len(key_parts[0]) > 0, f"Invalid key: {key!r}"
            if key_parts[0][0].isdigit():
                n, g, k = key_parts
                n = int(n)
                g = int(g)
                ng = (n, g)
                match k:
                    case "unprep_gate":
                        self.unprep_gate[ng] = v
                    case "depth":
                        self.scaled_depth[ng] = scale_depth(v, n)
                    case "observation":
                        # Modify in-place, this is fine since `data` is not re-used later.
                        convert_obs_to_sxzxz(v)
                        self.observation_sxzxz[ng] = v
                    case "global_n_idx":
                        self.global_n_idx[ng] = v
                    case "global_g_idx":
                        self.global_g_idx[ng] = v
            elif key_parts[0] == "global_n":
                _, n, k = key_parts
                n = int(n)
                match k:
                    case "layout":
                        eigval, eigvec = transform_graph(v)
                        self.eigval[n] = eigval
                        self.eigvec[n] = eigvec
            elif key_parts[0] == "global_g":
                _, g, k = key_parts
                g = int(g)
                match k:
                    case "gates":
                        self.gate[g] = v
                    case "gate_qubits":
                        self.gate_qubit[g] = v
        self.ng_list = list(self.scaled_depth.keys())
        self.ng_counts = np.array([v.shape[0] for v in self.scaled_depth.values()])
        self.num_samples = sum(self.ng_counts)
        self.idxs = {
            ng: np.arange(v.shape[0]) for ng, v in self.scaled_depth.items()}
        if validate:
            self.validate_data()
        self.set_batch_size(batch_size)

    def set_batch_size(self, batch_size):
        """
        Sets batch_size. Invalidates iterators.
        """
        self.batch_size = batch_size
        ng_num_batches = (self.ng_counts + batch_size - 1) // batch_size
        self.batch_ngs = [
            ng
            for ng, c in zip(self.ng_list, ng_num_batches)
            for _ in range(c)]

    def validate_data(self) -> None:
        """
        Validate internal consistency of loaded NPZ data.

        Hard failures raise ValueError/TypeError.
        """
        def _fail(msg: str) -> None:
            raise ValueError(f"LSPDataLoader.validate_data: {msg}")

        def _require(cond: bool, msg: str) -> None:
            if not cond:
                _fail(msg)

        def _is_int_dtype(a: np.ndarray) -> bool:
            return np.issubdtype(a.dtype, np.integer)

        # --- 1) Per-(n,g) key consistency / completeness ---
        per_ng_dicts = {
            "scaled_depth": self.scaled_depth,
            "unprep_gate": self.unprep_gate,
            "observation_sxzxz": self.observation_sxzxz,
            "global_n_idx": self.global_n_idx,
            "global_g_idx": self.global_g_idx,
        }
        ng_keys = set(self.scaled_depth.keys())
        _require(len(ng_keys) > 0,
                 "no (n,g) groups found (scaled_depth is empty).")

        for name, d in per_ng_dicts.items():
            keys = set(d.keys())
            if keys != ng_keys:
                msg = [f"per-ng keys mismatch for '{name}': "]
                missing = sorted(ng_keys - keys)
                extra = sorted(keys - ng_keys)
                for name, lst in [("missing", missing), ("extra", extra)]:
                    if not lst:
                        continue
                    msg.append(f"{name}=[")
                    msg.append(', '.join(f"({n},{g})" for n, g in lst[:10]))
                    if len(lst) > 10:
                        msg.append(", ...")
                    msg.append("]")
                    msg.append(" ")
                if msg[-1] == " ":
                    msg.pop()
                _fail(''.join(msg))

        # --- 2) Global-n eigen data sanity ---
        for n, eigval in self.eigval.items():
            _require(isinstance(n, int) and n > 0,
                     f"invalid n key in eigval: {n!r}")
            _require(isinstance(eigval, np.ndarray),
                     f"eigval[{n}] is not a numpy array.")
            _require(eigval.ndim == 2,
                     f"eigval[{n}] must be 2D (num_layouts, n); "
                     f"got shape {eigval.shape}.")
            _require(eigval.shape[1] == n,
                     f"eigval[{n}] second dim must equal n={n}; "
                     f"got {eigval.shape}.")
            _require(np.issubdtype(eigval.dtype, np.floating),
                     f"eigval[{n}] must be floating; got {eigval.dtype}.")

            eigvec = self.eigvec.get(n, None)
            _require(eigvec is not None, f"eigvec missing for n={n}.")
            _require(isinstance(eigvec, np.ndarray),
                     f"eigvec[{n}] is not a numpy array.")
            _require(eigvec.ndim == 3,
                     f"eigvec[{n}] must be 3D (num_layouts, n, n); "
                     f"got shape {eigvec.shape}.")
            _require(eigvec.shape[0] == eigval.shape[0],
                     f"eigvec[{n}] first dim must match eigval[{n}] num_layouts; "
                     f"eigvec={eigvec.shape[0]} eigval={eigval.shape[0]}.")
            _require(eigvec.shape[1] == n and eigvec.shape[2] == n,
                     f"eigvec[{n}] must have shape "
                     f"(num_layouts, {n}, {n}); got {eigvec.shape}.")
            _require(np.issubdtype(eigvec.dtype, np.floating),
                     f"eigvec[{n}] must be floating; got {eigvec.dtype}.")

            if np.any(~np.isfinite(eigval)):
                _fail(f"eigval[{n}] contains NaN/Inf.")
            if np.any(~np.isfinite(eigvec)):
                _fail(f"eigvec[{n}] contains NaN/Inf.")

        # --- 3) Global-g gate data sanity ---
        for g, gates in self.gate.items():
            _require(isinstance(g, int) and g > 0,
                     f"invalid g key in gate: {g!r}")
            _require(isinstance(gates, np.ndarray),
                     f"gate[{g}] is not a numpy array.")
            _require(gates.ndim == 2,
                     f"gate[{g}] must be 2D (num_variants, g); "
                     f"got shape {gates.shape}.")
            _require(gates.shape[1] == g,
                     f"gate[{g}] second dim must equal g={g}; got {gates.shape}.")

            gate_qubits = self.gate_qubit.get(g, None)
            _require(gate_qubits is not None,
                     f"gate_qubit missing for g={g}.")
            _require(isinstance(gate_qubits, np.ndarray),
                     f"gate_qubit[{g}] is not a numpy array.")
            # Iterator yields (bs, g, 2), so enforce that representation.
            _require(gate_qubits.ndim == 3,
                     f"gate_qubit[{g}] must be 3D (num_variants, g, 2); "
                     f"got shape {gate_qubits.shape}.")
            _require(gate_qubits.shape[1] == g and gate_qubits.shape[2] == 2,
                     f"gate_qubit[{g}] must have shape (num_variants, {g}, 2); "
                     f"got {gate_qubits.shape}.")
            _require(gate_qubits.shape[0] == gates.shape[0],
                     f"gate_qubit[{g}] num_variants must match gate[{g}]; "
                     f"gate_qubit={gate_qubits.shape[0]} gate={gates.shape[0]}.")

        # --- 4) Per-(n,g) arrays: length/dtype/bounds checks ---
        for ng in ng_keys:
            _require(isinstance(ng, tuple) and len(ng) == 2,
                     f"invalid ng key: {ng!r}")
            n, g = ng
            _require(isinstance(n, int) and n > 0, f"invalid n in ng={ng!r}")
            _require(isinstance(g, int) and g > 0, f"invalid g in ng={ng!r}")

            _require(n in self.eigval and n in self.eigvec,
                     f"missing eigen data for n={n} used by ng={ng}.")
            _require(g in self.gate and g in self.gate_qubit,
                     f"missing gate data for g={g} used by ng={ng}.")

            sd = self.scaled_depth[ng]
            up = self.unprep_gate[ng]
            obs = self.observation_sxzxz[ng]
            n_idx = self.global_n_idx[ng]
            g_idx = self.global_g_idx[ng]

            # First-dimension consistency (number of samples for this ng)
            _require(isinstance(sd, np.ndarray),
                     f"scaled_depth[{ng}] is not a numpy array.")
            m = sd.shape[0]
            _require(m > 0, f"scaled_depth[{ng}] has zero samples.")
            for name, arr in (
                    ("unprep_gate", up),
                    ("observation_sxzxz", obs),
                    ("global_n_idx", n_idx),
                    ("global_g_idx", g_idx)):
                _require(isinstance(arr, np.ndarray),
                         f"{name}[{ng}] is not a numpy array.")
                _require(arr.shape[0] == m,
                         f"{name}[{ng}] first dim mismatch: "
                         f"expected {m}, got {arr.shape[0]}.")

            # Dtype checks
            _require(np.issubdtype(sd.dtype, np.floating),
                     f"scaled_depth[{ng}] must be floating; got {sd.dtype}.")
            if np.any(~np.isfinite(sd)):
                _fail(f"scaled_depth[{ng}] contains NaN/Inf.")

            _require(obs.dtype == np.uint64,
                     f"observation_sxzxz[{ng}] must be uint64; "
                     f"got {obs.dtype}.")
            if not obs.flags["C_CONTIGUOUS"]:
                warnings.warn(
                    f"observation_sxzxz[{ng}] is not C-contiguous; "
                    "conversion/indexing may be slower.")

            _require(_is_int_dtype(n_idx),
                     f"global_n_idx[{ng}] must be integer dtype; got {n_idx.dtype}.")
            _require(_is_int_dtype(g_idx),
                     f"global_g_idx[{ng}] must be integer dtype; got {g_idx.dtype}.")
            _require(n_idx.ndim == 1,
                     f"global_n_idx[{ng}] should be 1D; got shape {n_idx.shape}.")
            _require(g_idx.ndim == 1,
                     f"global_g_idx[{ng}] should be 1D; got shape {g_idx.shape}.")

            # Bounds checks
            n_layouts = self.eigval[n].shape[0]
            g_variants = self.gate[g].shape[0]

            n_min = int(n_idx.min()) if n_idx.size else 0
            n_max = int(n_idx.max()) if n_idx.size else -1
            g_min = int(g_idx.min()) if g_idx.size else 0
            g_max = int(g_idx.max()) if g_idx.size else -1

            _require(n_min >= 0,
                     f"global_n_idx[{ng}] contains negative indices (min={n_min}).")
            _require(g_min >= 0,
                     f"global_g_idx[{ng}] contains negative indices (min={g_min}).")
            _require(n_max < n_layouts,
                     f"global_n_idx[{ng}] out of bounds: "
                     f"max={n_max}, but n_layouts={n_layouts}.")
            _require(g_max < g_variants,
                     f"global_g_idx[{ng}] out of bounds: "
                     f"max={g_max}, but g_variants={g_variants}.")

            # Dtype capacity checks (catches “this dtype can’t represent the needed range”)
            # This does not detect past wraparound, but prevents future silent overflow.
            if (np.issubdtype(n_idx.dtype, np.unsignedinteger)
                    or np.issubdtype(n_idx.dtype, np.signedinteger)):
                n_info = np.iinfo(n_idx.dtype)
                _require(n_layouts - 1 <= n_info.max,
                         f"global_n_idx[{ng}] dtype {n_idx.dtype} cannot "
                         f"represent up to n_layouts-1={n_layouts-1} "
                         f"(max={n_info.max}).")
                if np.issubdtype(n_idx.dtype, np.signedinteger):
                    _require(0 >= n_info.min,
                             f"global_n_idx[{ng}] dtype {n_idx.dtype} has "
                             f"unexpected min={n_info.min}.")

            if (np.issubdtype(g_idx.dtype, np.unsignedinteger)
                    or np.issubdtype(g_idx.dtype, np.signedinteger)):
                g_info = np.iinfo(g_idx.dtype)
                _require(g_variants - 1 <= g_info.max,
                         f"global_g_idx[{ng}] dtype {g_idx.dtype} cannot "
                         f"represent up to g_variants-1={g_variants-1} "
                         f"(max={g_info.max}).")
                if np.issubdtype(g_idx.dtype, np.signedinteger):
                    _require(0 >= g_info.min,
                             f"global_g_idx[{ng}] dtype {g_idx.dtype} has "
                             f"unexpected min={g_info.min}.")

        # --- 5) Determinism hint ---
        if not self.shuffle:
            if self.ng_list != sorted(self.ng_list):
                warnings.warn(
                    "shuffle=False but ng_list is not sorted; "
                    "iteration order may depend on NPZ key order.")

    def __len__(self):
        return len(self.batch_ngs)

    def __iter__(self):
        """
        Yields a batch of up to batch_size data points.

        The number of data points yielded is in [1, batch_size].

        Not thread-safe, having multiple iterators at the same time is
        not supported.
        """
        self.ng_to_pos = {ng: 0 for ng in self.ng_list}
        if self.shuffle:
            ii = self.rng.permutation(len(self.batch_ngs))
            self.batch_ngs = [self.batch_ngs[i] for i in ii]
            for v in self.idxs.values():
                self.rng.shuffle(v)
        for ng in self.batch_ngs:
            n, g = ng
            i0 = self.ng_to_pos[ng]
            i1 = i0 + self.batch_size
            self.ng_to_pos[ng] = i1
            ii = self.idxs[ng][i0:i1]
            n_ii = self.global_n_idx[ng][ii]
            g_ii = self.global_g_idx[ng][ii]
            yield {
                "eigval": self.eigval[n][n_ii], # (bs, n)
                "eigvec": self.eigvec[n][n_ii], # (bs, n, n)
                "gates": self.gate[g][g_ii], # (bs, g)
                "gate_qubits": self.gate_qubit[g][g_ii], # (bs, g, 2)
                "observation_sxzxz": self.observation_sxzxz[ng][ii],
                "unprep_gate": self.unprep_gate[ng][ii],
                "scaled_depth": self.scaled_depth[ng][ii]
            }
