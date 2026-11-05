"""
Regression test for `BatchSimulator.run_simulation`'s duplicate-handling logic,
and for `Path.__init__`'s matching `seen_sets` bookkeeping.

Uses a fake `lib.run_simulator_batch` (no compiled `libsim.so` needed) that
returns hand-built candidate states, so the test exercises exactly the
Python-side filtering/dedup logic without depending on the real C simulator.

Run: conda run -n torch-gpu python -m inference.test_infer_simulator_smoke
"""

import ctypes
import sys

import numpy as np
import torch

from inference.infer import BatchSimulator, Path

TS = 4  # state size per candidate


def make_fake_lib(candidates: np.ndarray, unprepped: np.ndarray):
    """candidates: (k, ts) int32 array. unprepped: (k,) bool array.
    Only used with bs=1, bw=1 fixtures (see `make_simulator`), so a single
    batched call's whole-grid output is just this one (i=0, j=0) result.
    """
    k = candidates.shape[0]
    # Keep numpy buffers alive for the lifetime of the fake lib object so the
    # ctypes pointers stay valid.
    candidates = np.ascontiguousarray(candidates, dtype=np.int32)
    unprepped = np.ascontiguousarray(unprepped, dtype=np.bool_)

    class FakeLib:
        def run_simulator_batch(self, n, bs, bw, layouts, num_gate_types, gate_sets,
                                 tableau_arr, num_applied_gates, gates):
            assert bs == 1 and bw == 1, "make_fake_lib only supports bs=1, bw=1 fixtures"
            result = BatchSimulator.Stabilizers()
            result.stabilizers = candidates.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            result.is_unprepped = unprepped.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
            result.n = n
            result.k = k
            return result

        def free_stabilizers_struct(self, result):
            pass

    return FakeLib(), candidates, unprepped


def make_simulator(candidates, unprepped):
    sim = BatchSimulator.__new__(BatchSimulator)
    sim.layouts = np.zeros((1, TS // 2, TS // 2), dtype=np.bool_)
    sim.gate_set = np.zeros((1, 3), dtype=np.int32)
    sim.lib, sim._candidates, sim._unprepped = make_fake_lib(candidates, unprepped)
    return sim


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    return cond


def main():
    ok = True

    # 4 candidates; candidate 2 is a byte-identical duplicate of candidate 0.
    candidates = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],  # duplicate of row 0
            [1, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    unprepped = np.array([False, False, False, False], dtype=np.bool_)
    filter_size = 3

    states = np.zeros((1, 1, TS), dtype=np.bool_)
    gates = np.zeros((1, 1, 4), dtype=np.int_)

    # --- remove_duplicates=True: duplicate (idx 2) should be skipped ---
    sim = make_simulator(candidates, unprepped)
    hash_sets = [set()]
    new_state, is_unprepped, is_duplicate, filtered_idxs = sim.run_simulation(
        states, gates, filter_size, hash_sets=hash_sets, torch_device="cpu", remove_duplicates=True
    )
    expected_idxs = torch.tensor([[[0, 1, 3]]])
    ok &= check(
        "remove_duplicates=True skips the byte-identical duplicate",
        torch.equal(filtered_idxs, expected_idxs),
    )
    ok &= check(
        "remove_duplicates=True: new_state matches expected candidates",
        torch.equal(
            new_state[0, 0].to(torch.int32),
            torch.tensor(candidates[[0, 1, 3]], dtype=torch.int32),
        ),
    )

    # --- remove_duplicates=False: all candidates (including the duplicate) kept, in order ---
    sim = make_simulator(candidates, unprepped)
    hash_sets = [set()]
    new_state, is_unprepped, is_duplicate, filtered_idxs = sim.run_simulation(
        states, gates, filter_size, hash_sets=hash_sets, torch_device="cpu", remove_duplicates=False
    )
    expected_idxs_no_dedup = torch.tensor([[[0, 1, 2]]])
    ok &= check(
        "remove_duplicates=False keeps first `filter_size` candidates verbatim",
        torch.equal(filtered_idxs, expected_idxs_no_dedup),
    )
    ok &= check(
        "remove_duplicates=False: new_state matches first filter_size candidates (incl. duplicate)",
        torch.equal(
            new_state[0, 0].to(torch.int32),
            torch.tensor(candidates[[0, 1, 2]], dtype=torch.int32),
        ),
    )
    ok &= check(
        "remove_duplicates=False does not populate hash_sets (dedup bookkeeping skipped)",
        len(hash_sets[0]) == 0,
    )

    # --- remove_duplicates=False across repeated calls with the same hash_sets stays consistent ---
    sim = make_simulator(candidates, unprepped)
    hash_sets = [set()]
    for _ in range(3):
        _, _, _, filtered_idxs = sim.run_simulation(
            states, gates, filter_size, hash_sets=hash_sets, torch_device="cpu", remove_duplicates=False
        )
        ok &= check(
            "remove_duplicates=False is stable across repeated calls",
            torch.equal(filtered_idxs, expected_idxs_no_dedup),
        )

    # --- is_unprepped is threaded through filtering correctly ---
    unprepped2 = np.array([False, True, False, False], dtype=np.bool_)
    sim = make_simulator(candidates, unprepped2)
    _, is_unprepped, _, filtered_idxs = sim.run_simulation(
        states, gates, filter_size, hash_sets=[set()], torch_device="cpu", remove_duplicates=False
    )
    ok &= check(
        "is_unprepped is gathered according to filtered_idxs",
        torch.equal(is_unprepped[0, 0], torch.tensor([False, True, False])),
    )

    # --- layouts/gate_sets/tableau_arr/gates are laid out correctly for every
    # (i, j) slot within the single batched `run_simulator_batch` call (this
    # used to check per-(i,j) hoisting across bs*bw separate calls; now there
    # is exactly one call carrying the whole grid, so this checks that the
    # grid itself is assembled/decoded correctly instead). ---
    bs2, bw2, n2, num_gates2 = 2, 3, TS // 2, 4
    recorded = {}

    class RecordingLib:
        def run_simulator_batch(self, n_, bs_, bw_, layouts_ptr, num_gate_types, gate_sets_ptr,
                                 tableau_ptr, num_applied_gates, gates_ptr):
            recorded["n"] = n_
            recorded["bs"] = bs_
            recorded["bw"] = bw_
            recorded["num_gate_types"] = num_gate_types
            recorded["num_applied_gates"] = num_applied_gates
            recorded["layouts"] = np.ctypeslib.as_array(
                layouts_ptr, shape=(bs_ * n_ * n_,)
            ).copy().reshape(bs_, n_, n_)
            recorded["gate_sets"] = np.ctypeslib.as_array(
                gate_sets_ptr, shape=(bs_ * num_gate_types,)
            ).copy().reshape(bs_, num_gate_types)
            recorded["tableau"] = np.ctypeslib.as_array(
                tableau_ptr, shape=(bs_ * bw_ * TS,)
            ).copy().reshape(bs_, bw_, TS)
            recorded["gates"] = np.ctypeslib.as_array(
                gates_ptr, shape=(bs_ * bw_ * num_applied_gates,)
            ).copy().reshape(bs_, bw_, num_applied_gates)

            total = bs_ * bw_ * num_applied_gates
            candidates_local = np.zeros((total, TS), dtype=np.int32)
            unprepped_local = np.zeros((total,), dtype=np.bool_)
            result = BatchSimulator.Stabilizers()
            result.stabilizers = candidates_local.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            result.is_unprepped = unprepped_local.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
            result.n = n_
            result.k = total
            self._keepalive = (candidates_local, unprepped_local)
            return result

        def free_stabilizers_struct(self, result):
            pass

    sim2 = BatchSimulator.__new__(BatchSimulator)
    sim2.layouts = np.stack(
        [np.zeros((n2, n2), dtype=np.bool_), np.ones((n2, n2), dtype=np.bool_)]
    )
    sim2.gate_set = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int32)
    sim2.lib = RecordingLib()
    # Distinct per-(i,j) content (not all-zero) so mis-indexing the grid would
    # actually be caught instead of trivially matching.
    states2 = np.zeros((bs2, bw2, TS), dtype=np.bool_)
    for i in range(bs2):
        for j in range(bw2):
            states2[i, j, 0] = (i + j) % 2 == 0
    gates2 = np.zeros((bs2, bw2, num_gates2), dtype=np.int_)
    for i in range(bs2):
        for j in range(bw2):
            gates2[i, j] = np.arange(num_gates2) + 10 * i + j
    sim2.run_simulation(
        states2, gates2, filter_size, hash_sets=[set(), set()],
        torch_device="cpu", remove_duplicates=False,
    )
    ok &= check(
        "run_simulator_batch is called exactly once (not once per (i, j))",
        recorded.get("bs") == bs2 and recorded.get("bw") == bw2,
    )
    ok &= check(
        "batched layouts/gate_sets match self.layouts/self.gate_set per batch row",
        all(
            np.array_equal(recorded["layouts"][i], sim2.layouts[i])
            and np.array_equal(recorded["gate_sets"][i], sim2.gate_set[i])
            for i in range(bs2)
        )
        and recorded["num_gate_types"] == sim2.gate_set.shape[-1],
    )
    ok &= check(
        "batched tableau_arr/gates match states2/gates2 for every (i, j), including j > 0",
        np.array_equal(recorded["tableau"], states2)
        and np.array_equal(recorded["gates"], gates2.astype(np.int32)),
    )

    # --- Path.update_states keeps `depths` aligned with `identifier`/`gates`/
    # `observations` per flattened position (regression test for a bug where
    # `self.depths` alone was `.transpose(0, 2)`'d during the batch-mask step
    # while the other three parallel tensors weren't, silently scrambling
    # which candidate's depth-history ended up at which flattened index
    # whenever bs > 1). ---
    def make_update_states_fixture(bs3, bw3, ke3, mark_unprepped_batch_idx=None):
        ts3 = TS
        depths_init = torch.zeros((bs3, bw3, 1), dtype=torch.float32)
        for i in range(bs3):
            for j in range(bw3):
                depths_init[i, j, 0] = i * 10 + j

        path = Path.__new__(Path)
        path.n = ts3 // 2
        path.observations = torch.zeros((bs3 * bw3, ts3), dtype=torch.bool)
        path.depths = depths_init.reshape(bs3 * bw3, -1)
        path.gates = None
        path.width = bw3
        path.bs = bs3
        path.unprepped = None
        path.unprepped_list = None
        path.identifier = torch.arange(bs3).unsqueeze(1).expand(bs3, bw3).clone()
        path.remove_duplicates = False
        path.seen_sets = [set() for _ in range(bs3)]

        class FakeSim:
            def run_simulation(self, states, gates, filter_size, hash_sets=None,
                                torch_device="cpu", remove_duplicates=False):
                bs_, bw_ = states.shape[0], states.shape[1]
                new_state = torch.zeros((bs_, bw_, ke3, ts3), dtype=torch.bool)
                is_unprepped = torch.zeros((bs_, bw_, ke3), dtype=torch.bool)
                if mark_unprepped_batch_idx is not None:
                    is_unprepped[mark_unprepped_batch_idx, 0, 0] = True
                is_duplicate = torch.zeros((bs_, bw_, ke3), dtype=torch.bool)
                filtered_idxs = torch.arange(ke3).reshape(1, 1, -1).expand(bs_, bw_, -1).clone()
                return new_state, is_unprepped, is_duplicate, filtered_idxs

        gate_predictions = torch.rand((bs3, bw3, 16))
        unprepped_states = [None] * bs3
        path.update_states(gate_predictions, unprepped_states, FakeSim(), ke3)
        return path, unprepped_states

    def check_depths_aligned(path):
        identifier_flat = path.identifier.reshape(-1)
        depths_flat = path.depths.reshape(-1)
        return all(
            round(depths_flat[idx].item()) // 10 == int(identifier_flat[idx].item())
            for idx in range(identifier_flat.shape[0])
        )

    path_no_removal, _ = make_update_states_fixture(bs3=2, bw3=1, ke3=2)
    ok &= check(
        "update_states keeps depths aligned with identifier (no batch removal)",
        check_depths_aligned(path_no_removal),
    )

    path_with_removal, unprepped3 = make_update_states_fixture(
        bs3=4, bw3=2, ke3=3, mark_unprepped_batch_idx=2
    )
    ok &= check(
        "update_states keeps depths aligned with identifier (with batch removal)",
        check_depths_aligned(path_with_removal),
    )
    ok &= check(
        "update_states still extracts the unprepped batch row correctly",
        unprepped3[2] is not None and all(unprepped3[i] is None for i in [0, 1, 3]),
    )

    # --- Path.__init__ only populates seen_sets when remove_duplicates=True ---
    observations = torch.zeros((2, TS), dtype=torch.bool)
    observations[1, 0] = True
    path_no_dedup = Path(
        n=TS // 2, observations=observations, depths=None, gates=None,
        width=1, bs=2, remove_duplicates=False,
    )
    ok &= check(
        "Path.__init__ skips seen_sets population when remove_duplicates=False",
        all(len(s) == 0 for s in path_no_dedup.seen_sets),
    )
    path_dedup = Path(
        n=TS // 2, observations=observations, depths=None, gates=None,
        width=1, bs=2, remove_duplicates=True,
    )
    ok &= check(
        "Path.__init__ populates seen_sets when remove_duplicates=True",
        all(len(s) == 1 for s in path_dedup.seen_sets),
    )

    # --- run_simulation accumulates on CPU and transfers to `torch_device` once;
    # verify the returned tensors actually land on a non-CPU device and match the
    # CPU-computed values exactly (the existing checks above all use
    # torch_device="cpu", so they wouldn't catch a device-placement bug here). ---
    if torch.cuda.is_available():
        sim_cpu = make_simulator(candidates, unprepped)
        cpu_out = sim_cpu.run_simulation(
            states, gates, filter_size, hash_sets=[set()],
            torch_device="cpu", remove_duplicates=True,
        )
        sim_cuda = make_simulator(candidates, unprepped)
        cuda_out = sim_cuda.run_simulation(
            states, gates, filter_size, hash_sets=[set()],
            torch_device="cuda", remove_duplicates=True,
        )
        ok &= check(
            "run_simulation(torch_device='cuda') returns tensors on cuda",
            all(t.device.type == "cuda" for t in cuda_out),
        )
        ok &= check(
            "run_simulation(torch_device='cuda') matches CPU computation",
            all(torch.equal(a.cpu(), b) for a, b in zip(cuda_out, cpu_out)),
        )
    else:
        print("[SKIP] run_simulation(torch_device='cuda') checks (no CUDA device available)")

    if not ok:
        print("\nSMOKE TEST FAILED")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
