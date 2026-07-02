"""
Regression test for `BatchSimulator.run_simulation`'s duplicate-handling logic,
and for `Path.__init__`'s matching `seen_sets` bookkeeping.

Uses a fake `lib.run_simulator` (no compiled `libsim.so` needed) that returns
hand-built candidate states, so the test exercises exactly the Python-side
filtering/dedup logic without depending on the real C simulator.

Run: conda run -n torch-gpu python inference/test_infer_simulator_smoke.py
"""

import ctypes
import sys

import numpy as np
import torch

from inference.infer import BatchSimulator, Path

TS = 4  # state size per candidate


def make_fake_lib(candidates: np.ndarray, unprepped: np.ndarray):
    """candidates: (k, ts) int32 array. unprepped: (k,) bool array."""
    k = candidates.shape[0]
    # Keep numpy buffers alive for the lifetime of the fake lib object so the
    # ctypes pointers stay valid.
    candidates = np.ascontiguousarray(candidates, dtype=np.int32)
    unprepped = np.ascontiguousarray(unprepped, dtype=np.bool_)

    class FakeLib:
        def run_simulator(self, n, layout, num_gate_types, gate_set, state, num_applied_gates, gates):
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

    # --- layout/gate_set/num_gate_types (hoisted out of the inner `j` loop) are still
    # correct for every (i, j), including j > 0 ---
    bs2, bw2, n2 = 2, 3, TS // 2
    recorded_calls = []

    class RecordingLib:
        def run_simulator(self, n_, layout, num_gate_types, gate_set, state, num_applied_gates, gates_ptr):
            layout_vals = np.ctypeslib.as_array(layout, shape=(n_ * n_,)).copy()
            gate_set_vals = np.ctypeslib.as_array(gate_set, shape=(num_gate_types,)).copy()
            recorded_calls.append((layout_vals, gate_set_vals, num_gate_types))
            k = num_applied_gates
            candidates_local = np.zeros((k, TS), dtype=np.int32)
            unprepped_local = np.zeros((k,), dtype=np.bool_)
            result = BatchSimulator.Stabilizers()
            result.stabilizers = candidates_local.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            result.is_unprepped = unprepped_local.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
            result.n = n_
            result.k = k
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
    states2 = np.zeros((bs2, bw2, TS), dtype=np.bool_)
    gates2 = np.zeros((bs2, bw2, 4), dtype=np.int_)
    sim2.run_simulation(
        states2, gates2, filter_size, hash_sets=[set(), set()],
        torch_device="cpu", remove_duplicates=False,
    )
    ok &= check(
        "hoisted layout/gate_set are correct for every (i, j) call, including j > 0",
        len(recorded_calls) == bs2 * bw2
        and all(
            np.array_equal(layout_vals, sim2.layouts[i].reshape(-1))
            and np.array_equal(gate_set_vals, sim2.gate_set[i])
            and num_gate_types == sim2.gate_set.shape[-1]
            for idx, (layout_vals, gate_set_vals, num_gate_types) in enumerate(recorded_calls)
            for i in [idx // bw2]
        ),
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
