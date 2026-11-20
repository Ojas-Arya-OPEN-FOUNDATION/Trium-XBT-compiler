"""
forgebench_xbt.py -- Functional Parity & Wall-Clock Verification for ML Workloads.

Executes real bit-exact functional parity checks using the Python golden model
(xbt_planar_dot64) for a 768x768 ternary projection and 64x64 / 128x128 GEMMs.
Does NOT report hardcoded RTL cycle counts as measured hardware results.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple
from verification.golden_model.xbt_golden import xbt_planar_dot64


def generate_ternary_matrix(rows: int, cols: int, seed: int = 42) -> List[List[int]]:
    rng = random.Random(seed)
    return [[rng.choice((-1, 0, 1)) for _ in range(cols)] for _ in range(rows)]


def generate_ternary_vector(size: int, seed: int = 123) -> List[int]:
    rng = random.Random(seed)
    return [rng.choice((-1, 0, 1)) for _ in range(size)]


def pack_trits_to_rails64(trits: List[int]) -> Tuple[List[int], List[int]]:
    p_words, n_words = [], []
    for w in range((len(trits) + 63) // 64):
        p = 0
        n = 0
        for i, t in enumerate(trits[w * 64 : (w + 1) * 64]):
            if t == 1:
                p |= 1 << i
            elif t == -1:
                n |= 1 << i
        p_words.append(p)
        n_words.append(n)
    return p_words, n_words


def run_768_projection_benchmark() -> Dict[str, Any]:
    dim = 768
    W = generate_ternary_matrix(dim, dim, seed=2026)
    x = generate_ternary_vector(dim, seed=777)

    t0 = time.perf_counter()
    y_ref = [sum(row[j] * x[j] for j in range(dim)) for row in W]
    ref_ms = (time.perf_counter() - t0) * 1e3

    p_x, n_x = pack_trits_to_rails64(x)
    num_chunks = len(p_x)

    t1 = time.perf_counter()
    y_xbt = []
    for i in range(dim):
        p_w, n_w = pack_trits_to_rails64(W[i])
        y_xbt.append(sum(xbt_planar_dot64(p_w[c], n_w[c], p_x[c], n_x[c]) for c in range(num_chunks)))
    xbt_ms = (time.perf_counter() - t1) * 1e3

    diffs = sum(1 for a, b in zip(y_xbt, y_ref) if a != b)
    assert diffs == 0, f"768 projection mismatch: {diffs} differences found!"

    return {
        "workload": "768 -> 768 Ternary Linear Projection",
        "mode": "python_golden_model_functional_parity",
        "parameters": dim * dim,
        "dot64_invocations": dim * num_chunks,
        "python_ref_ms": round(ref_ms, 2),
        "python_xbt_model_ms": round(xbt_ms, 2),
        "bit_exact_parity": "PASS (0 mismatches across all 768 elements)",
        "rtl_cycles_measured": False,
        "status": "PASS",
    }


def run_gemm_microbenchmarks() -> List[Dict[str, Any]]:
    results = []
    for N in (64, 128):
        A = generate_ternary_matrix(N, N, seed=N)
        B = generate_ternary_matrix(N, N, seed=N + 1)
        B_T = [[B[r][c] for r in range(N)] for c in range(N)]

        C_ref = [[sum(A[i][k] * B_T[j][k] for k in range(N)) for j in range(N)] for i in range(N)]
        chunks = (N + 63) // 64
        packed_a = [pack_trits_to_rails64(A[i]) for i in range(N)]
        packed_b = [pack_trits_to_rails64(B_T[j]) for j in range(N)]

        t0 = time.perf_counter()
        C_xbt = [
            [sum(xbt_planar_dot64(packed_a[i][0][c], packed_a[i][1][c], packed_b[j][0][c], packed_b[j][1][c]) for c in range(chunks)) for j in range(N)]
            for i in range(N)
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        errors = sum(1 for i in range(N) for j in range(N) if C_xbt[i][j] != C_ref[i][j])
        assert errors == 0, f"GEMM {N}x{N} mismatch: {errors} errors!"
        results.append({
            "size": f"{N}x{N}",
            "mac_ops": N * N * N,
            "python_model_ms": round(elapsed_ms, 2),
            "parity": "PASS (bit-exact)",
            "status": "PASS",
        })
    return results


def run_forgebench_suite() -> Dict[str, Any]:
    print("=" * 80)
    print("=== ForgeBench Functional Parity Suite (Python Golden Model) ===")
    print("=" * 80)
    proj = run_768_projection_benchmark()
    print(f"  [PASS] {proj['workload']}: {proj['bit_exact_parity']} ({proj['python_xbt_model_ms']} ms in Python model)")
    gemm = run_gemm_microbenchmarks()
    for g in gemm:
        print(f"  [PASS] Ternary GEMM {g['size']}: {g['mac_ops']:,} MACs | {g['parity']} ({g['python_model_ms']} ms)")
    print("  [NOTE] RTL cycle counts are not measured here; use Verilator/Icarus via tools/run_hardware_benchmarks.py.")
    return {"projection_768": proj, "gemm": gemm, "status": "PASS"}


if __name__ == "__main__":
    run_forgebench_suite()
