#!/usr/bin/env python3
"""
Heavy Computational CPU Benchmark in Python:
Standard Python / NumPy vs Trium SDK 1.2 JIT Native Execution
"""

import sys
import time
import math
from pathlib import Path

# Add ffi/python to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ffi/python'))
import trium
import bt_ffi

def run_python_benchmark():
    print("======================================================================")
    print("  PYTHON SDK BENCHMARK: Pure Python vs Trium SDK JIT Native Execution")
    print("  Workload: 1,000,000 Elements Vectorized Polynomial & Arithmetic Map")
    print("======================================================================")

    N = 1_000_000
    inputs = [(i % 1000) - 500 for i in range(N)]
    lib_path = str(Path(__file__).resolve().parents[1] / 'build/libtrium_sdk.dylib')

    # 1. Pure Python baseline
    print(f"\n[Pure Python] Running polynomial transform over {N:,} elements...")
    t0 = time.perf_counter()
    res_py = [((x * x) - (3 * x) + 7) for x in inputs]
    t1 = time.perf_counter()
    py_ms = (t1 - t0) * 1000.0

    # 2. Trium SDK JIT Native Compilation & Vector Map
    print(f"[Trium SDK] JIT compiling to native machine code and mapping over {N:,} elements...")
    code = """
    func poly(x: int) -> int {
        return (x * x) - (3 * x) + 7;
    }
    """
    with trium.Context(library=lib_path, max_elements=1000000, operation_budget=100000000) as ctx:
        with ctx.compile(code) as module:
            t2 = time.perf_counter()
            res_trium, stats = module.map('poly', inputs, backend='native')
            t3 = time.perf_counter()
            trium_ms = (t3 - t2) * 1000.0

    # Parity check
    assert res_py == res_trium, "Results do not match!"

    print("\n--- Python Performance Results ---")
    print(f"  Pure Python Execution:       {py_ms:8.2f} ms ({N / (py_ms / 1000.0) / 1e6:6.2f} M elements/sec)")
    print(f"  Trium SDK JIT Native Map:    {trium_ms:8.2f} ms ({N / (trium_ms / 1000.0) / 1e6:6.2f} M elements/sec)")
    print(f"  Kernel Execution Time:       {stats['execute_ms']:8.2f} ms")
    print(f"  Total Speedup:               {py_ms / trium_ms:8.2f}x faster than standard Python!")
    print(f"  Verification:                100% bit-exact output parity across 1,000,000 elements!")

    # 3. Native RMSNorm via bt_ffi
    print("\n----------------------------------------------------------------------")
    print("  Trium Native Fast RMSNorm Benchmark (1,000,000 elements)")
    print("----------------------------------------------------------------------")
    float_inputs = [float(x) / 100.0 for x in inputs]
    
    t4 = time.perf_counter()
    norm_out = bt_ffi.xbt_rmsnorm(float_inputs, None, 1e-5)
    t5 = time.perf_counter()
    rmsnorm_ms = (t5 - t4) * 1000.0

    print(f"  Trium Native RMSNorm Time:   {rmsnorm_ms:8.2f} ms ({N / (rmsnorm_ms / 1000.0) / 1e6:6.2f} M floats/sec)")
    print("======================================================================\n")

if __name__ == '__main__':
    run_python_benchmark()
