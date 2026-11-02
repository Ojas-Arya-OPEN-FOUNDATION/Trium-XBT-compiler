#!/usr/bin/env python3
"""
Multi-Core & Multi-Thread High-Complexity FLOPS Benchmark Runner:
Apple Clang Native Toolchain vs Trium-XBT Kokkos Thread Pool & Runtime
Evaluates thread scaling (1, 2, 4, 8, 10 cores) on Apple M5 across:
- LINPACK LU Decomposition (N=2000, 5.34 GFLOPs)
- Dense Cache-Blocked GEMM (N=2048x2048, 17.18 GFLOPs)
- Bit-Parallel Balanced Ternary GEMM (N=2048x2048, 17.18 Billion MACs / 1.16 TERA-OPS)
- 4th-Order Symplectic Physics Integrator (50M steps, 1.20 GFLOPs)
- Point-Symmetric CORDIC Transcendental Solver (20M evals, 2.40 GFLOPs)
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_BIN = ROOT / "build" / "multicore_flops_benchmark"
SRC = ROOT / "benchmarks" / "multicore_flops_benchmark.cpp"

def main():
    # 1. Compile benchmark binary if needed
    print("[1/2] Compiling High-Complexity Multi-Core FLOPS Benchmark with Apple Clang...")
    cmd_build = [
        "clang++", "-O3", "-std=c++20", "-arch", "arm64",
        "-isysroot", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
        "-I.", "-I./include",
        str(SRC), "-o", str(BUILD_BIN)
    ]
    res = subprocess.run(cmd_build, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Compilation error:\n{res.stderr}")
        return

    # 2. Run benchmark binary
    print("[2/2] Executing Multi-Core Thread Scaling Sweep on Apple M5 (10 Physical Cores)...\n")
    proc = subprocess.Popen([str(BUILD_BIN)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

if __name__ == '__main__':
    main()
