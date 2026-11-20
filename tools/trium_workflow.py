#!/usr/bin/env python3
"""
trium_workflow.py -- Unified Dual-Track CLI workflow driver for Trium-XBT.

Exposes both tracks cleanly without moving existing files or altering CMake:
  - Track 1 (Present-Day Hardware): CPU / LLVM / SDK / Ternary BLAS
  - Track 2 (Next-Gen Hardware):    YAML targets, RISC-V Xbt, RTL, Chipyard, Yosys

Subcommands:
  status                         Inspect Track 1 & Track 2 assets and detected tools
  track1-check                   Build bt-compiler (--test) & libtrium_sdk.so + SDK tests
  track1-bench                   Build & run tools/ternary_bench.cpp --check (-O3 -march=native)
  track2-targets                 Run tools/gen_targets.py --check & check_generated_backends.py
  track2-scaffold <target_name>  Scaffold a new backend package via tools/gen_backend.py
  track2-verify [--require-eda]  Run tools/run_hardware_benchmarks.py
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build" / "workflow_track1"


def _find_cxx() -> str:
    for cand in ("g++", "clang++", "c++"):
        path = shutil.which(cand)
        if path:
            return path
    raise RuntimeError("No C++20 compiler found (tried g++, clang++, c++)")


def _run(cmd: List[str], *, cwd: Path = ROOT, env: Optional[dict] = None) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return proc.returncode


def _check_tool(name: str) -> str:
    path = shutil.which(name)
    return f"found ({path})" if path else "missing"


def _check_path(rel: str) -> str:
    p = ROOT / rel
    return "present" if p.exists() else "missing"


def cmd_status(_args: argparse.Namespace) -> int:
    print("=" * 76)
    print("Trium-XBT Dual-Track Workflow Status")
    print("=" * 76)
    print(f"Repository : {ROOT}")
    print(f"Host       : {platform.system()} {platform.machine()} (Python {platform.python_version()})")
    print()

    print("[Track 1] Present-Day Hardware (CPU / LLVM / SDK / BLAS)")
    t1_assets = [
        ("Compiler driver", "bt_compiler.cpp"),
        ("MLIR / bt-opt", "tools/bt-opt/bt-opt.cpp"),
        ("SDK C ABI", "include/Ternary/SDK/sdk.h"),
        ("SDK C++20 Header", "include/Ternary/SDK/SDK.hpp"),
        ("SDK Runtime", "lib/SDK/sdk.cpp"),
        ("Python FFI", "ffi/python/trium/__init__.py"),
        ("Rust FFI", "ffi/rust/src/lib.rs"),
        ("Ternary BLAS", "ternary_xbt_blas.hpp"),
        ("Benchmark Harness", "tools/ternary_bench.cpp"),
    ]
    for label, rel in t1_assets:
        print(f"  - {label:20s}: {_check_path(rel):8s} ({rel})")

    print("  Host toolchain:")
    for tool in ("g++", "clang++", "cmake", "ninja", "llvm-config", "mlir-opt", "rustc", "cargo"):
        print(f"    * {tool:14s}: {_check_tool(tool)}")
    print()

    print("[Track 2] Next-Generation Ternary & RISC-V Xbt Hardware")
    targets = sorted((ROOT / "targets").glob("*.yaml")) if (ROOT / "targets").exists() else []
    print(f"  - YAML Targets ({len(targets)}) : {', '.join(t.stem for t in targets) or 'none'}")
    t2_assets = [
        ("Generated Targets", "include/Ternary/Target/GeneratedTargets.hpp"),
        ("Target Generator", "tools/gen_targets.py"),
        ("Backend Scaffolder", "tools/gen_backend.py"),
        ("RISC-V Xbt RTL", "hw/riscv_xbt_extension.sv"),
        ("Trium CPU Core RTL", "hw/trium_cpu_core.sv"),
        ("RTL Testbench", "hw/tb_riscv_xbt_alu.sv"),
        ("Yosys Synth Script", "hw/synth_xbt_yosys.ys"),
        ("Chipyard RoCC", "verification/chipyard/XbtRoCCAccelerator.scala"),
        ("Chipyard Adapter", "verification/chipyard/riscv_xbt_rocc_adapter.sv"),
        ("Golden Model", "verification/golden_model/xbt_golden.py"),
    ]
    for label, rel in t2_assets:
        print(f"  - {label:20s}: {_check_path(rel):8s} ({rel})")

    print("  EDA & Formal tools:")
    for tool in ("verilator", "iverilog", "vvp", "yosys", "z3", "lean"):
        print(f"    * {tool:14s}: {_check_tool(tool)}")
    print("=" * 76, flush=True)
    return 0


def cmd_track1_check(_args: argparse.Namespace) -> int:
    cxx = _find_cxx()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    bt_compiler = BUILD_DIR / "bt-compiler"
    sdk_lib = BUILD_DIR / "libtrium_sdk.so"
    host_bin = BUILD_DIR / "unit_sdk_host"
    gemm_bin = BUILD_DIR / "unit_sdk_gemm_layout"

    steps = [
        (
            "Build bt-compiler",
            [
                cxx, "-std=c++20", "-O2", "-I.", "-Iinclude",
                "bt_compiler.cpp", "-lpthread", "-ldl", "-o", str(bt_compiler),
            ],
        ),
        (
            "Run bt-compiler --test",
            [str(bt_compiler), "--test"],
        ),
        (
            "Build libtrium_sdk.so",
            [
                cxx, "-std=c++20", "-O2", "-shared", "-fPIC", "-march=native",
                "-I.", "-Iinclude",
                "lib/SDK/sdk.cpp",
                "lib/SDK/VulkanRuntime.cpp",
                "lib/SDK/NativeRuntime.cpp",
                "lib/SDK/NumericalRuntime.cpp",
                "-lpthread", "-ldl",
                "-o", str(sdk_lib),
            ],
        ),
        (
            "Run Python SDK contracts (test/SDK/contracts.py)",
            [sys.executable, "test/SDK/contracts.py", str(sdk_lib)],
        ),
        (
            "Build C++ SDK host test (test/SDK/host.cpp)",
            [
                cxx, "-std=c++20", "-O2", "-I.", "-Iinclude",
                "test/SDK/host.cpp",
                f"-L{BUILD_DIR}", f"-Wl,-rpath,{BUILD_DIR}",
                "-ltrium_sdk", "-lpthread", "-ldl",
                "-o", str(host_bin),
            ],
        ),
        (
            "Run C++ SDK host test",
            [str(host_bin)],
        ),
        (
            "Build C++ GEMM layout test (test/SDK/gemm_layout.cpp)",
            [
                cxx, "-std=c++20", "-O2", "-I.", "-Iinclude",
                "test/SDK/gemm_layout.cpp",
                f"-L{BUILD_DIR}", f"-Wl,-rpath,{BUILD_DIR}",
                "-ltrium_sdk", "-lpthread", "-ldl",
                "-o", str(gemm_bin),
            ],
        ),
        (
            "Run C++ GEMM layout test",
            [str(gemm_bin)],
        ),
    ]

    for title, cmd in steps:
        print(f"\n==> [Track 1] {title}", flush=True)
        rc = _run(cmd)
        if rc != 0:
            print(f"FAIL: {title} exited with code {rc}", file=sys.stderr, flush=True)
            return rc

    print("\nPASS: Track 1 check completed successfully.", flush=True)
    return 0


def cmd_track1_bench(args: argparse.Namespace) -> int:
    cxx = _find_cxx()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    bench_bin = BUILD_DIR / "ternary_bench"

    print("==> [Track 1] Build tools/ternary_bench.cpp (-O3 -march=native)", flush=True)
    rc = _run([
        cxx, "-std=c++20", "-O3", "-march=native",
        "-I.", "-Iinclude", "-Itools",
        "tools/ternary_bench.cpp", "-lpthread",
        "-o", str(bench_bin),
    ])
    if rc != 0:
        return rc

    print("\n==> [Track 1] Run ternary_bench --check", flush=True)
    run_cmd = [str(bench_bin), "--check"] + list(args.extra)
    return _run(run_cmd)


def cmd_track2_targets(_args: argparse.Namespace) -> int:
    print("==> [Track 2] Validate YAML targets (tools/gen_targets.py --check)", flush=True)
    rc = _run([sys.executable, "tools/gen_targets.py", "--check"])
    if rc != 0:
        return rc

    print("\n==> [Track 2] Verify generated backend packages (tools/check_generated_backends.py)", flush=True)
    env = os.environ.copy()
    local_tmp: Optional[Path] = None
    if "TMPDIR" not in env:
        local_tmp = ROOT / "build" / ".workflow_tmp"
        local_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(local_tmp)
    try:
        return _run([sys.executable, "tools/check_generated_backends.py"], env=env)
    finally:
        if local_tmp is not None:
            shutil.rmtree(local_tmp, ignore_errors=True)


def cmd_track2_scaffold(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "tools/gen_backend.py", args.target_name]
    if args.out:
        cmd.extend(["--out", args.out])
    print(f"==> [Track 2] Scaffold hardware backend for '{args.target_name}'", flush=True)
    return _run(cmd)


def cmd_track2_verify(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "tools/run_hardware_benchmarks.py"]
    if args.require_eda:
        cmd.append("--require-eda")
    if args.no_write_log:
        cmd.append("--no-write-log")
    print("==> [Track 2] Run hardware verification & benchmarks", flush=True)
    return _run(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trium_workflow.py",
        description="Unified Dual-Track workflow driver for Trium-XBT.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show Track 1 & Track 2 status and detected host/EDA tools")
    p_status.set_defaults(func=cmd_status)

    p_t1_check = sub.add_parser("track1-check", help="Build bt-compiler + libtrium_sdk.so and run Track 1 SDK tests")
    p_t1_check.set_defaults(func=cmd_track1_check)

    p_t1_bench = sub.add_parser("track1-bench", help="Compile and run tools/ternary_bench.cpp --check with -O3 -march=native")
    p_t1_bench.add_argument("extra", nargs="*", help="Optional extra arguments passed to ternary_bench")
    p_t1_bench.set_defaults(func=cmd_track1_bench)

    p_t2_targets = sub.add_parser("track2-targets", help="Run gen_targets.py --check and check_generated_backends.py")
    p_t2_targets.set_defaults(func=cmd_track2_targets)

    p_t2_scaffold = sub.add_parser("track2-scaffold", help="Scaffold a new hardware backend package from targets/<name>.yaml")
    p_t2_scaffold.add_argument("target_name", help="Target name or alias defined in targets/*.yaml")
    p_t2_scaffold.add_argument("--out", default=None, help="Optional output directory (default: build/backends)")
    p_t2_scaffold.set_defaults(func=cmd_track2_scaffold)

    p_t2_verify = sub.add_parser("track2-verify", help="Run tools/run_hardware_benchmarks.py")
    p_t2_verify.add_argument("--require-eda", action="store_true", help="Exit 77 if Verilator/Icarus or Yosys is missing")
    p_t2_verify.add_argument("--no-write-log", action="store_true", help="Do not write JSON report into verification/regression_logs/")
    p_t2_verify.set_defaults(func=cmd_track2_verify)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
