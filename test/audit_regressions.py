#!/usr/bin/env python3
"""Regression checks for the deep-audit compiler findings (C-01, C-02, C-03, C-04).

C-01: an int39 add/sub that leaves the balanced 39-trit domain must fail on the
      reference path (and must never silently saturate).
C-02: a failing cross-compilation backend must fail the driver and must not write
      a synthetic assembly stub.
C-03: arithmetic verification asserts checked contract (overflow error) and never
      encodes silent clamping.
C-04: compiled code and interpreter accept opt-in execution budget (--execution-budget=N)
      to bound loop back-edges.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = Path(sys.argv[1]).resolve()
H = (3**39 - 1) // 2


def run(args, env=None):
    return subprocess.run([str(compiler), *args], capture_output=True, text=True,
                          env=env, timeout=180, cwd=compiler.parent)


with tempfile.TemporaryDirectory(prefix="bt-audit-regressions-") as directory:
    work = Path(directory)

    # C-01: Checked overflow on addition and subtraction
    overflow = work / "overflow.bt"
    overflow.write_text(f"print(({H} + 1));\n")
    result = run(["--core-run", str(overflow)])
    assert result.returncode == 1, (
        "C-01: int39 overflow must be an error", result.returncode, result.stdout, result.stderr)
    assert "overflow" in result.stderr.lower(), result.stderr

    negative = work / "negative.bt"
    negative.write_text(f"print((-{H} - 1));\n")
    result = run(["--core-run", str(negative)])
    assert result.returncode == 1, (
        "C-01: negative int39 overflow must be an error", result.returncode, result.stdout, result.stderr)

    ordinary = work / "ordinary.bt"
    ordinary.write_text("print(2 + 3);\n")
    result = run(["--core-run", str(ordinary)])
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.split()[0] == "5", result.stdout

    bounded = work / "bounded.bt"
    bounded.write_text(f"print({H});\n")
    result = run(["--core-run", str(bounded)])
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)

    # C-02: a backend that always fails must not be papered over.
    fake_bin = work / "bin"
    fake_bin.mkdir()
    fake_clang = fake_bin / "clang"
    fake_clang.write_text("#!/bin/sh\nexit 1\n")
    fake_clang.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")

    assembly = work / "out.s"
    result = run(["--target=riscv64", "-S", str(ordinary), "-o", str(assembly)], env=env)
    assert result.returncode != 0, (
        "C-02: a failed backend must not be reported as success", result.returncode, result.stdout, result.stderr)
    if assembly.exists():
        text = assembly.read_text(errors="replace")
        assert "li a0, 0" not in text and "bt_main:" not in text, (
            "C-02: a synthetic assembly stub was written")

    # C-03: benchmark executable verifies checked contract and 0 clamping
    bench_bin = compiler.parent / "benchmark_arithmetic"
    if bench_bin.exists():
        b_res = subprocess.run([str(bench_bin), "1"], capture_output=True, text=True, timeout=60)
        assert b_res.returncode == 0, ("C-03: benchmark_arithmetic failed", b_res.stderr)
        assert "0 silent saturation" in b_res.stdout, "C-03: must assert 0 silent saturation"
        assert "checked range guard" in b_res.stdout, "C-03: must verify checked lowering"

    # C-04: Loop execution budget (--execution-budget=N)
    infloop = work / "infloop.bt"
    infloop.write_text("while (1) { }\n")

    # 1) Interpreter execution budget
    res_inf_core = run(["--execution-budget=25", "--core-run", str(infloop)])
    assert res_inf_core.returncode == 1, "C-04: interpreter infinite loop must terminate with error"
    assert "budget exceeded" in res_inf_core.stderr.lower(), res_inf_core.stderr

    # 2) LLVM IR lowering contains back-edge tick checks
    res_llvm = run(["--execution-budget=25", "--emit-llvm", str(infloop)])
    assert res_llvm.returncode == 0, res_llvm.stderr
    assert "@bt.core.ticks" in res_llvm.stdout, "C-04: IR must contain @bt.core.ticks counter"
    assert "icmp ugt i64" in res_llvm.stdout, "C-04: IR must contain tick upper bound comparison"

    # 3) Finite loop completes when under budget
    finiteloop = work / "finite.bt"
    finiteloop.write_text("var i = 0;\nwhile (i < 5) { i = i + 1; }\nprint(i);\n")
    res_fin_core = run(["--execution-budget=200", "--core-run", str(finiteloop)])
    assert res_fin_core.returncode == 0, (res_fin_core.returncode, res_fin_core.stderr)
    assert res_fin_core.stdout.split()[0] == "5", res_fin_core.stdout

    # 4) Native AOT execution budget halts infinite loop
    res_inf_native = run(["--execution-budget=25", "--native", str(infloop)])
    assert res_inf_native.returncode == 1, "C-04: native AOT infinite loop must terminate with error"

print("PASS: compiler audit regressions (C-01 overflow contract, C-02 honest backend, C-03 benchmark contract, C-04 loop budget)")
