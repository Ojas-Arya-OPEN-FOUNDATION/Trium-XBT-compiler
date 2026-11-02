#!/usr/bin/env python3
"""Regression checks for the deep-audit compiler findings (C-01, C-02).

C-01: an int39 add/sub that leaves the balanced 39-trit domain must fail on the
      reference path (and must never silently saturate).
C-02: a failing cross-compilation backend must fail the driver and must not write
      a synthetic assembly stub.
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

print("PASS: compiler audit regressions (C-01 overflow contract, C-02 no fabricated output)")
