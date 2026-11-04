#!/usr/bin/env python3
"""
End-to-end: .bt source -> frontend -> bt dialect -> bt-opt lowering passes.

The readiness audit found the frontend and the `bt` dialect were two
unconnected compilers: bt-opt could only be fed hand-written .mlir files, so
the five lowering passes were unreachable from user code.

This test drives the joined pipeline with both real binaries and checks that
the passes actually fire on output that came from .bt source.

Requires BT_COMPILER and BT_OPT (set by CTest). Skips (exit 77) when either is
missing or when the compiler was built without the dialect.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = """
// Minimal program that the dialect can lower completely.
func add_const(x:int)->int {
    return x + 3;
}
func sub_const(x:int)->int {
    return x - 40;
}
"""

# What a correct lowering must contain.
EXPECTED_OPS = ["bt.const", "bt.btha", "bt.csa_reduce"]
# Passes that must fire in --convert-all.
EXPECTED_LOWERINGS = [
    "ConvertBTHAPass",
    "ConvertCSAReducePass",
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    compiler = os.environ.get("BT_COMPILER")
    bt_opt = os.environ.get("BT_OPT")
    if not compiler or not bt_opt:
        print("SKIP: BT_COMPILER / BT_OPT not set")
        return 77
    if not (Path(compiler).exists() and Path(bt_opt).exists()):
        print("SKIP: bt-compiler or bt-opt not built")
        return 77

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "pipeline.bt"
        src.write_text(SOURCE)
        mlir_out = Path(td) / "pipeline.mlir"

        # 1. Frontend -> bt dialect
        r = run([compiler, "--emit-bt-dialect", str(src), "-o", str(mlir_out)])
        if r.returncode != 0:
            if "requires a build with" in r.stderr:
                print("SKIP: compiler built without the bt dialect")
                return 77
            print(f"FAIL: --emit-bt-dialect failed\n{r.stderr}")
            return 1

        text = mlir_out.read_text()
        for op in EXPECTED_OPS:
            if op not in text:
                print(f"FAIL: frontend output missing {op}")
                print(text)
                return 1
        if "module @" not in text:
            print("FAIL: no module emitted")
            return 1
        print(f"  frontend -> dialect: {len(text.splitlines())} lines, "
              f"{text.count('bt.')} bt ops")

        # 2. bt-opt must accept it -- this is the loader the driver uses, so
        #    a rejection means the frontend and dialect are still disjoint.
        r = run([bt_opt, str(mlir_out)])
        if r.returncode != 0:
            print(f"FAIL: bt-opt rejected frontend output\n{r.stderr}")
            return 1
        print("  bt-opt accepts frontend output")

        # 3. The lowering passes must fire on it.
        r = run([bt_opt, "--convert-all", str(mlir_out)])
        if r.returncode != 0:
            print(f"FAIL: --convert-all failed\n{r.stderr}")
            return 1
        lowered = r.stdout
        for pass_name in EXPECTED_LOWERINGS:
            if pass_name not in lowered:
                print(f"FAIL: {pass_name} did not fire on frontend output")
                print(lowered)
                return 1
        print(f"  --convert-all fired: {', '.join(EXPECTED_LOWERINGS)}")

        # 4. The canonical lowering bodies must be appended.
        for fn in ("bt_btha_lowering", "bt_csa_reduce_canonical"):
            if f"func.func @{fn}" not in lowered:
                print(f"FAIL: canonical lowering @{fn} not emitted")
                return 1
        print("  canonical lowering bodies appended")

    print("\nFRONTEND -> DIALECT PIPELINE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
