#!/usr/bin/env python3
"""
run_xbt_codegen_bench.py -- Compiles and executes tools/bench_xbt_codegen.cpp,
verifies tools/trium_cc/trium_cc.py relational operators and dot-product
vectorization on real C snippets, and writes a timestamped JSON report into
verification/regression_logs/xbt_codegen_bench.json.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.trium_cc.trium_cc import HAVE_PYCPARSER, TriumCC  # noqa: E402


def run_cpp_benchmark() -> Dict[str, Any]:
    src = REPO_ROOT / "tools" / "bench_xbt_codegen.cpp"
    if not src.exists():
        raise FileNotFoundError(f"Missing C++ benchmark source: {src}")

    with tempfile.TemporaryDirectory(prefix=".tmp_xbt_bench_", dir=str(REPO_ROOT)) as tmpdir:
        exe = Path(tmpdir) / "bench_xbt_codegen"
        compile_cmd = [
            "g++",
            "-std=c++20",
            "-O2",
            f"-I{REPO_ROOT}",
            f"-I{REPO_ROOT / 'include'}",
            str(src),
            "-o",
            str(exe),
        ]
        t_compile_0 = time.perf_counter()
        comp = subprocess.run(
            compile_cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        compile_ms = (time.perf_counter() - t_compile_0) * 1000.0
        if comp.returncode != 0:
            raise RuntimeError(
                f"Failed to compile {src} (exit {comp.returncode}):\n{comp.stderr}"
            )

        t_run_0 = time.perf_counter()
        run = subprocess.run(
            [str(exe)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        run_ms = (time.perf_counter() - t_run_0) * 1000.0
        if run.returncode != 0:
            raise RuntimeError(
                f"Benchmark binary failed (exit {run.returncode}):\n"
                f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"
            )

        payload = json.loads(run.stdout)
        payload["compile_ms"] = round(compile_ms, 3)
        payload["wall_run_ms"] = round(run_ms, 3)
        return payload


def _extract_mnemonics(asm_text: str) -> List[str]:
    mnemonics: List[str] = []
    for raw in asm_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.endswith(":"):
            continue
        tok = line.split()[0]
        if tok.startswith("xbt."):
            mnemonics.append(tok)
    return mnemonics


def _eval_cmov_sequence(cmovs: List[str], diff_sign: int) -> int:
    res = -1
    for op in cmovs:
        if op == "xbt.cmovp" and diff_sign > 0:
            res = 1
        elif op == "xbt.cmovn" and diff_sign < 0:
            res = 1
        elif op == "xbt.cmovz" and diff_sign == 0:
            res = 1
    return res


def verify_trium_cc() -> Dict[str, Any]:
    if not HAVE_PYCPARSER:
        return {
            "status": "skipped",
            "reason": "pycparser_not_installed",
            "all_passed": False,
        }

    cc = TriumCC(optimize_dot=True)

    # 1. Verify all 6 relational operators produce distinct, branchless cmov* sequences
    rel_specs: List[Tuple[str, str, List[str]]] = [
        ("gt", ">", ["xbt.cmovp"]),
        ("lt", "<", ["xbt.cmovn"]),
        ("eq", "==", ["xbt.cmovz"]),
        ("ne", "!=", ["xbt.cmovp", "xbt.cmovn"]),
        ("ge", ">=", ["xbt.cmovp", "xbt.cmovz"]),
        ("le", "<=", ["xbt.cmovn", "xbt.cmovz"]),
    ]
    rel_results: Dict[str, Any] = {}
    cmov_signatures: Dict[str, Tuple[str, ...]] = {}
    rel_all_passed = True

    test_pairs = [(5, 3), (3, 3), (2, 7)]
    py_ops = {
        ">": lambda a, b: 1 if a > b else -1,
        "<": lambda a, b: 1 if a < b else -1,
        "==": lambda a, b: 1 if a == b else -1,
        "!=": lambda a, b: 1 if a != b else -1,
        ">=": lambda a, b: 1 if a >= b else -1,
        "<=": lambda a, b: 1 if a <= b else -1,
    }

    for name, op, expected_cmovs in rel_specs:
        c_src = f"int rel_{name}(int a, int b) {{ return a {op} b; }}\n"
        asm = cc.compile_c_source(c_src)
        mnems = _extract_mnemonics(asm)
        cmovs = [m for m in mnems if m.startswith("xbt.cmov")]
        has_diff = ("xbt.neg" in mnems) and ("xbt.ksadd" in mnems) and ("xbt.sign" in mnems)
        has_seed_false = bool(re.search(r"xbt\.li\s+tr\d+,\s*-1\b", asm))
        has_seed_true = bool(re.search(r"xbt\.li\s+tr\d+,\s*1\b", asm))
        has_branches = any(m in ("xbt.beq", "xbt.bne", "xbt.blt", "xbt.bge") for m in mnems)

        truth_table_ok = True
        for lhs, rhs in test_pairs:
            s = 1 if lhs > rhs else (-1 if lhs < rhs else 0)
            if _eval_cmov_sequence(cmovs, s) != py_ops[op](lhs, rhs):
                truth_table_ok = False

        passed = (
            has_diff
            and has_seed_false
            and has_seed_true
            and (not has_branches)
            and (cmovs == expected_cmovs)
            and truth_table_ok
        )
        if not passed:
            rel_all_passed = False

        cmov_signatures[op] = tuple(cmovs)
        rel_results[op] = {
            "passed": passed,
            "emitted_cmovs": cmovs,
            "expected_cmovs": expected_cmovs,
            "branchless": not has_branches,
            "truth_table_ok": truth_table_ok,
        }

    all_distinct = len(set(cmov_signatures.values())) == 6
    if not all_distinct:
        rel_all_passed = False

    # 2. Verify hardened _try_vectorize_dot across valid and invalid C loops
    vec_cases = [
        (
            "const_64_exact",
            "int f(int *a, int *b) { int s = 0; for (int i = 0; i < 64; i++) s += a[i] * b[i]; return s; }",
            {"expect_dot64_count": 1, "expect_tail": False},
        ),
        (
            "const_768_unrolled",
            "int f(int *a, int *b) { int s = 0; for (int i = 0; i < 768; i++) s += a[i] * b[i]; return s; }",
            {"expect_dot64_count": 12, "expect_tail": False, "expect_ptr_step_64": True},
        ),
        (
            "const_100_with_tail",
            "int f(int *a, int *b) { int s = 0; for (int i = 0; i < 100; i++) s += a[i] * b[i]; return s; }",
            {"expect_dot64_count": 1, "expect_tail": True, "expect_tail_count": 36},
        ),
        (
            "const_128_stride_plus_eq_1",
            "int f(int *a, int *b) { int s = 0; for (int i = 0; i < 128; i += 1) s += a[i] * b[i]; return s; }",
            {"expect_dot64_count": 2, "expect_tail": False, "expect_ptr_step_64": True},
        ),
        (
            "runtime_var_n",
            "int f(int *a, int *b, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }",
            {"expect_dot64_count": 1, "expect_tail": True, "expect_runtime_loop": True, "expect_ptr_step_64": True},
        ),
        (
            "reject_const_32_below_64",
            "int f(int *a, int *b) { int s = 0; for (int i = 0; i < 32; i++) s += a[i] * b[i]; return s; }",
            {"expect_dot64_count": 0},
        ),
        (
            "reject_stride_2",
            "int f(int *a, int *b) { int s = 0; for (int i = 0; i < 128; i += 2) s += a[i] * b[i]; return s; }",
            {"expect_dot64_count": 0},
        ),
        (
            "reject_mismatched_index",
            "int f(int *a, int *b, int j) { int s = 0; for (int i = 0; i < 128; i++) s += a[i] * b[j]; return s; }",
            {"expect_dot64_count": 0},
        ),
        (
            "reject_offset_index",
            "int f(int *a, int *b) { int s = 0; for (int i = 0; i < 128; i++) s += a[i] * b[i + 1]; return s; }",
            {"expect_dot64_count": 0},
        ),
    ]

    vec_results: Dict[str, Any] = {}
    vec_all_passed = True

    for case_name, c_src, checks in vec_cases:
        asm = cc.compile_c_source(c_src)
        mnems = _extract_mnemonics(asm)
        dot64_count = mnems.count("xbt.dot64")
        has_tail = ".dot_tail_cond_" in asm
        has_ptr_step_64 = bool(re.search(r"xbt\.addi\s+tr20,\s*tr20,\s*64\b", asm)) and bool(
            re.search(r"xbt\.addi\s+tr21,\s*tr21,\s*64\b", asm)
        )
        has_runtime_loop = ".dot64_vec_loop_" in asm and bool(
            re.search(r"xbt\.addi\s+tr24,\s*tr24,\s*-64\b", asm)
        )

        passed = dot64_count == checks["expect_dot64_count"]
        if "expect_tail" in checks:
            passed = passed and (has_tail == checks["expect_tail"])
        if "expect_tail_count" in checks:
            tail_n = checks["expect_tail_count"]
            passed = passed and bool(re.search(rf"xbt\.li\s+tr24,\s*{tail_n}\b", asm))
        if checks.get("expect_ptr_step_64"):
            passed = passed and has_ptr_step_64
        if checks.get("expect_runtime_loop"):
            passed = passed and has_runtime_loop

        if not passed:
            vec_all_passed = False

        vec_results[case_name] = {
            "passed": passed,
            "dot64_count": dot64_count,
            "has_scalar_tail": has_tail,
            "has_ptr_step_64": has_ptr_step_64,
            "has_runtime_loop": has_runtime_loop,
        }

    return {
        "status": "passed" if (rel_all_passed and vec_all_passed) else "failed",
        "pycparser_available": True,
        "all_passed": rel_all_passed and vec_all_passed,
        "relational_operators": {
            "all_passed": rel_all_passed,
            "all_6_distinct": all_distinct,
            "operators": rel_results,
        },
        "dot_product_vectorization": {
            "all_passed": vec_all_passed,
            "cases": vec_results,
        },
    }


def main() -> int:
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cpp_bench = run_cpp_benchmark()
    trium_cc_verify = verify_trium_cc()

    all_passed = bool(cpp_bench.get("all_verified")) and bool(trium_cc_verify.get("all_passed"))
    report: Dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "all_passed": all_passed,
        "cpp_codegen_benchmark": cpp_bench,
        "trium_cc_verification": trium_cc_verify,
    }

    out_path = REPO_ROOT / "verification" / "regression_logs" / "xbt_codegen_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(json.dumps(report, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
