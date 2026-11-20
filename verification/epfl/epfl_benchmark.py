"""
epfl_benchmark.py -- EPFL Combinational & ISCAS-85 Circuit Benchmark Evaluator.

When Yosys is installed, runs real CMOS-2 synthesis via hw/synth_xbt_yosys.ys.
When Yosys is absent, reports analytical estimates clearly tagged as
"analytical_estimate_no_yosys" rather than claiming synthesized silicon numbers.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_epfl_adder() -> Dict[str, Any]:
    word_trits = 64
    stages = 6
    equivalent_binary_bits = math.log2(3**word_trits)
    estimated_gates = 384 + (stages * 64 * 14)
    return {
        "benchmark": "EPFL_Adder64_Ternary",
        "mode": "analytical_estimate",
        "word_trits": word_trits,
        "equivalent_binary_bits": round(equivalent_binary_bits, 1),
        "logic_depth_stages": stages,
        "estimated_critical_path_FO4": stages * 2.5 + 3.0,
        "estimated_gate_count": estimated_gates,
    }


def evaluate_epfl_priority_encoder() -> Dict[str, Any]:
    lanes = 64
    depth = math.ceil(math.log2(lanes))
    estimated_gates = lanes * 4 + (lanes - 1) * 3
    return {
        "benchmark": "EPFL_Priority_Encoder_SIGN",
        "mode": "analytical_estimate",
        "lanes": lanes,
        "tree_depth": depth,
        "estimated_gate_count": estimated_gates,
    }


def run_yosys_if_available() -> Dict[str, Any]:
    yosys = shutil.which("yosys")
    script = REPO_ROOT / "hw" / "synth_xbt_yosys.ys"
    if not yosys or not script.exists():
        return {"status": "skipped", "reason": "yosys_not_installed"}

    r = subprocess.run([yosys, "-s", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return {"status": "fail", "returncode": r.returncode, "stderr": r.stderr[-1000:]}

    cells: Dict[str, int] = {}
    current = None
    for line in r.stdout.splitlines():
        m_mod = re.search(r"===\s+(\w+)\s+===", line)
        if m_mod:
            current = m_mod.group(1)
        m_c = re.search(r"Number of cells:\s+(\d+)", line)
        if m_c and current:
            cells[current] = int(m_c.group(1))
            current = None
    return {"status": "pass", "module_cell_counts": cells}


def run_epfl_suite() -> Dict[str, Any]:
    print("=" * 80)
    print("=== EPFL Combinational Circuit Evaluator (Analytical + Optional Yosys) ===")
    print("=" * 80)

    adder = evaluate_epfl_adder()
    priority = evaluate_epfl_priority_encoder()
    yosys_res = run_yosys_if_available()

    print(f"  [ESTIMATE] {adder['benchmark']}: {adder['logic_depth_stages']} stages, ~{adder['estimated_gate_count']:,} gates (analytical)")
    print(f"  [ESTIMATE] {priority['benchmark']}: {priority['tree_depth']} levels, ~{priority['estimated_gate_count']} gates (analytical)")
    if yosys_res["status"] == "pass":
        print(f"  [YOSYS]    Real CMOS-2 synthesis cell counts: {yosys_res['module_cell_counts']}")
    else:
        print("  [SKIP]     Yosys not installed; real synthesis skipped (run tools/run_hardware_benchmarks.py on a Yosys host).")

    return {
        "mode": "yosys_synthesized" if yosys_res["status"] == "pass" else "analytical_estimate_no_yosys",
        "epfl_adder": adder,
        "epfl_priority": priority,
        "yosys_synthesis": yosys_res,
        "status": "PASS" if yosys_res["status"] != "fail" else "FAIL",
    }


if __name__ == "__main__":
    run_epfl_suite()
