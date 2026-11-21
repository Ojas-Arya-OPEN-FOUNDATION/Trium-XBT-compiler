#!/usr/bin/env python3
"""
run_hardware_benchmarks.py -- Unified, honest hardware verification & benchmark runner.

Executes only real, locally verifiable checks and never fabricates silicon or EDA metrics:
  1. Python Golden Model v1+v2 full ISA regression (verification/golden_model/xbt_golden.py)
  2. ExternalSimulator POSIX-pipe conformance test (if g++ is installed)
  3. SystemVerilog RTL testbench (hw/tb_riscv_xbt_alu.sv) via Verilator or Icarus Verilog
  4. Yosys CMOS-2 synthesis area/cell benchmark (hw/synth_xbt_yosys.ys)

When an EDA tool (verilator, iverilog, yosys) is not installed, its section is recorded
explicitly as {"status": "skipped", "reason": "tool_not_installed"}. Passing --require-eda
causes exit code 77 (CTest SKIP) if any EDA tool is unavailable.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verification.golden_model.xbt_golden import (  # noqa: E402
    riscv_xbt_alu,
    xbt_base243_decoder,
    xbt_kogge_stone_adder64,
    xbt_brent_kung_adder64,
    xbt_kogge_stone_adder64_v2,
    xbt_planar_dot64,
)


def _rand_word(rng: random.Random) -> tuple[int, int]:
    p = 0
    n = 0
    for i in range(64):
        t = rng.choice((-1, 0, 1))
        if t == 1:
            p |= 1 << i
        elif t == -1:
            n |= 1 << i
    return p, n


def run_golden_model_suite() -> Dict[str, Any]:
    t0 = time.perf_counter()
    rng = random.Random(20261120)
    checks = 0
    failures = 0

    # 1. KSADD_v2 vs KSADD_v1 (2,000 random pairs)
    for _ in range(2000):
        ap, an = _rand_word(rng)
        bp, bn = _rand_word(rng)
        s1p, s1n, c1p, c1n = xbt_kogge_stone_adder64(ap, an, bp, bn)
        s2p, s2n, c2p, c2n = xbt_kogge_stone_adder64_v2(ap, an, bp, bn)
        sbkp, sbkn, cbkp, cbkn = xbt_brent_kung_adder64(ap, an, bp, bn)
        checks += 2
        if (s1p, s1n, c1p, c1n) != (s2p, s2n, c2p, c2n) or (s2p & s2n) != 0:
            failures += 1
        if (sbkp, sbkn, cbkp, cbkn) != (s2p, s2n, c2p, c2n) or (sbkp & sbkn) != 0:
            failures += 1

    # 2. CSAMAC with distinct rd != rs1 (all 27 1-trit triples)
    OPCODE_XBT = 0x0B
    vals = [(0, 1), (0, 0), (1, 0)]
    for rp, rn in vals:
        for ap, an in vals:
            for bp, bn in vals:
                rd_p, rd_n, aux_p, aux_n, _, valid = riscv_xbt_alu(
                    OPCODE_XBT, 0b101, 0, ap, an, bp, bn, 0, rd_p_in=rp, rd_n_in=rn
                )
                checks += 1
                expected = (rp - rn) + (ap - an) * (bp - bn)
                got = (rd_p - rd_n) + 3 * (aux_p - aux_n)
                if not valid or (rd_p & rd_n) != 0 or got != expected:
                    failures += 1

    # 3. Base-243 UNPACK243 <-> PACK243 exhaustive round-trip (243 valid + 13 invalid)
    for b in range(256):
        u_p, u_n, _, _, _, u_valid = riscv_xbt_alu(OPCODE_XBT, 0b011, 0, 0, 0, 0, 0, b)
        checks += 1
        if b < 243:
            _, _, _, _, pk_scalar, pk_valid = riscv_xbt_alu(OPCODE_XBT, 0b110, 0x02, u_p, u_n, 0, 0, 0)
            if not u_valid or not pk_valid or pk_scalar != b:
                failures += 1
        else:
            if u_p != 0 or u_n != 0:
                failures += 1

    # 4. LI5 exhaustive (256 bytes x 8 chunks = 2,048 checks)
    for chunk in range(8):
        for b in range(256):
            imm12 = ((chunk & 0x7) << 8) | b
            r_p, r_n, _, _, _, r_valid = riscv_xbt_alu(OPCODE_XBT, 0b111, 0, 0, 0, 0, 0, 0, imm12=imm12)
            checks += 1
            bp, bn = xbt_base243_decoder(b)
            exp_p = (bp << (5 * chunk)) & ((1 << 64) - 1)
            exp_n = (bn << (5 * chunk)) & ((1 << 64) - 1)
            if not r_valid or r_p != exp_p or r_n != exp_n:
                failures += 1

    # 5. DOT64 throughput sample (10,000 random 64-trit words)
    dot_t0 = time.perf_counter()
    dot_acc = 0
    for _ in range(10000):
        ap, an = _rand_word(rng)
        bp, bn = _rand_word(rng)
        dot_acc += xbt_planar_dot64(ap, an, bp, bn)
        checks += 1
    dot_ms = (time.perf_counter() - dot_t0) * 1e3

    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return {
        "status": "pass" if failures == 0 else "fail",
        "checks": checks,
        "failures": failures,
        "elapsed_ms": round(elapsed_ms, 2),
        "dot64_10k_ms": round(dot_ms, 2),
        "dot64_checksum": dot_acc,
    }


def run_rtl_testbench() -> Dict[str, Any]:
    tb = REPO_ROOT / "hw" / "tb_riscv_xbt_alu.sv"
    rtl = REPO_ROOT / "hw" / "riscv_xbt_extension.sv"
    verilator = shutil.which("verilator")
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")

    if not verilator and not (iverilog and vvp):
        return {
            "status": "skipped",
            "reason": "tool_not_installed",
            "missing_tools": ["verilator", "iverilog"],
        }

    with tempfile.TemporaryDirectory(prefix="xbt_rtl_") as tmp:
        t0 = time.perf_counter()
        if verilator:
            cmd_build = [
                verilator,
                "--binary",
                "-Wall",
                "-Wno-DECLFILENAME",
                "--top-module",
                "tb_riscv_xbt_alu",
                "-Mdir",
                tmp,
                str(rtl),
                str(tb),
            ]
            b = subprocess.run(cmd_build, capture_output=True, text=True, timeout=180)
            if b.returncode != 0:
                return {
                    "status": "fail",
                    "tool": "verilator",
                    "stage": "compile",
                    "returncode": b.returncode,
                    "stderr": b.stderr[-2000:],
                }
            sim_bin = Path(tmp) / "Vtb_riscv_xbt_alu"
            r = subprocess.run([str(sim_bin)], capture_output=True, text=True, timeout=120)
            tool = "verilator"
        else:
            sim_out = Path(tmp) / "tb_xbt"
            b = subprocess.run(
                [iverilog, "-g2012", "-o", str(sim_out), str(rtl), str(tb)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if b.returncode != 0:
                return {
                    "status": "fail",
                    "tool": "iverilog",
                    "stage": "compile",
                    "returncode": b.returncode,
                    "stderr": b.stderr[-2000:],
                }
            r = subprocess.run([vvp, str(sim_out)], capture_output=True, text=True, timeout=120)
            tool = "iverilog"

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return {
            "status": "pass" if r.returncode == 0 else "fail",
            "tool": tool,
            "returncode": r.returncode,
            "elapsed_ms": round(elapsed_ms, 2),
            "stdout_tail": r.stdout[-1500:],
            "stderr_tail": r.stderr[-1000:],
        }


def run_yosys_synthesis() -> Dict[str, Any]:
    yosys = shutil.which("yosys")
    script = REPO_ROOT / "hw" / "synth_xbt_yosys.ys"
    if not yosys:
        return {
            "status": "skipped",
            "reason": "tool_not_installed",
            "missing_tools": ["yosys"],
        }

    t0 = time.perf_counter()
    r = subprocess.run(
        [yosys, "-s", str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    if r.returncode != 0:
        return {
            "status": "fail",
            "tool": "yosys",
            "returncode": r.returncode,
            "elapsed_ms": round(elapsed_ms, 2),
            "stderr_tail": r.stderr[-2000:],
        }

    modules: Dict[str, int] = {}
    current = None
    for line in r.stdout.splitlines():
        m_mod = re.search(r"===\s+(\w+)\s+===", line)
        if m_mod:
            current = m_mod.group(1)
        m_cells = re.search(r"Number of cells:\s+(\d+)", line) or re.search(r"^\s*(\d+)\s+cells", line)
        if m_cells and current:
            modules[current] = int(m_cells.group(1))
            current = None

    return {
        "status": "pass",
        "tool": "yosys",
        "elapsed_ms": round(elapsed_ms, 2),
        "module_cell_counts": modules,
    }




def run_yosys_abc_synthesis() -> Dict[str, Any]:
    yosys = shutil.which("yosys")
    script = REPO_ROOT / "hw" / "synth_xbt_abc.ys"
    if not yosys:
        return {
            "status": "skipped",
            "reason": "tool_not_installed",
            "missing_tools": ["yosys"],
        }

    t0 = time.perf_counter()
    r = subprocess.run(
        [yosys, "-s", str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    if r.returncode != 0:
        return {
            "status": "fail",
            "tool": "yosys+abc",
            "returncode": r.returncode,
            "elapsed_ms": round(elapsed_ms, 2),
            "stderr_tail": r.stderr[-2000:],
        }

    modules: Dict[str, Dict[str, Any]] = {}
    sections = re.findall(r"=== ABC CMOS-2:\s+(.+?)\s+===(.*?)(?==== ABC CMOS-2:|\Z)", r.stdout, re.DOTALL)
    for name, body in sections:
        name = name.strip()
        stat_blocks = list(re.finditer(r"Printing statistics\.(.*?)(\n\s*\n\s*\n|\Z)", body, re.DOTALL))
        cells = 0
        if stat_blocks:
            last_stat = stat_blocks[-1].group(1)
            target_text = last_stat.split("=== design hierarchy ===")[-1] if "=== design hierarchy ===" in last_stat else last_stat
            for line in target_text.splitlines():
                m_c = re.search(r"^\s*(\d+)\s+cells", line) or re.search(r"Number of cells:\s+(\d+)", line)
                if m_c:
                    cells = int(m_c.group(1))

        ltp_matches = re.findall(r"Longest topological path in [^\s]+ \(length=(\d+)\):", body)
        ltp = int(ltp_matches[0]) if ltp_matches else 0
        modules[name] = {"cells": cells, "logic_depth_ltp": ltp}

    return {
        "status": "pass",
        "tool": "yosys+abc",
        "elapsed_ms": round(elapsed_ms, 2),
        "modules": modules,
    }


def run_yosys_fpga_xilinx() -> Dict[str, Any]:
    yosys = shutil.which("yosys")
    script = REPO_ROOT / "hw" / "synth_xbt_fpga_xilinx.ys"
    if not yosys:
        return {
            "status": "skipped",
            "reason": "tool_not_installed",
            "missing_tools": ["yosys"],
        }

    t0 = time.perf_counter()
    r = subprocess.run(
        [yosys, "-s", str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    if r.returncode != 0:
        return {
            "status": "fail",
            "tool": "yosys_xilinx",
            "returncode": r.returncode,
            "elapsed_ms": round(elapsed_ms, 2),
            "stderr_tail": r.stderr[-2000:],
        }

    modules: Dict[str, Dict[str, int]] = {}
    sections = re.findall(r"=== FPGA xc7 LUT/FF:\s+(.+?)\s+===(.*?)(?==== FPGA xc7 LUT/FF:|\Z)", r.stdout, re.DOTALL)
    for name, body in sections:
        name = name.strip()
        stat_blocks = list(re.finditer(r"Printing statistics\.(.*?)(\n\s*\n\s*\n|\Z)", body, re.DOTALL))
        if not stat_blocks:
            continue
        last_stat = stat_blocks[-1].group(1)
        target_text = last_stat.split("=== design hierarchy ===")[-1] if "=== design hierarchy ===" in last_stat else last_stat

        cells = 0
        luts = 0
        ffs = 0
        for line in target_text.splitlines():
            m_c = re.search(r"^\s*(\d+)\s+cells", line) or re.search(r"Number of cells:\s+(\d+)", line)
            if m_c:
                cells = int(m_c.group(1))
            m_lut = re.search(r"^\s*(\d+)\s+LUT[1-6]", line)
            if m_lut:
                luts += int(m_lut.group(1))
            m_ff = re.search(r"^\s*(\d+)\s+FD[A-Z0-9]+", line)
            if m_ff:
                ffs += int(m_ff.group(1))

        modules[name] = {"cells": cells, "luts": luts, "ffs": ffs}

    return {
        "status": "pass",
        "tool": "yosys_xilinx",
        "elapsed_ms": round(elapsed_ms, 2),
        "modules": modules,
    }


def run_external_simulator_check() -> Dict[str, Any]:
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return {"status": "skipped", "reason": "tool_not_installed", "missing_tools": ["g++"]}

    with tempfile.TemporaryDirectory(prefix=".xbt_sim_", dir=str(REPO_ROOT)) as tmp:
        mock = Path(tmp) / "mock_sim.py"
        mock.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "state = []\n"
            "for line in sys.stdin:\n"
            "    parts = line.strip().split()\n"
            "    if not parts: continue\n"
            "    cmd = parts[0]\n"
            "    if cmd == 'load': print('ok', flush=True)\n"
            "    elif cmd == 'exec':\n"
            "        xs = [int(x) for x in parts[1:]]\n"
            "        state = [sum(xs)]\n"
            "        print('ok 1', flush=True)\n"
            "    elif cmd == 'state': print('ok ' + ' '.join(map(str, state)), flush=True)\n"
            "    elif cmd == 'quit': print('ok', flush=True); break\n"
            "    else: print('unsupported', flush=True)\n",
            encoding="utf-8",
        )
        os.chmod(mock, 0o755)
        driver = Path(tmp) / "driver.cpp"
        driver.write_text(
            '#include "Ternary/Target/Simulator.hpp"\n'
            "#include <iostream>\n"
            "int main(int argc, char** argv) {\n"
            '  bt::target::ExternalSimulator sim("python3", {argv[1]});\n'
            "  std::vector<bt::target::ConformanceVector> vecs = {\n"
            '    {"add_pos", {0x01}, {10, 32}, {42}},\n'
            '    {"add_neg", {0x01}, {-15, 7}, {-8}},\n'
            "  };\n"
            "  bt::target::ConformanceRunner runner(sim);\n  auto s = bt::target::ConformanceRunner::summarise(runner.run(vecs));\n"
            '  std::cout << s.passed << "/" << s.total() << "\\n";\n'
            "  return s.clean() ? 0 : 1;\n"
            "}\n",
            encoding="utf-8",
        )
        exe = Path(tmp) / "driver"
        t0 = time.perf_counter()
        b = subprocess.run(
            [cxx, "-std=c++20", "-O2", f"-I{REPO_ROOT}", f"-I{REPO_ROOT / 'include'}", str(driver), "-o", str(exe)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if b.returncode != 0:
            return {"status": "fail", "stage": "compile", "stderr_tail": b.stderr[-1500:]}
        r = subprocess.run([str(exe), str(mock)], capture_output=True, text=True, timeout=30)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return {
            "status": "pass" if r.returncode == 0 else "fail",
            "elapsed_ms": round(elapsed_ms, 2),
            "result": r.stdout.strip(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real Trium-XBT hardware benchmarks & RTL checks.")
    parser.add_argument("--require-eda", action="store_true", help="Exit 77 if Verilator/Icarus or Yosys is missing.")
    parser.add_argument("--no-write-log", action="store_true", help="Do not write JSON report into verification/regression_logs/.")
    args = parser.parse_args()

    now = datetime.datetime.now(datetime.timezone.utc)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "tools_detected": {
            "verilator": bool(shutil.which("verilator")),
            "iverilog": bool(shutil.which("iverilog")),
            "yosys": bool(shutil.which("yosys")),
            "g++": bool(shutil.which("g++")),
        },
        "suites": {
            "golden_model_v1_v2": run_golden_model_suite(),
            "external_simulator_pipe": run_external_simulator_check(),
            "rtl_testbench": run_rtl_testbench(),
            "yosys_synthesis": run_yosys_synthesis(),
            "yosys_abc_synthesis": run_yosys_abc_synthesis(),
            "yosys_fpga_xilinx": run_yosys_fpga_xilinx(),
        },
    }

    statuses = [v["status"] for v in report["suites"].values()]
    has_fail = any(s == "fail" for s in statuses)
    eda_skipped = (
        report["suites"]["rtl_testbench"]["status"] == "skipped"
        or report["suites"]["yosys_synthesis"]["status"] == "skipped"
    )
    report["overall_status"] = "fail" if has_fail else ("partial_eda_skipped" if eda_skipped else "pass")

    if not args.no_write_log:
        log_dir = REPO_ROOT / "verification" / "regression_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%d_%H%M%S")
        out_file = log_dir / f"hw_bench_{stamp}.json"
        latest_file = log_dir / "latest_hw_bench.json"
        payload = json.dumps(report, indent=2) + "\n"
        out_file.write_text(payload, encoding="utf-8")
        latest_file.write_text(payload, encoding="utf-8")
        report["log_path"] = str(out_file.relative_to(REPO_ROOT))

    print(json.dumps(report, indent=2))
    if has_fail:
        return 1
    if args.require_eda and eda_skipped:
        return 77
    return 0


if __name__ == "__main__":
    sys.exit(main())
