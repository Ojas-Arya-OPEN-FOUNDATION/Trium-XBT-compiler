"""
rebel6_comparison_bench.py -- Analytical & Micro-Decode Comparison: USN REBEL-6 vs Trium-XBT.

Clearly separates:
  1. Literature baseline figures published by Soppe/RV32IToREBEL (not executed here).
  2. Analytical 768-trit dot-product instruction-count ratio (scalar loop vs 64-trit DOT64).
  3. Live Python golden-model verification & 243-word decode timing.
"""

from __future__ import annotations

import time
from typing import Any, Dict
from verification.golden_model.xbt_golden import xbt_base243_decoder, xbt_planar_dot64


def count_ternary_tritflips_24(prev_val: int, next_val: int) -> int:
    def to_trits(v: int):
        trits = [0] * 24
        neg = v < 0
        v = abs(v)
        idx = 0
        while v != 0 and idx < 24:
            r = v % 3
            if r == 1:
                val = 1
            elif r == 2:
                val = -1
                v += 1
            else:
                val = 0
            trits[idx] = -val if neg else val
            v //= 3
            idx += 1
        return trits

    return sum(1 for a, b in zip(to_trits(prev_val), to_trits(next_val)) if a != b)


def run_rebel6_comparison() -> Dict[str, Any]:
    print("=" * 80)
    print("=== USN REBEL-6 vs. Trium-XBT (Literature Baselines + Analytical Ratio + Decode) ===")
    print("=" * 80)

    # 1. Published literature baselines (Soppe/RV32IToREBEL) -- NOT measured locally
    literature_baseline = {
        "source": "Soppe/RV32IToREBEL (published baseline, not executed locally)",
        "rv32ui_rv32i_insns": 10_688,
        "rv32ui_rebel6_insns": 9_359,
        "rv32ui_rv32i_flips": 40_600,
        "rv32ui_rebel6_flips": 19_425,
        "dhry_rv32i_insns": 2_408_092,
        "dhry_rebel6_insns": 2_404_087,
        "dhry_rv32i_flips": 9_488_104,
        "dhry_rebel6_flips": 6_489_886,
    }

    # 2. Analytical 768-trit dot-product instruction count ratio + live DOT64 functional check
    rebel6_dot768_insns_est = 768 * 6
    trium_dot768_insns_est = (768 // 64) * 2
    insn_ratio = rebel6_dot768_insns_est / trium_dot768_insns_est

    ap, an = 0xAAAAAAAAAAAAAAAA, 0x5555555555555555
    bp, bn = 0xAAAAAAAAAAAAAAAA, 0x5555555555555555
    acc = sum(xbt_planar_dot64(ap, an, bp, bn) for _ in range(12))
    assert acc == 768, f"Expected 768, got {acc}"

    # 3. Live Python decode comparison (243 words)
    t0 = time.perf_counter()
    for b in range(243):
        _ = count_ternary_tritflips_24(0, b - 121)
    rebel_decode_us = (time.perf_counter() - t0) * 1e6

    t1 = time.perf_counter()
    for b in range(243):
        p, n = xbt_base243_decoder(b)
        assert (p & n) == 0
    trium_decode_us = (time.perf_counter() - t1) * 1e6

    print(f"  [LITERATURE]  USN REBEL-6 baseline (not run here): RV32UI {literature_baseline['rv32ui_rv32i_insns']:,}->{literature_baseline['rv32ui_rebel6_insns']:,} insns")
    print(f"  [ANALYTICAL]  768-trit dot ratio: {rebel6_dot768_insns_est:,} scalar vs {trium_dot768_insns_est} DOT64+ADD ({insn_ratio:.1f}x analytical)")
    print(f"  [MEASURED]    243-word Python decode: div/mod-3 {rebel_decode_us:.1f} us vs Base-243 reciprocal {trium_decode_us:.1f} us")
    return {
        "mode": "literature_plus_analytical_plus_python_decode",
        "literature_baseline": literature_baseline,
        "dot768_analytical_ratio": insn_ratio,
        "decode_243_divmod_us": round(rebel_decode_us, 2),
        "decode_243_base243_us": round(trium_decode_us, 2),
        "status": "PASS",
    }


if __name__ == "__main__":
    run_rebel6_comparison()
