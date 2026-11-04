#!/usr/bin/env python3
"""
Generate conformance vectors for the RISC-V Xbt instruction set.

A conformance vector is an input/expected-output pair that ANY implementation
must reproduce: a future backend, an RTL simulation, an FPGA, or a physical
ternary chip. This is the contract that lets hardware be qualified against
the compiler without reading the compiler.

Vectors are derived from the *mathematical specification*, never from the
hardware model -- otherwise a bug in the hardware would simply be codified
into the vectors. The specification is the one verified in verify_isa.py.

Outputs
-------
  vectors/xbt_conformance.json   machine-readable, for external tools
  vectors/xbt_conformance.hpp    C++ table, compiled into the test suite

Coverage
--------
  exhaustive   every input combination, where the space is small enough
               (BTHA 9, BTFA 27, CSAMAC 27, NEG 3, UNPACK243 256 byte values)
  adversarial  boundary and structured patterns for the 64-trit word
               operations, whose input space (3**128) cannot be enumerated
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .z3core import (
    TRITS, trit_add, trit_add3, trit_neg, trit_mul,
    word_value, encode_word, balanced_range,
    WIDTH,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "vectors"

M64 = (1 << 64) - 1


def _u64(v: int) -> int:
    return v & M64


def _rails_of(trits) -> tuple[int, int]:
    P = N = 0
    for i, t in enumerate(trits):
        if t == 1:
            P |= 1 << i
        elif t == -1:
            N |= 1 << i
    return P, N


def _patterns(width: int) -> list[list[int]]:
    """Structured and boundary trit words for the word-level operations."""
    lo, hi = balanced_range(width)
    pats: list[list[int]] = []
    # all-zero, all-plus, all-minus
    pats.append([0] * width)
    pats.append([1] * width)
    pats.append([-1] * width)
    # alternating patterns
    pats.append([1 if i % 2 == 0 else -1 for i in range(width)])
    pats.append([-1 if i % 2 == 0 else 1 for i in range(width)])
    # single-trit at each end and in the middle
    for pos in (0, 1, width // 2, width - 2, width - 1):
        for v in (1, -1):
            w = [0] * width
            w[pos] = v
            pats.append(w)
    # numeric boundaries
    for v in (lo, lo + 1, -1, 0, 1, hi - 1, hi):
        pats.append(encode_word(v, width))
    # carry-chain stress: long runs that force carries to propagate
    pats.append([1] * (width - 1) + [0])
    pats.append([-1] * (width - 1) + [0])
    pats.append([1] * 32 + [-1] * 32)
    pats.append([-1] * 32 + [1] * 32)
    # powers of three
    for e in range(0, width, max(1, width // 16)):
        pats.append(encode_word(3 ** e, width))
        pats.append(encode_word(-(3 ** e), width))
    return pats


def _randoms(width: int, n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    lo, hi = balanced_range(width)
    out = []
    for _ in range(n):
        out.append(encode_word(rng.randint(lo, hi), width))
    return out


def build_vectors(width: int = WIDTH, n_random: int = 200,
                  seed: int = 20261104) -> dict[str, Any]:
    rng = random.Random(seed)
    vec: dict[str, Any] = {
        "isa": "riscv-xbt",
        "opcode": "0x0B (custom-0)",
        "width": width,
        "note": "Expected values come from the verified trit-level "
                "specification, not from any hardware implementation.",
        "groups": [],
    }

    def group(name, mode, cases, fields):
        vec["groups"].append({
            "op": name, "mode": mode, "count": len(cases),
            "fields": fields, "cases": cases,
        })

    # ---- BTHA : exhaustive over 9 trit pairs ------------------------------
    cases = []
    for a, b in [(x, y) for x in TRITS for y in TRITS]:
        s, c = trit_add(a, b)
        cases.append({"a": a, "b": b, "sum": s, "carry": c})
    group("BTHA", "exhaustive", cases, ["a", "b", "sum", "carry"])

    # ---- BTFA : exhaustive over 27 trit triples ---------------------------
    cases = []
    for a in TRITS:
        for b in TRITS:
            for d in TRITS:
                s, c = trit_add3(a, b, d)
                cases.append({"a": a, "b": b, "d": d, "sum": s, "carry": c})
    group("BTFA", "exhaustive", cases, ["a", "b", "d", "sum", "carry"])

    # ---- NEG : exhaustive over 3 trits ------------------------------------
    cases = [{"a": a, "neg": trit_neg(a)} for a in TRITS]
    group("NEG", "exhaustive", cases, ["a", "neg"])

    # ---- CSAMAC : exhaustive over 27 trit triples -------------------------
    # Documented contract: rd + (rs1 * rs2) == sum + 3*carry
    cases = []
    for a in TRITS:
        for b in TRITS:
            for rd in TRITS:
                s, c = trit_add3(rd, trit_mul(a, b), 0)
                cases.append({"rs1": a, "rs2": b, "rd": rd, "sum": s, "carry": c})
    group("CSAMAC", "exhaustive", cases, ["rs1", "rs2", "rd", "sum", "carry"])

    # ---- UNPACK243 : all 256 byte values ----------------------------------
    cases = []
    for b in range(256):
        if b >= 243:
            cases.append({"byte": b, "valid": False, "value": None})
        else:
            cases.append({"byte": b, "valid": True, "value": b - 121})
    group("UNPACK243", "exhaustive", cases, ["byte", "valid", "value"])

    # ---- KSADD : structured + random word pairs ---------------------------
    pats = _patterns(width)
    rnds = _randoms(width, n_random, seed)
    cases = []
    P = 3 ** width
    half = (P - 1) // 2
    for aw in pats + rnds:
        for bw in rnds[:40] + pats[:8]:
            A, B = word_value(aw), word_value(bw)
            T = A + B
            s = ((T + half) % P) - half
            cout = (T - s) // P
            ap, an = _rails_of(aw)
            bp, bn = _rails_of(bw)
            sp, sn = _rails_of(encode_word(s, width))
            cases.append({"a_p": ap, "a_n": an, "b_p": bp, "b_n": bn,
                          "sum_p": sp, "sum_n": sn, "cout": cout})
    group("KSADD", "adversarial", cases,
          ["a_p", "a_n", "b_p", "b_n", "sum_p", "sum_n", "cout"])

    # ---- DOT64 : structured + random word pairs ---------------------------
    cases = []
    for aw in pats + rnds:
        for bw in rnds[:40] + pats[:8]:
            dot = sum(t * u for t, u in zip(aw, bw))
            ap, an = _rails_of(aw)
            bp, bn = _rails_of(bw)
            cases.append({"a_p": ap, "a_n": an, "b_p": bp, "b_n": bn, "acc": dot})
    group("DOT64", "adversarial", cases, ["a_p", "a_n", "b_p", "b_n", "acc"])

    # ---- word-level NEG ---------------------------------------------------
    cases = []
    for aw in pats + rnds[:60]:
        ap, an = _rails_of(aw)
        cases.append({"a_p": ap, "a_n": an, "neg_p": an, "neg_n": ap})
    group("NEG_WORD", "adversarial", cases, ["a_p", "a_n", "neg_p", "neg_n"])

    # ---- word-level BTHA / BTFA / CSAMAC ----------------------------------
    for name, fn in (("BTHA_WORD", lambda a, b: trit_add(a, b)),
                     ("BTFA_WORD", lambda a, b: trit_add3(a, b, 0)),
                     ("CSAMAC_WORD", lambda a, b: trit_add3(0, trit_mul(a, b), 0))):
        cases = []
        for aw in pats + rnds[:60]:
            for bw in rnds[:20] + pats[:6]:
                sp, sn = 0, 0
                cp, cn = 0, 0
                for i in range(width):
                    s, c = fn(aw[i], bw[i])
                    if s == 1:
                        sp |= 1 << i
                    elif s == -1:
                        sn |= 1 << i
                    if c == 1:
                        cp |= 1 << i
                    elif c == -1:
                        cn |= 1 << i
                ap, an = _rails_of(aw)
                bp, bn = _rails_of(bw)
                cases.append({"a_p": ap, "a_n": an, "b_p": bp, "b_n": bn,
                              "sum_p": sp, "sum_n": sn,
                              "carry_p": cp, "carry_n": cn})
        group(name, "adversarial", cases,
              ["a_p", "a_n", "b_p", "b_n", "sum_p", "sum_n", "carry_p", "carry_n"])

    return vec


def write_json(vec: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "xbt_conformance.json"
    p.write_text(json.dumps(vec, indent=1))
    return p


def write_header(vec: dict[str, Any]) -> Path:
    """Emit a self-contained C++ table for the test suite."""
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "// Generated by verification/gen_vectors.py -- DO NOT EDIT.",
        "//",
        "// Conformance vectors for the RISC-V \"Xbt\" custom-0 extension.",
        "// Expected values are derived from the verified balanced-ternary",
        "// specification, never from the hardware model.",
        "//",
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        "namespace bt::hw::conformance {",
        "",
    ]
    for g in vec["groups"]:
        name = g["op"]
        fields = g["fields"]
        lines.append(f"// ---- {name}  ({g['mode']}, {g['count']} cases) ----")
        lines.append(f"struct {name}Case {{")
        for f in fields:
            if f == "valid":
                lines.append("    bool valid;")
            elif f.endswith("_p") or f.endswith("_n"):
                # dual-rail masks are unsigned 64-bit bit fields
                lines.append(f"    uint64_t {f};")
            else:
                lines.append(f"    int64_t {f};")
        lines.append("};")
        lines.append(f"inline const {name}Case* {name}_cases() {{")
        lines.append(f"    static const {name}Case k[] = {{")
        for c in g["cases"]:
            vals = []
            for f in fields:
                v = c[f]
                if f == "valid":
                    vals.append("true" if v else "false")
                elif v is None:
                    vals.append("0")
                elif f.endswith("_p") or f.endswith("_n"):
                    vals.append(f"{int(v) & M64}ULL")
                else:
                    vals.append(f"{int(v)}LL")
            lines.append("        {" + ", ".join(vals) + "},")
        lines.append("    };")
        lines.append("    return k;")
        lines.append("}")
        lines.append(f"inline constexpr size_t {name}_count() {{ return {g['count']}; }}")
        lines.append("")
    lines.append("} // namespace bt::hw::conformance")
    p = OUT / "xbt_conformance.hpp"
    p.write_text("\n".join(lines) + "\n")
    return p


def main() -> int:
    vec = build_vectors()
    j = write_json(vec)
    h = write_header(vec)
    total = sum(g["count"] for g in vec["groups"])
    print(f"wrote {j}")
    print(f"wrote {h}")
    for g in vec["groups"]:
        print(f"  {g['op']:<12} {g['mode']:<12} {g['count']:>7} cases")
    print(f"total: {total} vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
