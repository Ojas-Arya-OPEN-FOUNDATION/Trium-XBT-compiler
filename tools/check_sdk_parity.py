#!/usr/bin/env python3
"""
SDK parity matrix and enforcement (docs/SCALING_PLAN.md 8, item 1 and 5).

The plan treats an asymmetry between the C, Python and Rust surfaces as a
bug, and requires "a CI check that fails when a capability exists in one
language but not another".

This script:
  1. extracts the public surface of each language binding
  2. maps them onto a canonical capability list
  3. emits docs/SDK_PARITY.md
  4. fails on any UNJUSTIFIED asymmetry

An asymmetry is justified only if it is listed in KNOWN_ASYMMETRIES below
with a reason. An unlisted gap is a failure -- otherwise the allow-list
quietly becomes the place where parity goes to die.

Run:  python3 tools/check_sdk_parity.py            # check + regenerate
      python3 tools/check_sdk_parity.py --check    # check only
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C_HEADER = ROOT / "include" / "Ternary" / "SDK" / "sdk.h"
PY_DIR = ROOT / "ffi" / "python" / "trium"
RS_DIR = ROOT / "ffi" / "rust" / "src"
OUT_DOC = ROOT / "docs" / "SDK_PARITY.md"


# ---------------------------------------------------------------------------
# Capability definitions: canonical name -> (C symbol, Python, Rust)
# A language entry is a tuple of alternatives; any match counts.
# ---------------------------------------------------------------------------

CAPABILITIES = {
    "context":        (("bt_sdk_context_create",), ("Context",), ("new",)),
    "compile":        (("bt_sdk_compile",), ("compile",), ("compile",)),
    "invoke":         (("bt_sdk_invoke", "bt_sdk_invoke_ex"), ("invoke",), ("invoke",)),
    "map":            (("bt_sdk_map",), ("map", "map_array"), ("map",)),
    "info":           (("bt_sdk_info",), ("info",), ("info",)),
    "module_info":    (("bt_sdk_module_info",), ("exports",), ("info",)),
    "matmul":         (("bt_sdk_matmul",), ("matmul",), ("matmul",)),
    "gemm_ternary":   (("bt_sdk_gemm_ternary",), ("gemm_ternary",), ("gemm_ternary",)),
    "pack_matrix_a":  ((None,), ("pack_matrix_a",), ("pack_matrix_a",)),
    "pack_matrix_b":  ((None,), ("pack_matrix_b",), ("pack_matrix_b",)),
    "encode_gemm":    ((None,), ("encode_gemm",), ("encode_gemm",)),
    "cache_info":     (("bt_sdk_cache_info",), ("cache_info",), ("cache_info",)),
    "cache_clear":    (("bt_sdk_cache_clear",), ("clear_cache",), ("clear_cache",)),
    "devices":        (("bt_sdk_devices",), ("devices",), (None,)),
    "shader":         (("bt_sdk_shader",), ("shader",), (None,)),
    "dual_encode":    (("bt_sdk_dual_encode",), ("encode",), (None,)),
    "dual_decode":    (("bt_sdk_dual_decode",), ("decode",), (None,)),
    "float_from_double": (("bt_sdk_float_from_double",), ("float_from_double",), ("from_f64",)),
    "float_to_double":   (("bt_sdk_float_to_double",), ("float_to_double",), ("to_f64",)),
    "float_apply":    (("bt_sdk_float_apply",), ("float_apply",), (None,)),
    "float_reduce":   (("bt_sdk_float_reduce",), ("float_reduce",), (None,)),
    "float_powi":     (("bt_sdk_float_powi",), ("float_powi",), ("pow_int",)),
    "float_from_ratio": (("bt_sdk_float_from_ratio",), ("float_from_ratio",), (None,)),
    "float_polynomial": (("bt_sdk_float_polynomial",), ("float_polynomial",), (None,)),
    "is_ternary":     ((None,), ("is_ternary",), ("is_ternary",)),
    "release":        (("bt_sdk_release",), ("close",), (None,)),
    "metrics":        (("bt_sdk_metrics",), ("Metrics",), (None,)),
}

# Asymmetries that are deliberate, with the reason. Anything NOT here is a
# failure. Review this list whenever a capability is added.
KNOWN_ASYMMETRIES = {
    ("pack_matrix_a", "C"):
        "matrix packing is a host-side convenience; the C surface exposes the "
        "raw GEMM and leaves packing to the caller",
    ("pack_matrix_b", "C"):
        "as pack_matrix_a",
    ("encode_gemm", "C"):
        "convenience helper that packs both GEMM operands at once so the "
        "layout is correct by construction; the C ABI exposes the raw "
        "gemm_ternary and leaves packing to the caller",
    ("devices", "Rust"):
        "device enumeration not yet ported to the Rust binding",
    ("shader", "Rust"):
        "shader emission not yet ported to the Rust binding",
    ("dual_encode", "Rust"):
        "dual-rail encode/decode is available through the ternary core "
        "functions (bt_swar_word64_to_int64) but not as a named SDK call",
    ("dual_decode", "Rust"):
        "as dual_encode",
    ("float_apply", "Rust"):
        "elementwise float application not yet ported",
    ("float_reduce", "Rust"):
        "float reduction not yet ported",
    ("float_from_ratio", "Rust"):
        "ratio constructor not yet ported",
    ("float_polynomial", "Rust"):
        "polynomial evaluation not yet ported",
    ("is_ternary", "C"):
        "predicate is a Python convenience for routing decisions; the C ABI "
        "takes an explicit element-width argument instead",
    ("release", "Rust"):
        "Rust releases through Drop; no explicit call needed or exposed",
    ("metrics", "C"):
        "metrics are surfaced through bt_sdk_cache_info and bt_sdk_info "
        "rather than a dedicated handle",
    ("metrics", "Rust"):
        "not yet ported",
}


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

def c_symbols() -> set[str]:
    if not C_HEADER.exists():
        return set()
    return set(re.findall(r"\bbt_sdk_[a-z0-9_]+\b", C_HEADER.read_text()))


def python_surface() -> set[str]:
    if not PY_DIR.exists():
        return set()
    out = set()
    for f in PY_DIR.glob("*.py"):
        text = f.read_text()
        out |= set(re.findall(r"^\s{0,4}def\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                              text, re.M))
        out |= set(re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)",
                              text, re.M))
    return out


def rust_surface() -> set[str]:
    if not RS_DIR.exists():
        return set()
    out = set()
    for f in RS_DIR.glob("*.rs"):
        text = f.read_text()
        out |= set(re.findall(r"\bpub fn\s+([a-z_][a-z0-9_]*)", text))
    return out


# ---------------------------------------------------------------------------

def build_matrix():
    cs, py, rs = c_symbols(), python_surface(), rust_surface()
    matrix = {}
    for cap, (c_alts, py_alts, rs_alts) in CAPABILITIES.items():
        matrix[cap] = {
            "C": any(a and a in cs for a in c_alts),
            "Python": any(a and a in py for a in py_alts),
            "Rust": any(a and a in rs for a in rs_alts),
        }
    return matrix, cs, py, rs


def find_violations(matrix):
    bad = []
    for cap, langs in sorted(matrix.items()):
        present = [l for l in ("C", "Python", "Rust") if langs[l]]
        if not present:
            continue                      # nowhere at all: not an asymmetry
        for lang in ("C", "Python", "Rust"):
            if langs[lang]:
                continue
            if (cap, lang) in KNOWN_ASYMMETRIES:
                continue
            bad.append((cap, lang, present))
    return bad


def render_doc(matrix, cs, py, rs, violations) -> str:
    L = []
    w = L.append
    w("# SDK parity matrix")
    w("")
    w("Generated by `tools/check_sdk_parity.py`. Do not edit by hand.")
    w("")
    w("docs/SCALING_PLAN.md 8 treats an asymmetry between the C, Python and "
      "Rust surfaces as a bug. This table is the parity matrix; CI fails on "
      "any asymmetry that is not explicitly justified below.")
    w("")
    w(f"- C surface: `{len(cs)}` symbols in `include/Ternary/SDK/sdk.h`")
    w(f"- Python: `{len(py)}` public names in `ffi/python/trium/`")
    w(f"- Rust: `{len(rs)}` `pub fn` in `ffi/rust/src/`")
    w("")

    w("## Matrix")
    w("")
    w("| Capability | C | Python | Rust |")
    w("|---|---|---|---|")
    for cap in sorted(matrix):
        row = matrix[cap]
        cells = []
        for lang in ("C", "Python", "Rust"):
            if row[lang]:
                cells.append("yes")
            elif (cap, lang) in KNOWN_ASYMMETRIES:
                cells.append("**no** \\*")
            else:
                cells.append("NO")
        w(f"| `{cap}` | " + " | ".join(cells) + " |")
    w("")
    w("\\* documented and justified below.")
    w("")

    w("## Justified asymmetries")
    w("")
    w("Each of these is a deliberate gap with a reason. Anything not on this "
      "list is a CI failure.")
    w("")
    w("| Capability | Missing from | Reason |")
    w("|---|---|---|")
    for (cap, lang), reason in sorted(KNOWN_ASYMMETRIES.items()):
        w(f"| `{cap}` | {lang} | {reason} |")
    w("")

    total = len(matrix)
    full = sum(1 for r in matrix.values() if all(r.values()))
    w("## Status")
    w("")
    w(f"- **{full} of {total}** capabilities are present in all three "
      f"languages.")
    w(f"- **{len(KNOWN_ASYMMETRIES)}** asymmetries are documented as "
      f"justified.")
    w(f"- **{len(violations)}** unjustified asymmetries "
      f"{'(CI FAILURE)' if violations else '(none)'}")
    w("")
    if violations:
        w("### Unjustified asymmetries")
        w("")
        for cap, lang, present in violations:
            w(f"- `{cap}` is present in {', '.join(present)} but missing "
              f"from **{lang}**, with no justification recorded.")
        w("")
    return "\n".join(L)


def main() -> int:
    check_only = "--check" in sys.argv

    matrix, cs, py, rs = build_matrix()
    violations = find_violations(matrix)

    if check_only:
        if violations:
            print("SDK parity: FAIL", file=sys.stderr)
            for cap, lang, present in violations:
                print(f"  {cap}: present in {', '.join(present)}, "
                      f"missing from {lang} (no justification recorded)",
                      file=sys.stderr)
            return 1
        print(f"SDK parity: OK ({len(matrix)} capabilities, "
              f"{len(KNOWN_ASYMMETRIES)} justified asymmetries)")
        return 0

    doc = render_doc(matrix, cs, py, rs, violations)
    OUT_DOC.write_text(doc)
    print(f"wrote {OUT_DOC.relative_to(ROOT)}")
    full = sum(1 for r in matrix.values() if all(r.values()))
    print(f"  {full}/{len(matrix)} capabilities in all three languages")
    print(f"  {len(KNOWN_ASYMMETRIES)} justified asymmetries")
    if violations:
        print(f"  {len(violations)} UNJUSTIFIED asymmetries (CI will fail):")
        for cap, lang, present in violations:
            print(f"    {cap} missing from {lang}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
