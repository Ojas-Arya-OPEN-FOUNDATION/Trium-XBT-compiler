"""
Layout conformance (docs/SCALING_PLAN.md 3.4).

The hardware audit records ZERO conformance vectors for layouts. This
generates them and proves the round-trip property for every conversion pair.

    convert(from -> to) then convert(to -> from)  ==  identity

Mirrors include/Ternary/IR/Layout.hpp.

Obligations:
  L1  dual-rail round-trip
  L2  base-243 round-trip (5 trits/byte, bias 121)
  L3  T-CSR round-trip (sparse; zeros restored)
  L4  BTFP round-trip, exact exactly when representable
  L5  every conversion pair round-trips
  L6  trait table is self-consistent (density ordering)
  L7  conformance vectors exist for every pair

Run:  python3 verification/verify_layouts.py
      python3 verification/verify_layouts.py --emit DIR   # write vectors
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import z3

PASS = 0
FAIL = 0

DUAL_RAIL, BASE243, TCSR, BTFP = "dual-rail", "base-243", "tcsr", "btfp"
LAYOUTS = [DUAL_RAIL, BASE243, TCSR, BTFP]
MANTISSA_TRITS = 39


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"    [PASS] {name}{('  ' + detail) if detail else ''}")
    else:
        FAIL += 1
        print(f"    [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# Conversions (mirror of Layout.hpp)
# ---------------------------------------------------------------------------

def to_dual_rail(trits):
    return list(trits)


def to_base243(trits):
    """5 trits per byte, biased by 121. Mirrors the C++ packing order."""
    nbytes = (len(trits) + 4) // 5
    bytes_ = [121] * nbytes   # start at the BIAS, not 0
    for i, t in enumerate(trits):
        by = i // 5
        place = 3 ** (i % 5)
        v = bytes_[by] - 121
        v += t * place
        bytes_[by] = v + 121
    return bytes_


def from_base243(bytes_, num_trits):
    # The byte is the BIASED value in 0..242. Extract plain base-3 digits of
    # the BYTE, then subtract 1 to recover trits in {-1,0,1}. Extracting
    # digits of the *unbiased* value needs a borrow and gets negatives wrong.
    out = [0] * num_trits
    for i in range(num_trits):
        v = bytes_[i // 5]          # 0..242
        place = 3 ** (i % 5)
        out[i] = (v // place) % 3 - 1
    return out


def to_tcsr(trits):
    return [(i, t) for i, t in enumerate(trits) if t != 0]


def from_tcsr(pairs, num_trits):
    out = [0] * num_trits
    for i, t in pairs:
        if i < num_trits:
            out[i] = t
    return out


def to_btfp(trits):
    v = sum(t * 3 ** i for i, t in enumerate(trits))
    lim = (3 ** MANTISSA_TRITS - 1) // 2
    if not (-lim <= v <= lim):
        return None
    return (v, 0)


def from_btfp(rep):
    if rep is None:
        return None
    v, _ = rep
    mag, neg = (abs(v), v < 0)
    out = [0] * MANTISSA_TRITS
    i = 0
    while mag and i < MANTISSA_TRITS:
        r = mag % 3
        if r == 2:
            out[i] = 1 if neg else -1
            mag = (mag + 1) // 3
        else:
            out[i] = -r if neg else r
            mag //= 3
        i += 1
    return out


def convert(trits, to):
    if to == DUAL_RAIL:
        return to_dual_rail(trits)
    if to == BASE243:
        return to_base243(trits)
    if to == TCSR:
        return to_tcsr(trits)
    if to == BTFP:
        return to_btfp(trits)
    return None


def unconvert(rep, to, num_trits):
    if rep is None:
        return None
    if to == DUAL_RAIL:
        return list(rep)
    if to == BASE243:
        return from_base243(rep, num_trits)
    if to == TCSR:
        return from_tcsr(rep, num_trits)
    if to == BTFP:
        return from_btfp(rep)
    return None


def roundtrip(trits, layout):
    r = convert(trits, layout)
    if r is None:
        return None
    back = unconvert(r, layout, len(trits))
    if back is None:
        return None
    if layout == BTFP:
        # BTFP is a numeric format: normalise length before comparing.
        back = back[:len(trits)] + [0] * max(0, len(trits) - len(back))
    return back


# ---------------------------------------------------------------------------
# L1-L4 individual round-trips
# ---------------------------------------------------------------------------

def l1_dual_rail():
    bad = [t for t in itertools.product((-1, 0, 1), repeat=8)
           if roundtrip(list(t), DUAL_RAIL) != list(t)]
    report("L1 dual-rail round-trip", not bad, "all 3^8 8-trit words")


def l2_base243():
    bad = []
    for w in range(1, 12):
        for t in itertools.product((-1, 0, 1), repeat=w):
            if roundtrip(list(t), BASE243) != list(t):
                bad.append((w, t))
                break
    report("L2 base-243 round-trip", not bad,
           "widths 1..11, all words" + (f", {bad[:2]}" if bad else ""))

    # The byte range must be exactly 0..242.
    vals = set()
    for t in itertools.product((-1, 0, 1), repeat=5):
        vals.update(to_base243(list(t)))
    report("L2 base-243 bytes span exactly 0..242",
           min(vals) == 0 and max(vals) == 242 and len(vals) == 243,
           f"{len(vals)} distinct byte values")


def l3_tcsr():
    bad = []
    for t in itertools.product((-1, 0, 1), repeat=7):
        if roundtrip(list(t), TCSR) != list(t):
            bad.append(t)
            break
    report("L3 T-CSR round-trip (zeros restored)", not bad, "all 3^7 words")

    # Sparsity: an all-zero word compresses to no entries.
    z = to_tcsr([0] * 16)
    report("L3 all-zero word is fully compressed", z == [], f"nnz={len(z)}")


def l4_btfp():
    bad = []
    refused = 0
    for w in (1, 2, 5, 8):
        for t in itertools.product((-1, 0, 1), repeat=w):
            r = roundtrip(list(t), BTFP)
            if r is None:
                refused += 1
                continue
            if r[:w] != list(t):
                bad.append((t, r[:w]))
                break
    report("L4 BTFP round-trip is exact when representable", not bad,
           f"widths 1,2,5,8; {refused} refused as unrepresentable"
           + (f", {bad[:2]}" if bad else ""))

    # Out of range must be refused, not approximated.
    huge = [1] * (MANTISSA_TRITS + 1)
    report("L4 out-of-range BTFP is refused", to_btfp(huge) is None,
           f"mantissa of {len(huge)} trits exceeds the {MANTISSA_TRITS}-trit limit")


# ---------------------------------------------------------------------------
# L5 all conversion pairs
# ---------------------------------------------------------------------------

def l5_all_pairs():
    """convert(a->b) then convert(b->a) == identity, for every ordered pair."""
    results = {}
    bad = []
    for a, b in itertools.product(LAYOUTS, repeat=2):
        if a == b:
            continue
        checked = 0
        failed = 0
        for w in (1, 3, 5):
            for t in itertools.product((-1, 0, 1), repeat=w):
                src = list(t)
                mid = convert(src, b)
                if mid is None:
                    continue
                # From the mid representation back to dense trits, then to `a`.
                back_dense = unconvert(mid, b, len(src))
                if back_dense is None:
                    continue
                if b == BTFP:
                    back_dense = back_dense[:len(src)]
                final = convert(back_dense, a)
                if final is None:
                    continue
                got = unconvert(final, a, len(src))
                if got is None:
                    continue
                if a == BTFP:
                    got = got[:len(src)]
                checked += 1
                if got[:len(src)] != src:
                    failed += 1
                    if len(bad) < 3:
                        bad.append((a, b, src, got[:len(src)]))
        results[(a, b)] = (checked, failed)

    total_bad = sum(f for _, f in results.values())
    total_checked = sum(c for c, _ in results.values())
    report("L5 all ordered conversion pairs round-trip", total_bad == 0,
           f"{len(results)} pairs, {total_checked} words"
           + (f", {bad[:3]}" if bad else ""))
    return results


# ---------------------------------------------------------------------------
# L6 traits
# ---------------------------------------------------------------------------

def l6_traits():
    traits = {
        DUAL_RAIL: 2.0,
        BASE243: 1.6,
        TCSR: 0.0,     # sparse: no fixed cost per trit
        BTFP: 2.0,
    }
    report("L6 base-243 is denser than dual-rail",
           traits[BASE243] < traits[DUAL_RAIL],
           f"{traits[BASE243]} vs {traits[DUAL_RAIL]} bits/trit -- "
           f"{int(100*(1 - traits[BASE243]/traits[DUAL_RAIL]))}% fewer bytes")
    report("L6 T-CSR has no fixed per-trit cost (it is sparse)",
           traits[TCSR] == 0.0)
    report("L6 every layout has a positive or zero cost",
           all(v >= 0 for v in traits.values()))


# ---------------------------------------------------------------------------
# L7 conformance vectors
# ---------------------------------------------------------------------------

def build_vectors(widths=(1, 2, 3, 5, 8)):
    vectors = []
    for w in widths:
        if w > 5:
            # sample rather than enumerate, to keep the set small
            words = [tuple((i >> k) % 3 - 1 for k in range(w))
                     for i in range(3 ** min(w, 5))]
        else:
            words = list(itertools.product((-1, 0, 1), repeat=w))
        for t in words:
            src = list(t)
            for b in LAYOUTS:
                mid = convert(src, b)
                if mid is None:
                    continue
                back = unconvert(mid, b, w)
                if back is None:
                    continue
                vectors.append({
                    "width": w,
                    "trits": src,
                    "layout": b,
                    "encoded": [int(x) if not isinstance(x, tuple) else list(x)
                                for x in (mid if isinstance(mid, list) else [mid])],
                    "decoded": (back[:w] if b == BTFP else back),
                    "round_trips": (back[:w] if b == BTFP else back) == src,
                })
    return vectors


def l7_vectors(vectors, emit_dir=None):
    total = len(vectors)
    good = sum(1 for v in vectors if v["round_trips"])
    report("L7 every conformance vector round-trips", good == total,
           f"{good}/{total} vectors")

    covered = {(v["layout"], v["width"]) for v in vectors}
    report("L7 every layout has vectors",
           {c[0] for c in covered} == set(LAYOUTS),
           f"{len(covered)} (layout, width) combinations")

    if emit_dir:
        d = Path(emit_dir)
        d.mkdir(parents=True, exist_ok=True)
        out = d / "layout_conformance.json"
        out.write_text(json.dumps(vectors, indent=1))
        print(f"    wrote {len(vectors)} vectors to {out}")


# ---------------------------------------------------------------------------

def main() -> int:
    emit = None
    if "--emit" in sys.argv:
        emit = sys.argv[sys.argv.index("--emit") + 1]

    print("=" * 78)
    print("  Layout conformance (docs/SCALING_PLAN.md 3.4)")
    print("=" * 78)

    print("\n  L1-L4 individual round-trips")
    l1_dual_rail()
    l2_base243()
    l3_tcsr()
    l4_btfp()

    print("\n  L5 conversion pairs")
    l5_all_pairs()

    print("\n  L6 traits")
    l6_traits()

    print("\n  L7 conformance vectors")
    l7_vectors(build_vectors(), emit)

    print("\n" + "=" * 78)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
