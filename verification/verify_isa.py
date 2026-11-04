#!/usr/bin/env python3
"""Verify the RISC-V Xbt instruction set against balanced-ternary mathematics."""
import argparse, sys, time
from .xbt_isa import (build_isa, verify_pack_unpack_roundtrip,
                      verify_li5)
from .rocq_bridge import check_rocq_matches_rtl
from .engine import (verify_exhaustive, verify_symbolic,
                      VerificationReport, ProofResult)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=int, default=300_000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    isa = build_isa(args.width)
    report = VerificationReport()
    t0 = time.time()
    for op in isa:
        if args.only and op.name != args.only:
            continue
        if op.exhaustive:
            report.results.append(verify_exhaustive(isa, op, args.timeout))
        if op.symbolic:
            report.results.append(verify_symbolic(isa, op, args.timeout))

    if not args.quiet:
        print(f"Xbt instruction set verification  (width={args.width}, z3 "
              f"{__import__('z3').get_version_string()})")
        print("=" * 78)
        cur = None
        for r in report.results:
            if r.op != cur:
                cur = r.op
                print(f"\n  {cur}")
            mark = "PASS" if r.proved else "FAIL"
            print(f"    [{mark}] {r.strategy:<11} {r.detail}  ({r.seconds:.2f}s)")
            if not r.proved and r.counterexample:
                print(f"           counterexample: {r.counterexample}")
        print("\n" + "=" * 78)

    # Cross-operation property: the base-243 codec must round-trip.
    if not args.quiet:
        print("\n  PACK243/UNPACK243 round-trip (exhaustive)")
    rt = verify_pack_unpack_roundtrip(5)
    if not args.quiet:
        if rt:
            print(f"    [FAIL] {len(rt)} round-trip errors, first: {rt[0]}")
        else:
            print("    [PASS] pack(unpack(b))==b for all 243 bytes; "
                  "unpack(pack(v))==v for all 243 values")
    if rt:
        report.results.append(ProofResult("PACK243/UNPACK243", "roundtrip",
                                          False, f"{len(rt)} errors: {rt[0]}"))
    else:
        report.results.append(ProofResult("PACK243/UNPACK243", "roundtrip",
                                          True, "exhaustive over the 243-value domain"))

    # LI5: exhaustive over every byte value x every shift.
    if not args.quiet:
        print("\n  LI5 constant materialisation (exhaustive, 256 bytes x 8 chunks)")
    lerr = verify_li5()
    if not args.quiet:
        if lerr:
            print(f"    [FAIL] {len(lerr)} errors, first: {lerr[0]}")
        else:
            print("    [PASS] (byte-121)*3**shamt for all 243 valid bytes x 8 shifts; "
                  "13 invalid bytes rejected")
    report.results.append(ProofResult(
        "LI5", "roundtrip", not lerr,
        "exhaustive over 256 bytes x 8 chunks" if not lerr else f"{len(lerr)} errors: {lerr[0]}"))

    # Rocq <-> RTL: the gate networks the theorems are stated about must be
    # the ones the RTL actually implements.
    if not args.quiet:
        print("\n  Rocq <-> RTL gate-level bridge")
    bridge_errs = check_rocq_matches_rtl(verbose=not args.quiet)
    if not args.quiet:
        for e in bridge_errs[:10]:
            print(f"    [FAIL] {e}")
        if not bridge_errs:
            print("    [PASS] Rocq btha/rail_mul definitions match the RTL model")
    report.results.append(ProofResult(
        "Rocq/RTL bridge", "bridge", not bridge_errs,
        "gate definitions agree on all rail combinations" if not bridge_errs
        else f"{len(bridge_errs)} discrepancies: {bridge_errs[0]}"))

    n = len(report.results)
    f = len(report.failures)
    print(f"{n - f}/{n} proofs passed in {time.time()-t0:.2f}s")
    if report.failures:
        print("\nFAILED:")
        for r in report.failures:
            print(f"  {r.op}/{r.strategy}: {r.detail}")
    return 0 if report.ok else 1

if __name__ == "__main__":
    sys.exit(main())
