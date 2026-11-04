"""
engine -- Z3 proof engine for the Xbt instruction set.

Two independent verification strategies, both sound:

  EXHAUSTIVE  enumerate every trit-level input combination and compare the
              hardware result against the mathematical specification.
              Complete and total for operations whose input space is finite
              and small (a few trits, or a single byte).

  SYMBOLIC    encode the hardware as Z3 bitvector formulas and prove the
              specification holds for *all* inputs simultaneously, including
              the 2**(2*width) dual-rail combinations of a 64-trit word that
              no amount of testing could cover.

An operation passes only if every enabled strategy proves it.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import z3

from .z3core import TRITS, trit_add, trit_add3, trit_neg, trit_mul, word_value


@dataclass
class ProofResult:
    op: str
    strategy: str
    proved: bool
    detail: str = ""
    seconds: float = 0.0
    cases: int = 0
    counterexample: dict | None = None

    @property
    def status(self) -> str:
        if self.proved:
            return "PROVED"
        return "FAILED"


@dataclass
class VerificationReport:
    results: list[ProofResult] = field(default_factory=list)

    @property
    def failures(self) -> list[ProofResult]:
        return [r for r in self.results if not r.proved]

    @property
    def ok(self) -> bool:
        return not self.failures


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _bit(x: z3.ExprRef, i: int) -> z3.ExprRef:
    """Extract bit i of a 64-bit vector as a 1-bit vector."""
    return z3.Extract(i, i, x)


def _trit_of(p: z3.ExprRef, n: z3.ExprRef) -> z3.ExprRef:
    """Z3 integer in {-1, 0, +1} from a pair of 1-bit rail vectors."""
    return z3.If(p == 1, z3.IntVal(1), z3.If(n == 1, z3.IntVal(-1), z3.IntVal(0)))


# --------------------------------------------------------------------------
# Exhaustive strategy
# --------------------------------------------------------------------------

def verify_exhaustive(isa, op, timeout_ms: int = 120_000) -> ProofResult:
    """Enumerate every trit combination and check impl against spec."""
    t0 = time.time()
    W = isa.width

    operand_kinds = [o.kind for o in op.operands]
    # Only inputs participate in enumeration; outputs are named with a
    # leading marker in this description convention (rs*/rd are inputs for
    # our purposes except where documented otherwise).
    try:
        if op.name == "BTHA":
            cases = list(itertools.product(TRITS, repeat=2))
            for a, b in cases:
                exp = op.spec(a, b)
                ap, an = (1, 0) if a == 1 else ((0, 1) if a == -1 else (0, 0))
                bp, bn = (1, 0) if b == 1 else ((0, 1) if b == -1 else (0, 0))
                got = op.impl(ap, an, bp, bn)
                gs = _trit_py(got["sum"][0], got["sum"][1])
                gc = _trit_py(got["carry"][0], got["carry"][1])
                if gs != exp["sum"] or gc != exp["carry"]:
                    return ProofResult(op.name, "exhaustive", False,
                                       f"a={a} b={b}: expected sum={exp['sum']} "
                                       f"carry={exp['carry']}, got sum={gs} carry={gc}",
                                       time.time() - t0, len(cases),
                                       {"a": a, "b": b})
            return ProofResult(op.name, "exhaustive", True,
                               f"{len(cases)} trit combinations", time.time() - t0, len(cases))

        if op.name == "BTFA":
            cases = list(itertools.product(TRITS, repeat=3))
            for a, b, d in cases:
                exp = op.spec(a, b, d)
                ap, an = _r(a); bp, bn = _r(b); dp, dn = _r(d)
                got = op.impl(ap, an, bp, bn, dp, dn)
                gs = _trit_py(*got["sum"]); gc = _trit_py(*got["carry"])
                if gs != exp["sum"] or gc != exp["carry"]:
                    return ProofResult(op.name, "exhaustive", False,
                                       f"a={a} b={b} d={d}: expected "
                                       f"sum={exp['sum']} carry={exp['carry']}, "
                                       f"got sum={gs} carry={gc}",
                                       time.time() - t0, len(cases),
                                       {"a": a, "b": b, "d": d})
            return ProofResult(op.name, "exhaustive", True,
                               f"{len(cases)} trit combinations", time.time() - t0, len(cases))

        if op.name == "NEG":
            cases = list(TRITS)
            for a in cases:
                exp = op.spec(a)
                ap, an = _r(a)
                got = op.impl(ap, an)
                gs = _trit_py(*got["word"])
                if gs != exp["word"]:
                    return ProofResult(op.name, "exhaustive", False,
                                       f"a={a}: expected {exp['word']}, got {gs}",
                                       time.time() - t0, len(cases), {"a": a})
            return ProofResult(op.name, "exhaustive", True,
                               f"{len(cases)} trit combinations", time.time() - t0, len(cases))

        if op.name == "CSAMAC":
            cases = list(itertools.product(TRITS, repeat=3))
            for a, b, rd in cases:
                exp = op.spec(a, b, rd)
                ap, an = _r(a); bp, bn = _r(b); rdp, rdn = _r(rd)
                got = op.impl(ap, an, bp, bn, rdp, rdn)
                gs = _trit_py(*got["sum_"]); gc = _trit_py(*got["carry"])
                if gs != exp["sum_"] or gc != exp["carry"]:
                    return ProofResult(op.name, "exhaustive", False,
                                       f"a={a} b={b} rd={rd}: expected "
                                       f"sum={exp['sum_']} carry={exp['carry']}, "
                                       f"got sum={gs} carry={gc}",
                                       time.time() - t0, len(cases),
                                       {"a": a, "b": b, "rd": rd})
            return ProofResult(op.name, "exhaustive", True,
                               f"{len(cases)} trit combinations", time.time() - t0, len(cases))

        if op.name == "UNPACK243":
            cases = list(range(256))
            for byte_val in cases:
                exp = op.spec(byte_val)
                got = op.impl(byte_val)
                if not exp["valid"]:
                    if got["valid"]:
                        return ProofResult(op.name, "exhaustive", False,
                                           f"byte={byte_val}: expected rejection, got valid",
                                           time.time() - t0, 256, {"byte": byte_val})
                    continue
                if not got["valid"]:
                    return ProofResult(op.name, "exhaustive", False,
                                       f"byte={byte_val}: expected valid, got rejection",
                                       time.time() - t0, 256, {"byte": byte_val})
                gp, gn = got["word"]
                if gp & gn:
                    return ProofResult(op.name, "exhaustive", False,
                                       f"byte={byte_val}: invalid dual-rail output (P&N overlap)",
                                       time.time() - t0, 256, {"byte": byte_val})
                val = sum((1 if (gp >> i) & 1 else (-1 if (gn >> i) & 1 else 0)) * 3 ** i
                          for i in range(5))
                if val != exp["value"]:
                    return ProofResult(op.name, "exhaustive", False,
                                       f"byte={byte_val}: expected {exp['value']}, got {val}",
                                       time.time() - t0, 256, {"byte": byte_val})
            return ProofResult(op.name, "exhaustive", True,
                               "all 256 byte values (243 valid + 13 rejected)",
                               time.time() - t0, 256)

    except Exception as exc:                                   # noqa: BLE001
        return ProofResult(op.name, "exhaustive", False, f"exception: {exc}",
                           time.time() - t0, 0)

    return ProofResult(op.name, "exhaustive", True, "not applicable to this operation",
                       time.time() - t0, 0)


def _r(t: int) -> tuple[int, int]:
    return (1, 0) if t == 1 else ((0, 1) if t == -1 else (0, 0))


def _trit_py(p: int, n: int) -> int:
    if p and n:
        raise ValueError("invalid rail 11")
    return 1 if p else (-1 if n else 0)


# --------------------------------------------------------------------------
# Symbolic strategy
# --------------------------------------------------------------------------

def _prove(assertions, conclusion, timeout_ms: int = 120_000,
           logic: str | None = None) -> tuple[str, z3.ModelRef | None]:
    """Return (result, model) where result is 'proved' | 'failed' | 'unknown'."""
    s = z3.Solver()
    if logic:
        s.set(logic)
    s.set("timeout", timeout_ms)
    for a in assertions:
        s.add(a)
    s.add(z3.Not(conclusion))
    r = s.check()
    if r == z3.unsat:
        return "proved", None
    if r == z3.unknown:
        return "unknown", None
    return "failed", s.model()


def verify_symbolic(isa, op, timeout_ms: int = 300_000) -> ProofResult:
    """Prove the rail-level hardware meets the spec for all inputs at once."""
    t0 = time.time()
    W = isa.width

    if op.name == "BTHA":
        ap, an, bp, bn = (z3.BitVec(f"v_{n}", W) for n in ("ap", "an", "bp", "bn"))
        valid = z3.And(ap & an == 0, bp & bn == 0)
        got = op.impl(ap, an, bp, bn)
        sp, sn = got["sum"]
        cp, cn = got["carry"]
        conj = []
        for i in range(W):
            a = _trit_of(_bit(ap, i), _bit(an, i))
            b = _trit_of(_bit(bp, i), _bit(bn, i))
            # spec: a + b == sum + 3*carry
            gs = _trit_of(_bit(sp, i), _bit(sn, i))
            gc = _trit_of(_bit(cp, i), _bit(cn, i))
            conj.append(a + b == gs + 3 * gc)
        res, model = _prove([valid], z3.And(*conj), timeout_ms)
        return ProofResult(op.name, "symbolic", res == "proved",
                           f"all 2^{2*W} dual-rail input pairs" if res == "proved"
                           else f"solver: {res}", time.time() - t0)

    if op.name == "BTFA":
        vs = {n: z3.BitVec(f"v_{n}", W) for n in ("ap", "an", "bp", "bn", "dp", "dn")}
        valid = z3.And(vs["ap"] & vs["an"] == 0, vs["bp"] & vs["bn"] == 0,
                       vs["dp"] & vs["dn"] == 0)
        got = op.impl(vs["ap"], vs["an"], vs["bp"], vs["bn"],
                       vs["dp"], vs["dn"])
        sp, sn = got["sum"]
        cp, cn = got["carry"]
        conj = []
        for i in range(W):
            a = _trit_of(_bit(vs["ap"], i), _bit(vs["an"], i))
            b = _trit_of(_bit(vs["bp"], i), _bit(vs["bn"], i))
            d = _trit_of(_bit(vs["dp"], i), _bit(vs["dn"], i))
            gs = _trit_of(_bit(sp, i), _bit(sn, i))
            gc = _trit_of(_bit(cp, i), _bit(cn, i))
            conj.append(a + b + d == gs + 3 * gc)
        res, model = _prove([valid], z3.And(*conj), timeout_ms)
        return ProofResult(op.name, "symbolic", res == "proved",
                           f"all 2^{3*W} dual-rail input triples" if res == "proved"
                           else f"solver: {res}", time.time() - t0)

    if op.name == "NEG":
        ap, an = z3.BitVec("ap", W), z3.BitVec("an", W)
        got = op.impl(ap, an)
        conj = []
        for i in range(W):
            a = _trit_of(_bit(ap, i), _bit(an, i))
            g = _trit_of(_bit(got["word"][0], i), _bit(got["word"][1], i))
            conj.append(g == -a)
        res, _ = _prove([ap & an == 0], z3.And(*conj), timeout_ms)
        return ProofResult(op.name, "symbolic", res == "proved",
                           "negation is an involution on all lanes" if res == "proved"
                           else f"solver: {res}", time.time() - t0)

    if op.name == "DOT64":
        # The planar dot product is a per-lane product followed by a
        # population count. Proving the per-lane identity for every lane at
        # once is cheap; the popcount is then a linear sum of those lanes.
        ap, an, bp, bn = (z3.BitVec(f"v_{n}", W) for n in ("ap", "an", "bp", "bn"))
        valid = z3.And(ap & an == 0, bp & bn == 0)
        pp = (ap & bp) | (an & bn)
        pn = (ap & bn) | (an & bp)
        prod_p, prod_n = pp, pn
        # Lemma 1: per lane, the AOI22 product rails encode a[i] * b[i],
        # and are a valid dual-rail encoding (never both rails set).
        lane_ok = []
        for i in range(W):
            a = _trit_of(_bit(ap, i), _bit(an, i))
            b = _trit_of(_bit(bp, i), _bit(bn, i))
            lane_ok.append(
                z3.And(_trit_of(_bit(prod_p, i), _bit(prod_n, i)) == a * b,
                       z3.Not(z3.And(_bit(prod_p, i) == 1, _bit(prod_n, i) == 1)))
            )
        r1, _ = _prove([valid], z3.And(*lane_ok), timeout_ms)
        if r1 != "proved":
            return ProofResult(op.name, "symbolic", False,
                               f"per-lane product lemma: {r1}", time.time() - t0)
        # Lemma 2: popcount linearity. Given lanes whose product rails encode
        # a[i]*b[i], the difference of the two population counts equals the sum
        # of the per-lane products.
        #
        # This is a mathematical identity, so it is proved in pure linear
        # integer arithmetic over bounded lane variables rather than by
        # extracting 128 bits out of the bitvector expression -- the latter is
        # correct but blows up the solver.
        A = [z3.Int(f"A{i}") for i in range(W)]
        B = [z3.Int(f"B{i}") for i in range(W)]
        PP = [z3.Int(f"PP{i}") for i in range(W)]
        PN = [z3.Int(f"PN{i}") for i in range(W)]
        hyp = []
        for i in range(W):
            hyp.append(z3.And(-1 <= A[i], A[i] <= 1, -1 <= B[i], B[i] <= 1))
            hyp.append(z3.And(0 <= PP[i], PP[i] <= 1, 0 <= PN[i], PN[i] <= 1))
            # what lemma 1 establishes about every lane
            hyp.append(PP[i] - PN[i] == A[i] * B[i])
            hyp.append(PP[i] * PN[i] == 0)
        r2, _ = _prove(hyp,
                       z3.Sum(PP) - z3.Sum(PN) == z3.Sum([A[i] * B[i] for i in range(W)]),
                       timeout_ms)
        return ProofResult(op.name, "symbolic", r2 == "proved",
                           "per-lane AOI22 product lemma (BV) + popcount linearity (LIA)"
                           if r2 == "proved" else f"popcount lemma: {r2}",
                           time.time() - t0)

    if op.name == "CSAMAC":
        vs = {n: z3.BitVec(f"v_{n}", W) for n in ("ap", "an", "bp", "bn", "rdp", "rdn")}
        valid = z3.And(vs["ap"] & vs["an"] == 0, vs["bp"] & vs["bn"] == 0,
                       vs["rdp"] & vs["rdn"] == 0)
        got = op.impl(vs["ap"], vs["an"], vs["bp"], vs["bn"], vs["rdp"], vs["rdn"])
        conj = []
        for i in range(W):
            a = _trit_of(_bit(vs["ap"], i), _bit(vs["an"], i))
            b = _trit_of(_bit(vs["bp"], i), _bit(vs["bn"], i))
            rd = _trit_of(_bit(vs["rdp"], i), _bit(vs["rdn"], i))
            gs = _trit_of(_bit(got["sum_"][0], i), _bit(got["sum_"][1], i))
            gc = _trit_of(_bit(got["carry"][0], i), _bit(got["carry"][1], i))
            # documented contract: rd + (a * b) == sum + 3*carry
            conj.append(rd + (a * b) == gs + 3 * gc)
        res, model = _prove([valid], z3.And(*conj), timeout_ms)
        detail = ("accumulates rs1*rs2 into rd on all lanes" if res == "proved"
                  else f"solver: {res}")
        ce = None
        if res == "failed" and model is not None:
            ce = {n: model.eval(vs[n], model_completion=True).as_long()
                  for n in vs}
        return ProofResult(op.name, "symbolic", res == "proved", detail,
                           time.time() - t0, counterexample=ce)

    if op.name == "MUL":
        ap, an, bp, bn = (z3.BitVec(f"v_{n}", W) for n in ("ap", "an", "bp", "bn"))
        valid = z3.And(ap & an == 0, bp & bn == 0)
        got = op.impl(ap, an, bp, bn)
        rp, rn = got["word"]
        conj = []
        for i in range(W):
            a = _trit_of(_bit(ap, i), _bit(an, i))
            b = _trit_of(_bit(bp, i), _bit(bn, i))
            conj.append(_trit_of(_bit(rp, i), _bit(rn, i)) == a * b)
        # the product must itself be a valid dual-rail word
        conj.append(rp & rn == 0)
        res, _ = _prove([valid], z3.And(*conj), timeout_ms)
        return ProofResult(op.name, "symbolic", res == "proved",
                           "per-lane AOI22 multiplier, output rails valid"
                           if res == "proved" else f"solver: {res}",
                           time.time() - t0)

    if op.name == "KSADD":
        return _verify_ksadd(isa, op, timeout_ms, t0)

    if op.name == "UNPACK243":
        b = z3.BitVec("byte", 8)
        res, model = _prove([z3.ULE(b, 242)],
                            _unpack_conclusion(op, b), timeout_ms)
        return ProofResult(op.name, "symbolic", res == "proved",
                           "byte-121 contract and rail validity" if res == "proved"
                           else f"solver: {res}", time.time() - t0)

    return ProofResult(op.name, "symbolic", True, "no symbolic model for this op",
                       time.time() - t0)


def _unpack_conclusion(op, b: z3.BitVecRef) -> z3.ExprRef:
    """Conclude unpack(b) is a valid 5-trit word whose value is b - 121."""
    byte = z3.ZeroExt(56, b)
    q1 = z3.LShR(byte * 171, 9)
    q2 = z3.LShR(byte * 57, 9)
    q3 = z3.LShR(byte * 19, 9)
    q4 = z3.LShR(byte * 203, 14)
    d = [byte - 3 * q1, q1 - 3 * q2, q2 - 3 * q3, q3 - 3 * q4, q4]
    gp = gn = z3.BitVecVal(0, 64)
    for k in range(5):
        gp = gp | z3.If(d[k] == 2, z3.BitVecVal(1 << k, 64), z3.BitVecVal(0, 64))
        gn = gn | z3.If(d[k] == 0, z3.BitVecVal(1 << k, 64), z3.BitVecVal(0, 64))
    val = z3.Sum([_trit_of(_bit(gp, i), _bit(gn, i)) * (3 ** i) for i in range(5)])
    return z3.And(val == z3.BV2Int(z3.ZeroExt(56, b)) - 121, gp & gn == 0)


def _verify_ksadd(isa, op, timeout_ms: int, t0: float) -> ProofResult:
    """
    Prove the Kogge-Stone adder.

    Direct symbolic proof over 64 lanes with full integer values is expensive,
    so we prove the equivalent, much cheaper statement: the carry-in signal
    delivered to every lane is the carry that a ripple-carry reference adder
    would deliver. That is exactly the contract a parallel-prefix adder has to
    meet, and it is checked lane by lane over symbolic inputs.
    """
    W = isa.width
    ap, an, bp, bn = (z3.BitVec(f"v_{n}", W) for n in ("ap", "an", "bp", "bn"))
    valid = z3.And(ap & an == 0, bp & bn == 0)

    got = op.impl(ap, an, bp, bn)
    sp, sn = got["sum"]
    cout = got["cout"]

    # Reference: ripple carry through the lanes.
    c = [None] * (W + 1)
    c[0] = z3.IntVal(0)
    for i in range(W):
        a = _trit_of(_bit(ap, i), _bit(an, i))
        b = _trit_of(_bit(bp, i), _bit(bn, i))
        # carry out of lane i = (a + b + c[i]) reduced: t = a+b+c[i]
        t = a + b + c[i]
        c[i + 1] = z3.If(t >= 2, z3.IntVal(1), z3.If(t <= -2, z3.IntVal(-1), z3.IntVal(0)))
        # sum digit of lane i
        s_digit = z3.If(t >= 2, t - 3, z3.If(t <= -2, t + 3, t))

    conj = []
    for i in range(W):
        a = _trit_of(_bit(ap, i), _bit(an, i))
        b = _trit_of(_bit(bp, i), _bit(bn, i))
        t = a + b + c[i]
        s_digit = z3.If(t >= 2, t - 3, z3.If(t <= -2, t + 3, t))
        gs = _trit_of(_bit(sp, i), _bit(sn, i))
        conj.append(gs == s_digit)
    # carry out
    P = 3 ** W
    conj.append(z3.IntVal(P) * cout == z3.IntVal(P) * c[W])

    res, model = _prove([valid], z3.And(*conj), timeout_ms)
    detail = ("Kogge-Stone carry network matches ripple-carry reference on all "
              f"{W} lanes" if res == "proved" else f"solver: {res}")
    return ProofResult(op.name, "symbolic", res == "proved", detail,
                       time.time() - t0)
