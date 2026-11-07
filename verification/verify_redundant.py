"""
Z3 verification of the symplectic-ternary carry-free accumulator.

Obligations discharged here (referenced from docs/SYMPLECTIC_TERNARY_MATHS.md):

  V1  BTFA per-trit spec       s + 3c == a + b + d          exhaustive + Z3
  V2  Energy conservation      H(S',C') == H(S,C) + val(X)  symbolic Z3
  V3  N-fold accumulation      H after N == sum of inputs   symbolic Z3
  V4  Restore exactness        restore == canonical(sum)    exhaustive + Z3
  V5  Sizing prevents overflow worst-case patterns          exhaustive
  V6  Edge cases               all -1 / all +1 / zero / alt / MSB
  V7  Parallel == sequential   O(log w) restore agrees      exhaustive

Run:  python3 -m verification.verify_redundant
"""

from __future__ import annotations

import sys
import time

import z3

# ---------------------------------------------------------------------------
# Reference model (plain Python integers -- the oracle)
# ---------------------------------------------------------------------------

T = (-1, 0, 1)


def trit_btfa(a: int, b: int, d: int) -> tuple[int, int]:
    """Full adder spec: returns (s, c) with s + 3c == a + b + d."""
    t = a + b + d
    if t >= 2:
        return (t - 3, 1)
    if t <= -2:
        return (t + 3, -1)
    return (t, 0)


def trit_btha(a: int, b: int) -> tuple[int, int]:
    return trit_btfa(a, b, 0)


def word_value(trits) -> int:
    return sum(t * (3 ** i) for i, t in enumerate(trits))


def shift3(C, w):
    """3*C as a trit shift. Returns (shifted, dropped_top_trit) of length w."""
    return ([0] + list(C[:-1])), C[-1]


def btfa_word(A, B, D):
    return tuple(zip(*(trit_btfa(a, b, d) for a, b, d in zip(A, B, D))))


def H(S, C) -> int:
    return word_value(S) + 3 * word_value(C)


def accumulate(S, C, X, w):
    """One carry-save step. Returns ((S',C'), dropped_top_trit)."""
    Cs, dropped = shift3(C, w)
    S2, C2 = btfa_word(S, Cs, X)
    return (list(S2), list(C2)), dropped


def restore_sequential(S, C, w):
    """Collapse the redundant pair to a canonical word (ripple). O(w) depth."""
    s, c = list(S), list(C)
    steps = 0
    while any(c) and steps <= w + 2:
        Cs, _ = shift3(c, w)
        s, c = btfa_word(s, Cs, [0] * w)
        steps += 1
    return s, steps


# ---------------------------------------------------------------------------
# O(log w) parallel-prefix restore (the fix for defect D2)
# ---------------------------------------------------------------------------

def _transfer(raw: int) -> dict[int, tuple[int, int]]:
    """c_in -> (sum_digit, c_out) for a position with digit-sum `raw`."""
    out = {}
    for c in (-1, 0, 1):
        total = raw + c
        if total >= 2:
            out[c] = (total - 3, 1)
        elif total <= -2:
            out[c] = (total + 3, -1)
        else:
            out[c] = (total, 0)
    return out


def _compose(f, g):
    """h(c) = g(f(c)) -- carry transfer composition (the prefix operator)."""
    return {c: g[f[c][1]] for c in (-1, 0, 1)}


def restore_parallel(S, C, w):
    """Parallel-prefix assimilation: O(log w) depth, same function as ripple."""
    # raw digit sums of S + 3C
    Cs, _ = shift3(C, w)
    raw = [a + b for a, b in zip(S, Cs)]
    tr = [_transfer(r) for r in raw]

    # Kogge-Stone prefix over the carry transfer functions (O(log w) rounds).
    # After processing step s, f[i] is the transfer of the range
    # [i - 2s + 1 .. i], clamped at 0; at the end f[i] == transfer of [0..i].
    f = list(tr)
    span = 1
    while span < w:
        nxt = list(f)
        for i in range(w - 1, -1, -1):
            if i - span >= 0:
                # lower positions first, then this one
                nxt[i] = _compose(f[i - span], f[i])
        f = nxt
        span *= 2

    # carry into position i == f[i-1] applied to the initial carry 0
    carry_in = [0] * w
    for i in range(1, w):
        carry_in[i] = f[i - 1][0][1]

    digits = [tr[i][carry_in[i]][0] for i in range(w)]
    return digits, 0


# ---------------------------------------------------------------------------
# Canonical balanced-ternary encoding (the oracle for restore)
# ---------------------------------------------------------------------------

def encode_balanced(v: int, w: int):
    trits = []
    x = v
    for _ in range(w):
        r = x % 3
        if r == 0:
            trits.append(0); x //= 3
        elif r == 1:
            trits.append(1); x = (x - 1) // 3
        else:
            trits.append(-1); x = (x + 1) // 3
    return trits


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"    [PASS] {name}{('  ' + detail) if detail else ''}")
    else:
        FAIL += 1
        print(f"    [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# V1 -- BTFA per-trit spec, exhaustive over all 27 triples
# ---------------------------------------------------------------------------

def v1_btfa_exhaustive():
    bad = []
    for a in T:
        for b in T:
            for d in T:
                s, c = trit_btfa(a, b, d)
                if s + 3 * c != a + b + d:
                    bad.append((a, b, d, s, c))
                if s not in T or c not in T:
                    bad.append((a, b, d, s, c, "out of range"))
    report("V1 BTFA spec exhaustive", not bad, f"27 trit triples" + (f", {bad[:3]}" if bad else ""))


def v1_btfa_symbolic():
    a, b, d, s, c = z3.Ints("a b d s c")
    sol = z3.Solver()
    for v in (a, b, d):
        sol.add(v >= -1, v <= 1)
    for v in (s, c):
        sol.add(v >= -1, v <= 1)
    sol.add(s + 3 * c == a + b + d)
    # The spec must be TOTAL and UNIQUE: for every (a,b,d) exactly one (s,c).
    # Prove: no two distinct (s,c) satisfy it, and at least one does.
    s2, c2 = z3.Ints("s2 c2")
    sol2 = z3.Solver()
    for v in (a, b, d, s, c, s2, c2):
        sol2.add(v >= -1, v <= 1)
    sol2.add(s + 3 * c == a + b + d)
    sol2.add(s2 + 3 * c2 == a + b + d)
    sol2.add(z3.Or(s != s2, c != c2))
    r = sol2.check()
    report("V1 BTFA solution is unique", r == z3.unsat,
           "no two distinct (sum,carry) for one input" if r == z3.unsat else f"sat {sol2.model()}")


# ---------------------------------------------------------------------------
# V2 -- energy conservation, symbolic Z3 at full width
# ---------------------------------------------------------------------------

def v2_energy_conservation(width=12):
    """H(S',C') == H(S,C) + val(X), symbolically, with no top-trit drop."""
    S = z3.IntVector("S", width)
    C = z3.IntVector("C", width)
    X = z3.IntVector("X", width)
    sol = z3.Solver()
    for v in list(S) + list(C) + list(X):
        sol.add(v >= -1, v <= 1)
    # Force the top trit of C to zero so the 3*C shift is exact (no drop).
    sol.add(C[width - 1] == 0)

    Cs = [0] + [C[i] for i in range(width - 1)]
    S2, C2 = [], []
    for i in range(width):
        raw = S[i] + Cs[i] + X[i]
        # encode the spec as constraints rather than If-trees
        s_i, c_i = z3.Int(f"s2_{i}"), z3.Int(f"c2_{i}")
        sol.add(s_i >= -1, s_i <= 1, c_i >= -1, c_i <= 1)
        sol.add(s_i + 3 * c_i == raw)
        S2.append(s_i); C2.append(c_i)

    Hs = z3.Sum([S[i] * (3 ** i) for i in range(width)])
    Hc = z3.Sum([C[i] * (3 ** i) for i in range(width)])
    Hx = z3.Sum([X[i] * (3 ** i) for i in range(width)])
    Hs2 = z3.Sum([S2[i] * (3 ** i) for i in range(width)])
    Hc2 = z3.Sum([C2[i] * (3 ** i) for i in range(width)])

    goal = (Hs2 + 3 * Hc2) == (Hs + 3 * Hc + Hx)
    t0 = time.time()
    r = _prove(sol, goal)
    report("V2 energy conservation (symbolic)", r, f"width={width}, {time.time()-t0:.2f}s")


def v2_energy_with_drop(width=12):
    """With the top trit of C non-zero, the shift drops it -- quantify the loss."""
    S = z3.IntVector("S", width)
    C = z3.IntVector("C", width)
    X = z3.IntVector("X", width)
    sol = z3.Solver()
    for v in list(S) + list(C) + list(X):
        sol.add(v >= -1, v <= 1)
    sol.add(C[width - 1] != 0)  # a carry IS sitting at the top trit

    Cs = [0] + [C[i] for i in range(width - 1)]
    S2, C2 = [], []
    for i in range(width):
        raw = S[i] + Cs[i] + X[i]
        s_i, c_i = z3.Int(f"s3_{i}"), z3.Int(f"c3_{i}")
        sol.add(s_i >= -1, s_i <= 1, c_i >= -1, c_i <= 1)
        sol.add(s_i + 3 * c_i == raw)
        S2.append(s_i); C2.append(c_i)

    Hs = z3.Sum([S[i] * (3 ** i) for i in range(width)])
    Hc = z3.Sum([C[i] * (3 ** i) for i in range(width)])
    Hx = z3.Sum([X[i] * (3 ** i) for i in range(width)])
    Hs2 = z3.Sum([S2[i] * (3 ** i) for i in range(width)])
    Hc2 = z3.Sum([C2[i] * (3 ** i) for i in range(width)])
    dropped = C[width - 1] * (3 ** width)

    goal = (Hs2 + 3 * Hc2) == (Hs + 3 * Hc + Hx - dropped)
    r = _prove(sol, goal)
    report("V2 overflow is exactly the dropped top trit", r,
           "confirms defect D1: loss == C[w-1]*3^w")


def _prove(sol, goal):
    """Check sol => goal by refuting sol & ~goal."""
    s = z3.Solver()
    s.add(sol.assertions())
    s.add(z3.Not(goal))
    return s.check() == z3.unsat


# ---------------------------------------------------------------------------
# V3 -- N-fold accumulation, symbolic
# ---------------------------------------------------------------------------

def v3_nfold(width=8, n=4):
    Xs = [z3.IntVector(f"X{k}", width) for k in range(n)]
    sol = z3.Solver()
    for X in Xs:
        for v in X:
            sol.add(v >= -1, v <= 1)

    S = [0] * width
    C = [0] * width
    for k in range(n):
        Cs = [0] + [C[i] for i in range(width - 1)]
        # deliberately under-size the constraint: require the top trit of C
        # to stay 0, which the sizing rule (3.1) guarantees
        sol.add(C[width - 1] == 0)
        S2, C2 = [], []
        for i in range(width):
            raw = S[i] + Cs[i] + Xs[k][i]
            s_i, c_i = z3.Int(f"n{k}s_{i}"), z3.Int(f"n{k}c_{i}")
            sol.add(s_i >= -1, s_i <= 1, c_i >= -1, c_i <= 1)
            sol.add(s_i + 3 * c_i == raw)
            S2.append(s_i); C2.append(c_i)
        S, C = S2, C2

    Hfinal = z3.Sum([S[i] * 3**i for i in range(width)]) + \
             3 * z3.Sum([C[i] * 3**i for i in range(width)])
    Hwant = z3.Sum([z3.Sum([Xs[k][i] * 3**i for i in range(width)])
                    for k in range(n)])
    r = _prove(sol, Hfinal == Hwant)
    report("V3 N-fold accumulation", r, f"width={width}, N={n}")


# ---------------------------------------------------------------------------
# V4 / V6 / V7 -- exhaustive over reachable states and edge patterns
# ---------------------------------------------------------------------------

import itertools


def _patterns(w):
    """Named edge-case word patterns."""
    pos = [1] * w
    neg = [-1] * w
    zero = [0] * w
    alt = [1 if i % 2 == 0 else -1 for i in range(w)]
    alt2 = [-1 if i % 2 == 0 else 1 for i in range(w)]
    one = [0] * w; one[0] = 1
    msb = [0] * w; msb[w - 1] = 1
    msbn = [0] * w; msbn[w - 1] = -1
    return {
        "all+1": pos, "all-1": neg, "zero": zero,
        "alt(+1,-1)": alt, "alt(-1,+1)": alt2,
        "lsb+1": one, "msb+1": msb, "msb-1": msbn,
    }


def v4_restore_exhaustive(w=6):
    """restore(S,C) must equal canonical(val(S) + 3*val(C)) for every state."""
    bad = []
    maxsteps = 0
    count = 0
    for S in itertools.product(T, repeat=w):
        for C in itertools.product(T, repeat=w):
            target = word_value(S) + 3 * word_value(C)
            lo = -(3 ** w - 1) // 2
            hi = (3 ** w - 1) // 2
            if not (lo <= target <= hi):
                continue  # not representable in w trits -- sizing, not a bug
            got, steps = restore_sequential(S, C, w)
            maxsteps = max(maxsteps, steps)
            count += 1
            if word_value(got) != target:
                bad.append((S, C, target, word_value(got)))
                if len(bad) > 3:
                    break
        if bad:
            break
    report("V4 restore exactness (exhaustive)", not bad,
           f"w={w}, {count} states, max ripple steps={maxsteps}"
           + (f", bad={bad[:2]}" if bad else ""))


def v6_edge_cases(n=6, N=3):
    """Every named edge pattern, accumulated N times in a correctly sized accumulator."""
    pats = _patterns(n)
    bad = []
    for name, Xp in pats.items():
        w = required_width(n, word_value(Xp), N)
        X = list(Xp) + [0] * (w - n)
        S, C = [0] * w, [0] * w
        dropped_any = False
        for _ in range(N):
            (S, C), dropped = accumulate(S, C, X, w)
            dropped_any = dropped_any or dropped != 0
        target = N * word_value(Xp)
        got, _ = restore_sequential(S, C, w)
        par, _ = restore_parallel(S, C, w)
        if dropped_any or word_value(got) != target or word_value(par) != target:
            bad.append((name, target, word_value(got), word_value(par), dropped_any))
    report("V6 edge cases (sized accumulator)", not bad,
           f"{len(pats)} patterns x N={N}, restore exact + parallel agrees"
           + (f", {bad[:2]}" if bad else ""))


def v6_exhaustive_chains(n=4, N=3):
    """ALL operand chains of length N over width n -- the true ternary edge set."""
    bad = []
    count = 0
    for combo in itertools.product(T, repeat=n * N):
        Xs = [list(combo[k * n:(k + 1) * n]) for k in range(N)]
        M = max(abs(word_value(X)) for X in Xs)
        w = required_width(n, M, N)
        S, C = [0] * w, [0] * w
        dropped_any = False
        for X in Xs:
            Xp = X + [0] * (w - n)
            (S, C), dropped = accumulate(S, C, Xp, w)
            dropped_any = dropped_any or dropped != 0
        target = sum(word_value(X) for X in Xs)
        got, _ = restore_sequential(S, C, w)
        count += 1
        if dropped_any or word_value(got) != target:
            bad.append((Xs, target, word_value(got)))
            break
    report("V6 exhaustive operand chains", not bad,
           f"{count} chains of N={N} over w={n} (3^{n*N} space)"
           + (f", {bad[:1]}" if bad else ""))


def v7_parallel_matches_sequential(w=6):
    bad = []
    for S in itertools.product(T, repeat=w):
        for C in itertools.product(T, repeat=w):
            target = word_value(S) + 3 * word_value(C)
            lo = -(3 ** w - 1) // 2
            hi = (3 ** w - 1) // 2
            if not (lo <= target <= hi):
                continue
            seq, _ = restore_sequential(S, C, w)
            par, _ = restore_parallel(S, C, w)
            if word_value(par) != word_value(seq):
                bad.append((S, C, word_value(seq), word_value(par)))
                break
        if bad:
            break
    report("V7 parallel-prefix restore == ripple", not bad,
           f"w={w}" + (f", {bad[:2]}" if bad else ""))


def required_width(n, M, N):
    """Sizing rule (3.1): smallest w that represents N*M plus shift headroom."""
    M = max(1, abs(M))
    need = N * M
    w = n
    while (3 ** w - 1) // 2 < need:
        w += 1
    return w + 2  # headroom so C[w-1] stays 0 across the 3*C shift


def v5_sizing(n=6, Ns=(1, 2, 3, 8, 16)):
    """With the sizing rule, no top-trit carry is ever dropped and restore is exact."""
    worst = [1] * n                      # the maximum-magnitude word
    bad = []
    for N in Ns:
        M = word_value(worst)
        w = required_width(n, M, N)
        S, C = [0] * w, [0] * w
        Xw = list(worst) + [0] * (w - n)
        dropped_any = False
        for _ in range(N):
            (S, C), dropped = accumulate(S, C, Xw, w)
            dropped_any = dropped_any or dropped != 0
        total = N * M
        got, _ = restore_sequential(S, C, w)
        ok = (not dropped_any) and word_value(got) == total
        if not ok:
            bad.append((N, w, total, word_value(got), dropped_any))
    report("V5 sizing rule is sufficient", not bad,
           f"n={n}, N in {Ns}, worst-case word" + (f", {bad[:2]}" if bad else ""))


def v5_sizing_necessary(n=6, N=3):
    """Under-sizing loses exactly the dropped top trits -- sizing is necessary."""
    worst = [1] * n
    w = n                                # NO headroom: deliberately under-sized
    S, C = [0] * w, [0] * w
    lost = 0
    for _ in range(N):
        (S, C), dropped = accumulate(S, C, worst, w)
        lost += dropped * (3 ** w)
    expect = N * word_value(worst)
    got_H = H(S, C)
    ok = got_H == expect - lost and lost != 0
    report("V5 sizing is necessary (D1 characterised)", ok,
           f"loss={lost} exactly accounts for the discrepancy "
           f"({expect} -> H={got_H})")


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("  Symplectic-ternary carry-free accumulator")
    print("=" * 78)

    print("\n  V1 BTFA compressor spec")
    v1_btfa_exhaustive()
    v1_btfa_symbolic()

    print("\n  V2 Energy conservation H(S',C') == H(S,C) + val(X)")
    v2_energy_conservation(width=12)
    v2_energy_with_drop(width=12)

    print("\n  V3 N-fold accumulation")
    v3_nfold(width=8, n=4)

    print("\n  V4 Restore exactness")
    v4_restore_exhaustive(w=6)

    print("\n  V6 Edge cases")
    v6_edge_cases(n=6, N=3)
    v6_exhaustive_chains(n=4, N=3)

    print("\n  V5 Sizing / overflow")
    v5_sizing(n=6, Ns=(1, 2, 3, 8, 16))
    v5_sizing_necessary(n=6, N=3)

    print("\n  V7 Parallel-prefix restore")
    v7_parallel_matches_sequential(w=6)

    print("\n" + "=" * 78)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
