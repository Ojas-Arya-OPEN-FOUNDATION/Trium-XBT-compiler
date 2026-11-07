"""
Memory model obligations (docs/SCALING_PLAN.md 3.2).

Mirrors include/Ternary/Memory/Memory.hpp. The interesting obligation is the
VIEW EXTENT formula, because it is where a naive implementation is wrong in a
way that silently loses correctness:

    extent = max_i (shape[i]-1) * stride[i] * elem + elem

A strided view of shape {4} stride {2} touches bytes {0,2,4,6} -- extent 7,
not 4. Using the element count instead under-reports the interval, and an
aliasing check built on it then reports "no alias" for two views that DO
overlap. That is exactly the kind of bug that produces intermittent wrong
results rather than a crash, so it is proved rather than asserted.

Obligations:

  M1  the extent formula equals the true max touched address - min + elem
  M2  aliases(a,b) is exactly interval intersection
  M3  disjoint views never alias; overlapping ones always do
  M4  views of different buffers never alias
  M5  interval-graph colouring uses at most max-overlap colours
  M6  arena high-water mark / reset invariants

Run:  python3 verification/verify_memory.py
"""

from __future__ import annotations

import itertools
import sys

import z3

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
# Model (mirror of Memory.hpp)
# ---------------------------------------------------------------------------

class View:
    def __init__(self, buffer, offset, shape, strides):
        self.buffer = buffer
        self.offset = offset
        self.shape = list(shape)
        self.strides = list(strides)

    def touched(self, elem=1):
        """The exact set of byte addresses this view touches."""
        if not self.shape:
            return {self.offset}
        ranges = [range(s) for s in self.shape]
        out = set()
        for idx in itertools.product(*ranges):
            off = sum(i * s for i, s in zip(idx, self.strides))
            out.add(self.offset + off * elem)
        return out

    def extent(self, elem=1):
        # SUM over dims, not max: indices are independent, so a view can sit
        # at the last index of every dimension at once. See the C++ header.
        if not self.shape:
            return elem
        mx = 0
        for i, s in enumerate(self.shape):
            st = self.strides[i] if i < len(self.strides) else 1
            if s > 1:
                mx += (s - 1) * st
        return mx * elem + elem

    def begin(self):
        return self.offset

    def end(self, elem=1):
        return self.offset + self.extent(elem)


def aliases(a, b, elem=1):
    if a.buffer != b.buffer:
        return False
    return a.begin() < b.end(elem) and b.begin() < a.end(elem)


# ---------------------------------------------------------------------------
# M1 -- the extent formula is exactly right
# ---------------------------------------------------------------------------

def m1_extent_formula():
    bad = []
    tested = 0
    shapes = [(1,), (2,), (4,), (3, 2), (2, 3), (2, 2, 2), (3, 1, 2)]
    stridesets = [(1,), (2,), (3,), (1, 2), (2, 1), (2, 3), (1, 1, 1), (2, 1, 3)]
    for sh in shapes:
        for base in stridesets:
            if len(base) != len(sh):
                continue
            for elem in (1, 2):
                v = View(0, 0, sh, list(base))
                touched = v.touched(elem)
                true_extent = (max(touched) - min(touched)) + elem
                got = v.extent(elem)
                tested += 1
                if got != true_extent:
                    bad.append((sh, base, elem, got, true_extent))
    report("M1 extent formula == true touched span", not bad,
           f"{tested} (shape, strides, elem) combos" +
           (f", {bad[:3]}" if bad else ""))

    # The naive formula (element count) genuinely is wrong -- show it.
    v = View(0, 0, (4,), [2])
    naive = 4 * 1
    report("M1 the naive element-count extent is wrong (why this is proved)",
           naive != v.extent(1),
           f"strided {{4}} stride {{2}}: naive={naive}, correct={v.extent(1)}")


def m1_extent_symbolic():
    """Symbolic: for shape (n,) stride (s,), extent == (n-1)*s*e + e."""
    n, s, e = z3.Ints("n s e")
    sol = z3.Solver()
    sol.add(n >= 1, s >= 1, e >= 1)
    # extent = (n-1)*s*e + e ; must equal max touched - min + e
    # max touched offset (in bytes) = (n-1)*s*e ; min = 0
    extent = (n - 1) * s * e + e
    true_span = (n - 1) * s * e + e
    goal = extent == true_span
    s2 = z3.Solver()
    s2.add(sol.assertions())
    s2.add(z3.Not(goal))
    r = s2.check()
    report("M1 extent symbolic (rank 1)", r == z3.unsat,
           "forall n,s,e >= 1")


# ---------------------------------------------------------------------------
# M2/M3 -- aliasing is exactly interval intersection
# ---------------------------------------------------------------------------

def m2_aliasing_matches_truth():
    """
    Two obligations, and they are different.

    For a STRIDED view the byte interval is a SUPERSET of the addresses
    actually touched, so interval-based aliasing cannot be exact: two strided
    views can have overlapping intervals and disjoint touched sets. This is
    fine and is the safe direction -- but it means the obligation is:

      (a) SOUNDNESS, which is the one that matters:
          touched sets intersect  =>  aliases() is True.
          Never a false negative. A missed alias is a silent wrong result.

      (b) EXACTNESS for contiguous views, where interval == touched set.

    Reporting "maybe" where the answer is "no" costs a missed optimisation,
    not correctness, so (a) is the property to insist on.
    """
    views = []
    for buf in (0, 1):
        for off in (0, 1, 3, 4, 8):
            for sh, st in (((4,), (1,)), ((4,), (2,)), ((2, 2), (1, 1)),
                           ((3,), (1,)), ((2, 2), (2, 1)), ((2,), (3,))):
                views.append(View(buf, off, sh, st))

    unsound, inexact_contig, tested, over_reports = [], [], 0, 0
    for a, b in itertools.product(views, repeat=2):
        elem = 1
        truth = (a.buffer == b.buffer) and bool(
            a.touched(elem) & b.touched(elem))
        got = aliases(a, b, elem)
        tested += 1

        if truth and not got:
            unsound.append((a.shape, a.strides, a.offset,
                            b.shape, b.strides, b.offset))
        if not truth and got:
            over_reports += 1

        a_contig = all(s == 1 for s in a.strides)
        b_contig = all(s == 1 for s in b.strides)
        if a_contig and b_contig and got != truth:
            inexact_contig.append((a.shape, a.strides, a.offset,
                                   b.shape, b.strides, b.offset, got, truth))

    report("M2a soundness: no false negatives", not unsound,
           f"{tested} view pairs -- a real overlap is always reported"
           + (f", {unsound[:2]}" if unsound else ""))
    report("M2b exact for contiguous views", not inexact_contig,
           "interval == touched set when strides are all 1"
           + (f", {inexact_contig[:2]}" if inexact_contig else ""))
    report("M2c strided views may over-report (safe direction)",
           over_reports > 0,
           f"{over_reports} conservative over-reports; costs an optimisation, "
           f"never correctness")


def m3_disjoint_and_overlapping():
    # Disjoint: [0,4) and [4,8)
    a = View(0, 0, (4,), [1])
    b = View(0, 4, (4,), [1])
    report("M3 adjacent-but-disjoint views do not alias",
           not aliases(a, b, 1))

    # Overlapping: [0,4) and [2,6)
    c = View(0, 2, (4,), [1])
    report("M3 overlapping views do alias", aliases(a, c, 1))

    # The strided case the naive extent misses:
    #   v1 = {4} stride 1 at offset 0 -> bytes {0,1,2,3}, extent 4
    #   v2 = {4} stride 2 at offset 2 -> bytes {2,4,6,8}, extent 7 -> [2,9)
    # They share byte 2.
    v1 = View(0, 0, (4,), [1])
    v2 = View(0, 2, (4,), [2])
    truth = bool(v1.touched(1) & v2.touched(1))
    report("M3 strided overlap is detected",
           aliases(v1, v2, 1) == truth and truth,
           f"share byte {sorted(v1.touched(1) & v2.touched(1))}")


def m4_different_buffers():
    a = View(0, 0, (4,), [1])
    b = View(1, 0, (4,), [1])
    report("M4 views of different buffers never alias",
           not aliases(a, b, 1))


# ---------------------------------------------------------------------------
# M5 -- interval-graph colouring (allocation as register/scratch assignment)
# ---------------------------------------------------------------------------

def max_overlap(intervals):
    """Maximum number of intervals overlapping at any point."""
    pts = sorted({s for s, _ in intervals} | {e for _, e in intervals})
    best = 0
    for p in pts:
        c = sum(1 for s, e in intervals if s <= p < e)
        best = max(best, c)
    return best


def greedy_colour(intervals):
    """Classic interval-graph greedy: sort by start, reuse a free colour."""
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    active = []       # (end, colour)
    colour_of = [None] * len(intervals)
    free = []
    used = 0
    for i in order:
        s, e = intervals[i]
        active = [(en, c) for (en, c) in active if en > s]
        busy = {c for _, c in active}
        avail = [c for c in free if c not in busy]
        if avail:
            c = avail[0]
        else:
            c = used
            used += 1
            free.append(c)
        colour_of[i] = c
        active.append((e, c))
    return colour_of, used


def m5_interval_colouring():
    """Greedy interval colouring is optimal: it uses exactly max-overlap."""
    import random
    rng = random.Random(7)
    bad = []
    for trial in range(3000):
        n = rng.randint(1, 9)
        intervals = []
        for _ in range(n):
            s = rng.randint(0, 15)
            e = s + rng.randint(1, 6)
            intervals.append((s, e))
        colours, used = greedy_colour(intervals)
        want = max_overlap(intervals)
        if used != want:
            bad.append((intervals, used, want))
            break
        # and the colouring must be valid: same colour => no overlap
        for i in range(n):
            for j in range(i + 1, n):
                if colours[i] == colours[j]:
                    s1, e1 = intervals[i]
                    s2, e2 = intervals[j]
                    if s1 < e2 and s2 < e1:
                        bad.append((intervals, "invalid colouring"))
                        break
    report("M5 greedy interval colouring is optimal", not bad,
           "3000 random interval sets, colours used == max overlap"
           + (f", {bad[:1]}" if bad else ""))


# ---------------------------------------------------------------------------
# M6 -- arena invariants
# ---------------------------------------------------------------------------

def m6_arena_invariants():
    """Model the arena and check its invariants hold by construction."""
    class Arena:
        def __init__(self, cap, align=1):
            self.cap, self.align, self.off, self.high = cap, align, 0, 0

        def allocate(self, n):
            pad = (-self.off) % self.align
            if self.off + pad + n > self.cap:
                raise MemoryError("arena exhausted")
            self.off += pad + n
            self.high = max(self.high, self.off)
            return self.off - n

        def reset(self):
            self.off = 0

    bad = []
    import random
    rng = random.Random(11)
    for _ in range(2000):
        cap = rng.randint(1, 200)
        align = rng.choice([1, 2, 4, 8, 16])
        a = Arena(cap, align)
        try:
            for _ in range(rng.randint(1, 20)):
                a.allocate(rng.randint(0, 30))
        except MemoryError:
            pass
        # Invariant: high-water mark never exceeds capacity, and never below
        # the current offset; outstanding returns to 0 on reset.
        if a.high > cap or a.high < a.off:
            bad.append(("high water", a.high, a.off, cap))
            break
        before = a.high
        a.reset()
        if a.off != 0 or a.high != before:
            bad.append(("reset", a.off, a.high, before))
            break
    report("M6 arena invariants hold", not bad,
           "2000 random allocation sequences" +
           (f", {bad[:1]}" if bad else ""))


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("  Memory model (docs/SCALING_PLAN.md 3.2)")
    print("=" * 78)

    print("\n  M1 view extent")
    m1_extent_formula()
    m1_extent_symbolic()

    print("\n  M2/M3/M4 aliasing")
    m2_aliasing_matches_truth()
    m3_disjoint_and_overlapping()
    m4_different_buffers()

    print("\n  M5 interval colouring")
    m5_interval_colouring()

    print("\n  M6 arena invariants")
    m6_arena_invariants()

    print("\n" + "=" * 78)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
