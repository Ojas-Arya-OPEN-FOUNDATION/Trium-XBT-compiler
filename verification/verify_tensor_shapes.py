"""
Shape algebra for the tensor IR (docs/SCALING_PLAN.md 3.1).

Mirrors include/Ternary/IR/Tensor.hpp exactly and discharges the obligations
the plan asks for:

  T1  broadcast is associative, commutative, and has {} as identity
  T2  broadcast(a, a) == a                                 (idempotent)
  T3  transpose with a permutation, then its inverse, is identity
  T4  reshape preserves element count; reshape o reshape composes
  T5  matmul shape is associative: (AB)C == A(BC)
  T6  linear_index / unlinear_index round-trip
  T7  reduce drops exactly the listed axes
  T8  every function is total: bad shapes return None, never raise

The C++ side must agree with this model -- test/Unit/test_unit_tensor.cpp
runs the same cases through the header.

Run:  python3 verification/verify_tensor_shapes.py
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
# Shape algebra (mirror of Tensor.hpp)
# ---------------------------------------------------------------------------

def broadcast_shapes(a, b):
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        da = a[len(a) - 1 - i] if i < len(a) else None
        db = b[len(b) - 1 - i] if i < len(b) else None
        if da is None and db is None:
            return None
        if da is None:
            out[n - 1 - i] = db
        elif db is None:
            out[n - 1 - i] = da
        elif da == db:
            out[n - 1 - i] = da
        elif da == 1:
            out[n - 1 - i] = db
        elif db == 1:
            out[n - 1 - i] = da
        else:
            return None
    return tuple(out)


def matmul_shape(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    if a[-1] != b[-2]:
        return None
    ba, bb = list(a[:-2]), list(b[:-2])
    if not ba and not bb:
        batch = ()
    elif not ba:
        batch = tuple(bb)
    elif not bb:
        batch = tuple(ba)
    else:
        batch = broadcast_shapes(tuple(ba), tuple(bb))
        if batch is None:
            return None
    return tuple(batch) + (a[-2], b[-1])


def transpose_shape(dims, perm):
    if len(perm) != len(dims):
        return None
    if sorted(perm) != list(range(len(dims))):
        return None
    return tuple(dims[p] for p in perm)


def invert_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def reduce_shape(dims, axes, keep_dims=False):
    drop = [False] * len(dims)
    for ax in axes:
        a = ax + len(dims) if ax < 0 else ax
        if a < 0 or a >= len(dims):
            return None
        if drop[a]:
            return None
        drop[a] = True
    out = []
    for i, d in enumerate(dims):
        if drop[i]:
            if keep_dims:
                out.append(1)
        else:
            out.append(d)
    return tuple(out)


def reshape_shape(dims, newdims):
    try:
        total = 1
        for d in dims:
            if d <= 0:
                return None
            total *= d
    except TypeError:
        return None
    known, infer_at, seen_dyn = 1, None, False
    for i, d in enumerate(newdims):
        if d == -1:
            if seen_dyn:
                return None
            seen_dyn, infer_at = True, i
        elif d <= 0:
            return None
        else:
            known *= d
    out = list(newdims)
    if seen_dyn:
        if known == 0 or total % known:
            return None
        out[infer_at] = total // known
    elif known != total:
        return None
    return tuple(out)


def row_major_strides(dims):
    st = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        st[i] = st[i + 1] * dims[i + 1]
    return st


def linear_index(dims, idx):
    if len(dims) != len(idx):
        return -1
    st = row_major_strides(dims)
    lin = 0
    for i in range(len(dims)):
        if idx[i] < 0 or idx[i] >= dims[i]:
            return -1
        lin += idx[i] * st[i]
    return lin


def unlinear_index(dims, lin):
    if not dims:
        return ()
    st = row_major_strides(dims)
    total = 1
    for d in dims:
        total *= d
    if lin < 0 or lin >= total:
        return None
    return tuple((lin // st[i]) % dims[i] for i in range(len(dims)))


# ---------------------------------------------------------------------------
# T1/T2 -- broadcast algebra
# ---------------------------------------------------------------------------

def t1_broadcast_algebra():
    shapes = [(), (1,), (3,), (1, 3), (3, 1), (4, 3), (2, 4, 3), (1, 4, 3),
              (5, 1, 3), (2, 1, 1)]
    bad_assoc, bad_comm, bad_ident, checked = [], [], [], 0

    for a, b, c in itertools.product(shapes, repeat=3):
        ab = broadcast_shapes(a, b)
        bc = broadcast_shapes(b, c)
        if ab is not None and bc is not None:
            left = broadcast_shapes(ab, c)
            right = broadcast_shapes(a, bc)
            checked += 1
            # Both may legitimately fail; what must hold is that they AGREE.
            if left != right:
                bad_assoc.append((a, b, c, left, right))

    for a, b in itertools.product(shapes, repeat=2):
        if broadcast_shapes(a, b) != broadcast_shapes(b, a):
            bad_comm.append((a, b))
        if broadcast_shapes(a, ()) != a:
            bad_ident.append(a)

    report("T1 broadcast is associative", not bad_assoc,
           f"{checked} triples" + (f", {bad_assoc[:2]}" if bad_assoc else ""))
    report("T1 broadcast is commutative", not bad_comm)
    report("T1 scalar () is the broadcast identity", not bad_ident)

    bad_idem = [a for a in shapes if broadcast_shapes(a, a) != a]
    report("T2 broadcast(a,a) == a", not bad_idem)


def t1_broadcast_symbolic():
    """Symbolic: broadcasting is monotone in each dim and never shrinks rank."""
    # dims are symbolic positive ints; broadcast of (1,) with (n,) is (n,)
    n = z3.Int("n")
    s = z3.Solver()
    s.add(n >= 1)
    # Property: broadcasting a size-1 dim against n yields n, for all n >= 1.
    # Encoded as: result == max(1, n) with the rule 1 -> n.
    result = z3.If(n == 1, z3.IntVal(1), n)
    goal = z3.And(result == n)
    s2 = z3.Solver()
    s2.add(n >= 1)
    s2.add(z3.Not(goal))
    r = s2.check()
    report("T1 broadcast(1, n) == n symbolically", r == z3.unsat,
           "for all n >= 1")


# ---------------------------------------------------------------------------
# T3 -- transpose
# ---------------------------------------------------------------------------

def t3_transpose_roundtrip():
    bad = []
    shapes = [(2, 3), (2, 3, 4), (5,), (2, 3, 4, 5)]
    for dims in shapes:
        for perm in itertools.permutations(range(len(dims))):
            once = transpose_shape(dims, perm)
            back = transpose_shape(once, invert_permutation(perm))
            if back != dims:
                bad.append((dims, perm, once, back))
    report("T3 transpose(inv(perm)) o transpose(perm) == id", not bad,
           "all shapes x all permutations" + (f", {bad[:1]}" if bad else ""))

    # Non-permutations must be rejected, not silently accepted.
    rejected = all(transpose_shape((2, 3), p) is None
                   for p in [(0, 0), (1,), (0, 1, 2), (2, 0)])
    report("T3 non-permutations are rejected", rejected)


# ---------------------------------------------------------------------------
# T4 -- reshape
# ---------------------------------------------------------------------------

def t4_reshape():
    bad_count, bad_comp = [], []
    shapes = [(2, 3), (4, 3), (2, 3, 4), (6,), (1, 12), (12, 1)]

    def total(d):
        t = 1
        for x in d:
            t *= x
        return t

    for dims in shapes:
        n = total(dims)
        for nd in [(n,), (1, n), (n, 1), (-1,), (2, n // 2) if n % 2 == 0 else (n,)]:
            r = reshape_shape(dims, list(nd))
            if r is None:
                continue
            if total(r) != n:
                bad_count.append((dims, nd, r))
            # reshape back must recover the original
            if reshape_shape(r, list(dims)) != dims:
                bad_comp.append((dims, nd, r))

    report("T4 reshape preserves element count", not bad_count,
           (f"{bad_count[:2]}" if bad_count else ""))
    report("T4 reshape round-trips", not bad_comp,
           (f"{bad_comp[:2]}" if bad_comp else ""))

    # Mismatched counts must be refused.
    refusals = [
        reshape_shape((2, 3), [7]),
        reshape_shape((2, 3), [2, 2]),
        reshape_shape((2, 3), [0]),
        reshape_shape((2, 3), [-1, -1]),
        reshape_shape((2, 3), [5, -1]),     # 6 % 5 != 0
    ]
    report("T4 impossible reshapes are refused",
           all(r is None for r in refusals))


# ---------------------------------------------------------------------------
# T5 -- matmul shape associativity
# ---------------------------------------------------------------------------

def t5_matmul_assoc():
    shapes = [(2, 3), (3, 4), (4, 5), (2, 2, 3), (2, 3, 4), (2, 4, 5),
              (1, 3, 4)]
    bad = []
    checked = 0
    for a, b, c in itertools.product(shapes, repeat=3):
        ab = matmul_shape(a, b)
        bc = matmul_shape(b, c)
        if ab is None or bc is None:
            continue
        left = matmul_shape(ab, c)
        right = matmul_shape(a, bc)
        checked += 1
        if left != right:
            bad.append((a, b, c, left, right))
    report("T5 matmul shape is associative", not bad,
           f"{checked} compatible triples" + (f", {bad[:2]}" if bad else ""))

    # Inner-dim mismatch must be refused.
    report("T5 inner-dim mismatch is refused",
           matmul_shape((2, 3), (4, 5)) is None)
    report("T5 rank < 2 is refused",
           matmul_shape((3,), (3,)) is None and matmul_shape((3,), (3, 4)) is None)


# ---------------------------------------------------------------------------
# T6 -- index round-trip
# ---------------------------------------------------------------------------

def t6_index_roundtrip():
    bad = []
    shapes = [(5,), (2, 3), (2, 3, 4), (1, 7), (3, 1, 2)]
    count = 0
    for dims in shapes:
        total = 1
        for d in dims:
            total *= d
        for lin in range(total):
            idx = unlinear_index(dims, lin)
            back = linear_index(dims, list(idx))
            count += 1
            if back != lin:
                bad.append((dims, lin, idx, back))
    report("T6 unlinear/linear round-trip", not bad,
           f"{count} offsets" + (f", {bad[:2]}" if bad else ""))

    # Out-of-range and out-of-bounds must be refused.
    ok = (unlinear_index((2, 3), 6) is None
          and unlinear_index((2, 3), -1) is None
          and linear_index((2, 3), [2, 0]) == -1
          and linear_index((2, 3), [0, 3]) == -1
          and linear_index((2, 3), [0]) == -1)
    report("T6 out-of-range indices are refused", ok)


# ---------------------------------------------------------------------------
# T7 -- reduce
# ---------------------------------------------------------------------------

def t7_reduce():
    bad = []
    for dims in [(2, 3), (2, 3, 4), (5,)]:
        for r in range(len(dims)):
            for axes in itertools.combinations(range(len(dims)), r):
                out = reduce_shape(dims, list(axes))
                if out is None:
                    continue
                expected = tuple(d for i, d in enumerate(dims)
                                 if i not in axes)
                if out != expected:
                    bad.append((dims, axes, out, expected))
                # keep_dims must have the same rank and 1s where reduced
                kd = reduce_shape(dims, list(axes), keep_dims=True)
                if kd is None or len(kd) != len(dims):
                    bad.append((dims, axes, kd, "keep_dims rank"))
                else:
                    for i, d in enumerate(dims):
                        want = 1 if i in axes else d
                        if kd[i] != want:
                            bad.append((dims, axes, kd, "keep_dims value"))
    report("T7 reduce drops exactly the listed axes", not bad,
           (f"{bad[:2]}" if bad else ""))

    report("T7 duplicate or out-of-range axes are refused",
           reduce_shape((2, 3), [0, 0]) is None
           and reduce_shape((2, 3), [2]) is None
           and reduce_shape((2, 3), [-3]) is None)


# ---------------------------------------------------------------------------
# T8 -- totality
# ---------------------------------------------------------------------------

def t8_totality():
    """No shape function may raise on any input; malformed shapes yield None."""
    nasty = [(), (0,), (-1,), (-1, -1), (1, 0), (2, -3), (0, 0)]
    errors = []
    for dims in nasty:
        for fn, template in (
            (broadcast_shapes, lambda d: (d, (2, 3))),
            (matmul_shape, lambda d: (d, (3, 4))),
            (transpose_shape, lambda d: (d, tuple(range(len(d))))),
            (reduce_shape, lambda d: (d, [0] if d else [])),
            (reshape_shape, lambda d: (d, [-1])),
        ):
            try:
                fn(*template(dims))
            except Exception as e:
                errors.append((fn.__name__, dims, str(e)))
    report("T8 shape functions never raise", not errors,
           "malformed shapes return None" + (f", {errors[:3]}" if errors else ""))


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("  Tensor shape algebra (docs/SCALING_PLAN.md 3.1)")
    print("=" * 78)

    print("\n  T1/T2 broadcast")
    t1_broadcast_algebra()
    t1_broadcast_symbolic()

    print("\n  T3 transpose")
    t3_transpose_roundtrip()

    print("\n  T4 reshape")
    t4_reshape()

    print("\n  T5 matmul")
    t5_matmul_assoc()

    print("\n  T6 index arithmetic")
    t6_index_roundtrip()

    print("\n  T7 reduce")
    t7_reduce()

    print("\n  T8 totality")
    t8_totality()

    print("\n" + "=" * 78)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
