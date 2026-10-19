#!/usr/bin/env python3
"""
Formal Z3 SMT Verification of Balanced Euclidean Division (`divmod_balanced`).
Proves:
  1. Division identity: a == q * b + r and minimal-magnitude bound 2 * |r| <= |b| for all b != 0.
  2. Strict uniqueness of (q, r) for every odd divisor b (including all powers of 3, b = 3^k).
  3. Zero-bias odd symmetry: divmod(-a, b) == (-q, -r) and divmod(a, -b) == (-q, r).
"""

import z3


def abs_z3(x):
    return z3.If(x >= 0, x, -x)


def sgn_z3(x):
    return z3.If(x > 0, z3.IntVal(1), z3.If(x < 0, z3.IntVal(-1), z3.IntVal(0)))


def divmod_balanced_spec(a, b, q0, r0):
    """
    Given truncated division (q0, r0) satisfying:
      a == q0 * b + r0,  |r0| < |b|,  and r0 has the same sign as a (or 0),
    returns balanced (q, r) via nearest-integer adjustment.
    """
    abs_b = abs_z3(b)
    sgn_b = sgn_z3(b)

    cond_hi = (2 * r0 > abs_b)
    cond_lo = (2 * r0 < -abs_b)

    q = z3.If(cond_hi, q0 + sgn_b, z3.If(cond_lo, q0 - sgn_b, q0))
    r = z3.If(cond_hi, r0 - abs_b, z3.If(cond_lo, r0 + abs_b, r0))
    return q, r


def verify_divmod_invariant_and_uniqueness():
    print("[1/3] Proving balanced Euclidean divmod identity a = q*b + r and 2|r| <= |b|...")
    s = z3.Solver()
    a, b, q0, r0 = z3.Ints("a b q0 r0")

    abs_b = abs_z3(b)
    # Preconditions of C++ truncated division: a = q0*b + r0, |r0| < |b|, sign(r0) matches sign(a)
    s.add(b != 0)
    s.add(a == q0 * b + r0)
    s.add(abs_z3(r0) < abs_b)
    s.add(z3.If(a >= 0, r0 >= 0, r0 <= 0))

    q, r = divmod_balanced_spec(a, b, q0, r0)

    s.push()
    s.add(z3.Or(a != q * b + r, 2 * abs_z3(r) > abs_b))
    assert s.check() == z3.unsat, "Balanced divmod invariant failed!"
    s.pop()
    print("  -> UNSAT (Verified: a == q*b + r and 2|r| <= |b| for all integers a, b != 0).")

    print("[2/3] Proving strict uniqueness of balanced (q, r) for any odd divisor b = 2*k + 1...")
    s_uniq = z3.Solver()
    a_u, k, q1, r1, q2, r2 = z3.Ints("a_u k q1 r1 q2 r2")
    b_odd = 2 * k + 1
    s_uniq.add(b_odd != 0)
    s_uniq.add(a_u == q1 * b_odd + r1, 2 * abs_z3(r1) <= abs_z3(b_odd))
    s_uniq.add(a_u == q2 * b_odd + r2, 2 * abs_z3(r2) <= abs_z3(b_odd))
    s_uniq.add(z3.Or(q1 != q2, r1 != r2))
    assert s_uniq.check() == z3.unsat, "Uniqueness for odd divisor failed!"
    print("  -> UNSAT (Verified: for every odd divisor b, balanced (q, r) is strictly unique).")


def verify_odd_symmetry():
    print("[3/3] Proving odd symmetry divmod(-a, b) == (-q, -r) and divmod(a, -b) == (-q, r)...")
    s = z3.Solver()
    a, b, q0, r0 = z3.Ints("a b q0 r0")
    abs_b = abs_z3(b)

    s.add(b != 0)
    s.add(a == q0 * b + r0)
    s.add(abs_z3(r0) < abs_b)
    s.add(z3.If(a >= 0, r0 >= 0, r0 <= 0))

    q_pos, r_pos = divmod_balanced_spec(a, b, q0, r0)
    # For -a divided by b, truncated quotient is -q0 and truncated remainder is -r0
    q_nega, r_nega = divmod_balanced_spec(-a, b, -q0, -r0)
    # For a divided by -b, truncated quotient is -q0 and truncated remainder is r0
    q_negb, r_negb = divmod_balanced_spec(a, -b, -q0, r0)

    s.push()
    s.add(z3.Or(q_nega != -q_pos, r_nega != -r_pos, q_negb != -q_pos, r_negb != r_pos))
    assert s.check() == z3.unsat, "Odd symmetry of balanced divmod failed!"
    s.pop()
    print("  -> UNSAT (Verified: divmod(-a, b) = (-q, -r) and divmod(a, -b) = (-q, r) hold universally).")


if __name__ == "__main__":
    print("=== Z3 SMT Formal Verification: TritBigInt Balanced Euclidean DivMod ===")
    verify_divmod_invariant_and_uniqueness()
    verify_odd_symmetry()
    print("ALL BALANCED DIVMOD Z3 PROOFS SUCCEEDED.")
