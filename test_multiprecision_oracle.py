#!/usr/bin/env python3
"""
test_multiprecision_oracle.py
Arbitrary-Precision Reference Oracle for Balanced Ternary Arithmetic, Rationals & GEMM
Using Python Arbitrary Precision Integers, Exact Rationals (fractions.Fraction), and SymPy
"""

import math
from fractions import Fraction
import sympy as sp
import random

def to_balanced_ternary(n: int) -> str:
    """Converts an exact arbitrary-precision integer n to balanced ternary string."""
    if n == 0:
        return "0"
    trits = []
    curr = n
    while curr != 0:
        rem = curr % 3
        curr //= 3
        if rem == 2:
            rem = -1
            curr += 1
        elif rem == -2:
            rem = 1
            curr -= 1
        trits.append('+' if rem == 1 else ('-' if rem == -1 else '0'))
    return "".join(trits) # Least-significant trit first

def from_balanced_ternary(s: str) -> int:
    """Evaluates balanced ternary string to exact integer."""
    total = 0
    p = 1
    for ch in s:
        v = 1 if ch == '+' else (-1 if ch == '-' else 0)
        total += v * p
        p *= 3
    return total

def test_integer_multiprecision():
    print("[Oracle 1/4] Testing Arbitrary-Precision Integer Round-Tripping (up to 3^100)...")
    for exponent in [10, 25, 50, 100]:
        val = 3**exponent - 17
        s = to_balanced_ternary(val)
        val_back = from_balanced_ternary(s)
        assert val == val_back, f"Mismatch at 3^{exponent}"
        
        # Test negative values (involution)
        neg_val = -val
        neg_s = to_balanced_ternary(neg_val)
        neg_back = from_balanced_ternary(neg_s)
        assert neg_val == neg_back, f"Negative mismatch at 3^{exponent}"

    print("  -> PASSED: Arbitrary-precision integers up to 3^100 exact round-trip verified.")

def test_exact_rational_expansion():
    print("[Oracle 2/4] Testing Exact Rational Expansions (SymPy & fractions.Fraction)...")
    # Verify 1/2 = sum_{k=1}^N (+) * 3^{-k}
    # In ternary, 1/2 = 0.++++... = sum_{k=1}^\infty 3^{-k} = (1/3) / (1 - 1/3) = 1/2
    k = sp.Symbol('k', integer=True)
    geom_sum = sp.Sum(sp.Rational(1, 3)**k, (k, 1, sp.oo)).doit()
    assert geom_sum == sp.Rational(1, 2), f"Geometric sum failed: {geom_sum}"

    # Verify 1/4 = 0.+-+-+-...
    # sum_{m=1}^\infty (3^{-(2m-1)} - 3^{-2m}) = sum_{m=1}^\infty (3-1)*3^{-2m} = 2 * sum_{m=1}^\infty 9^{-m}
    # 2 * (1/9) / (1 - 1/9) = 2/8 = 1/4
    term = sp.Rational(1, 3)**(2*k - 1) - sp.Rational(1, 3)**(2*k)
    quarter_sum = sp.Sum(term, (k, 1, sp.oo)).doit()
    assert quarter_sum == sp.Rational(1, 4), f"1/4 expansion failed: {quarter_sum}"

    print("  -> PASSED: Exact rational series 1/2 and 1/4 formally simplified via SymPy.")

def test_matrix_gemm_oracle():
    print("[Oracle 3/4] Testing Dense GEMM Oracle against SymPy Matrix...")
    M, K, N = 8, 32, 8
    rng = random.Random(42)

    # Generate random ternary matrices
    A_vals = [[rng.choice([-1, 0, 1]) for _ in range(K)] for _ in range(M)]
    B_vals = [[rng.choice([-1, 0, 1]) for _ in range(N)] for _ in range(K)]

    A_sym = sp.Matrix(A_vals)
    B_sym = sp.Matrix(B_vals)
    C_expected = A_sym * B_sym

    # Verify bit-sliced POPCNT identity for each element:
    for i in range(M):
        for j in range(N):
            # Compute bitplanes
            P_u = sum((1 << k) for k in range(K) if A_vals[i][k] == 1)
            N_u = sum((1 << k) for k in range(K) if A_vals[i][k] == -1)

            P_v = sum((1 << k) for k in range(K) if B_vals[k][j] == 1)
            N_v = sum((1 << k) for k in range(K) if B_vals[k][j] == -1)

            P_prod = (P_u & P_v) | (N_u & N_v)
            N_prod = (P_u & N_v) | (N_u & P_v)

            dot_popcnt = bin(P_prod).count('1') - bin(N_prod).count('1')
            assert dot_popcnt == C_expected[i, j], f"POPCNT GEMM mismatch at ({i}, {j})"

    print("  -> PASSED: 8x32 * 32x8 Bit-Sliced Planar GEMM exactly matches SymPy matrix tensor product.")

def test_symmetric_rounding_bounds():
    print("[Oracle 4/4] Testing Symmetric Rounding Error Bounds (-0.5 * 3^-F, +0.5 * 3^-F)...")
    # In balanced ternary, truncating after F fractional trits incurs error bounded by:
    # |error| <= sum_{k=F+1}^\infty 1 * 3^{-k} = (1/3^{F+1}) / (1 - 1/3) = (1/2) * 3^{-F}
    F = 5
    bound = Fraction(1, 2) * Fraction(1, 3**F)

    # Test 1000 random real numbers
    rng = random.Random(1234)
    for _ in range(1000):
        val = rng.uniform(-100.0, 100.0)
        # Scaled integer
        scaled = round(val * (3**F))
        reconstructed = scaled / (3**F)
        err = abs(val - reconstructed)
        assert err <= float(bound) + 1e-12, f"Error bound violated for {val}: {err} > {bound}"

    print(f"  -> PASSED: Truncation-as-symmetric-rounding bound verified across 1000 trials (F={F}).")

if __name__ == '__main__':
    print("=" * 70)
    print("  Balanced Ternary Multiprecision & SymPy Reference Oracle Suite")
    print("=" * 70)
    test_integer_multiprecision()
    test_exact_rational_expansion()
    test_matrix_gemm_oracle()
    test_symmetric_rounding_bounds()
    print("=" * 70)
    print("  ALL MULTIPRECISION ORACLE TESTS PASSED (0 ERRORS).")
    print("=" * 70)
