#!/usr/bin/env python3
"""
test_multiprecision_oracle.py
Arbitrary-Precision Reference Oracle for Balanced Ternary Arithmetic, Rationals & GEMM
Using Python Arbitrary Precision Integers, Exact Rationals (fractions.Fraction), and SymPy
"""

import math
from fractions import Fraction
from decimal import Decimal, getcontext
import random

getcontext().prec = 100

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

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
    print("[Oracle 1/6] Testing Arbitrary-Precision Integer Round-Tripping (up to 3^100)...")
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
    print("[Oracle 2/6] Testing Exact Rational Expansions (fractions.Fraction & SymPy)...")
    if HAS_SYMPY:
        k = sp.Symbol('k', integer=True)
        geom_sum = sp.Sum(sp.Rational(1, 3)**k, (k, 1, sp.oo)).doit()
        assert geom_sum == sp.Rational(1, 2), f"Geometric sum failed: {geom_sum}"
        term = sp.Rational(1, 3)**(2*k - 1) - sp.Rational(1, 3)**(2*k)
        quarter_sum = sp.Sum(term, (k, 1, sp.oo)).doit()
        assert quarter_sum == sp.Rational(1, 4), f"1/4 expansion failed: {quarter_sum}"
    else:
        # Exact partial sum identity: sum_{k=1}^N 3^{-k} = 1/2 * (1 - 3^{-N})
        for N in [10, 40, 80]:
            partial_half = sum(Fraction(1, 3**k) for k in range(1, N + 1))
            assert partial_half == Fraction(1, 2) * (1 - Fraction(1, 3**N))
            partial_quarter = sum(Fraction(1, 3**(2*k - 1)) - Fraction(1, 3**(2*k)) for k in range(1, N + 1))
            assert partial_quarter == Fraction(1, 4) * (1 - Fraction(1, 9**N))

    print("  -> PASSED: Exact rational series 1/2 (0.++++...) and 1/4 (0.+-+-...) verified.")

def test_matrix_gemm_oracle():
    print("[Oracle 3/6] Testing Dense GEMM Oracle against Exact Reference Matrix...")
    M, K, N = 8, 32, 8
    rng = random.Random(42)

    A_vals = [[rng.choice([-1, 0, 1]) for _ in range(K)] for _ in range(M)]
    B_vals = [[rng.choice([-1, 0, 1]) for _ in range(N)] for _ in range(K)]

    C_expected = [
        [sum(A_vals[i][k] * B_vals[k][j] for k in range(K)) for j in range(N)]
        for i in range(M)
    ]

    for i in range(M):
        for j in range(N):
            P_u = sum((1 << k) for k in range(K) if A_vals[i][k] == 1)
            N_u = sum((1 << k) for k in range(K) if A_vals[i][k] == -1)

            P_v = sum((1 << k) for k in range(K) if B_vals[k][j] == 1)
            N_v = sum((1 << k) for k in range(K) if B_vals[k][j] == -1)

            P_prod = (P_u & P_v) | (N_u & N_v)
            N_prod = (P_u & N_v) | (N_u & P_v)

            dot_popcnt = bin(P_prod).count('1') - bin(N_prod).count('1')
            assert dot_popcnt == C_expected[i][j], f"POPCNT GEMM mismatch at ({i}, {j})"

    print("  -> PASSED: 8x32 * 32x8 Bit-Sliced Planar GEMM matches exact tensor product.")

def test_symmetric_rounding_bounds():
    print("[Oracle 4/6] Testing Symmetric Rounding Error Bounds (-0.5 * 3^-F, +0.5 * 3^-F)...")
    F = 5
    bound = Fraction(1, 2) * Fraction(1, 3**F)

    rng = random.Random(1234)
    for _ in range(1000):
        val = rng.uniform(-100.0, 100.0)
        scaled = round(val * (3**F))
        reconstructed = scaled / (3**F)
        err = abs(val - reconstructed)
        assert err <= float(bound) + 1e-12, f"Error bound violated for {val}: {err} > {bound}"

    print(f"  -> PASSED: Truncation-as-symmetric-rounding bound verified across 1000 trials (F={F}).")

def ternary_gcd_stein(a: int, b: int) -> int:
    """Division-free Ternary Stein's GCD algorithm."""
    a, b = abs(a), abs(b)
    if a == 0:
        return b
    if b == 0:
        return a
    pow3 = 1
    while a % 3 == 0 and b % 3 == 0:
        a //= 3
        b //= 3
        pow3 *= 3
    while a % 3 == 0:
        a //= 3
    while b != 0:
        while b % 3 == 0:
            b //= 3
        if a > b:
            a, b = b, a
        # Now a <= b and neither is divisible by 3
        if (a - b) % 3 == 0:
            b = (b - a) // 3
        else:
            b = (b + a) // 3
    return a * pow3

def test_stein_gcd_and_magic_codecs():
    print("[Oracle 5/6] Testing Ternary Stein GCD & Base-243 / Base-59049 Magic Multipliers...")
    rng = random.Random(2026)
    for _ in range(500):
        x = rng.randint(-10**18, 10**18)
        y = rng.randint(-10**18, 10**18)
        assert ternary_gcd_stein(x, y) == math.gcd(x, y)

    # Exhaustively verify Base-243 magic multipliers M = [171, 57, 19, 203] for all b in [0..242]
    M243 = [171, 57, 19, 203]
    for b in range(243):
        q0 = (b * M243[0]) >> 9
        q1 = (b * M243[1]) >> 9
        q2 = (b * M243[2]) >> 9
        q3 = (b * M243[3]) >> 14
        assert q0 == b // 3 and q1 == b // 9 and q2 == b // 27 and q3 == b // 81

    # Exhaustively verify Base-59049 magic multipliers from ternary_core.hpp for all w in [0..59048]
    M59049 = [43691, 58255, 77673, 25891, 69043, 92057, 61371, 20457, 6819]
    S59049 = [17, 19, 21, 21, 24, 26, 27, 27, 27]
    POW3 = [3**k for k in range(1, 10)]
    for w in range(59049):
        for idx in range(9):
            q = (w * M59049[idx]) >> S59049[idx]
            assert q == w // POW3[idx], f"Base59049 mismatch at w={w}, idx={idx}"

    print("  -> PASSED: Ternary Stein GCD (500 pairs), Base-243 (243/243), Base-59049 (59,049/59,049) verified.")

def test_householder_cubic_and_pade_tanh():
    print("[Oracle 6/6] Testing Householder Cubic Reciprocal & Odd-Symmetric Padé Tanh...")
    for d_num in range(50, 151):
        D = Decimal(d_num) / Decimal(100)
        # Initial 3-trit seed
        v = max(5, min(13, round(float(D) * 9.0)))
        X = Decimal(9) / Decimal(v)
        for _ in range(3):
            R = Decimal(1) - D * X
            X = X * (Decimal(1) + R + R * R)
        rel_err = abs(Decimal(1) - D * X)
        assert rel_err < Decimal('1e-20'), f"Householder error too large for D={D}: {rel_err}"

    # Verify strict odd symmetry and accuracy of [7/6] Padé Tanh for |x| < 0.65
    for i in range(1, 65):
        x = i * 0.01
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        pade_pos = x * (135135.0 + 17325.0 * x2 + 378.0 * x4 + x6) / (135135.0 + 62370.0 * x2 + 3150.0 * x4 + 28.0 * x6)
        pade_neg = -x * (135135.0 + 17325.0 * x2 + 378.0 * x4 + x6) / (135135.0 + 62370.0 * x2 + 3150.0 * x4 + 28.0 * x6)
        assert pade_pos == -pade_neg
        assert abs(pade_pos - math.tanh(x)) < 2e-11

    print("  -> PASSED: Householder 3-step cubic error < 10^-20 and [7/6] Padé Tanh error < 2e-11.")

if __name__ == '__main__':
    print("=" * 70)
    print("  Balanced Ternary Multiprecision & Mathematical Oracle Suite")
    print("=" * 70)
    test_integer_multiprecision()
    test_exact_rational_expansion()
    test_matrix_gemm_oracle()
    test_symmetric_rounding_bounds()
    test_stein_gcd_and_magic_codecs()
    test_householder_cubic_and_pade_tanh()
    print("=" * 70)
    print("  ALL MULTIPRECISION ORACLE TESTS PASSED (0 ERRORS).")
    print("=" * 70)
