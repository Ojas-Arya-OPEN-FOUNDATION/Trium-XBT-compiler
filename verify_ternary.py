"""
Empirical Verification & Testing of Balanced Ternary Algorithms & Bit-Slicing
Author: Antigravity Prime Research Engine
"""

import random
import math

# --- 1. Balanced Ternary Integer Conversion ---

def int_to_balanced_ternary(n: int) -> list[int]:
    """Converts any integer n into balanced ternary digits (least significant first)."""
    if n == 0:
        return [0]
    trits = []
    curr = n
    while curr != 0:
        rem = curr % 3
        curr = curr // 3
        if rem == 2:
            rem = -1
            curr += 1
        elif rem == -2:
            rem = 1
            curr -= 1
        trits.append(rem)
    return trits

def balanced_ternary_to_int(trits: list[int]) -> int:
    """Converts balanced ternary digits (least significant first) back to integer."""
    val = 0
    power = 1
    for t in trits:
        val += t * power
        power *= 3
    return val

# --- 2. Arbitrary Real / Float to Balanced Ternary Fixed-Point ---

def float_to_balanced_ternary(x: float, precision_trits: int = 15) -> tuple[list[int], int]:
    """
    Converts any floating point number x into a balanced ternary fixed point representation:
    Returns (trits, scale_exp) where x ≈ sum(t_i * 3^(i - scale_exp)).
    """
    # Scale x by 3^precision_trits and round symmetrically to nearest integer
    scaled = round(x * (3 ** precision_trits))
    trits = int_to_balanced_ternary(scaled)
    return trits, precision_trits

def balanced_ternary_to_float(trits: list[int], scale_exp: int) -> float:
    scaled_int = balanced_ternary_to_int(trits)
    return scaled_int / (3 ** scale_exp)

# --- 3. Packing Trits into Bytes (Base-243: 5 trits per byte) ---

def pack_5trits_to_byte(trits5: list[int]) -> int:
    """Packs 5 trits (each in {-1, 0, 1}) into a single unsigned 8-bit byte [0..242]."""
    assert len(trits5) == 5
    byte_val = 0
    mult = 1
    for t in trits5:
        # map t in {-1, 0, 1} to standard ternary digit in {0, 1, 2}
        digit = t + 1
        byte_val += digit * mult
        mult *= 3
    return byte_val

def unpack_byte_to_5trits(byte_val: int) -> list[int]:
    """Unpacks an 8-bit byte [0..242] back into 5 balanced trits."""
    assert 0 <= byte_val < 243
    trits = []
    curr = byte_val
    for _ in range(5):
        digit = curr % 3
        curr = curr // 3
        trits.append(digit - 1)
    return trits

# --- 4. Dual-Bitplane 2-Bit Representation & Vector Popcount Dot Product ---

def trits_to_bitplane(trit_array: list[int]) -> tuple[int, int]:
    """
    Converts an array of trits (length <= 64) into two 64-bit unsigned integers:
    P (positive bitmask) and N (negative bitmask).
    """
    P = 0
    N = 0
    for i, t in enumerate(trit_array):
        if t == 1:
            P |= (1 << i)
        elif t == -1:
            N |= (1 << i)
    return P, N

def bitplane_dot_product(P_u: int, N_u: int, P_v: int, N_v: int) -> int:
    """
    Computes dot product using bitwise logic and popcount:
    P_dot = (P_u & P_v) | (N_u & N_v)
    N_dot = (P_u & N_v) | (N_u & P_v)
    result = popcount(P_dot) - popcount(N_dot)
    """
    P_dot = (P_u & P_v) | (N_u & N_v)
    N_dot = (P_u & N_v) | (N_u & P_v)
    pop_p = bin(P_dot).count('1')
    pop_n = bin(N_dot).count('1')
    return pop_p - pop_n

# --- 5. Unit Tests & Mathematical Verification ---

def run_tests():
    print("=== Testing Balanced Ternary Conversions ===")
    test_integers = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, 13, -13, 42, -42, 1000, -1000, 2026]
    for n in test_integers:
        trits = int_to_balanced_ternary(n)
        rec = balanced_ternary_to_int(trits)
        assert rec == n, f"Mismatch for {n}: got {rec}"
    print(f"PASS: Verified {len(test_integers)} integer conversions.")

    print("\n=== Testing Real/Float Balanced Ternary Approximation ===")
    test_floats = [0.0, 0.5, -0.5, 3.14159265, -2.71828, 0.1, -0.99]
    for x in test_floats:
        trits, exp = float_to_balanced_ternary(x, precision_trits=16)
        rec = balanced_ternary_to_float(trits, exp)
        err = abs(x - rec)
        assert err < 1e-7, f"Float error too large for {x}: err={err}"
        print(f"  Float {x:10.5f} -> {len(trits)} trits -> Decoded {rec:10.5f} (error: {err:.2e})")
    print("PASS: Verified float approximation.")

    print("\n=== Testing Base-243 5-Trits-per-Byte Packing ===")
    import itertools
    all_5trits = list(itertools.product([-1, 0, 1], repeat=5))
    assert len(all_5trits) == 243
    for t5 in all_5trits:
        b = pack_5trits_to_byte(list(t5))
        assert 0 <= b < 243
        unpacked = unpack_byte_to_5trits(b)
        assert tuple(unpacked) == t5
    print(f"PASS: Verified all 243 5-trit byte packing combinations.")

    print("\n=== Testing Bit-Sliced Popcount Vector Dot Product ===")
    random.seed(42)
    for trial in range(100):
        # Generate random ternary vectors of length 64
        u = random.choices([-1, 0, 1], k=64)
        v = random.choices([-1, 0, 1], k=64)
        expected_dot = sum(ui * vi for ui, vi in zip(u, v))
        
        P_u, N_u = trits_to_bitplane(u)
        P_v, N_v = trits_to_bitplane(v)
        simd_dot = bitplane_dot_product(P_u, N_u, P_v, N_v)
        
        assert simd_dot == expected_dot, f"Dot product mismatch: {simd_dot} vs {expected_dot}"
    print("PASS: Verified 100 random 64-element bit-sliced popcount dot products.")

    print("\n=== ALL EMPIRICAL VERIFICATIONS PASSED ===")

if __name__ == "__main__":
    run_tests()
