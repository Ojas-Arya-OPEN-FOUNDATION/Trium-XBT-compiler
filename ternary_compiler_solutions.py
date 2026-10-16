"""
Ternary Compiler Solutions & Architecture Harness
Implements:
  1. Exact Rational / Repeating Fraction Cycle Detection Engine (Solving Problem 3)
  2. Base-243 In-Register SIMD Decompression Emulator (Solving Problem 1)
  3. Radix-3 Parallel-Prefix Carry-Lookahead Adder (Solving Problem 2)
  4. Sub-Byte Fat Pointer & Branchless 3-Way Selector (Solving Blockers A & B)
"""

import math

# ==============================================================================
# 1. Solving Problem 3: Exact Rational & Repeating Fraction Cycle Detection
# ==============================================================================

class ExactRational:
    """Exact rational number p/q with integer arithmetic (Zero Float Drift)."""
    def __init__(self, p: int, q: int = 1):
        if q == 0:
            raise ZeroDivisionError("Denominator cannot be zero.")
        if q < 0:
            p, q = -p, -q
        g = math.gcd(abs(p), q)
        self.p = p // g
        self.q = q // g

    def __add__(self, other: 'ExactRational') -> 'ExactRational':
        return ExactRational(self.p * other.q + other.p * self.q, self.q * other.q)

    def __sub__(self, other: 'ExactRational') -> 'ExactRational':
        return ExactRational(self.p * other.q - other.p * self.q, self.q * other.q)

    def __mul__(self, other: 'ExactRational') -> 'ExactRational':
        return ExactRational(self.p * other.p, self.q * other.q)

    def __truediv__(self, other: 'ExactRational') -> 'ExactRational':
        return ExactRational(self.p * other.q, self.q * other.p)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExactRational):
            return False
        return self.p == other.p and self.q == other.q

    def __repr__(self) -> str:
        return f"{self.p}/{self.q}" if self.q != 1 else f"{self.p}"

def rational_to_balanced_ternary_expansion(rat: ExactRational, max_digits: int = 30):
    """
    Detects non-repeating prefix and strictly repeating period in base 3.
    Returns: (integer_part, prefix_trits, repeating_period_trits)
    """
    integer_part = round(rat.p / rat.q) # balanced integer part
    # remainder fraction r in [-1/2, 1/2]
    rem_p = rat.p - integer_part * rat.q
    rem_q = rat.q

    seen_remainders = {} # rem -> position
    trits = []
    cycle_start = -1

    curr_p = rem_p
    for i in range(max_digits):
        if curr_p == 0:
            # Terminates!
            return integer_part, trits, []
        if curr_p in seen_remainders:
            cycle_start = seen_remainders[curr_p]
            break
        seen_remainders[curr_p] = i

        # Next trit in balanced ternary
        # Multiply by 3 and round to nearest integer {-1, 0, 1}
        mult_p = curr_p * 3
        digit = round(mult_p / rem_q)
        if digit > 1:
            digit = 1
        elif digit < -1:
            digit = -1
        trits.append(digit)
        curr_p = mult_p - digit * rem_q

    if cycle_start != -1:
        prefix = trits[:cycle_start]
        period = trits[cycle_start:]
        return integer_part, prefix, period
    return integer_part, trits, []

# ==============================================================================
# 2. Solving Problem 1: Fast Base-243 In-Register SIMD Decompression
# ==============================================================================

# 243-entry precomputed table unpacked into Positive and Negative 5-bit masks
DECODE_LUT_P = [0] * 243
DECODE_LUT_N = [0] * 243

for b in range(243):
    curr = b
    p_mask = 0
    n_mask = 0
    for i in range(5):
        digit = curr % 3
        curr //= 3
        trit = digit - 1
        if trit == 1:
            p_mask |= (1 << i)
        elif trit == -1:
            n_mask |= (1 << i)
    DECODE_LUT_P[b] = p_mask
    DECODE_LUT_N[b] = n_mask

def decompress_base243_byte(byte_val: int) -> tuple[int, int]:
    """1-cycle table lookup mimicking x86 vpshufb / ARM tbl decompression."""
    return DECODE_LUT_P[byte_val], DECODE_LUT_N[byte_val]

# ==============================================================================
# 3. Solving Problem 2: Radix-3 Parallel-Prefix Kogge-Stone Adder
# ==============================================================================

class CarryOp:
    """Ternary Carry Transfer Operator: maps C_in {-1, 0, 1} to C_out."""
    def __init__(self, on_neg: int, on_zero: int, on_pos: int):
        self.on_neg = on_neg
        self.on_zero = on_zero
        self.on_pos = on_pos

    def apply(self, c_in: int) -> int:
        if c_in == -1: return self.on_neg
        if c_in == 0:  return self.on_zero
        if c_in == 1:  return self.on_pos
        raise ValueError(f"Invalid carry trit: {c_in}")

    def __matmul__(self, other: 'CarryOp') -> 'CarryOp':
        """Associative composition (self ∘ other)."""
        return CarryOp(
            on_neg=self.apply(other.on_neg),
            on_zero=self.apply(other.on_zero),
            on_pos=self.apply(other.on_pos)
        )

def make_carry_op(a: int, b: int) -> CarryOp:
    s = a + b
    def calc_c(total):
        if total >= 2: return 1
        if total <= -2: return -1
        return 0
    return CarryOp(calc_c(s - 1), calc_c(s), calc_c(s + 1))

def parallel_prefix_kogge_stone_add(A: list[int], B: list[int]) -> list[int]:
    """Computes A + B for ternary vectors in O(log2 N) prefix depth."""
    N = len(A)
    # Stage 1: Generate leaf carry operators
    ops = [make_carry_op(A[i], B[i]) for i in range(N)]

    # Stage 2: Kogge-Stone parallel prefix network
    step = 1
    prefix_ops = list(ops)
    while step < N:
        next_ops = list(prefix_ops)
        for i in range(step, N):
            next_ops[i] = prefix_ops[i] @ prefix_ops[i - step]
        prefix_ops = next_ops
        step *= 2

    # Stage 3: Extract carries and compute sum trits
    carries = [0] * (N + 1)
    for i in range(N):
        carries[i + 1] = prefix_ops[i].apply(0) # with C_in = 0

    sum_trits = []
    for i in range(N):
        total = A[i] + B[i] + carries[i]
        c_out = carries[i + 1]
        s_trit = total - 3 * c_out
        sum_trits.append(s_trit)

    if carries[N] != 0:
        sum_trits.append(carries[N])
    return sum_trits

# ==============================================================================
# 4. Solving Blockers A & B: Fat Pointers & Branchless 3-Way Selection
# ==============================================================================

class TritFatPointer:
    """Sub-byte addressable pointer for a 5-trits-per-byte packed array."""
    def __init__(self, buffer: bytearray, trit_index: int):
        self.buffer = buffer
        self.trit_index = trit_index

    def get(self) -> int:
        byte_idx = self.trit_index // 5
        offset = self.trit_index % 5
        byte_val = self.buffer[byte_idx]
        p_mask, n_mask = decompress_base243_byte(byte_val)
        p_bit = (p_mask >> offset) & 1
        n_bit = (n_mask >> offset) & 1
        return p_bit - n_bit

def branchless_ternary_select(cond_trit: int, if_pos: int, if_zero: int, if_neg: int) -> int:
    """Branchless conditional selection (lowered to hardware cmov/csel)."""
    # cond_trit in {-1, 0, 1}
    # (cond > 0) ? if_pos : ((cond < 0) ? if_neg : if_zero)
    is_pos = int(cond_trit > 0)
    is_neg = int(cond_trit < 0)
    is_zero = 1 - (is_pos | is_neg)
    return is_pos * if_pos + is_neg * if_neg + is_zero * if_zero

# ==============================================================================
# 5. Verification Test Suite
# ==============================================================================

def run_tests():
    print("=== Testing Exact Rational & Fraction Cycle Detection (Problem 3) ===")
    half = ExactRational(1, 2)
    one = half + half
    assert one == ExactRational(1, 1), f"Expected 1/1, got {one}"
    print(f"PASS: 1/2 + 1/2 = {one} (Zero floating drift!)")

    int_part, prefix, period = rational_to_balanced_ternary_expansion(half)
    print(f"Fraction 1/2 in Balanced Ternary: {int_part} + prefix={prefix}, repeating_period={period}")
    # 1/2 in balanced ternary has repeating period [+1]
    assert period == [1], f"Expected repeating period [1], got {period}"
    print("PASS: Exact cycle detection identified 1/2 as repeating [1] (+).")

    fourth = ExactRational(1, 4)
    int_p4, pref4, period4 = rational_to_balanced_ternary_expansion(fourth)
    print(f"Fraction 1/4 in Balanced Ternary: {int_p4} + prefix={pref4}, repeating_period={period4}")
    assert period4 == [1, -1], f"Expected repeating period [1, -1] (+ -), got {period4}"
    print("PASS: Exact cycle detection identified 1/4 as repeating [1, -1] (+ -).")

    print("\n=== Testing Radix-3 Parallel Prefix Adder (Problem 2) ===")
    import random
    random.seed(1337)
    for _ in range(50):
        # Generate 8-trit numbers
        A = random.choices([-1, 0, 1], k=8)
        B = random.choices([-1, 0, 1], k=8)
        val_A = sum(t * (3 ** i) for i, t in enumerate(A))
        val_B = sum(t * (3 ** i) for i, t in enumerate(B))
        expected_sum = val_A + val_B

        sum_trits = parallel_prefix_kogge_stone_add(A, B)
        actual_sum = sum(t * (3 ** i) for i, t in enumerate(sum_trits))
        assert actual_sum == expected_sum, f"Mismatch: {actual_sum} vs {expected_sum}"
    print("PASS: Verified 50 random 8-trit parallel prefix additions against reference.")

    print("\n=== Testing Fat Pointer Sub-Byte Addressing (Blocker A) ===")
    buf = bytearray(2)
    # Pack 10 trits into 2 bytes:
    # byte 0: trits [+1, -1, 0, +1, -1] -> digits [2, 0, 1, 2, 0] -> 2 + 0*3 + 1*9 + 2*27 + 0*81 = 65
    buf[0] = 2 + 0*3 + 1*9 + 2*27 + 0*81
    ptr = TritFatPointer(buf, trit_index=3)
    assert ptr.get() == 1, f"Expected +1 at index 3, got {ptr.get()}"
    ptr2 = TritFatPointer(buf, trit_index=1)
    assert ptr2.get() == -1, f"Expected -1 at index 1, got {ptr2.get()}"
    print("PASS: Sub-byte fat pointer correctly addressed individual trits in byte buffer.")

    print("\n=== Testing Branchless 3-Way Ternary Selector (Blocker B) ===")
    res_pos = branchless_ternary_select(1, 100, 200, 300)
    res_zero = branchless_ternary_select(0, 100, 200, 300)
    res_neg = branchless_ternary_select(-1, 100, 200, 300)
    assert res_pos == 100 and res_zero == 200 and res_neg == 300
    print("PASS: Branchless 3-way ternary selector passed without branching.")

    print("\n=== ALL COMPILER ARCHITECTURE SOLUTIONS PASSED ===")

if __name__ == "__main__":
    run_tests()
