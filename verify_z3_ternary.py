"""
Automated Formal SMT Verification Suite for Balanced Ternary Logic & Magic Multipliers
Author: Antigravity Prime Research Engine
Solver: Z3 SMT Solver (QF_BV & QF_LIA)
"""

import z3

def test_btha_soundness():
    """Formally proves BTHA using Z3 bit-blasting."""
    print(">>> Proving BTHA (Balanced Ternary Half-Adder) in Z3...")
    solver = z3.Solver()

    # Inputs: Booleans for positive and negative rails
    PA, NA = z3.Bools('PA NA')
    PB, NB = z3.Bools('PB NB')

    # Canonical invariant: disjointness
    solver.add(z3.Not(z3.And(PA, NA)))
    solver.add(z3.Not(z3.And(PB, NB)))

    # Dual-rail BTHA equations
    PC = z3.And(PA, PB)
    NC = z3.And(NA, NB)

    PS = z3.Or(z3.And(z3.Xor(PA, PB), z3.Not(z3.Or(NA, NB))), NC)
    NS = z3.Or(z3.And(z3.Xor(NA, NB), z3.Not(z3.Or(PA, PB))), PC)

    # 1. Prove disjointness of outputs
    # To prove property P, assert Not(P) and check for UNSAT
    solver.push()
    solver.add(z3.And(PC, NC))
    assert solver.check() == z3.unsat, "BTHA Carry disjointness failed!"
    solver.pop()

    solver.push()
    solver.add(z3.And(PS, NS))
    assert solver.check() == z3.unsat, "BTHA Sum disjointness failed!"
    solver.pop()

    # 2. Prove algebraic integer equivalence: Val(A) + Val(B) == 3*Val(C) + Val(S)
    val_A = z3.If(PA, 1, 0) - z3.If(NA, 1, 0)
    val_B = z3.If(PB, 1, 0) - z3.If(NB, 1, 0)
    val_C = z3.If(PC, 1, 0) - z3.If(NC, 1, 0)
    val_S = z3.If(PS, 1, 0) - z3.If(NS, 1, 0)

    solver.push()
    solver.add(val_A + val_B != 3 * val_C + val_S)
    assert solver.check() == z3.unsat, "BTHA Arithmetic soundness failed!"
    solver.pop()
    print("    [Z3 PROVED] BTHA satisfies output disjointness and exact algebraic soundness!")

def test_btfa_soundness():
    """Formally proves BTFA (3-to-2 Compressor) and the No-Overlap Carry Theorem in Z3."""
    print(">>> Proving BTFA (Balanced Ternary Full-Adder / 3-to-2 Compressor) in Z3...")
    solver = z3.Solver()

    PA, NA = z3.Bools('PA NA')
    PB, NB = z3.Bools('PB NB')
    PD, ND = z3.Bools('PD ND')

    solver.add(z3.Not(z3.And(PA, NA)))
    solver.add(z3.Not(z3.And(PB, NB)))
    solver.add(z3.Not(z3.And(PD, ND)))

    # Stage 1: Half-Adder(A, B) -> (C1, S1)
    PC1 = z3.And(PA, PB)
    NC1 = z3.And(NA, NB)
    PS1 = z3.Or(z3.And(z3.Xor(PA, PB), z3.Not(z3.Or(NA, NB))), NC1)
    NS1 = z3.Or(z3.And(z3.Xor(NA, NB), z3.Not(z3.Or(PA, PB))), PC1)

    # Stage 2: Half-Adder(S1, D) -> (C2, S)
    PC2 = z3.And(PS1, PD)
    NC2 = z3.And(NS1, ND)
    PS  = z3.Or(z3.And(z3.Xor(PS1, PD), z3.Not(z3.Or(NS1, ND))), NC2)
    NS  = z3.Or(z3.And(z3.Xor(NS1, ND), z3.Not(z3.Or(PS1, PD))), PC2)

    # 1. Prove the No-Overlap Carry Theorem: C1 and C2 NEVER share the same non-zero sign!
    # (PC1 & PC2 == 0) and (NC1 & NC2 == 0)
    solver.push()
    solver.add(z3.And(PC1, PC2))
    assert solver.check() == z3.unsat, "Carry positive overlap occurred!"
    solver.pop()

    solver.push()
    solver.add(z3.And(NC1, NC2))
    assert solver.check() == z3.unsat, "Carry negative overlap occurred!"
    solver.pop()

    # Merged Carry C
    PC = z3.And(z3.Or(PC1, PC2), z3.Not(z3.Or(NC1, NC2)))
    NC = z3.And(z3.Or(NC1, NC2), z3.Not(z3.Or(PC1, PC2)))

    # 2. Prove algebraic equivalence: Val(A) + Val(B) + Val(D) == 3*Val(C) + Val(S)
    val_A = z3.If(PA, 1, 0) - z3.If(NA, 1, 0)
    val_B = z3.If(PB, 1, 0) - z3.If(NB, 1, 0)
    val_D = z3.If(PD, 1, 0) - z3.If(ND, 1, 0)
    val_C = z3.If(PC, 1, 0) - z3.If(NC, 1, 0)
    val_S = z3.If(PS, 1, 0) - z3.If(NS, 1, 0)

    solver.push()
    solver.add(val_A + val_B + val_D != 3 * val_C + val_S)
    assert solver.check() == z3.unsat, "BTFA Arithmetic soundness failed!"
    solver.pop()
    print("    [Z3 PROVED] BTFA satisfies the No-Overlap Carry Theorem and exact 3-to-2 compression!")

def test_multiplication_soundness():
    """Formally proves Dual-Rail bitplane multiplication in Z3."""
    print(">>> Proving Trit Multiplication in Z3...")
    solver = z3.Solver()
    PA, NA = z3.Bools('PA NA')
    PB, NB = z3.Bools('PB NB')

    solver.add(z3.Not(z3.And(PA, NA)))
    solver.add(z3.Not(z3.And(PB, NB)))

    P_out = z3.Or(z3.And(PA, PB), z3.And(NA, NB))
    N_out = z3.Or(z3.And(PA, NB), z3.And(NA, PB))

    # Disjointness
    solver.push()
    solver.add(z3.And(P_out, N_out))
    assert solver.check() == z3.unsat, "Multiplication output disjointness failed!"
    solver.pop()

    # Algebraic soundness
    val_A = z3.If(PA, 1, 0) - z3.If(NA, 1, 0)
    val_B = z3.If(PB, 1, 0) - z3.If(NB, 1, 0)
    val_out = z3.If(P_out, 1, 0) - z3.If(N_out, 1, 0)

    solver.push()
    solver.add(val_A * val_B != val_out)
    assert solver.check() == z3.unsat, "Multiplication algebraic soundness failed!"
    solver.pop()
    print("    [Z3 PROVED] Dual-Rail bitplane multiplication is algebraically exact!")

def test_magic_multipliers():
    """Formally proves that division-free reciprocal multipliers are exact for all bytes in [0..242]."""
    print(">>> Proving Base-243 Division-Free Magic Multipliers in Z3...")
    solver = z3.Solver()
    B = z3.BitVec('B', 32)
    solver.add(z3.UGE(B, 0), z3.ULE(B, 242))

    # Real integer divisions
    q1_real = z3.UDiv(B, 3)
    q2_real = z3.UDiv(B, 9)
    q3_real = z3.UDiv(B, 27)
    q4_real = z3.UDiv(B, 81)

    # Magic multiplier formulas
    q1_magic = (B * 171) >> 9
    q2_magic = (B * 57) >> 9
    q3_magic = (B * 19) >> 9
    q4_magic = (B * 203) >> 14

    solver.push()
    solver.add(z3.Or(
        q1_real != q1_magic,
        q2_real != q2_magic,
        q3_real != q3_magic,
        q4_real != q4_magic
    ))
    assert solver.check() == z3.unsat, "Magic multiplier equivalence failed!"
    solver.pop()
    print("    [Z3 PROVED] Magic multipliers (171, 57, 19, 203) are bit-exact for all 243 byte states!")

def test_lut_hex_constants():
    """Proves that 3-input LUT constants 0x14, 0x54, 0xCA, 0x96 evaluate correctly."""
    print(">>> Proving Hardware 3-Input LUT Hex Constants in Z3...")
    # Basis vectors
    mask_a = 0xF0
    mask_b = 0xCC
    mask_c = 0xAA

    # 1. BTHA Sum filter: (a ^ b) & ~c
    assert ((mask_a ^ mask_b) & ~mask_c) & 0xFF == 0x14, "immLut 0x14 mismatch!"

    # 2. BTFA Carry merge: (a | b) & ~c
    assert ((mask_a | mask_b) & ~mask_c) & 0xFF == 0x54, "immLut 0x54 mismatch!"

    # 3. Kleene 2-way select: (a & b) | (~a & c)
    assert ((mask_a & mask_b) | (~mask_a & mask_c)) & 0xFF == 0xCA, "immLut 0xCA mismatch!"

    # 4. XOR-Accumulate: a ^ b ^ c
    assert (mask_a ^ mask_b ^ mask_c) & 0xFF == 0x96, "immLut 0x96 mismatch!"
    print("    [VERIFIED] All hardware immLut constants (0x14, 0x54, 0xCA, 0x96) verified bit-exact!")

if __name__ == "__main__":
    print("==================================================================")
    print("  Z3 SMT Formal Verification: Balanced Ternary Logic & Circuits  ")
    print("==================================================================")
    test_btha_soundness()
    test_btfa_soundness()
    test_multiplication_soundness()
    test_magic_multipliers()
    test_lut_hex_constants()
    print("==================================================================")
    print("  ALL Z3 SMT THEOREMS MATHEMATICALLY PROVEN WITH ZERO COUNTEREXAMPLES!")
    print("==================================================================")
