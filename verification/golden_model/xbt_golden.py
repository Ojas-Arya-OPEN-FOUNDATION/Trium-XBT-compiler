"""
xbt_golden.py -- Bit-exact gate-level Python golden model of hw/riscv_xbt_extension.sv.

This model is NOT an abstract mathematical specification. It is an exact line-by-line
transcription of the SystemVerilog combinational logic and registers, using only 64-bit
bitwise operations (&, |, ^, ~, <<, >>) and integer arithmetic.

This serves as the primary Golden Model against which:
  1. The SystemVerilog RTL simulation (via PyRTL / Verilator) is diffed bit-by-bit.
  2. The transistor-level netlist is cross-checked.
  3. The mathematical ISA specification (xbt_isa.py) is formally verified.

References:
  - hw/riscv_xbt_extension.sv (SystemVerilog RTL)
  - Ankhdjet & TernaryCore bit-exact verification methodology
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1


def u64(x: int) -> int:
    """Mask to 64-bit unsigned integer."""
    return x & M64


def s64(x: int) -> int:
    """Interpret 64-bit unsigned as signed 64-bit int."""
    val = x & M64
    if val >= (1 << 63):
        return val - (1 << 64)
    return val


# -----------------------------------------------------------------------------
# 1. 64-Trit Dual-Rail Balanced Ternary Half-Adder (xbt_btha64)
#    Matches hw/riscv_xbt_extension.sv lines 21-37
# -----------------------------------------------------------------------------
def xbt_btha64(a_p: int, a_n: int, b_p: int, b_n: int) -> Tuple[int, int, int, int]:
    """
    Combinational 6-gate network per trit lane.
    Returns: (sum_p, sum_n, carry_p, carry_n)
    """
    a_p = u64(a_p)
    a_n = u64(a_n)
    b_p = u64(b_p)
    b_n = u64(b_n)

    carry_p = u64(a_p & b_p)
    carry_n = u64(a_n & b_n)
    sum_p = u64(((a_p ^ b_p) & ~(a_n | b_n)) | carry_n)
    sum_n = u64(((a_n ^ b_n) & ~(a_p | b_p)) | carry_p)

    return sum_p, sum_n, carry_p, carry_n


# -----------------------------------------------------------------------------
# 2. 64-Trit 6-Stage Parallel-Prefix Kogge-Stone Adder (xbt_kogge_stone_adder64)
#    Matches hw/riscv_xbt_extension.sv lines 44-134
# -----------------------------------------------------------------------------
def xbt_kogge_stone_adder64(a_p: int, a_n: int, b_p: int, b_n: int) -> Tuple[int, int, int, int]:
    """
    Computes exact carry-lookahead across 64 trits in 6 Kogge-Stone stages.
    Returns: (sum_p, sum_n, cout_p, cout_n)
    """
    a_p = u64(a_p)
    a_n = u64(a_n)
    b_p = u64(b_p)
    b_n = u64(b_n)

    # Stage 0: Initial per-trit BTHA
    s1_p, s1_n, c1_p, c1_n = xbt_btha64(a_p, a_n, b_p, b_n)

    # Transfer functions across 7 levels (0..6)
    # yp = F(+1), y0 = F(0), ym = F(-1)
    yp_p = [0] * 7
    yp_n = [0] * 7
    y0_p = [0] * 7
    y0_n = [0] * 7
    ym_p = [0] * 7
    ym_n = [0] * 7

    # Level 0 initialization
    yp_p[0] = u64((a_p | b_p) & ~(a_n | b_n))
    yp_n[0] = 0
    y0_p[0] = c1_p
    y0_n[0] = c1_n
    ym_p[0] = 0
    ym_n[0] = u64((a_n | b_n) & ~(a_p | b_p))

    # 6 Kogge-Stone stages: d = 1, 2, 4, 8, 16, 32
    for s in range(6):
        d = 1 << s
        low_mask = (1 << d) - 1

        lp_p = u64((yp_p[s] << d) | low_mask)
        lp_n = u64(yp_n[s] << d)
        l0_p = u64(y0_p[s] << d)
        l0_n = u64(y0_n[s] << d)
        lm_p = u64(ym_p[s] << d)
        lm_n = u64((ym_n[s] << d) | low_mask)

        zp = u64(~(lp_p | lp_n))
        z0 = u64(~(l0_p | l0_n))
        zm = u64(~(lm_p | lm_n))

        # Compose H o L for all 64 lanes in parallel
        yp_p[s + 1] = u64((lp_p & yp_p[s]) | (zp & y0_p[s]) | (lp_n & ym_p[s]))
        yp_n[s + 1] = u64((lp_p & yp_n[s]) | (zp & y0_n[s]) | (lp_n & ym_n[s]))

        y0_p[s + 1] = u64((l0_p & yp_p[s]) | (z0 & y0_p[s]) | (l0_n & ym_p[s]))
        y0_n[s + 1] = u64((l0_p & yp_n[s]) | (z0 & y0_n[s]) | (l0_n & ym_n[s]))

        ym_p[s + 1] = u64((lm_p & yp_p[s]) | (zm & y0_p[s]) | (lm_n & ym_p[s]))
        ym_n[s + 1] = u64((lm_p & yp_n[s]) | (zm & y0_n[s]) | (lm_n & ym_n[s]))

    # Final Sum resolution
    cin_p = u64(y0_p[6] << 1)
    cin_n = u64(y0_n[6] << 1)
    cout_p = (y0_p[6] >> 63) & 1
    cout_n = (y0_n[6] >> 63) & 1

    sum_p = u64(((s1_p ^ cin_p) & ~(s1_n | cin_n)) | (s1_n & cin_n))
    sum_n = u64(((s1_n ^ cin_n) & ~(s1_p | cin_p)) | (s1_p & cin_p))

    return sum_p, sum_n, cout_p, cout_n


# -----------------------------------------------------------------------------
# 2b. Upgraded 64-Trit Kogge-Stone Adder v2 (xbt_kogge_stone_adder64_v2)
#     Matches hw/riscv_xbt_extension.sv lines 140-219
# -----------------------------------------------------------------------------
def xbt_kogge_stone_adder64_v2(a_p: int, a_n: int, b_p: int, b_n: int,
                               cin_p: int = 0, cin_n: int = 0) -> Tuple[int, int, int, int]:
    """Monotone Lattice Absorption Kogge-Stone Adder with explicit carry-in."""
    a_p = u64(a_p)
    a_n = u64(a_n)
    b_p = u64(b_p)
    b_n = u64(b_n)
    cin_p = cin_p & 1
    cin_n = cin_n & 1

    d_p, d_n, c1_p, c1_n = xbt_btha64(a_p, a_n, b_p, b_n)

    # Fold cin into bit 0 of Stage 0
    bit0_cp = (d_p & 1) & cin_p
    bit0_cn = (d_n & 1) & cin_n
    bit0_sp = (((d_p & 1) ^ cin_p) & ~((d_n & 1) | cin_n)) | bit0_cn
    bit0_sn = (((d_n & 1) ^ cin_n) & ~((d_p & 1) | cin_p)) | bit0_cp

    s1_p = u64((d_p & ~1) | bit0_sp)
    s1_n = u64((d_n & ~1) | bit0_sn)

    f0_p = [0] * 6
    f0_n = [0] * 6
    fp_p = [0] * 6
    fp_n = [0] * 6
    fm_p = [0] * 6
    fm_n = [0] * 6

    f0_p[0] = u64((c1_p & ~1) | (((c1_p & 1) | bit0_cp) & ~((c1_n & 1) | bit0_cn)))
    f0_n[0] = u64((c1_n & ~1) | (((c1_n & 1) | bit0_cn) & ~((c1_p & 1) | bit0_cp)))
    fp_p[0] = u64((a_p | b_p) & ~(a_n | b_n))
    fp_n[0] = 0
    fm_p[0] = 0
    fm_n[0] = u64((a_n | b_n) & ~(a_p | b_p))

    for s in range(5):
        shift = 1 << s
        l_fm_p = u64(fm_p[s] << shift)
        l_fm_n = u64(fm_n[s] << shift)
        l_f0_p = u64(f0_p[s] << shift)
        l_f0_n = u64(f0_n[s] << shift)
        l_fp_p = u64(fp_p[s] << shift)
        l_fp_n = u64(fp_n[s] << shift)

        fm_p[s + 1] = u64(fm_p[s] | (f0_p[s] & ~l_fm_n) | (fp_p[s] & l_fm_p))
        fm_n[s + 1] = u64(fp_n[s] | (f0_n[s] & ~l_fm_p) | (fm_n[s] & l_fm_n))
        f0_p[s + 1] = u64(fm_p[s] | (f0_p[s] & ~l_f0_n) | (fp_p[s] & l_f0_p))
        f0_n[s + 1] = u64(fp_n[s] | (f0_n[s] & ~l_f0_p) | (fm_n[s] & l_f0_n))
        fp_p[s + 1] = u64(fm_p[s] | (f0_p[s] & ~l_fp_n) | (fp_p[s] & l_fp_p))
        fp_n[s + 1] = u64(fp_n[s] | (f0_n[s] & ~l_fp_p) | (fm_n[s] & l_fm_n))

    # Stage 5
    final_cout_p = u64(fm_p[5] | (f0_p[5] & ~(u64(f0_n[5] << 32))) | (fp_p[5] & (u64(f0_p[5] << 32))))
    final_cout_n = u64(fp_n[5] | (f0_n[5] & ~(u64(f0_p[5] << 32))) | (fm_n[5] & (u64(f0_n[5] << 32))))
    cout_p = (final_cout_p >> 63) & 1
    cout_n = (final_cout_n >> 63) & 1

    c_in_vec_p = u64(final_cout_p << 1)
    c_in_vec_n = u64(final_cout_n << 1)
    sum_p = u64(((s1_p ^ c_in_vec_p) & ~(s1_n | c_in_vec_n)) | (s1_n & c_in_vec_n))
    sum_n = u64(((s1_n ^ c_in_vec_n) & ~(s1_p | c_in_vec_p)) | (s1_p & c_in_vec_p))

    return sum_p, sum_n, cout_p, cout_n


# -----------------------------------------------------------------------------
# 3. 64-Trit Planar Dot Product Unit (xbt_planar_dot64)
#    Matches hw/riscv_xbt_extension.sv lines 227-250
# -----------------------------------------------------------------------------
def xbt_planar_dot64(a_p: int, a_n: int, b_p: int, b_n: int) -> int:
    """
    Computes prod_p and prod_n across 64 lanes and popcounts them.
    Returns: signed 64-bit integer dot_out in range [-64, +64].
    """
    a_p = u64(a_p)
    a_n = u64(a_n)
    b_p = u64(b_p)
    b_n = u64(b_n)

    prod_p = u64((a_p & b_p) | (a_n & b_n))
    prod_n = u64((a_p & b_n) | (a_n & b_p))

    cnt_p = bin(prod_p).count("1")
    cnt_n = bin(prod_n).count("1")

    return cnt_p - cnt_n


# -----------------------------------------------------------------------------
# 3b. CMOS Compound AOI22 Planar Dot Product Unit (xbt_aoi22_planar_dot64)
#     Matches hw/riscv_xbt_extension.sv lines 259-290
# -----------------------------------------------------------------------------
def xbt_aoi22_planar_dot64(a_p: int, a_n: int, b_p: int, b_n: int) -> Tuple[int, int, int, int, int]:
    """Compound AOI22 gate level model with active-low and active-high rails."""
    a_p = u64(a_p)
    a_n = u64(a_n)
    b_p = u64(b_p)
    b_n = u64(b_n)

    prod_p_bar = u64(~((a_p & b_p) | (a_n & b_n)))
    prod_n_bar = u64(~((a_p & b_n) | (a_n & b_p)))

    prod_p = u64(~prod_p_bar)
    prod_n = u64(~prod_n_bar)

    cnt_p = bin(prod_p).count("1")
    cnt_n = bin(prod_n).count("1")
    dot_out = cnt_p - cnt_n

    return prod_p_bar, prod_n_bar, prod_p, prod_n, dot_out


# -----------------------------------------------------------------------------
# 4. Division-Free Base-243 Hardware Unpacker (xbt_base243_decoder)
#    Matches hw/riscv_xbt_extension.sv lines 456-491
# -----------------------------------------------------------------------------
def xbt_base243_decoder(packed_byte: int) -> Tuple[int, int]:
    """
    Unpacks an 8-bit Base-243 byte [0..242] into 5 Dual-Rail trits in 1 cycle
    using constant reciprocal multipliers (171, 57, 19, 203).
    Returns: (out_p, out_n)
    """
    byte_val = packed_byte & 0xFF
    if byte_val >= 243:
        return 0, 0

    b32 = byte_val & M32
    q1 = ((b32 * 171) & M32) >> 9
    q2 = ((b32 * 57) & M32) >> 9
    q3 = ((b32 * 19) & M32) >> 9
    q4 = ((b32 * 203) & M32) >> 14

    d = [0] * 5
    d[0] = (b32 - q1 * 3) & M32
    d[1] = (q1 - q2 * 3) & M32
    d[2] = (q2 - q3 * 3) & M32
    d[3] = (q3 - q4 * 3) & M32
    d[4] = q4 & M32

    out_p = 0
    out_n = 0
    for k in range(5):
        if d[k] == 2:
            out_p |= 1 << k
        elif d[k] == 0:
            out_n |= 1 << k

    return out_p, out_n


# -----------------------------------------------------------------------------
# 5. Top-Level RISC-V Xbt Execution Unit (riscv_xbt_alu)
#    Matches hw/riscv_xbt_extension.sv lines 503-613
# -----------------------------------------------------------------------------
OPCODE_CUSTOM0 = 0x0B


def riscv_xbt_alu(opcode: int, funct3: int, funct7: int,
                  rs1_p: int, rs1_n: int,
                  rs2_p: int, rs2_n: int,
                  rs1_scalar: int = 0) -> Tuple[int, int, int, int, int, bool]:
    """
    Top-level ALU execution unit.
    Returns: (rd_p, rd_n, rd_aux_p, rd_aux_n, rd_scalar, valid_out)
    """
    opcode = opcode & 0x7F
    funct3 = funct3 & 0x07
    funct7 = funct7 & 0x7F
    rs1_p = u64(rs1_p)
    rs1_n = u64(rs1_n)
    rs2_p = u64(rs2_p)
    rs2_n = u64(rs2_n)
    rs1_scalar = u64(rs1_scalar)

    rd_p = 0
    rd_n = 0
    rd_aux_p = 0
    rd_aux_n = 0
    rd_scalar = 0
    valid_out = (opcode == OPCODE_CUSTOM0)

    if opcode == OPCODE_CUSTOM0:
        if funct3 == 0b000:  # XBT.BTHA
            sp, sn, cp, cn = xbt_btha64(rs1_p, rs1_n, rs2_p, rs2_n)
            rd_p = sp
            rd_n = sn
            rd_aux_p = cp
            rd_aux_n = cn
        elif funct3 == 0b001:  # XBT.KSADD
            sp, sn, cp, cn = xbt_kogge_stone_adder64(rs1_p, rs1_n, rs2_p, rs2_n)
            rd_p = sp
            rd_n = sn
            rd_aux_p = cp & 1
            rd_aux_n = cn & 1
        elif funct3 == 0b010:  # XBT.DOT64
            rd_scalar = xbt_planar_dot64(rs1_p, rs1_n, rs2_p, rs2_n)
        elif funct3 == 0b011:  # XBT.UNPACK243
            up, un = xbt_base243_decoder(rs1_scalar & 0xFF)
            rd_p = up
            rd_n = un
        elif funct3 == 0b100:  # XBT.NEG (0-gate wire swap)
            rd_p = rs1_n
            rd_n = rs1_p
        elif funct3 == 0b101:  # XBT.CSAMAC
            prod_p = u64((rs1_p & rs2_p) | (rs1_n & rs2_n))
            prod_n = u64((rs1_p & rs2_n) | (rs1_n & rs2_p))
            rd_p = u64(((prod_p ^ rs1_p) & ~(prod_n | rs1_n)) | (prod_n & rs1_n))
            rd_n = u64(((prod_n ^ rs1_n) & ~(prod_p | rs1_p)) | (prod_p & rs1_p))
            rd_aux_p = u64(prod_p & rs1_p)
            rd_aux_n = u64(prod_n & rs1_n)
        elif funct3 == 0b110:  # GROUP2 extended ops
            if funct7 == 0x01:  # MUL (per-lane AOI22 product)
                rd_p = u64((rs1_p & rs2_p) | (rs1_n & rs2_n))
                rd_n = u64((rs1_p & rs2_n) | (rs1_n & rs2_p))
            elif funct7 == 0x03:  # SIGN priority encoder
                for i in range(63, -1, -1):
                    p = (rs1_p >> i) & 1
                    n = (rs1_n >> i) & 1
                    if p or n:
                        if p:
                            rd_p = 1
                        else:
                            rd_n = 1
                        break
        else:
            valid_out = False

    return rd_p, rd_n, rd_aux_p, rd_aux_n, rd_scalar, valid_out


# -----------------------------------------------------------------------------
# 6. Pipelined Systolic BTFA Compressor (xbt_systolic_btfa_unit)
#    Matches hw/riscv_xbt_extension.sv lines 358-449
# -----------------------------------------------------------------------------
def xbt_systolic_btfa_unit(a_p: int, a_n: int, b_p: int, b_n: int, c_p: int, c_n: int) -> Tuple[int, int, int, int]:
    """Combinational core of systolic BTFA compressor unit."""
    s1_p, s1_n, c1_p, c1_n = xbt_btha64(a_p, a_n, b_p, b_n)
    s2_p, s2_n, c2_p, c2_n = xbt_btha64(s1_p, s1_n, c_p, c_n)

    sum_p = s2_p
    sum_n = s2_n
    carry_p = u64((c1_p | c2_p) & ~(c1_n | c2_n))
    carry_n = u64((c1_n | c2_n) & ~(c1_p | c2_p))

    return sum_p, sum_n, carry_p, carry_n
