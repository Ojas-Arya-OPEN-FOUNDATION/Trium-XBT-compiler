"""
pyrtl_xbt.py -- Cycle-accurate gate-level PyRTL hardware simulation netlist
for the RISC-V Xbt Extension (hw/riscv_xbt_extension.sv).

Builds hardware blocks using PyRTL gates, wires, and muxes:
  - xbt_btha64_block
  - xbt_kogge_stone_adder64_block
  - xbt_planar_dot64_block
  - xbt_base243_decoder_block
  - riscv_xbt_alu_block (top-level execution unit)
"""

from __future__ import annotations
import pyrtl
from typing import Tuple, Dict, Any


def build_btha64_block() -> Tuple[pyrtl.Block, pyrtl.Simulation]:
    """Build isolated PyRTL hardware simulation block for xbt_btha64."""
    block = pyrtl.Block()
    with pyrtl.set_working_block(block):
        ap = pyrtl.Input(64, 'a_p')
        an = pyrtl.Input(64, 'a_n')
        bp = pyrtl.Input(64, 'b_p')
        bn = pyrtl.Input(64, 'b_n')

        sp = pyrtl.Output(64, 'sum_p')
        sn = pyrtl.Output(64, 'sum_n')
        cp = pyrtl.Output(64, 'carry_p')
        cn = pyrtl.Output(64, 'carry_n')

        # Gate logic exactly as in hw/riscv_xbt_extension.sv
        carry_p = ap & bp
        carry_n = an & bn
        sum_p = ((ap ^ bp) & ~(an | bn)) | carry_n
        sum_n = ((an ^ bn) & ~(ap | bp)) | carry_p

        cp <<= carry_p
        cn <<= carry_n
        sp <<= sum_p
        sn <<= sum_n

    trace = pyrtl.SimulationTrace(block=block)
    sim = pyrtl.Simulation(tracer=trace, block=block)
    return block, sim


def build_dot64_block() -> Tuple[pyrtl.Block, pyrtl.Simulation]:
    """Build isolated PyRTL hardware simulation block for xbt_planar_dot64."""
    block = pyrtl.Block()
    with pyrtl.set_working_block(block):
        ap = pyrtl.Input(64, 'a_p')
        an = pyrtl.Input(64, 'a_n')
        bp = pyrtl.Input(64, 'b_p')
        bn = pyrtl.Input(64, 'b_n')

        dot_out = pyrtl.Output(64, 'dot_out')

        prod_p = (ap & bp) | (an & bn)
        prod_n = (ap & bn) | (an & bp)

        # Tree reduction popcount (7-bit counters for 64 inputs)
        p_bits = [prod_p[i].zero_extended(7) for i in range(64)]
        n_bits = [prod_n[i].zero_extended(7) for i in range(64)]

        cnt_p = pyrtl.tree_reduce(lambda x, y: x + y, p_bits)
        cnt_n = pyrtl.tree_reduce(lambda x, y: x + y, n_bits)

        # Pad to 64 bits and subtract (two's complement)
        p_ext = cnt_p.zero_extended(64)
        n_ext = cnt_n.zero_extended(64)
        dot_out <<= p_ext - n_ext

    trace = pyrtl.SimulationTrace(block=block)
    sim = pyrtl.Simulation(tracer=trace, block=block)
    return block, sim


def build_unpack243_block() -> Tuple[pyrtl.Block, pyrtl.Simulation]:
    """Build isolated PyRTL hardware simulation block for xbt_base243_decoder."""
    block = pyrtl.Block()
    with pyrtl.set_working_block(block):
        byte_in = pyrtl.Input(8, 'packed_byte')
        out_p = pyrtl.Output(64, 'out_p')
        out_n = pyrtl.Output(64, 'out_n')

        # Reciprocals: 171, 57, 19, 203
        b32 = byte_in.zero_extended(32)
        q1 = (b32 * pyrtl.Const(171, 32))[9:41].zero_extended(32)
        q2 = (b32 * pyrtl.Const(57, 32))[9:41].zero_extended(32)
        q3 = (b32 * pyrtl.Const(19, 32))[9:41].zero_extended(32)
        q4 = (b32 * pyrtl.Const(203, 32))[14:46].zero_extended(32)

        d0 = (b32 - (q1 * pyrtl.Const(3, 32))[0:32])[0:32]
        d1 = (q1 - (q2 * pyrtl.Const(3, 32))[0:32])[0:32]
        d2 = (q2 - (q3 * pyrtl.Const(3, 32))[0:32])[0:32]
        d3 = (q3 - (q4 * pyrtl.Const(3, 32))[0:32])[0:32]
        d4 = q4[0:32]

        digits = [d0[0:2], d1[0:2], d2[0:2], d3[0:2], d4[0:2]]

        p_bits = []
        n_bits = []
        for d in digits:
            p_bits.append(d == pyrtl.Const(2, 2))
            n_bits.append(d == pyrtl.Const(0, 2))

        # Check byte < 243
        valid = byte_in < pyrtl.Const(243, 8)

        # Pack lower 5 bits into 64-bit vector
        p_vec = pyrtl.concat_list(p_bits).zero_extended(64)
        n_vec = pyrtl.concat_list(n_bits).zero_extended(64)

        out_p <<= pyrtl.select(valid, p_vec, pyrtl.Const(0, 64))
        out_n <<= pyrtl.select(valid, n_vec, pyrtl.Const(0, 64))

    trace = pyrtl.SimulationTrace(block=block)
    sim = pyrtl.Simulation(tracer=trace, block=block)
    return block, sim


def build_xbt_alu_block() -> Tuple[pyrtl.Block, pyrtl.Simulation]:
    """
    Build top-level PyRTL hardware simulation block for riscv_xbt_alu.
    Handles opcode=0x0B with funct3 dispatch:
      3'b000: BTHA
      3'b001: KSADD (direct lookahead tree)
      3'b010: DOT64
      3'b011: UNPACK243
      3'b100: NEG
      3'b101: CSAMAC
    """
    block = pyrtl.Block()
    with pyrtl.set_working_block(block):
        opcode = pyrtl.Input(7, 'opcode')
        funct3 = pyrtl.Input(3, 'funct3')
        funct7 = pyrtl.Input(7, 'funct7')
        rs1_p = pyrtl.Input(64, 'rs1_p')
        rs1_n = pyrtl.Input(64, 'rs1_n')
        rs2_p = pyrtl.Input(64, 'rs2_p')
        rs2_n = pyrtl.Input(64, 'rs2_n')
        rs1_scalar = pyrtl.Input(64, 'rs1_scalar')

        rd_p = pyrtl.Output(64, 'rd_p')
        rd_n = pyrtl.Output(64, 'rd_n')
        rd_aux_p = pyrtl.Output(64, 'rd_aux_p')
        rd_aux_n = pyrtl.Output(64, 'rd_aux_n')
        rd_scalar = pyrtl.Output(64, 'rd_scalar')
        valid_out = pyrtl.Output(1, 'valid_out')

        is_custom0 = (opcode == pyrtl.Const(0x0B, 7))
        valid_out <<= is_custom0

        # Sub-unit 1: BTHA
        btha_cp = rs1_p & rs2_p
        btha_cn = rs1_n & rs2_n
        btha_sp = ((rs1_p ^ rs2_p) & ~(rs1_n | rs2_n)) | btha_cn
        btha_sn = ((rs1_n ^ rs2_n) & ~(rs1_p | rs2_p)) | btha_cp

        # Sub-unit 2: Kogge-Stone Adder (6 stages)
        yp_p = [None] * 7
        yp_n = [None] * 7
        y0_p = [None] * 7
        y0_n = [None] * 7
        ym_p = [None] * 7
        ym_n = [None] * 7

        yp_p[0] = (rs1_p | rs2_p) & ~(rs1_n | rs2_n)
        yp_n[0] = pyrtl.Const(0, 64)
        y0_p[0] = btha_cp
        y0_n[0] = btha_cn
        ym_p[0] = pyrtl.Const(0, 64)
        ym_n[0] = (rs1_n | rs2_n) & ~(rs1_p | rs2_p)

        for s in range(6):
            d = 1 << s
            low_mask = pyrtl.Const((1 << d) - 1, d)

            lp_p = pyrtl.concat(yp_p[s][0:64-d], low_mask)
            lp_n = pyrtl.concat(yp_n[s][0:64-d], pyrtl.Const(0, d))
            l0_p = pyrtl.concat(y0_p[s][0:64-d], pyrtl.Const(0, d))
            l0_n = pyrtl.concat(y0_n[s][0:64-d], pyrtl.Const(0, d))
            lm_p = pyrtl.concat(ym_p[s][0:64-d], pyrtl.Const(0, d))
            lm_n = pyrtl.concat(ym_n[s][0:64-d], low_mask)

            zp = ~(lp_p | lp_n)
            z0 = ~(l0_p | l0_n)
            zm = ~(lm_p | lm_n)

            yp_p[s+1] = (lp_p & yp_p[s]) | (zp & y0_p[s]) | (lp_n & ym_p[s])
            yp_n[s+1] = (lp_p & yp_n[s]) | (zp & y0_n[s]) | (lp_n & ym_n[s])
            y0_p[s+1] = (l0_p & yp_p[s]) | (z0 & y0_p[s]) | (l0_n & ym_p[s])
            y0_n[s+1] = (l0_p & yp_n[s]) | (z0 & y0_n[s]) | (l0_n & ym_n[s])
            ym_p[s+1] = (lm_p & yp_p[s]) | (zm & y0_p[s]) | (lm_n & ym_p[s])
            ym_n[s+1] = (lm_p & yp_n[s]) | (zm & y0_n[s]) | (lm_n & ym_n[s])

        ks_cin_p = pyrtl.concat(y0_p[6][0:63], pyrtl.Const(0, 1))
        ks_cin_n = pyrtl.concat(y0_n[6][0:63], pyrtl.Const(0, 1))
        ks_cout_p = y0_p[6][63]
        ks_cout_n = y0_n[6][63]

        ks_sp = ((btha_sp ^ ks_cin_p) & ~(btha_sn | ks_cin_n)) | (btha_sn & ks_cin_n)
        ks_sn = ((btha_sn ^ ks_cin_n) & ~(btha_sp | ks_cin_p)) | (btha_sp & ks_cin_p)

        # Sub-unit 3: Planar Dot64
        dot_pp = (rs1_p & rs2_p) | (rs1_n & rs2_n)
        dot_pn = (rs1_p & rs2_n) | (rs1_n & rs2_p)
        cnt_p = pyrtl.tree_reduce(lambda x, y: x + y, [dot_pp[i].zero_extended(7) for i in range(64)]).zero_extended(64)
        cnt_n = pyrtl.tree_reduce(lambda x, y: x + y, [dot_pn[i].zero_extended(7) for i in range(64)]).zero_extended(64)
        dot_res = cnt_p - cnt_n

        # Sub-unit 4: Base-243 Unpack
        byte_val = rs1_scalar[0:8]
        b32 = byte_val.zero_extended(32)
        q1 = (b32 * pyrtl.Const(171, 32))[9:41].zero_extended(32)
        q2 = (b32 * pyrtl.Const(57, 32))[9:41].zero_extended(32)
        q3 = (b32 * pyrtl.Const(19, 32))[9:41].zero_extended(32)
        q4 = (b32 * pyrtl.Const(203, 32))[14:46].zero_extended(32)
        d0 = (b32 - (q1 * pyrtl.Const(3, 32))[0:32])[0:32]
        d1 = (q1 - (q2 * pyrtl.Const(3, 32))[0:32])[0:32]
        d2 = (q2 - (q3 * pyrtl.Const(3, 32))[0:32])[0:32]
        d3 = (q3 - (q4 * pyrtl.Const(3, 32))[0:32])[0:32]
        d4 = q4[0:32]
        up_bits = [(d == pyrtl.Const(2, 2)) for d in [d0[0:2], d1[0:2], d2[0:2], d3[0:2], d4[0:2]]]
        un_bits = [(d == pyrtl.Const(0, 2)) for d in [d0[0:2], d1[0:2], d2[0:2], d3[0:2], d4[0:2]]]
        unpack_valid = byte_val < pyrtl.Const(243, 8)
        raw_up = pyrtl.concat_list(up_bits).zero_extended(64)
        raw_un = pyrtl.concat_list(un_bits).zero_extended(64)
        up_res = pyrtl.select(unpack_valid, raw_up, pyrtl.Const(0, 64))
        un_res = pyrtl.select(unpack_valid, raw_un, pyrtl.Const(0, 64))

        # Sub-unit 5: NEG (wire swap)
        neg_p = rs1_n
        neg_n = rs1_p

        # Sub-unit 6: CSAMAC
        csa_pp = (rs1_p & rs2_p) | (rs1_n & rs2_n)
        csa_pn = (rs1_p & rs2_n) | (rs1_n & rs2_p)
        csa_sp = ((csa_pp ^ rs1_p) & ~(csa_pn | rs1_n)) | (csa_pn & rs1_n)
        csa_sn = ((csa_pn ^ rs1_n) & ~(csa_pp | rs1_p)) | (csa_pp & rs1_p)
        csa_cp = csa_pp & rs1_p
        csa_cn = csa_pn & rs1_n

        # Output MUX based on funct3
        f0 = (funct3 == pyrtl.Const(0, 3))
        f1 = (funct3 == pyrtl.Const(1, 3))
        f2 = (funct3 == pyrtl.Const(2, 3))
        f3 = (funct3 == pyrtl.Const(3, 3))
        f4 = (funct3 == pyrtl.Const(4, 3))
        f5 = (funct3 == pyrtl.Const(5, 3))

        zero64 = pyrtl.Const(0, 64)

        mux_rd_p = pyrtl.select(f0, btha_sp,
                   pyrtl.select(f1, ks_sp,
                   pyrtl.select(f3, up_res,
                   pyrtl.select(f4, neg_p,
                   pyrtl.select(f5, csa_sp, zero64)))))

        mux_rd_n = pyrtl.select(f0, btha_sn,
                   pyrtl.select(f1, ks_sn,
                   pyrtl.select(f3, un_res,
                   pyrtl.select(f4, neg_n,
                   pyrtl.select(f5, csa_sn, zero64)))))

        mux_aux_p = pyrtl.select(f0, btha_cp,
                    pyrtl.select(f1, ks_cout_p.zero_extended(64),
                    pyrtl.select(f5, csa_cp, zero64)))

        mux_aux_n = pyrtl.select(f0, btha_cn,
                    pyrtl.select(f1, ks_cout_n.zero_extended(64),
                    pyrtl.select(f5, csa_cn, zero64)))

        mux_scalar = pyrtl.select(f2, dot_res, zero64)

        rd_p <<= pyrtl.select(is_custom0, mux_rd_p, zero64)
        rd_n <<= pyrtl.select(is_custom0, mux_rd_n, zero64)
        rd_aux_p <<= pyrtl.select(is_custom0, mux_aux_p, zero64)
        rd_aux_n <<= pyrtl.select(is_custom0, mux_aux_n, zero64)
        rd_scalar <<= pyrtl.select(is_custom0, mux_scalar, zero64)

    trace = pyrtl.SimulationTrace(block=block)
    sim = pyrtl.Simulation(tracer=trace, block=block)
    return block, sim
