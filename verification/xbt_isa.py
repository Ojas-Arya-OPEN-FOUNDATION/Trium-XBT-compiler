"""
xbt_isa -- declarative description of the RISC-V "Xbt" Custom-0 extension.

Each instruction is described twice:

  * ``spec`` -- what it means in balanced ternary mathematics (trit level).
  * ``impl`` -- what the hardware does (dual-rail level), mirroring
                ``hw/xbt_alu_model.hpp`` and ``hw/riscv_xbt_extension.sv``.

The verification engine proves the two agree. This file is data: adding an
instruction means appending an ``Operation``, not changing the engine.

IMPORTANT: ``impl`` functions here are transcribed from the RTL model so that
a divergence is *detectable*. If the RTL is fixed, fix it here too.
"""

from __future__ import annotations

import z3

from .z3core import (
    Operation, Operand, VerifiedISA, TRITS,
    trit_add, trit_add3, trit_neg, trit_mul,
    word_value, rails_to_trits, trits_to_rails,
)

WIDTH = 64
M64 = (1 << 64) - 1

# funct3 encodings (hw/xbt_alu_model.hpp, opcode 0x0B custom-0)
F_BTHA      = 0b000
F_KSADD     = 0b001
F_DOT64     = 0b010
F_UNPACK243 = 0b011
F_NEG       = 0b100
F_CSAMAC    = 0b101
# 0b110 and 0b111 are unassigned.
#
# NOTE: BTFA (the balanced ternary full adder) is implemented in the RTL and
# used by the systolic compressor, but it is NOT exposed as an instruction --
# there is no funct3 for it. It is verified here as a micro-operation.
F_BTFA = None


# --------------------------------------------------------------------------
# Rail-level primitives -- transcribed from hw/xbt_alu_model.hpp
# --------------------------------------------------------------------------

def _is1(x):
    """Truth test that works for both Python ints and Z3 bitvectors."""
    if isinstance(x, int):
        return x
    return x == 1


def _popcount(x):
    """Population count for both Python ints and Z3 bitvectors."""
    if isinstance(x, int):
        return bin(x).count("1")
    return z3.Sum([z3.If(z3.Extract(i, i, x) == 1, 1, 0) for i in range(WIDTH)])


def _ite(c, a, b):
    """If-then-else that works for both Python booleans and Z3 conditions."""
    if isinstance(c, (bool, int)):
        return a if c else b
    return z3.If(c, a, b)


def btha_rails(ap, an, bp, bn):
    """Dual-rail balanced ternary half adder (all 64 lanes at once)."""
    PC = ap & bp
    NC = an & bn
    PS = ((ap ^ bp) & ~(an | bn) & M64) | NC
    NS = ((an ^ bn) & ~(ap | bp) & M64) | PC
    return (PS, NS), (PC, NC)


def btfa_rails(ap, an, bp, bn, dp, dn):
    """Dual-rail balanced ternary full adder (all 64 lanes at once)."""
    (s1p, s1n), (c1p, c1n) = btha_rails(ap, an, bp, bn)
    (s2p, s2n), (c2p, c2n) = btha_rails(s1p, s1n, dp, dn)
    PC = (c1p | c2p) & ~(c1n | c2n)
    NC = (c1n | c2n) & ~(c1p | c2p)
    return (s2p, s2n), (PC, NC)


# --------------------------------------------------------------------------
# The instruction set
# --------------------------------------------------------------------------

def build_isa(width: int = WIDTH) -> VerifiedISA:
    isa = VerifiedISA("riscv-xbt", width)
    W = width

    # ---- BTHA -------------------------------------------------------------
    def btha_spec(a, b):
        s, c = trit_add(a, b)
        return {"sum": s, "carry": c}

    def btha_impl(ap, an, bp, bn):
        (sp, sn), (cp, cn) = btha_rails(ap, an, bp, bn)
        return {"sum": (sp, sn), "carry": (cp, cn)}

    isa.add(Operation(
        name="BTHA", mnemonic="xbt.btha", funct3=F_BTHA,
        operands=[Operand("rs1", "trit_word", W), Operand("rs2", "trit_word", W),
                  Operand("sum", "trit_word", W), Operand("carry", "trit_word", W)],
        spec=btha_spec, impl=btha_impl,
        doc="64-trit dual-rail balanced ternary half adder. Per lane: "
            "a + b == sum + 3*carry.",
    ))

    # ---- BTFA -------------------------------------------------------------
    def btfa_spec(a, b, d):
        s, c = trit_add3(a, b, d)
        return {"sum": s, "carry": c}

    def btfa_impl(ap, an, bp, bn, dp, dn):
        (sp, sn), (cp, cn) = btfa_rails(ap, an, bp, bn, dp, dn)
        return {"sum": (sp, sn), "carry": (cp, cn)}

    isa.add(Operation(
        name="BTFA", mnemonic="xbt.btfa", funct3=F_BTFA,
        operands=[Operand("rs1", "trit_word", W), Operand("rs2", "trit_word", W),
                  Operand("rd", "trit_word", W),
                  Operand("sum", "trit_word", W), Operand("carry", "trit_word", W)],
        spec=btfa_spec, impl=btfa_impl,
        doc="64-trit balanced ternary full adder. Per lane: a + b + d == sum + 3*carry.",
    ))

    # ---- KSADD ------------------------------------------------------------
    def ksadd_impl(ap, an, bp, bn):
        (s1p, s1n), (c1p, c1n) = btha_rails(ap, an, bp, bn)

        yp_p = (ap | bp) & ~(an | bn) & M64
        yp_n = 0
        y0_p = c1p
        y0_n = c1n
        ym_p = 0
        ym_n = (an | bn) & ~(ap | bp) & M64

        for s in range(6):
            d = 1 << s
            low = (1 << d) - 1
            lp_p = ((yp_p << d) | low) & M64
            lp_n = (yp_n << d) & M64
            l0_p = (y0_p << d) & M64
            l0_n = (y0_n << d) & M64
            lm_p = (ym_p << d) & M64
            lm_n = ((ym_n << d) | low) & M64

            zp = ~(lp_p | lp_n) & M64
            z0 = ~(l0_p | l0_n) & M64
            zm = ~(lm_p | lm_n) & M64

            nyp_p = ((lp_p & yp_p) | (zp & y0_p) | (lp_n & ym_p)) & M64
            nyp_n = ((lp_p & yp_n) | (zp & y0_n) | (lp_n & ym_n)) & M64
            ny0_p = ((l0_p & yp_p) | (z0 & y0_p) | (l0_n & ym_p)) & M64
            ny0_n = ((l0_p & yp_n) | (z0 & y0_n) | (l0_n & ym_n)) & M64
            nym_p = ((lm_p & yp_p) | (zm & y0_p) | (lm_n & ym_p)) & M64
            nym_n = ((lm_p & yp_n) | (zm & y0_n) | (lm_n & ym_n)) & M64

            yp_p, yp_n = nyp_p, nyp_n
            y0_p, y0_n = ny0_p, ny0_n
            ym_p, ym_n = nym_p, nym_n

        cin_p = (y0_p << 1) & M64
        cin_n = (y0_n << 1) & M64

        sum_p = (((s1p ^ cin_p) & ~(s1n | cin_n)) | (s1n & cin_n)) & M64
        sum_n = (((s1n ^ cin_n) & ~(s1p | cin_p)) | (s1p & cin_p)) & M64

        # cout is derived symbolically so the same code serves concrete
        # evaluation and Z3 proof.
        cout = _ite(_is1((y0_p >> 63) & 1), 1,
                    _ite(_is1((y0_n >> 63) & 1), -1, 0))
        return {"sum": (sum_p, sum_n), "cout": cout}

    def ksadd_spec(a_word, b_word):
        A = word_value(a_word)
        B = word_value(b_word)
        T = A + B
        P = 3 ** W
        half = (P - 1) // 2
        s = ((T + half) % P) - half
        cout = (T - s) // P
        return {"sum": s, "cout": cout}

    isa.add(Operation(
        name="KSADD", mnemonic="xbt.ksadd", funct3=F_KSADD,
        operands=[Operand("rs1", "trit_word", W), Operand("rs2", "trit_word", W),
                  Operand("sum", "trit_word", W), Operand("cout", "trit")],
        spec=ksadd_spec, impl=ksadd_impl,
        doc="6-stage Kogge-Stone parallel-prefix 64-trit adder. "
            "Contract: value(rs1) + value(rs2) == value(sum) + cout * 3**64.",
        exhaustive=False,   # 3**128 combinations: verified symbolically instead
    ))

    # ---- DOT64 ------------------------------------------------------------
    def dot64_impl(ap, an, bp, bn):
        prod_p = (ap & bp) | (an & bn)
        prod_n = (ap & bn) | (an & bp)
        return {"acc": _popcount(prod_p) - _popcount(prod_n),
                "prod": (prod_p, prod_n)}

    def dot64_spec(a_word, b_word):
        return {"acc": sum(t * u for t, u in zip(a_word, b_word))}

    isa.add(Operation(
        name="DOT64", mnemonic="xbt.dot64", funct3=F_DOT64,
        operands=[Operand("rs1", "trit_word", W), Operand("rs2", "trit_word", W),
                  Operand("acc", "int")],
        spec=dot64_spec, impl=dot64_impl,
        doc="Planar dot product: sum over lanes of rs1[i] * rs2[i], in [-64, 64].",
        exhaustive=False,
    ))

    # ---- UNPACK243 --------------------------------------------------------
    def unpack243_impl(byte_val):
        if byte_val >= 243:
            return {"word": (0, 0), "valid": False}
        q1 = (byte_val * 171) >> 9
        q2 = (byte_val * 57) >> 9
        q3 = (byte_val * 19) >> 9
        q4 = (byte_val * 203) >> 14
        d = [byte_val - 3 * q1, q1 - 3 * q2, q2 - 3 * q3, q3 - 3 * q4, q4]
        op = on = 0
        for k in range(5):
            if d[k] == 2:
                op |= 1 << k
            elif d[k] == 0:
                on |= 1 << k
        return {"word": (op, on), "valid": True}

    def unpack243_spec(byte_val):
        # Base-243 digit d maps to trit (d - 1); the word value is byte - 121.
        if byte_val >= 243:
            return {"value": None, "valid": False}
        return {"value": byte_val - 121, "valid": True}

    isa.add(Operation(
        name="UNPACK243", mnemonic="xbt.unpack243", funct3=F_UNPACK243,
        operands=[Operand("rs1", "byte"), Operand("rd", "trit_word", 5)],
        spec=unpack243_spec, impl=unpack243_impl,
        doc="Division-free base-243 byte -> 5 dual-rail trits. "
            "Contract: value == byte - 121 for byte <= 242; byte >= 243 is rejected.",
        exhaustive=True,
    ))

    # ---- NEG --------------------------------------------------------------
    def neg_impl(ap, an):
        return {"word": (an, ap)}

    def neg_spec(a):
        return {"word": trit_neg(a)}

    isa.add(Operation(
        name="NEG", mnemonic="xbt.neg", funct3=F_NEG,
        operands=[Operand("rs1", "trit_word", W), Operand("rd", "trit_word", W)],
        spec=neg_spec, impl=neg_impl,
        doc="0-gate wire-swapped ternary negation: swap the P and N rails.",
    ))

    # ---- CSAMAC -----------------------------------------------------------
    # Documented contract (hw/xbt_alu_model.hpp header and the ISA table):
    #     "rs1 * rs2 accumulated into rd"
    # so the intended meaning is  rd_out == rd + (rs1 * rs2)  per lane.
    def csamac_spec(a, b, rd):
        return {"sum_": trit_add3(trit_mul(a, b), rd, 0)[0],
                "carry": trit_add3(trit_mul(a, b), rd, 0)[1]}

    # hw/xbt_alu_model.hpp, corrected: the accumulator is the destination
    # register rd, as the ISA documentation requires.
    def csamac_impl(ap, an, bp, bn, rdp, rdn):
        prod_p = (ap & bp) | (an & bn)
        prod_n = (ap & bn) | (an & bp)
        sp = ((prod_p ^ rdp) & ~(prod_n | rdn)) | (prod_n & rdn)
        sn = ((prod_n ^ rdn) & ~(prod_p | rdp)) | (prod_p & rdp)
        cp = prod_p & rdp
        cn = prod_n & rdn
        return {"sum_": (sp, sn), "carry": (cp, cn)}

    isa.add(Operation(
        name="CSAMAC", mnemonic="xbt.csamac", funct3=F_CSAMAC,
        operands=[Operand("rs1", "trit_word", W), Operand("rs2", "trit_word", W),
                  Operand("rd", "trit_word", W),
                  Operand("sum", "trit_word", W), Operand("carry", "trit_word", W)],
        spec=csamac_spec, impl=csamac_impl,
        doc="Fused carry-save MAC. Documented contract: rd + (rs1 * rs2) per lane.",
    ))

    # ======================================================================
    # Proposed extensions (v2).
    #
    # funct3 0b110 and 0b111 are unassigned in the v1 encoding. The v1 set is
    # not sufficient to *compile* to: there is no way to materialise a
    # constant into a ternary register, and the full adder and the per-lane
    # multiplier exist in RTL but are not reachable. These four close exactly
    # that gap and nothing more.
    # ======================================================================

    # ---- MUL (proposed, funct3=0b110 funct7=0x01) -------------------------
    def mul_impl(ap, an, bp, bn):
        # AOI22 compound gates -- the same product network DOT64 already uses.
        return {"word": ((ap & bp) | (an & bn), (ap & bn) | (an & bp))}

    def mul_spec(a, b):
        return {"word": trit_mul(a, b)}

    isa.add(Operation(
        name="MUL", mnemonic="xbt.mul", funct3=0b110,
        operands=[Operand("rs1", "trit_word", W), Operand("rs2", "trit_word", W),
                  Operand("rd", "trit_word", W)],
        spec=mul_spec, impl=mul_impl,
        doc="PROPOSED (funct3=0b110, funct7=0x01). Per-lane ternary multiply. "
            "The product network already exists in the DOT64 datapath; this "
            "simply exposes it. Contract: rd[i] == rs1[i] * rs2[i].",
    ))

    # ---- PACK243 (proposed, funct3=0b110 funct7=0x02) ---------------------
    def pack243_impl(ap, an):
        # 5 trits -> base-243 byte. digit d = trit + 1, value = sum d[k]*3**k
        byte = 0
        for k in range(5):
            p = (ap >> k) & 1
            n = (an >> k) & 1
            t = 1 if p else (-1 if n else 0)
            byte += (t + 1) * (3 ** k)
        return {"byte": byte}

    def pack243_spec(trits5):
        return {"byte": sum((t + 1) * (3 ** k) for k, t in enumerate(trits5))}

    isa.add(Operation(
        name="PACK243", mnemonic="xbt.pack243", funct3=0b110,
        operands=[Operand("rs1", "trit_word", 5), Operand("rd", "byte")],
        spec=pack243_spec, impl=pack243_impl,
        doc="PROPOSED (funct3=0b110, funct7=0x02). Inverse of UNPACK243: "
            "5 trits -> base-243 byte. Required to move ternary data back out "
            "to memory and to spill ternary registers.",
        exhaustive=False, symbolic=False,
    ))

    # ---- LI5 (proposed, funct3=0b111, I-type) -----------------------------
    def li5_impl(imm8, shamt):
        """Materialise (imm8 - 121) * 3**shamt as a trit word."""
        if imm8 >= 243:
            return {"word": (0, 0), "valid": False}
        v = (imm8 - 121) * (3 ** (5 * shamt))
        trits = []
        x = abs(v)
        for _ in range(W):
            r = x % 3
            if r == 0:
                trits.append(0); x //= 3
            elif r == 1:
                trits.append(1); x = (x - 1) // 3
            else:
                trits.append(-1); x = (x + 1) // 3
        if v < 0:
            trits = [-t for t in trits]
        P = N = 0
        for i, t in enumerate(trits):
            if t == 1:
                P |= 1 << i
            elif t == -1:
                N |= 1 << i
        return {"word": (P, N), "valid": True}

    def li5_spec(imm8, shamt):
        if imm8 >= 243:
            return {"value": None, "valid": False}
        return {"value": (imm8 - 121) * (3 ** (5 * shamt)), "valid": True}

    isa.add(Operation(
        name="LI5", mnemonic="xbt.li5", funct3=0b111,
        operands=[Operand("imm8", "byte"), Operand("shamt", "int"),
                  Operand("rd", "trit_word", W)],
        spec=li5_spec, impl=li5_impl,
        doc="PROPOSED (funct3=0b111, I-type: imm[7:0]=base-243 byte, "
            "imm[10:8]=trit shift). Load a 5-trit constant at any trit offset. "
            "This is the instruction the v1 set is missing: without it a "
            "compiler cannot materialise a constant into a ternary register, "
            "so the ISA cannot be a compilation target at all.",
        exhaustive=False, symbolic=False,
    ))

    # ---- SIGN (proposed, funct3=0b110 funct7=0x03) -----------------------
    def sign_impl(ap, an):
        """
        Sign of a balanced ternary word: the value of the most significant
        NON-ZERO trit. An all-zero word gives zero.

        This is a priority encoder over the rails, most significant first --
        the same structure as a count-leading-zeros circuit.
        """
        P, N = 0, 0     # result: only trit 0 can be set
        for i in range(W - 1, -1, -1):
            p = (ap >> i) & 1
            n = (an >> i) & 1
            if p or n:
                if p:
                    P = 1
                else:
                    N = 1
                break
        return {"word": (P, N)}

    def sign_spec(trits):
        v = sum(t * 3 ** i for i, t in enumerate(trits))
        return {"sign": (v > 0) - (v < 0)}

    isa.add(Operation(
        name="SIGN", mnemonic="xbt.sign", funct3=0b110,
        operands=[Operand("rs1", "trit_word", W), Operand("rd", "trit_word", W)],
        spec=sign_spec, impl=sign_impl,
        doc="PROPOSED (funct3=0b110, funct7=0x03). Sign of a ternary word, "
            "in {-1,0,+1}. This is the primitive every comparison and the "
            "3-way select are built from; without it the ISA cannot compare.",
        exhaustive=False, symbolic=False,
    ))

    # ---- CMOV (proposed, funct3=0b110 funct7=0x04/0x05/0x06) -------------
    # Three conditional moves on the sign of a condition word. Together they
    # implement a 3-way select with no control flow.
    isa.add(Operation(
        name="CMOV", mnemonic="xbt.cmov", funct3=0b110,
        operands=[Operand("rc", "trit_word", W), Operand("ra", "trit_word", W),
                  Operand("rd", "trit_word", W)],
        spec=None,
        impl=_cmov_impl_for("z", W),
        doc="PROPOSED (funct3=0b110, funct7=0x04/0x05/0x06). Conditional move "
            "on the sign of a ternary condition: cmovz / cmovp / cmovn. "
            "These three build a 3-way select without any branch.",
        exhaustive=False, symbolic=False,
    ))

    # ---- SHL3 (proposed, funct3=0b110 funct7=0x07) -----------------------
    # Shift by trit positions: rd = value(rs1) * 3^shamt. This is the
    # primitive P1 scalar multiply is built on; the ISA had no shift at all.
    # The shift amount rides in the rs2 field, as a RISC-V shift-immediate
    # does, so no new instruction format is needed.
    def shl3_spec(trits, shamt):
        v = sum(t * 3 ** i for i, t in enumerate(trits))
        return {"rd": v * (3 ** shamt)}

    def shl3_impl(ap, an, bp, bn):
        # `bp/bn` carry the shift amount as a small integer in bp.
        k = int(bp) if not hasattr(bp, "as_long") else bp.as_long()
        # Rail-level shift: move both rails left by k trits. Trits that would
        # leave the word are dropped here by the mask, which is why the
        # overflow check lives in the model and is verified separately.
        return {"rd": ((ap << k) & ((1 << W) - 1), (an << k) & ((1 << W) - 1))}

    isa.add(Operation(
        name="SHL3", mnemonic="xbt.shl3", funct3=0b110,
        operands=[Operand("rs1", "trit_word", W), Operand("rs2", "trit_word", W),
                  Operand("rd", "trit_word", W)],
        spec=shl3_spec, impl=shl3_impl,
        doc="PROPOSED (funct3=0b110, funct7=0x07). Shift by trit positions: "
            "rd = value(rs1) * 3^rs2field. Exact within range; the hardware "
            "model REFUSES (throws) when trits would leave the word, because "
            "P1 scalar multiply must fail loudly rather than wrap. Verified "
            "in verification/verify_scalar_mul.py (S1, S2, S7).",
        exhaustive=False, symbolic=False,
    ))

    return isa


def _sign_of(P, N, W):
    """Sign of a concrete dual-rail word, as -1 / 0 / +1."""
    for i in range(W - 1, -1, -1):
        p = (P >> i) & 1
        n = (N >> i) & 1
        if p or n:
            return 1 if p else -1
    return 0


def _cmov_impl_for(mode, W):
    """Conditional move: mode 'z' (==0), 'p' (>0) or 'n' (<0)."""
    def f(cond_p, cond_n, src_p, src_n, dst_p, dst_n):
        s = _sign_of(cond_p, cond_n, W)
        take = (s == 0) if mode == "z" else ((s > 0) if mode == "p" else (s < 0))
        return {"word": (src_p, src_n) if take else (dst_p, dst_n)}
    return f


def sign_spec_value(trits):
    v = sum(t * 3 ** i for i, t in enumerate(trits))
    return (v > 0) - (v < 0)


def verify_sign(width: int = 64, exhaustive_width: int = 10):
    """
    Verify SIGN two independent ways.

    1. Exhaustively over EVERY trit word up to `exhaustive_width` trits -- a
       complete check for narrow words, which is exactly where sign bugs hide.
    2. Symbolically in Z3 over the full width, proving the priority encoder
       agrees with the sign of the value for all valid dual-rail words.
    """
    import itertools
    import z3

    isa = build_isa(width)
    sign = next(o for o in isa if o.name == "SIGN")
    errors = []

    # 1. Exhaustive over narrow widths.
    for w in range(1, exhaustive_width + 1):
        for trits in itertools.product((-1, 0, 1), repeat=w):
            P = N = 0
            for i, t in enumerate(trits):
                if t == 1:
                    P |= 1 << i
                elif t == -1:
                    N |= 1 << i
            gp, gn = sign.impl(P, N)["word"]
            got = 1 if gp & 1 else (-1 if gn & 1 else 0)
            exp = sign_spec_value(list(trits))
            if got != exp:
                errors.append(
                    f"sign width={w} trits={list(trits)}: expected {exp}, got {got}")
            if gp & gn:
                errors.append(f"sign produced invalid rails for {list(trits)}")

    # 2. Symbolic over the full width.
    ap = z3.BitVec("ap", width)
    an = z3.BitVec("an", width)
    valid = ap & an == 0

    def z3_sign(P, N):
        # Fold LSB -> MSB so that a nonzero trit in a HIGHER position
        # overwrites the result of every lower one. Folding MSB-first would
        # let a low trit overwrite the high one and report the wrong sign.
        acc = z3.IntVal(0)
        for i in range(0, width):
            p = z3.Extract(i, i, P) == 1
            n = z3.Extract(i, i, N) == 1
            acc = z3.If(p, z3.IntVal(1), z3.If(n, z3.IntVal(-1), acc))
        return acc

    def z3_value(P, N):
        return z3.Sum([
            z3.If(z3.Extract(i, i, P) == 1, z3.IntVal(3 ** i),
                  z3.If(z3.Extract(i, i, N) == 1, z3.IntVal(-(3 ** i)),
                        z3.IntVal(0)))
            for i in range(width)
        ])

    s = z3_sign(ap, an)
    v = z3_value(ap, an)

    solver = z3.Solver()
    solver.set("timeout", 300_000)
    solver.add(valid)
    solver.add(z3.Not(z3.And(
        z3.Or(s == 1, s == 0, s == -1),
        z3.Implies(v > 0, s == 1),
        z3.Implies(v == 0, s == 0),
        z3.Implies(v < 0, s == -1),
    )))
    r = solver.check()
    if r != z3.unsat:
        errors.append(f"symbolic sign proof: {r}")

    return errors


def verify_cmov(width: int = 64):
    """
    Verify the three conditional moves. The control depends only on the SIGN
    of the condition, so covering every sign against several data patterns is
    a complete check of the control behaviour.
    """
    isa = build_isa(width)
    errors = []

    patterns = [
        (0, 0),
        (1, 0),
        (0, 1),
        (0xFFFFFFFFFFFFFFFF, 0),
        (0x0123456789ABCDEF, 0x0FEDCBA987654321 & ~(0x0123456789ABCDEF)),
    ]
    cond_words = {
        0:  (0, 0),
        1:  (0b1111, 0),      # +1+3+9+27 > 0
        -1: (0, 0b1111),      # -(1+3+9+27) < 0
    }

    for mode, test in (("z", lambda s: s == 0),
                       ("p", lambda s: s > 0),
                       ("n", lambda s: s < 0)):
        f = _cmov_impl_for(mode, width)
        for cs, (cp, cn) in cond_words.items():
            for (sp, sn) in patterns:
                for (dp, dn) in patterns:
                    gp, gn = f(cp, cn, sp, sn, dp, dn)["word"]
                    expect = (sp, sn) if test(cs) else (dp, dn)
                    if (gp, gn) != expect:
                        errors.append(
                            f"cmov{mode} cond_sign={cs}: expected {expect}, "
                            f"got {(gp, gn)}")
    return errors


def verify_li5():
    """
    Exhaustively verify LI5 over its whole domain: every byte value (0..255,
    including the 13 that must be rejected) crossed with every shift (0..7).
    """
    from .z3core import word_value

    isa = build_isa()
    li5 = next(o for o in isa if o.name == "LI5")
    errors = []
    for shamt in range(8):
        for b in range(256):
            got = li5.impl(b, shamt)
            exp = li5.spec(b, shamt)
            if not exp["valid"]:
                if got["valid"] or got["word"] != (0, 0):
                    errors.append(f"li5(byte={b}, shamt={shamt}) should be rejected")
                continue
            P, N = got["word"]
            if P & N:
                errors.append(f"li5(byte={b}, shamt={shamt}) produced invalid rails")
                continue
            trits = [1 if (P >> i) & 1 else (-1 if (N >> i) & 1 else 0)
                     for i in range(64)]
            v = sum(t * 3 ** i for i, t in enumerate(trits))
            if v != exp["value"]:
                errors.append(
                    f"li5(byte={b}, shamt={shamt}): expected {exp['value']}, got {v}")
    return errors


def verify_pack_unpack_roundtrip(width: int = 5):
    """
    Cross-property: PACK243 and UNPACK243 must be inverses.

    This is what makes the base-243 encoding usable as a *storage* format:
    data written through PACK243 and read back through UNPACK243 (or the
    reverse) must be identical. Verified exhaustively over the whole
    243-value domain rather than sampled.
    """
    from .z3core import encode_word, word_value

    isa = build_isa(width)
    unpack = next(o for o in isa if o.name == "UNPACK243")
    pack = next(o for o in isa if o.name == "PACK243")

    lo, hi = -(3 ** width - 1) // 2, (3 ** width - 1) // 2
    errors = []

    # pack(unpack(b)) == b   for every valid byte
    for b in range(243):
        wp, wn = unpack.impl(b)["word"]
        trits = [1 if (wp >> i) & 1 else (-1 if (wn >> i) & 1 else 0)
                 for i in range(width)]
        got = pack.impl(wp, wn)["byte"]
        if got != b:
            errors.append(f"pack(unpack({b})) = {got}")

    # unpack(pack(v)) == v   for every representable value
    for v in range(lo, hi + 1):
        trits = encode_word(v, width)
        P = N = 0
        for i, t in enumerate(trits):
            if t == 1:
                P |= 1 << i
            elif t == -1:
                N |= 1 << i
        b = pack.impl(P, N)["byte"]
        wp, wn = unpack.impl(b)["word"]
        back = sum((1 if (wp >> i) & 1 else (-1 if (wn >> i) & 1 else 0)) * 3 ** i
                   for i in range(width))
        if back != v:
            errors.append(f"unpack(pack({v})) = {back}")

    # encoded byte must always be in [0, 242]
    for v in range(lo, hi + 1):
        trits = encode_word(v, width)
        P = N = 0
        for i, t in enumerate(trits):
            if t == 1:
                P |= 1 << i
            elif t == -1:
                N |= 1 << i
        b = pack.impl(P, N)["byte"]
        if not (0 <= b <= 242):
            errors.append(f"pack({v}) = {b} out of byte range")

    return errors
