/**
 * @file hw/xbt_alu_model.hpp
 * @brief C++20 RISC-V "Xbt" Custom-0 (0x0B) instruction-set model
 *
 * Behavioural model of the instruction semantics implemented by
 * `hw/riscv_xbt_extension.sv` (IEEE 1800-2017 SystemVerilog). This is NOT an RTL
 * co-simulation: the two are checked against independent reference arithmetic, and
 * RTL simulation requires Verilator/Icarus (see docs/HARDWARE_GPU_QUALIFICATION.md).
 */

#ifndef XBT_ALU_MODEL_HPP
#define XBT_ALU_MODEL_HPP

#include "../ternary_core.hpp"
#include "../ternary_dense_algebra.hpp"

#include <cstdint>
#include <cstddef>
#include <array>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bt::hw {

// ==============================================================================
// 1. RISC-V "Xbt" Custom-0 Opcode & Funct3 Definitions
// ==============================================================================

static constexpr uint8_t OPCODE_XBT_CUSTOM0 = 0x0B; // 7'b0001011 (custom-0)

enum class XbtFunct3 : uint8_t {
    BTHA      = 0b000, // 1-cycle 64-trit Dual-Rail Balanced Ternary Half-Adder
    KSADD     = 0b001, // 1-cycle 64-trit 6-Stage Kogge-Stone Parallel Prefix Adder
    DOT64     = 0b010, // 1-cycle 64-trit Planar Dot Product -> Scalar Register
    UNPACK243 = 0b011, // 1-cycle Division-Free Base-243 Byte -> Dual-Rail Unpack
    NEG       = 0b100, // 0-gate Wire-Swapped Ternary Negation
    CSAMAC    = 0b101, // 1-cycle Fused Carry-Save MAC (rs1 * rs2 accumulated into rd, funct7)
    // ---- v2 extension (see docs/XBT_ISA_SPEC.md) ----
    // The v1 set cannot be a compilation target: it has no way to materialise
    // a constant into a ternary register, and the full adder and the per-lane
    // multiplier exist in RTL but are unreachable. These four close exactly
    // that gap. All are Z3-verified in verification/verify_isa.py.
    GROUP2    = 0b110, // R-type group; funct7 selects the operation
    LI5       = 0b111  // I-type: load a 5-trit constant at a trit offset
};

// Extended opcodes for XbtFunct3::GROUP2 (carried in funct7).
enum class XbtGroup2 : uint8_t {
    BTFA    = 0x00, // rd = sum(rs1, rs2, rd); carry -> last_carry
    MUL     = 0x01, // rd[i] = rs1[i] * rs2[i]   (per-lane, AOI22 product network)
    PACK243 = 0x02, // x[rd] = base-243 byte built from rs1[4:0]
    // ---- comparison / selection (see docs/XBT_ISA_SPEC.md) ----
    SIGN    = 0x03, // rd = sign(rs1) in {-1,0,+1}: the most significant
                    //      NON-ZERO trit, 0 for an all-zero word.
    CMOVZ   = 0x04, // if sign(rs1) == 0 then rd = rs2   (conditional move)
    CMOVP   = 0x05, // if sign(rs1) >  0 then rd = rs2
    CMOVN   = 0x06  // if sign(rs1) <  0 then rd = rs2
};

// Base-243: 5 trits per byte, digit d = trit + 1, so byte = value + 121.
static constexpr int XBT_BASE243_BIAS = 121;

struct DecodedXbtInsn {
    uint8_t opcode{0};
    uint8_t rd{0};
    uint8_t funct3{0};
    uint8_t rs1{0};
    uint8_t rs2{0};
    uint8_t funct7{0};

    constexpr bool is_valid_xbt() const noexcept {
        // funct3 <= 0b101 is the v1 set; 0b110/0b111 are the v2 extension.
        return opcode == OPCODE_XBT_CUSTOM0 && funct3 <= 0b111;
    }
};

/**
 * @brief Encodes a 32-bit RISC-V R-type instruction for the Xbt Custom-0 extension (opcode 0x0B).
 * Layout: [31:25]=funct7 | [24:20]=rs2 | [19:15]=rs1 | [14:12]=funct3 | [11:7]=rd | [6:0]=0x0B
 */
constexpr inline uint32_t encode_xbt_r(uint8_t funct7, uint8_t rs2, uint8_t rs1,
                                       uint8_t funct3, uint8_t rd) noexcept {
    return ((static_cast<uint32_t>(funct7) & 0x7FU) << 25) |
           ((static_cast<uint32_t>(rs2)    & 0x1FU) << 20) |
           ((static_cast<uint32_t>(rs1)    & 0x1FU) << 15) |
           ((static_cast<uint32_t>(funct3) & 0x07U) << 12) |
           ((static_cast<uint32_t>(rd)     & 0x1FU) << 7)  |
           (static_cast<uint32_t>(OPCODE_XBT_CUSTOM0) & 0x7FU);
}

/**
 * @brief Decodes a 32-bit RISC-V R-type instruction into its constituent fields.
 */
constexpr inline DecodedXbtInsn decode_xbt_r(uint32_t insn) noexcept {
    DecodedXbtInsn d{};
    d.opcode = static_cast<uint8_t>(insn & 0x7FU);
    d.rd     = static_cast<uint8_t>((insn >> 7)  & 0x1FU);
    d.funct3 = static_cast<uint8_t>((insn >> 12) & 0x07U);
    d.rs1    = static_cast<uint8_t>((insn >> 15) & 0x1FU);
    d.rs2    = static_cast<uint8_t>((insn >> 20) & 0x1FU);
    d.funct7 = static_cast<uint8_t>((insn >> 25) & 0x7FU);
    return d;
}

/**
 * @brief Encodes a 32-bit RISC-V I-type instruction for the Xbt extension.
 * Layout: [31:20]=imm | [19:15]=rs1 | [14:12]=funct3 | [11:7]=rd | [6:0]=0x0B
 * For LI5: imm[7:0] is a base-243 byte and imm[10:8] is a trit shift.
 */
constexpr inline uint32_t encode_xbt_i(uint16_t imm, uint8_t rs1,
                                       uint8_t funct3, uint8_t rd) noexcept {
    return ((static_cast<uint32_t>(imm)    & 0xFFFU) << 20) |
           ((static_cast<uint32_t>(rs1)    & 0x1FU)  << 15) |
           ((static_cast<uint32_t>(funct3) & 0x07U)  << 12) |
           ((static_cast<uint32_t>(rd)     & 0x1FU)  << 7)  |
           (static_cast<uint32_t>(OPCODE_XBT_CUSTOM0) & 0x7FU);
}

/** @brief Extracts the I-type immediate field (imm[11:0]). */
constexpr inline uint16_t xbt_i_immediate(uint32_t insn) noexcept {
    return static_cast<uint16_t>((insn >> 20) & 0xFFFU);
}

/**
 * @brief Disassembles a 32-bit Xbt instruction into standard RISC-V assembly + `.insn r` directive.
 */
inline std::string disassemble_xbt(uint32_t insn) {
    DecodedXbtInsn d = decode_xbt_r(insn);
    if (d.opcode != OPCODE_XBT_CUSTOM0) {
        return "unknown // non-xbt opcode";
    }

    std::ostringstream oss;
    switch (static_cast<XbtFunct3>(d.funct3)) {
        case XbtFunct3::BTHA:
            oss << "xbt.btha t" << +d.rd << ", t" << +d.rs1 << ", t" << +d.rs2;
            break;
        case XbtFunct3::KSADD:
            oss << "xbt.ksadd t" << +d.rd << ", t" << +d.rs1 << ", t" << +d.rs2;
            break;
        case XbtFunct3::DOT64:
            oss << "xbt.dot64 x" << +d.rd << ", t" << +d.rs1 << ", t" << +d.rs2;
            break;
        case XbtFunct3::UNPACK243:
            oss << "xbt.unpack243 t" << +d.rd << ", x" << +d.rs1;
            break;
        case XbtFunct3::NEG:
            oss << "xbt.neg t" << +d.rd << ", t" << +d.rs1;
            break;
        case XbtFunct3::CSAMAC:
            oss << "xbt.csamac t" << +d.rd << ", t" << +d.rs1 << ", t" << +d.rs2;
            break;
        default:
            return "unknown // invalid xbt funct3";
    }

    oss << " // .insn r 0x0b, " << +d.funct3 << ", " << +d.funct7
        << ", x" << +d.rd << ", x" << +d.rs1 << ", x" << +d.rs2;
    return oss.str();
}

// ==============================================================================
// 2. Cycle-Exact RTL Submodule Models (Bit-Identical to hw/riscv_xbt_extension.sv)
// ==============================================================================

struct KoggeStoneResult64 {
    TritWord64 sum;
    Trit       cout{Trit::Zero};
};

/**
 * @brief Bit-accurate model of `module xbt_kogge_stone_adder64` in SystemVerilog.
 * Executes 6 parallel-prefix doubling stages (d = 1, 2, 4, 8, 16, 32) on
 * 3-state monotone carry transfer functions (yp, y0, ym).
 */
inline KoggeStoneResult64 rtl_kogge_stone_adder64(const TritWord64& a, const TritWord64& b) noexcept {
    AdderOutput st0 = BTHA(a, b);
    uint64_t s1_p = st0.Sum.P;
    uint64_t s1_n = st0.Sum.N;
    uint64_t c1_p = st0.Carry.P;
    uint64_t c1_n = st0.Carry.N;

    // Level 0 transfer function F^{(0)}(c_in) for c_in in {+1, 0, -1}
    uint64_t yp_p = (a.P | b.P) & ~(a.N | b.N);
    uint64_t yp_n = 0ULL;
    uint64_t y0_p = c1_p;
    uint64_t y0_n = c1_n;
    uint64_t ym_p = 0ULL;
    uint64_t ym_n = (a.N | b.N) & ~(a.P | b.P);

    // 6 Kogge-Stone prefix doubling stages: d = 1, 2, 4, 8, 16, 32
    for (int s = 0; s < 6; ++s) {
        const int d = 1 << s;
        const uint64_t low_mask = (1ULL << d) - 1ULL;

        uint64_t lp_p = (yp_p << d) | low_mask;
        uint64_t lp_n = (yp_n << d);
        uint64_t l0_p = (y0_p << d);
        uint64_t l0_n = (y0_n << d);
        uint64_t lm_p = (ym_p << d);
        uint64_t lm_n = (ym_n << d) | low_mask;

        uint64_t zp = ~(lp_p | lp_n);
        uint64_t z0 = ~(l0_p | l0_n);
        uint64_t zm = ~(lm_p | lm_n);

        uint64_t next_yp_p = (lp_p & yp_p) | (zp & y0_p) | (lp_n & ym_p);
        uint64_t next_yp_n = (lp_p & yp_n) | (zp & y0_n) | (lp_n & ym_n);

        uint64_t next_y0_p = (l0_p & yp_p) | (z0 & y0_p) | (l0_n & ym_p);
        uint64_t next_y0_n = (l0_p & yp_n) | (z0 & y0_n) | (l0_n & ym_n);

        uint64_t next_ym_p = (lm_p & yp_p) | (zm & y0_p) | (lm_n & ym_p);
        uint64_t next_ym_n = (lm_p & yp_n) | (zm & y0_n) | (lm_n & ym_n);

        yp_p = next_yp_p; yp_n = next_yp_n;
        y0_p = next_y0_p; y0_n = next_y0_n;
        ym_p = next_ym_p; ym_n = next_ym_n;
    }

    uint64_t cin_p = y0_p << 1;
    uint64_t cin_n = y0_n << 1;

    uint64_t sum_p = ((s1_p ^ cin_p) & ~(s1_n | cin_n)) | (s1_n & cin_n);
    uint64_t sum_n = ((s1_n ^ cin_n) & ~(s1_p | cin_p)) | (s1_p & cin_p);

    Trit cout = Trit::Zero;
    if ((y0_p >> 63) & 1ULL) cout = Trit::Pos;
    else if ((y0_n >> 63) & 1ULL) cout = Trit::Neg;

    return { TritWord64(sum_p, sum_n), cout };
}

/**
 * @brief Bit-accurate model of `module xbt_base243_decoder` in SystemVerilog.
 * Unpacks an 8-bit Base-243 byte [0..242] into 5 Dual-Rail trits in a TritWord64.
 */
inline TritWord64 rtl_base243_decoder(uint8_t packed_byte) noexcept {
    if (packed_byte >= 243U) {
        return TritWord64(0, 0);
    }
    uint32_t b32 = packed_byte;
    uint32_t q1  = (b32 * 171U) >> 9;
    uint32_t q2  = (b32 * 57U)  >> 9;
    uint32_t q3  = (b32 * 19U)  >> 9;
    uint32_t q4  = (b32 * 203U) >> 14;

    uint32_t d[5] = {
        b32 - 3U * q1,
        q1  - 3U * q2,
        q2  - 3U * q3,
        q3  - 3U * q4,
        q4
    };

    uint64_t out_p = 0ULL;
    uint64_t out_n = 0ULL;
    for (int k = 0; k < 5; ++k) {
        if (d[k] == 2U) {
            out_p |= (1ULL << k);
        } else if (d[k] == 0U) {
            out_n |= (1ULL << k);
        }
    }
    return TritWord64(out_p, out_n);
}

/**
 * @brief Bit-accurate model of `module xbt_aoi22_planar_dot64` in SystemVerilog.
 * Computes product rails via CMOS compound AOI22 gates:
 *   prod_p_bar = ~((a.P & b.P) | (a.N & b.N))
 *   prod_n_bar = ~((a.P & b.N) | (a.N & b.P))
 * followed by parallel popcount dot product.
 */
inline int64_t rtl_aoi22_planar_dot64(const TritWord64& a, const TritWord64& b,
                                      TritWord64* prod_out = nullptr) noexcept {
    uint64_t prod_p = (a.P & b.P) | (a.N & b.N);
    uint64_t prod_n = (a.P & b.N) | (a.N & b.P);
    if (prod_out) {
        *prod_out = TritWord64(prod_p, prod_n);
    }
    return static_cast<int64_t>(TritWord64::popcount64(prod_p)) -
           static_cast<int64_t>(TritWord64::popcount64(prod_n));
}

/**
 * @brief Sign of a balanced ternary word: -1, 0 or +1.
 *
 * The sign is the value of the most significant NON-ZERO trit -- the lower
 * trits can never outweigh the leading one, since
 * sum_{i<j} 3**i = (3**j - 1)/2 < 3**j.
 *
 * This is the primitive every comparison and the 3-way select are built
 * from; without it the ISA cannot compare at all. Verified exhaustively for
 * narrow widths and symbolically at 64 trits in verification/verify_isa.py.
 */
inline int8_t ternary_word_sign(const TritWord64& w) noexcept {
    for (int i = 63; i >= 0; --i) {
        if ((w.P >> i) & 1ULL) return 1;
        if ((w.N >> i) & 1ULL) return -1;
    }
    return 0;
}

/** @brief Encodes a small integer in {-1,0,+1} as a dual-rail word. */
inline TritWord64 ternary_word_from_sign(int8_t s) noexcept {
    return (s > 0) ? TritWord64(1ULL, 0ULL)
         : (s < 0) ? TritWord64(0ULL, 1ULL)
                   : TritWord64(0ULL, 0ULL);
}

/**
 * @brief Shifts a dual-rail trit word left by whole trits (multiply by 3**n).
 *
 * Trit i moves to trit i+n; the low n trits become zero. This is the shift
 * used by the LI5 immediate field, and it is exact: no carry is generated
 * because no two trits ever land on the same position.
 */
inline TritWord64 shift_trits_left(const TritWord64& w, unsigned n) noexcept {
    if (n == 0) return w;
    if (n >= 64) return TritWord64(0, 0);
    return TritWord64(w.P << n, w.N << n);
}

// ==============================================================================
// 3. Cycle-Accurate RISC-V "Xbt" Coprocessor & Register File
// ==============================================================================

class RiscvXbtCoprocessor {
public:
    TritWord64 treg[32]{}; // 32 Dual-Rail 128-bit ternary registers (tr0..tr31)
    int64_t    xreg[32]{}; // 32 Scalar 64-bit RISC-V integer registers (x0..x31)
    TritWord64 last_carry{0, 0}; // Auxiliary carry output latch from BTHA / KSADD

    // Hardware Performance Counters
    uint64_t cycles{0};
    uint64_t instructions_retired{0};
    uint64_t zero_gate_negations{0};

    // Systolic BTFA Compressor Pipeline Stages
    struct SystolicBtfaStage {
        TritWord64 sum{0, 0};
        TritWord64 carry{0, 0};
        bool valid{false};
    };
    static constexpr size_t SYSTOLIC_STAGES = 2;
    std::array<SystolicBtfaStage, SYSTOLIC_STAGES> systolic_pipeline{};

    RiscvXbtCoprocessor() noexcept {
        reset();
    }

    void reset() noexcept {
        for (int i = 0; i < 32; ++i) {
            treg[i] = TritWord64(0, 0);
            xreg[i] = 0;
        }
        last_carry = TritWord64(0, 0);
        cycles = 0;
        instructions_retired = 0;
        zero_gate_negations = 0;
        flush_systolic_pipeline();
    }

    /**
     * @brief Read from 32 dual-rail ternary register file (tr0..tr31).
     * Supports 0-gate wire-swap negation bypass (rs_neg).
     * Register tr0 is hardwired to 0 (P=0, N=0).
     */
    TritWord64 read_tr(size_t idx, bool rs_neg = false) noexcept {
        if (idx >= 32) return TritWord64(0, 0);
        TritWord64 val = (idx == 0) ? TritWord64(0, 0) : treg[idx];
        if (rs_neg) {
            zero_gate_negations += 1;
            return TritWord64(val.N, val.P);
        }
        return val;
    }

    /**
     * @brief Write to 32 dual-rail ternary register file (tr0..tr31).
     * Writes to tr0 are discarded (hardwired to 0).
     */
    void write_tr(size_t idx, const TritWord64& val) noexcept {
        if (idx > 0 && idx < 32) {
            treg[idx] = val;
        }
    }

    /**
     * @brief Advance the pipelined Systolic BTFA compressor unit by 1 clock cycle.
     * Compresses 3 dual-rail inputs (A, B, C) via BTFA into (Sum, Carry)
     * and streams through the pipeline stages with zero carry ripples.
     */
    SystolicBtfaStage step_systolic_btfa(const TritWord64& a, const TritWord64& b,
                                        const TritWord64& c, bool valid_in = true) noexcept {
        AdderOutput fa = BTFA(a, b, c);

        for (size_t s = SYSTOLIC_STAGES - 1; s > 0; --s) {
            systolic_pipeline[s] = systolic_pipeline[s - 1];
        }
        systolic_pipeline[0] = SystolicBtfaStage{ fa.Sum, fa.Carry, valid_in };

        cycles += 1;
        if (valid_in) {
            instructions_retired += 1;
        }
        return systolic_pipeline[SYSTOLIC_STAGES - 1];
    }

    void flush_systolic_pipeline() noexcept {
        for (auto& stage : systolic_pipeline) {
            stage = SystolicBtfaStage{};
        }
    }

    /**
     * @brief Executes a single 32-bit encoded Xbt instruction bit-identically to
     *        `module riscv_xbt_alu` in `hw/riscv_xbt_extension.sv`.
     */
    void execute(uint32_t insn) {
        DecodedXbtInsn d = decode_xbt_r(insn);
        if (d.opcode != OPCODE_XBT_CUSTOM0) {
            throw std::invalid_argument("RiscvXbtCoprocessor::execute: Invalid Custom-0 opcode");
        }

        const TritWord64& rs1_t = treg[d.rs1];
        const TritWord64& rs2_t = treg[d.rs2];
        const int64_t     rs1_x = (d.rs1 == 0) ? 0 : xreg[d.rs1];

        switch (static_cast<XbtFunct3>(d.funct3)) {
            case XbtFunct3::BTHA: {
                AdderOutput ha = BTHA(rs1_t, rs2_t);
                treg[d.rd] = ha.Sum;
                last_carry = ha.Carry;
                if (d.funct7 > 0 && d.funct7 < 32) {
                    treg[d.funct7] = ha.Carry;
                }
                cycles += 1;
                instructions_retired += 1;
                break;
            }
            case XbtFunct3::KSADD: {
                KoggeStoneResult64 ks = rtl_kogge_stone_adder64(rs1_t, rs2_t);
                treg[d.rd] = ks.sum;
                last_carry = TritWord64(
                    ks.cout == Trit::Pos ? 1ULL : 0ULL,
                    ks.cout == Trit::Neg ? 1ULL : 0ULL
                );
                cycles += 1;
                instructions_retired += 1;
                break;
            }
            case XbtFunct3::DOT64: {
                int64_t dot = dotProduct64(rs1_t, rs2_t);
                if (d.rd != 0) {
                    xreg[d.rd] = dot;
                }
                cycles += 1;
                instructions_retired += 1;
                break;
            }
            case XbtFunct3::UNPACK243: {
                uint8_t byte_val = static_cast<uint8_t>(static_cast<uint64_t>(rs1_x) & 0xFFULL);
                treg[d.rd] = rtl_base243_decoder(byte_val);
                cycles += 1;
                instructions_retired += 1;
                break;
            }
            case XbtFunct3::NEG: {
                // 0-gate wire-swapped negation: out_p = in_n, out_n = in_p
                treg[d.rd] = TritWord64(rs1_t.N, rs1_t.P);
                zero_gate_negations += 1;
                instructions_retired += 1;
                break;
            }
            case XbtFunct3::CSAMAC: {
                // Fused Carry-Save MAC.
                //
                // Contract (see the XbtFunct3::CSAMAC documentation above):
                //   "rs1 * rs2 accumulated into rd", i.e. per lane
                //       rd_out == rd + (rs1 * rs2)
                // so the accumulator operand is the DESTINATION register.
                //
                // The accumulator was previously taken from rs1, which computed
                // rs1 + rs1*rs2 and silently ignored rd. Z3 verification
                // (verification/verify_isa.py) caught the divergence; the
                // behaviour below now matches the documented contract.
                const TritWord64& acc_t = treg[d.rd];
                uint64_t prod_p = (rs1_t.P & rs2_t.P) | (rs1_t.N & rs2_t.N);
                uint64_t prod_n = (rs1_t.P & rs2_t.N) | (rs1_t.N & rs2_t.P);
                uint64_t sp = ((prod_p ^ acc_t.P) & ~(prod_n | acc_t.N)) | (prod_n & acc_t.N);
                uint64_t sn = ((prod_n ^ acc_t.N) & ~(prod_p | acc_t.P)) | (prod_p & acc_t.P);
                uint64_t cp = prod_p & acc_t.P;
                uint64_t cn = prod_n & acc_t.N;
                treg[d.rd] = TritWord64(sp, sn);
                last_carry = TritWord64(cp, cn);
                if (d.funct7 > 0 && d.funct7 < 32) {
                    treg[d.funct7] = TritWord64(cp, cn);
                }
                cycles += 1;
                instructions_retired += 1;
                break;
            }

            // ---- v2 extension ------------------------------------------
            case XbtFunct3::GROUP2: {
                switch (static_cast<XbtGroup2>(d.funct7)) {
                    case XbtGroup2::BTFA: {
                        // rd = sum(rs1, rs2, rd); carry is latched.
                        AdderOutput fa = BTFA(rs1_t, rs2_t, treg[d.rd]);
                        treg[d.rd] = fa.Sum;
                        last_carry = fa.Carry;
                        break;
                    }
                    case XbtGroup2::MUL: {
                        // Per-lane ternary product via the AOI22 network.
                        TritWord64 p(rs1_t.P & rs2_t.P | rs1_t.N & rs2_t.N,
                                     rs1_t.P & rs2_t.N | rs1_t.N & rs2_t.P);
                        treg[d.rd] = p;
                        break;
                    }
                    case XbtGroup2::SIGN: {
                        // rd = sign(rs1), in {-1, 0, +1}.
                        treg[d.rd] = ternary_word_from_sign(
                            ternary_word_sign(rs1_t));
                        break;
                    }
                    case XbtGroup2::CMOVZ:
                    case XbtGroup2::CMOVP:
                    case XbtGroup2::CMOVN: {
                        // Conditional move on the sign of rs1. rd keeps its
                        // old value when the test fails, so the three modes
                        // compose into a 3-way select with no branch.
                        const int8_t s = ternary_word_sign(rs1_t);
                        const bool take =
                            (d.funct7 == static_cast<uint8_t>(XbtGroup2::CMOVZ)) ? (s == 0) :
                            (d.funct7 == static_cast<uint8_t>(XbtGroup2::CMOVP)) ? (s >  0) : (s <  0);
                        if (take) treg[d.rd] = rs2_t;
                        break;
                    }
                    case XbtGroup2::PACK243: {
                        // 5 trits -> base-243 byte. Inverse of UNPACK243.
                        int byte_val = XBT_BASE243_BIAS;
                        int place = 1;
                        for (int k = 0; k < 5; ++k) {
                            int t = 0;
                            if ((rs1_t.P >> k) & 1ULL)      t = 1;
                            else if ((rs1_t.N >> k) & 1ULL) t = -1;
                            byte_val += t * place;
                            place *= 3;
                        }
                        if (d.rd != 0) xreg[d.rd] = byte_val;
                        break;
                    }
                    default:
                        throw std::invalid_argument(
                            "RiscvXbtCoprocessor::execute: Unsupported Xbt group-2 opcode");
                }
                cycles += 1;
                instructions_retired += 1;
                break;
            }

            case XbtFunct3::LI5: {
                // I-type: rd = (imm[7:0] - 121) * 3**(5 * imm[10:8])
                //
                // imm[7:0]  is a base-243 byte carrying 5 trits.
                // imm[10:8] is a CHUNK INDEX, not a raw trit shift: the shift
                // is 5 trits per step. Eight chunks cover 40 trits, which is
                // exactly the width of an int39 scalar plus one.
                const uint16_t imm = xbt_i_immediate(insn);
                const uint8_t  byte_field = static_cast<uint8_t>(imm & 0xFFU);
                const unsigned chunk = static_cast<unsigned>((imm >> 8) & 0x7U);
                const unsigned shamt = 5u * chunk;

                if (byte_field >= 243U) {
                    // Out-of-range base-243 byte: decode to zero rather than
                    // producing an invalid dual-rail word.
                    treg[d.rd] = TritWord64(0, 0);
                } else {
                    treg[d.rd] = shift_trits_left(
                        rtl_base243_decoder(byte_field), shamt);
                }
                cycles += 1;
                instructions_retired += 1;
                break;
            }

            default:
                throw std::invalid_argument("RiscvXbtCoprocessor::execute: Unsupported Xbt funct3");
        }

        // Enforce RISC-V x0 hardwired zero invariant
        xreg[0] = 0;
    }

    void execute_program(const std::vector<uint32_t>& program) {
        for (uint32_t insn : program) {
            execute(insn);
        }
    }

    static std::string disassemble(uint32_t insn) {
        return disassemble_xbt(insn);
    }
};

} // namespace bt::hw

#endif // XBT_ALU_MODEL_HPP
