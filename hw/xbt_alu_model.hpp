/**
 * @file hw/xbt_alu_model.hpp
 * @brief C++20 Cycle-Exact RISC-V "Xbt" Custom-0 (0x0B) ISA Emulator & RTL Co-Simulator
 *
 * Bit-accurate behavioral & RTL co-simulation model corresponding to
 * `hw/riscv_xbt_extension.sv` (IEEE 1800-2017 SystemVerilog).
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
    NEG       = 0b100  // 0-gate Wire-Swapped Ternary Negation
};

struct DecodedXbtInsn {
    uint8_t opcode{0};
    uint8_t rd{0};
    uint8_t funct3{0};
    uint8_t rs1{0};
    uint8_t rs2{0};
    uint8_t funct7{0};

    constexpr bool is_valid_xbt() const noexcept {
        return opcode == OPCODE_XBT_CUSTOM0 && funct3 <= 0b100;
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

// ==============================================================================
// 3. Cycle-Accurate RISC-V "Xbt" Coprocessor & Register File
// ==============================================================================

class RiscvXbtCoprocessor {
public:
    TritWord64 treg[32]{}; // 32 Dual-Rail 128-bit ternary registers (t0..t31)
    int64_t    xreg[32]{}; // 32 Scalar 64-bit RISC-V integer registers (x0..x31)
    TritWord64 last_carry{0, 0}; // Auxiliary carry output latch from BTHA / KSADD

    // Hardware Performance Counters
    uint64_t cycles{0};
    uint64_t instructions_retired{0};
    uint64_t zero_gate_negations{0};

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
