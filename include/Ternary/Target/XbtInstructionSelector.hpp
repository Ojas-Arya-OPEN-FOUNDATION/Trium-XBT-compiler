#ifndef TERNARY_TARGET_XBT_INSTRUCTION_SELECTOR_HPP
#define TERNARY_TARGET_XBT_INSTRUCTION_SELECTOR_HPP

/**
 * @file include/Ternary/Target/XbtInstructionSelector.hpp
 * @brief Lowers the backend-neutral core IR to RISC-V "Xbt" custom-0 machine code.
 *
 * This is the component that turns the Xbt extension from a specification into
 * a compilation target. It consumes the same `bt::ir::Program` every other
 * backend consumes and emits real encoded instructions.
 *
 * Instruction set
 * ---------------
 *   v1    (existing RTL)     BTHA KSADD DOT64 UNPACK243 NEG CSAMAC
 *   v2    (proposed)         BTFA MUL PACK243 LI5
 *
 * The v2 instructions exist because the v1 set is not sufficient to compile
 * to: it provides no way to materialise a constant into a ternary register,
 * and the full adder and per-lane multiplier exist in RTL but are unreachable.
 * Every instruction this selector emits is verified in
 * `verification/verify_isa.py`; see `docs/XBT_ISA_SPEC.md`.
 *
 * Honest failure
 * --------------
 * Constructs this selector cannot lower raise `std::runtime_error` naming the
 * construct. It never emits a plausible-looking instruction sequence that
 * would compute the wrong thing -- the same principle the rest of the
 * compiler follows.
 */

#include "../IR/Core.hpp"
#include "../IR/Program.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt::target::xbt {

// --------------------------------------------------------------------------
// Encoding (must match hw/xbt_alu_model.hpp)
// --------------------------------------------------------------------------

constexpr uint8_t OPCODE_XBT_CUSTOM0 = 0x0B;

enum class Funct3 : uint8_t {
    BTHA      = 0b000,
    KSADD     = 0b001,
    DOT64     = 0b010,
    UNPACK243 = 0b011,
    NEG       = 0b100,
    CSAMAC    = 0b101,
    // v2 group: R-type with funct7 as an extended opcode
    GROUP2    = 0b110,
    // v2 group: I-type load
    LI5       = 0b111,
};

enum class Group2 : uint8_t {
    BTFA    = 0x00,   // rd = sum(rs1, rs2, rd); carry -> last_carry
    MUL     = 0x01,   // rd[i] = rs1[i] * rs2[i]
    PACK243 = 0x02,   // x[rd] = pack243(rs1[4:0])
};

// Base-243 packing: 5 trits per byte, digit d = trit + 1, bias 121.
constexpr int BASE243_BIAS = 121;
constexpr int TRITS_PER_BYTE = 5;
constexpr int MAX_LI5_CHUNK = 7;   // 8 chunks x 5 trits = 40 trits

// --------------------------------------------------------------------------
// Emitted instruction
// --------------------------------------------------------------------------

struct Insn {
    bool is_itype{false};
    uint8_t funct3{0};
    uint8_t funct7{0};      // group-2 extended opcode, or carry destination
    uint8_t rd{0};
    uint8_t rs1{0};
    uint8_t rs2{0};
    uint32_t imm{0};        // I-type: imm[7:0] = base-243 byte, imm[10:8] = shift
    std::string comment;

    /** Canonical RISC-V `.insn r` / `.insn i` directive text. */
    std::string text() const {
        std::string s = "  .insn ";
        if (is_itype) {
            s += "i 0x0b, " + std::to_string((int)funct3) + ", x" +
                 std::to_string((int)rd) + ", x" + std::to_string((int)rs1) +
                 ", " + std::to_string(imm);
        } else {
            s += "r 0x0b, " + std::to_string((int)funct3) + ", " +
                 std::to_string((int)funct7) + ", x" + std::to_string((int)rd) +
                 ", x" + std::to_string((int)rs1) + ", x" +
                 std::to_string((int)rs2);
        }
        if (!comment.empty()) s += "  // " + comment;
        return s;
    }
};

struct LoweredFunction {
    std::string name;
    std::vector<Insn> code;
    int ternary_regs_used{0};
    std::vector<std::string> notes;   // honest record of what was done
};

// --------------------------------------------------------------------------
// Selector
// --------------------------------------------------------------------------

class InstructionSelector {
public:
    // 32 ternary registers; t0 is hardwired to zero.
    static constexpr int REG_COUNT = 32;
    static constexpr int REG_ZERO = 0;

    explicit InstructionSelector(int available_regs = 16)
        : available_(available_regs) {
        if (available_ < 4 || available_ >= REG_COUNT)
            throw std::invalid_argument("xbt: register count out of range");
    }

    LoweredFunction lower_function(const ir::Function& f) {
        LoweredFunction out;
        out.name = f.name;
        reset();

        // Parameters arrive in t1..tN.
        for (const auto& p : f.parameters) {
            int r = alloc_reg();
            symbol_reg_[p.symbol] = r;
            out.code.push_back(make_note("param sym#" + std::to_string(p.symbol) +
                                         " -> t" + std::to_string(r)));
        }

        lower_statements(f.body, out);
        out.ternary_regs_used = next_reg_;
        return out;
    }

private:
    int available_;
    int next_reg_{1};                       // t0 reserved as zero
    std::map<uint32_t, int> symbol_reg_;    // IR symbol -> ternary register
    std::set<int> free_regs_;

    Insn make_note(const std::string& c) const {
        Insn i; i.comment = c; return i;
    }

    void reset() {
        next_reg_ = 1;
        symbol_reg_.clear();
        free_regs_.clear();
    }

    int alloc_reg() {
        if (!free_regs_.empty()) {
            int r = *free_regs_.begin();
            free_regs_.erase(free_regs_.begin());
            return r;
        }
        if (next_reg_ >= available_) {
            throw std::runtime_error(
                "xbt: out of ternary registers (limit " + std::to_string(available_) +
                "); spilling is not implemented");
        }
        return next_reg_++;
    }

    void free_reg(int r) {
        if (r > REG_ZERO) free_regs_.insert(r);
    }

    Insn make_r(Funct3 f3, uint8_t f7, int rd, int rs1, int rs2,
                const std::string& c) const {
        Insn i;
        i.funct3 = static_cast<uint8_t>(f3);
        i.funct7 = f7;
        i.rd = static_cast<uint8_t>(rd);
        i.rs1 = static_cast<uint8_t>(rs1);
        i.rs2 = static_cast<uint8_t>(rs2);
        i.comment = c;
        return i;
    }

    Insn make_i(Funct3 f3, int rd, int rs1, uint32_t imm,
                const std::string& c) const {
        Insn i;
        i.is_itype = true;
        i.funct3 = static_cast<uint8_t>(f3);
        i.rd = static_cast<uint8_t>(rd);
        i.rs1 = static_cast<uint8_t>(rs1);
        i.imm = imm;
        i.comment = c;
        return i;
    }

    /**
     * True when an expression is a literal in {-1, 0, +1}, i.e. occupies a
     * single trit. Only then is the lane-wise MUL safe for scalar use.
     */
    static bool is_single_trit(const ir::Expression& e) {
        return e && e->op == ir::Op::Integer && e->integer >= -1 && e->integer <= 1;
    }

    // ---- constant materialisation ---------------------------------------
    //
    // LI5 carries 5 trits at a trit offset. A 39-trit int39 constant is built
    // from ceil(39/5) = 8 LI5 instructions accumulated with KSADD.

    std::vector<Insn> materialise(int64_t value, int dst) {
        std::vector<Insn> code;
        std::vector<int64_t> chunks;
        int64_t v = value;
        // Decompose into balanced ternary digits.
        std::vector<int> trits;
        for (int i = 0; i < 40 && v != 0; ++i) {
            int64_t r = ((v % 3) + 3) % 3;
            if (r == 0)      { trits.push_back(0);  v /= 3; }
            else if (r == 1) { trits.push_back(1);  v = (v - 1) / 3; }
            else             { trits.push_back(-1); v = (v + 1) / 3; }
        }
        if (trits.empty()) {
            // Zero is digits {1,1,1,1,1}: byte 121, and byte - bias == 0.
            code.push_back(make_i(Funct3::LI5, dst, 0,
                                  pack_digits({1,1,1,1,1}, 0), "li5 t" +
                                  std::to_string(dst) + ", 0"));
            return code;
        }
        while (trits.size() % TRITS_PER_BYTE != 0) trits.push_back(0);

        bool first = true;
        int tmp = 0;
        for (size_t base = 0; base < trits.size(); base += TRITS_PER_BYTE) {
            std::vector<int> dig(TRITS_PER_BYTE);
            for (int k = 0; k < TRITS_PER_BYTE; ++k) {
                dig[k] = trits[base + k] + 1;          // trit -> digit {0,1,2}
            }
            uint32_t imm = pack_digits(dig, static_cast<int>(base));
            int target = first ? dst : (tmp ? tmp : (tmp = alloc_reg()));
            code.push_back(make_i(Funct3::LI5, target, 0, imm,
                                  "li5 t" + std::to_string(target) +
                                  ", digits@" + std::to_string(base)));
            if (!first) {
                code.push_back(make_r(Funct3::KSADD, 0, dst, dst, target,
                                      "ksadd t" + std::to_string(dst) + ", t" +
                                      std::to_string(dst) + ", t" +
                                      std::to_string(target)));
            }
            first = false;
        }
        if (tmp) free_reg(tmp);
        return code;
    }

    /**
     * imm[7:0]  = base-243 byte (5 trits, digit = trit + 1)
     * imm[10:8] = chunk index; the trit shift is 5 trits per chunk.
     *
     * Eight chunks cover 40 trits, one more than an int39 needs.
     */
    static uint32_t pack_digits(const std::vector<int>& dig, int trit_offset) {
        int byte = 0;
        int p = 1;
        for (int d : dig) { byte += d * p; p *= 3; }
        if (trit_offset % TRITS_PER_BYTE != 0)
            throw std::runtime_error("xbt: LI5 chunk must be 5-trit aligned");
        int chunk = trit_offset / TRITS_PER_BYTE;
        if (chunk < 0 || chunk > MAX_LI5_CHUNK)
            throw std::runtime_error("xbt: LI5 chunk index out of range");
        return static_cast<uint32_t>((byte & 0xFF) | ((chunk & 0x7) << 8));
    }

    // ---- expression lowering --------------------------------------------

    /** Lower an expression; result lands in a fresh register. */
    int lower_expr(const ir::Expression& e, LoweredFunction& out) {
        if (!e) throw std::invalid_argument("xbt: null expression");

        switch (e->op) {
        case ir::Op::Integer: {
            int r = alloc_reg();
            for (auto& i : materialise(e->integer, r)) out.code.push_back(i);
            return r;
        }
        case ir::Op::Parameter: {
            auto it = symbol_reg_.find(e->symbol);
            if (it == symbol_reg_.end())
                throw std::runtime_error("xbt: unknown parameter symbol #" +
                                         std::to_string(e->symbol));
            return it->second;
        }
        case ir::Op::Negate: {
            int a = lower_expr(e->operands.at(0), out);
            int d = alloc_reg();
            out.code.push_back(make_r(Funct3::NEG, 0, d, a, 0,
                                      "neg t" + std::to_string(d) + ", t" +
                                      std::to_string(a)));
            free_reg(a);
            return d;
        }
        case ir::Op::Add: {
            int a = lower_expr(e->operands.at(0), out);
            int b = lower_expr(e->operands.at(1), out);
            int d = alloc_reg();
            out.code.push_back(make_r(Funct3::KSADD, 0, d, a, b,
                                      "ksadd t" + std::to_string(d) + ", t" +
                                      std::to_string(a) + ", t" +
                                      std::to_string(b)));
            free_reg(a); free_reg(b);
            return d;
        }
        case ir::Op::Sub: {
            // a - b  ==  a + (-b); NEG is a zero-gate rail swap.
            int a = lower_expr(e->operands.at(0), out);
            int b = lower_expr(e->operands.at(1), out);
            int nb = alloc_reg();
            out.code.push_back(make_r(Funct3::NEG, 0, nb, b, 0,
                                      "neg t" + std::to_string(nb) + ", t" +
                                      std::to_string(b)));
            int d = alloc_reg();
            out.code.push_back(make_r(Funct3::KSADD, 0, d, a, nb,
                                      "ksadd t" + std::to_string(d) + ", t" +
                                      std::to_string(a) + ", t" +
                                      std::to_string(nb)));
            free_reg(a); free_reg(b); free_reg(nb);
            return d;
        }
        case ir::Op::Mul: {
            // The Xbt MUL is LANE-WISE: rd[i] = rs1[i] * rs2[i]. That is not
            // scalar multiplication, and using it as if it were silently
            // produces wrong answers whenever either operand needs more than
            // one trit (carries between lanes are simply dropped).
            //
            // So this is only safe when both operands are single-trit values.
            // Anything else is refused -- see docs/XBT_ISA_SPEC.md.
            if (!is_single_trit(e->operands.at(0)) ||
                !is_single_trit(e->operands.at(1))) {
                throw std::runtime_error(
                    "xbt: scalar multiplication is not implemented (the Xbt "
                    "MUL is lane-wise and drops inter-lane carries); refusing "
                    "to emit incorrect code");
            }
            int a = lower_expr(e->operands.at(0), out);
            int b = lower_expr(e->operands.at(1), out);
            int d = alloc_reg();
            out.code.push_back(make_r(Funct3::GROUP2,
                                      static_cast<uint8_t>(Group2::MUL),
                                      d, a, b,
                                      "xbt.mul t" + std::to_string(d) + ", t" +
                                      std::to_string(a) + ", t" +
                                      std::to_string(b)));
            free_reg(a); free_reg(b);
            return d;
        }
        default:
            throw std::runtime_error(
                "xbt: instruction selection not implemented for IR op #" +
                std::to_string(static_cast<int>(e->op)) +
                " -- refusing to emit a stub");
        }
    }

    void lower_statements(const std::vector<ir::Statement>& body,
                          LoweredFunction& out) {
        for (const auto& s : body) {
            switch (s.kind) {
            case ir::StatementKind::Let: {
                int r = lower_expr(s.value, out);
                auto [it, inserted] = symbol_reg_.emplace(s.symbol, r);
                if (!inserted) {
                    // Rebinding an existing symbol: overwrite its register.
                    it->second = r;
                }
                break;
            }
            case ir::StatementKind::Assign: {
                int r = lower_expr(s.value, out);
                symbol_reg_[s.symbol] = r;
                break;
            }
            case ir::StatementKind::Return: {
                if (s.value) {
                    int r = lower_expr(s.value, out);
                    // Result convention: t1 holds the return value.
                    if (r != 1) {
                        out.code.push_back(make_r(Funct3::KSADD, 0, 1, r, 0,
                                                  "move t1, t" + std::to_string(r)));
                    }
                }
                break;
            }
            case ir::StatementKind::Evaluate: {
                int r = lower_expr(s.value, out);
                free_reg(r);
                break;
            }
            case ir::StatementKind::Block:
                lower_statements(s.body, out);
                break;
            default:
                throw std::runtime_error(
                    "xbt: instruction selection not implemented for statement kind #" +
                    std::to_string(static_cast<int>(s.kind)) +
                    " -- refusing to emit a stub");
            }
        }
    }
};

// --------------------------------------------------------------------------
// Assembly emission
// --------------------------------------------------------------------------

inline std::string emit_assembly(const LoweredFunction& f) {
    std::string s;
    s += "    .text\n";
    s += "    .globl " + f.name + "\n";
    s += f.name + ":\n";
    for (const auto& i : f.code) s += i.text() + "\n";
    if (f.code.empty() || !f.code.back().comment.empty()) s += "  ret\n";
    s += "\n";
    return s;
}

/** Errors collected instead of thrown, so a whole program can be reported. */
struct LoweringResult {
    std::string assembly;
    std::vector<std::string> unlowered;   // functions that could not be lowered
    int functions{0};
    int instructions{0};

    bool complete() const { return unlowered.empty(); }
};

/**
 * Lower a whole core-IR program to Xbt assembly.
 *
 * Functions that cannot be lowered are reported rather than silently emitted
 * as something wrong; `complete()` says whether the whole program made it.
 */
inline LoweringResult lower_program_to_assembly(const ir::Program& program,
                                                int available_regs = 16) {
    LoweringResult out;
    out.assembly += "// Generated by Trium-XBT -- target riscv64-xbt\n";
    out.assembly += "// RISC-V \"Xbt\" custom-0 (opcode 0x0B), balanced ternary\n";
    out.assembly += "// Every instruction emitted here is Z3-verified against\n";
    out.assembly += "// balanced-ternary arithmetic: see verification/verify_isa.py\n";
    out.assembly += "\n";

    InstructionSelector sel(available_regs);
    for (const auto& f : program.functions) {
        ++out.functions;
        try {
            auto lowered = sel.lower_function(f);
            out.instructions += static_cast<int>(lowered.code.size());
            out.assembly += emit_assembly(lowered);
        } catch (const std::exception& e) {
            out.unlowered.push_back(f.name + ": " + e.what());
        }
    }
    return out;
}

} // namespace bt::target::xbt

#endif // TERNARY_TARGET_XBT_INSTRUCTION_SELECTOR_HPP
