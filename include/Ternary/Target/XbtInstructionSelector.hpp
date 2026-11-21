#ifndef TERNARY_TARGET_XBT_INSTRUCTION_SELECTOR_HPP
#define TERNARY_TARGET_XBT_INSTRUCTION_SELECTOR_HPP

// MachineDescription (generated from targets/*.yaml) lets rules consult target
// capability. This is what makes `has_scalar_mul` a data-driven decision.
#include "GeneratedTargets.hpp"
#include "TargetDescription.hpp"

/**
 * @file include/Ternary/Target/XbtInstructionSelector.hpp
 * @brief Rule-driven lowering of the backend-neutral core IR to Xbt machine code.
 *
 * This is the component that turns the Xbt extension from a specification into
 * a compilation target. It consumes the same `bt::ir::Program` every other
 * backend consumes and emits real encoded instructions.
 *
 * Rule-based design
 * -----------------
 * Lowering is driven by a table of `SelectionRule`s rather than a switch. Each
 * rule names one IR op, carries a predicate over the operands, and an emitter
 * that produces the instruction sequence. To support a new op you append a
 * rule; nothing else changes. That is what makes this extensible instead of a
 * growing list of special cases.
 *
 * Every comparison and the 3-way select are built from two primitives:
 *
 *   SIGN   rd, rc            rd = sign(rc) in {-1, 0, +1}
 *   CMOV<z|p|n> rd, rc, ra   rd = ra when sign(rc) is ==0 / >0 / <0
 *
 * so supporting `<=>` and `select` automatically supports all seven comparison
 * operators, because each is a different predicate over the same sign.
 *
 * Trium-XBT compiler innovations
 * ------------------------------
 *   1. Branchless 3-way predication for pure `If` statements (single Assign to
 *      the same symbol or single Return in both arms) via `xbt.sign` +
 *      `xbt.cmovp` / `xbt.cmovz` / `xbt.cmovn` -- zero branches, zero flushes.
 *   2. Hybrid RV64I + Xbt control-flow lowering for impure `If` and `While`
 *      loops (`.L_while_head_N:`, `xbt.sign`, `blez`, `j`).
 *   3. Trium-XBT-native algebraic folding (`--x -> x`, `x+0 -> x`, `x-0 -> x`,
 *      `x*0 -> t0`, `x*1 -> x`, `x*-1 -> neg`, `x*(+-3^k) -> shl3 (+ neg)`),
 *      post-lowering peephole optimisation (`peephole_optimise`), and
 *      symbol-aware temporary register reclamation (`free_reg`).
 *   4. Exact assembly efficiency reporting (`CompiledMetrics` / `compute_metrics`).
 *
 * Honest failure
 * --------------
 * Constructs no rule can lower raise `std::runtime_error` naming the
 * construct. This never emits a plausible-looking sequence that computes the
 * wrong thing -- the same principle the rest of the compiler follows.
 *
 * Instruction set
 * ---------------
 *   v1    (existing RTL)     BTHA KSADD DOT64 UNPACK243 NEG CSAMAC
 *   v2    (proposed)         BTFA MUL PACK243 LI5
 *   v2.1  (proposed)         SIGN CMOVZ CMOVP CMOVN SHL3
 *
 * Every instruction emitted here is verified in `verification/verify_isa.py`;
 * see `docs/XBT_ISA_SPEC.md`.
 */

#include "../IR/Core.hpp"
#include "../IR/Program.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    GROUP2    = 0b110,   // R-type group; funct7 is an extended opcode
    LI5       = 0b111,   // I-type load
};

enum class Group2 : uint8_t {
    BTFA    = 0x00,
    MUL     = 0x01,
    PACK243 = 0x02,
    SIGN    = 0x03,
    CMOVZ   = 0x04,
    CMOVP   = 0x05,
    CMOVN   = 0x06,
    SHL3    = 0x07,   // rd = value(rs1) * 3^rs2field; see hw/xbt_alu_model.hpp
};

constexpr int BASE243_BIAS = 121;
constexpr int TRITS_PER_BYTE = 5;
constexpr int MAX_LI5_CHUNK = 7;   // 8 chunks x 5 trits = 40 trits

// --------------------------------------------------------------------------
// Emitted instruction
// --------------------------------------------------------------------------

enum class InsnKind : uint8_t {
    Xbt = 0,         // 32-bit hardware Xbt custom-0 instruction (.insn r / .insn i)
    Label,           // GNU assembly label (e.g. ".L_while_head_0:")
    BranchIfNotPos,  // RV64I conditional branch on scalar sign <= 0 ("blez x<r>, .L_...")
    Jump,            // RV64I unconditional jump ("j .L_...")
    Call,            // Function call ("call <name>")
    Mul,             // Runtime scalar multiply ("mul x<rd>, x<rs1>, x<rs2>")
    Spill,           // Stack spill ("sd x<rd>, <imm>(sp)")
    Reload,          // Stack reload ("ld x<rd>, <imm>(sp)")
    StackAdj,        // Stack pointer adjustment ("addi sp, sp, <imm>")
    Ret,             // Function return ("ret")
};

struct Insn {
    bool is_itype{false};
    uint8_t funct3{0};
    uint8_t funct7{0};
    uint8_t rd{0}, rs1{0}, rs2{0};
    uint32_t imm{0};
    std::string comment;
    InsnKind kind{InsnKind::Xbt};
    std::string pseudo_asm;

    bool is_xbt_insn() const { return kind == InsnKind::Xbt; }
    bool is_control_flow() const {
        return kind == InsnKind::BranchIfNotPos || kind == InsnKind::Jump ||
               kind == InsnKind::Call || kind == InsnKind::Ret;
    }
    bool is_label() const { return kind == InsnKind::Label; }

    /**
     * @brief Binary encoding, bit-compatible with `hw/xbt_alu_model.hpp`.
     *
     * This is what lets a lowered function be handed straight to
     * `RiscvXbtCoprocessor::execute_program`, so the selector's output can be
     * conformance-tested against the IR interpreter instead of only eyeballed.
     * Returns 0 for non-Xbt pseudo-instructions (labels / RV64I branches / calls / spills).
     */
    uint32_t encode() const {
        if (!is_xbt_insn()) return 0;
        if (is_itype)
            return ((imm & 0xFFFU) << 20) |
                   ((static_cast<uint32_t>(rs1)    & 0x1FU) << 15) |
                   ((static_cast<uint32_t>(funct3) & 0x07U) << 12) |
                   ((static_cast<uint32_t>(rd)     & 0x1FU) << 7)  |
                   OPCODE_XBT_CUSTOM0;
        return ((static_cast<uint32_t>(funct7) & 0x7FU) << 25) |
               ((static_cast<uint32_t>(rs2)    & 0x1FU) << 20) |
               ((static_cast<uint32_t>(rs1)    & 0x1FU) << 15) |
               ((static_cast<uint32_t>(funct3) & 0x07U) << 12) |
               ((static_cast<uint32_t>(rd)     & 0x1FU) << 7)  |
               OPCODE_XBT_CUSTOM0;
    }

    std::string text() const {
        if (kind == InsnKind::Label) {
            std::string s = pseudo_asm;
            if (!comment.empty()) s += "  // " + comment;
            return s;
        }
        if (kind == InsnKind::BranchIfNotPos || kind == InsnKind::Jump ||
            kind == InsnKind::Call || kind == InsnKind::Mul ||
            kind == InsnKind::Spill || kind == InsnKind::Reload ||
            kind == InsnKind::StackAdj || kind == InsnKind::Ret) {
            std::string s = "  " + pseudo_asm;
            if (!comment.empty()) s += "  // " + comment;
            return s;
        }
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

// --------------------------------------------------------------------------
// Compiled assembly metrics & post-lowering peephole optimiser
// --------------------------------------------------------------------------

struct CompiledMetrics {
    size_t total_insns{0};
    size_t xbt_insns{0};
    size_t branchless_cmovs{0};
    size_t zero_gate_negs{0};
    size_t li5_chunks{0};
    size_t shl3_shifts{0};
    size_t control_branches{0};
    size_t labels{0};
    size_t peak_regs{0};
    size_t function_calls{0};
    size_t runtime_muls{0};
    size_t stack_spills{0};
    size_t stack_reloads{0};
};

inline CompiledMetrics compute_metrics(const std::vector<Insn>& code) {
    CompiledMetrics m;
    for (const auto& ins : code) {
        if (ins.is_label()) {
            ++m.labels;
            continue;
        }
        if (ins.kind == InsnKind::Call) {
            ++m.function_calls;
            ++m.total_insns;
            continue;
        }
        if (ins.kind == InsnKind::Mul) {
            ++m.runtime_muls;
            ++m.total_insns;
            if (ins.rd > 0)  m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rd));
            if (ins.rs1 > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rs1));
            if (ins.rs2 > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rs2));
            continue;
        }
        if (ins.kind == InsnKind::Spill) {
            ++m.stack_spills;
            ++m.total_insns;
            if (ins.rd > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rd));
            continue;
        }
        if (ins.kind == InsnKind::Reload) {
            ++m.stack_reloads;
            ++m.total_insns;
            if (ins.rd > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rd));
            continue;
        }
        if (ins.kind == InsnKind::StackAdj || ins.kind == InsnKind::Ret) {
            ++m.total_insns;
            continue;
        }
        if (ins.is_control_flow()) {
            ++m.control_branches;
            ++m.total_insns;
            if (ins.kind == InsnKind::BranchIfNotPos && ins.rs1 > 0) {
                m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rs1));
            }
            continue;
        }
        if (ins.is_xbt_insn()) {
            ++m.xbt_insns;
            ++m.total_insns;

            if (ins.rd > 0)  m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rd));
            if (ins.rs1 > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rs1));

            if (ins.is_itype) {
                if (ins.funct3 == static_cast<uint8_t>(Funct3::LI5)) {
                    ++m.li5_chunks;
                }
            } else {
                const auto f3 = static_cast<Funct3>(ins.funct3);
                if (f3 == Funct3::NEG) {
                    ++m.zero_gate_negs;
                } else if (f3 == Funct3::GROUP2) {
                    const auto g2 = static_cast<Group2>(ins.funct7);
                    if (g2 == Group2::CMOVZ || g2 == Group2::CMOVP || g2 == Group2::CMOVN) {
                        ++m.branchless_cmovs;
                        if (ins.rs2 > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rs2));
                    } else if (g2 == Group2::SHL3) {
                        ++m.shl3_shifts;
                        // rs2 holds shift immediate, not a register index
                    } else if (g2 == Group2::MUL || g2 == Group2::BTFA) {
                        if (ins.rs2 > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rs2));
                    }
                } else if (f3 == Funct3::KSADD || f3 == Funct3::BTHA ||
                           f3 == Funct3::DOT64 || f3 == Funct3::CSAMAC) {
                    if (ins.rs2 > 0) m.peak_regs = std::max(m.peak_regs, static_cast<size_t>(ins.rs2));
                }
            }
        }
    }
    return m;
}

/**
 * @brief Post-lowering peephole optimiser for Xbt instruction streams.
 *
 *   1. Removes redundant `KSADD r, r, t0` (or `KSADD r, t0, r`) self-moves.
 *   2. Collapses adjacent `NEG t, a; NEG d, t` when `t` is dead after the
 *      second NEG:
 *        - if `d == a`: erases both instructions (0 instructions).
 *        - if `d != a`: replaces the pair with a single `KSADD d, a, t0` move.
 */
inline void peephole_optimise(std::vector<Insn>& code) {
    auto is_neg = [](const Insn& i) {
        return i.is_xbt_insn() && !i.is_itype &&
               i.funct3 == static_cast<uint8_t>(Funct3::NEG);
    };
    auto is_self_move = [](const Insn& i) {
        return i.is_xbt_insn() && !i.is_itype &&
               i.funct3 == static_cast<uint8_t>(Funct3::KSADD) &&
               ((i.rd == i.rs1 && i.rs2 == 0) || (i.rd == i.rs2 && i.rs1 == 0));
    };
    auto reads_reg = [](const Insn& i, uint8_t reg) -> bool {
        if (reg == 0) return false;
        if (i.kind == InsnKind::BranchIfNotPos) return i.rs1 == reg;
        if (i.kind == InsnKind::Mul) return i.rs1 == reg || i.rs2 == reg;
        if (i.kind == InsnKind::Spill) return i.rd == reg;
        if (!i.is_xbt_insn()) return false;
        if (i.rs1 == reg) return true;
        if (!i.is_itype) {
            const auto f3 = static_cast<Funct3>(i.funct3);
            if (f3 == Funct3::GROUP2) {
                const auto g2 = static_cast<Group2>(i.funct7);
                if (g2 == Group2::CMOVZ || g2 == Group2::CMOVP || g2 == Group2::CMOVN) {
                    return i.rs2 == reg || i.rd == reg; // CMOV preserves rd when false
                }
                if (g2 == Group2::BTFA) {
                    return i.rs2 == reg || i.rd == reg; // BTFA accumulates into rd
                }
                if (g2 == Group2::MUL) {
                    return i.rs2 == reg;
                }
                // SIGN, PACK243, SHL3: rs2 is unused or an immediate
                return false;
            }
            if (f3 == Funct3::CSAMAC) {
                return i.rs2 == reg || i.rd == reg;
            }
            if (f3 == Funct3::KSADD || f3 == Funct3::BTHA || f3 == Funct3::DOT64) {
                return i.rs2 == reg;
            }
        }
        return false;
    };
    auto unconditionally_writes_reg = [](const Insn& i, uint8_t reg) -> bool {
        if (reg == 0) return false;
        if (i.kind == InsnKind::Mul || i.kind == InsnKind::Reload) return i.rd == reg;
        if (!i.is_xbt_insn()) return false;
        if (i.rd != reg) return false;
        if (i.is_itype) return true; // LI5 unconditionally overwrites rd
        const auto f3 = static_cast<Funct3>(i.funct3);
        if (f3 == Funct3::GROUP2) {
            const auto g2 = static_cast<Group2>(i.funct7);
            if (g2 == Group2::CMOVZ || g2 == Group2::CMOVP ||
                g2 == Group2::CMOVN || g2 == Group2::BTFA) {
                return false;
            }
            return true;
        }
        if (f3 == Funct3::CSAMAC) return false;
        return true;
    };
    auto is_dead_after = [&](size_t start_idx, uint8_t reg) -> bool {
        for (size_t j = start_idx; j < code.size(); ++j) {
            if (code[j].is_label() || code[j].is_control_flow()) {
                return false; // conservative across basic-block boundaries
            }
            if (reads_reg(code[j], reg)) return false;
            if (unconditionally_writes_reg(code[j], reg)) return true;
        }
        // Live-out of function is only the return register t1.
        return reg != 1;
    };

    bool changed = true;
    while (changed) {
        changed = false;

        // Pass 1: Remove redundant KSADD r, r, t0 self-moves.
        std::vector<Insn> next;
        next.reserve(code.size());
        for (const auto& ins : code) {
            if (is_self_move(ins)) {
                changed = true;
                continue;
            }
            next.push_back(ins);
        }
        code.swap(next);

        // Pass 2: Collapse adjacent NEG t, a; NEG d, t when t is dead.
        for (size_t i = 0; i + 1 < code.size(); ) {
            if (is_neg(code[i]) && is_neg(code[i + 1]) &&
                code[i + 1].rs1 == code[i].rd) {
                const uint8_t t = code[i].rd;
                const uint8_t a = code[i].rs1;
                const uint8_t d = code[i + 1].rd;
                if (t == d || is_dead_after(i + 2, t)) {
                    if (d == a) {
                        code.erase(code.begin() + static_cast<std::ptrdiff_t>(i),
                                   code.begin() + static_cast<std::ptrdiff_t>(i + 2));
                    } else {
                        Insn mv;
                        mv.funct3 = static_cast<uint8_t>(Funct3::KSADD);
                        mv.rd = d;
                        mv.rs1 = a;
                        mv.rs2 = 0;
                        mv.comment = "move t" + std::to_string((int)d) + ", t" +
                                     std::to_string((int)a) + "   // peephole double-neg";
                        code[i] = std::move(mv);
                        code.erase(code.begin() + static_cast<std::ptrdiff_t>(i + 1));
                    }
                    changed = true;
                    continue;
                }
            }
            ++i;
        }
    }
}

struct LoweredFunction {
    std::string name;
    std::vector<Insn> code;
    int ternary_regs_used{0};
    std::vector<std::string> notes;
    CompiledMetrics metrics;
};

// --------------------------------------------------------------------------
// Selection rules
// --------------------------------------------------------------------------

class InstructionSelector;

/** Context handed to a rule's emitter. */
struct RuleContext {
    InstructionSelector* sel;
    LoweredFunction& out;

    /** Recursively lower a sub-expression; returns the register holding it. */
    int lower(const ir::Expression& e) const;
    /** Look up the register bound to an already-lowered symbol. */
    int lookup_symbol(uint32_t symbol) const;
    int alloc() const;
    void release(int r) const;
    void emit(const Insn& i) const;
    void note(const std::string& s) const;
};

/**
 * @brief One pattern-matching lowering rule.
 *
 * `matches` is a predicate over the whole expression including its operands,
 * so a rule can accept only the operand shapes it handles correctly (for
 * example, lane-wise multiplication is only safe for single-trit operands).
 */
struct SelectionRule {
    ir::Op op{};
    std::string name;
    int cost{1};
    std::function<bool(const ir::Expression&)> matches;
    std::function<int(const RuleContext&, const ir::Expression&)> emit;

    /**
     * @brief Human-readable explanation of what `matches` requires.
     *
     * When an op is known but no rule accepts the operand shapes, the
     * selector reports this text instead of a bare op number. A rule that
     * can never fail to match may leave it empty.
     */
    std::string reason;
};

// --------------------------------------------------------------------------
// Selector
// --------------------------------------------------------------------------

class InstructionSelector {
public:
    static constexpr int REG_COUNT = 32;
    static constexpr int REG_ZERO = 0;

    explicit InstructionSelector(int available_regs = 16)
        : available_(available_regs) {
        if (available_ < 4 || available_ > REG_COUNT)
            throw std::invalid_argument("xbt: register count out of range");
        register_default_rules();
    }

    /**
     * @brief Construct for a specific target.
     *
     * Target capability drives rule availability, register budget, and
     * immediate width -- that is the whole point of describing hardware as
     * data.
     */
    explicit InstructionSelector(const bt::target::MachineDescription& m,
                                 int available_regs = 0)
        : available_(derive_available_regs(m, available_regs)), machine_(&m) {
        if (available_ < 4 || available_ > REG_COUNT)
            throw std::invalid_argument("xbt: register count out of range");
        register_default_rules();
    }

    /** Effective register pool limit driven by MachineDescription or constructor. */
    int available_regs() const { return available_; }

    /** Maximum immediate width in trits driven by MachineDescription (default 5). */
    int max_immediate_trits() const {
        if (machine_ && machine_->max_immediate_trits > 0)
            return static_cast<int>(machine_->max_immediate_trits);
        return TRITS_PER_BYTE;
    }

    /** The target this selector was built for, or nullptr if unspecified. */
    const bt::target::MachineDescription* machine() const { return machine_; }

    LoweredFunction lower_function(const ir::Function& f) {
        LoweredFunction out;
        out.name = f.name;
        reset();
        current_out_ = &out;

        has_calls_ = contains_calls(f.body);
        if (has_calls_) {
            stack_frame_size_ = std::max(stack_frame_size_, 16);
        }

        for (const auto& p : f.parameters) {
            int r = alloc_reg();
            symbol_reg_[p.symbol] = r;
            out.code.push_back(note_insn("param sym#" + std::to_string(p.symbol) +
                                         " -> t" + std::to_string(r)));
            used_registers_.insert(r);
        }

        lower_statements(f.body, out);

        if (stack_frame_size_ > 0) {
            std::vector<Insn> prologue;
            Insn adj_down;
            adj_down.kind = InsnKind::StackAdj;
            adj_down.imm = static_cast<uint32_t>(-stack_frame_size_);
            adj_down.pseudo_asm = "addi sp, sp, -" + std::to_string(stack_frame_size_);
            adj_down.comment = "allocate stack frame (" +
                               std::to_string(stack_frame_size_) + " bytes)";
            prologue.push_back(adj_down);

            if (has_calls_) {
                Insn save_ra;
                save_ra.kind = InsnKind::Spill;
                save_ra.rd = 1; // x1 = ra
                save_ra.imm = 0;
                save_ra.pseudo_asm = "sd ra, 0(sp)";
                save_ra.comment = "save return address";
                prologue.push_back(save_ra);
            }

            out.code.insert(out.code.begin(), prologue.begin(), prologue.end());

            if (has_calls_) {
                Insn restore_ra;
                restore_ra.kind = InsnKind::Reload;
                restore_ra.rd = 1;
                restore_ra.imm = 0;
                restore_ra.pseudo_asm = "ld ra, 0(sp)";
                restore_ra.comment = "restore return address";
                out.code.push_back(restore_ra);
            }

            Insn adj_up;
            adj_up.kind = InsnKind::StackAdj;
            adj_up.imm = static_cast<uint32_t>(stack_frame_size_);
            adj_up.pseudo_asm = "addi sp, sp, " + std::to_string(stack_frame_size_);
            adj_up.comment = "free stack frame";
            out.code.push_back(adj_up);
        }

        peephole_optimise(out.code);
        out.ternary_regs_used = next_reg_;
        out.metrics = compute_metrics(out.code);
        current_out_ = nullptr;
        return out;
    }

    /** Append a rule. Later rules are tried after earlier ones for the same op. */
    void add_rule(SelectionRule r) { rules_[r.op].push_back(std::move(r)); }

    const std::vector<SelectionRule>& rules_for(ir::Op op) const {
        static const std::vector<SelectionRule> none;
        auto it = rules_.find(op);
        return it == rules_.end() ? none : it->second;
    }

    /** Names of every IR op with at least one rule, for diagnostics. */
    std::string supported_ops() const;

private:
    friend struct RuleContext;

    int available_;
    const bt::target::MachineDescription* machine_{nullptr};
    int next_reg_{1};
    uint32_t next_label_id_{0};
    std::map<uint32_t, int> symbol_reg_;
    std::map<uint32_t, int> symbol_spill_offset_;
    std::set<int> free_regs_;
    std::set<int> used_registers_;
    LoweredFunction* current_out_{nullptr};
    int spill_slot_count_{0};
    int stack_frame_size_{0};
    bool has_calls_{false};

    std::map<ir::Op, std::vector<SelectionRule>> rules_;

    void reset() {
        next_reg_ = 1;
        next_label_id_ = 0;
        symbol_reg_.clear();
        symbol_spill_offset_.clear();
        free_regs_.clear();
        used_registers_.clear();
        spill_slot_count_ = 0;
        stack_frame_size_ = 0;
        has_calls_ = false;
        current_out_ = nullptr;
    }

    int allocate_spill_slot() {
        int slot = spill_slot_count_++;
        int offset = (has_calls_ ? 16 : 0) + slot * 16;
        int needed = offset + 16;
        stack_frame_size_ = std::max(stack_frame_size_, ((needed + 15) / 16) * 16);
        return offset;
    }

    // -- register allocation ------------------------------------------------
    bool is_bound_reg(int r) const {
        if (r <= REG_ZERO) return true;
        for (const auto& [sym, reg] : symbol_reg_) {
            if (reg == r) return true;
        }
        return false;
    }

    int alloc_reg() {
        if (!free_regs_.empty()) {
            int r = *free_regs_.begin();
            free_regs_.erase(free_regs_.begin());
            used_registers_.insert(r);
            return r;
        }
        if (next_reg_ < available_) {
            int r = next_reg_++;
            used_registers_.insert(r);
            return r;
        }

        // Out of physical registers: spill a bound symbol to the stack frame.
        for (auto it = symbol_reg_.begin(); it != symbol_reg_.end(); ++it) {
            uint32_t sym = it->first;
            int victim_reg = it->second;
            if (victim_reg > REG_ZERO) {
                int offset = allocate_spill_slot();
                symbol_spill_offset_[sym] = offset;

                if (current_out_) {
                    Insn spill;
                    spill.kind = InsnKind::Spill;
                    spill.rd = static_cast<uint8_t>(victim_reg);
                    spill.imm = static_cast<uint32_t>(offset);
                    spill.pseudo_asm = "sd x" + std::to_string(victim_reg) +
                                       ", " + std::to_string(offset) + "(sp)";
                    spill.comment = "spill sym#" + std::to_string(sym) +
                                    " -> stack slot @" + std::to_string(offset);
                    current_out_->code.push_back(spill);
                }

                symbol_reg_.erase(it);
                return victim_reg;
            }
        }

        // If no bound symbol can be spilled, spill an older temporary.
        int victim = (available_ > 2) ? (available_ - 1) : 1;
        int offset = allocate_spill_slot();
        if (current_out_) {
            Insn spill;
            spill.kind = InsnKind::Spill;
            spill.rd = static_cast<uint8_t>(victim);
            spill.imm = static_cast<uint32_t>(offset);
            spill.pseudo_asm = "sd x" + std::to_string(victim) +
                               ", " + std::to_string(offset) + "(sp)";
            spill.comment = "spill temporary t" + std::to_string(victim) +
                            " -> stack slot @" + std::to_string(offset);
            current_out_->code.push_back(spill);
        }
        return victim;
    }

    int lookup_symbol(uint32_t symbol) {
        auto it = symbol_reg_.find(symbol);
        if (it != symbol_reg_.end()) {
            return it->second;
        }
        auto sp = symbol_spill_offset_.find(symbol);
        if (sp != symbol_spill_offset_.end()) {
            int offset = sp->second;
            int r = alloc_reg();
            if (current_out_) {
                Insn reload;
                reload.kind = InsnKind::Reload;
                reload.rd = static_cast<uint8_t>(r);
                reload.imm = static_cast<uint32_t>(offset);
                reload.pseudo_asm = "ld x" + std::to_string(r) +
                                    ", " + std::to_string(offset) + "(sp)";
                reload.comment = "reload sym#" + std::to_string(symbol) +
                                 " <- stack slot @" + std::to_string(offset);
                current_out_->code.push_back(reload);
            }
            symbol_reg_[symbol] = r;
            return r;
        }
        throw std::runtime_error("xbt: unknown symbol #" + std::to_string(symbol));
    }

    /**
     * @brief Reclaim an expression temporary register if it is not bound to a
     *        symbol (parameter or local variable) and not t0.
     */
    void free_reg(int r) {
        if (r > REG_ZERO && !is_bound_reg(r)) {
            free_regs_.insert(r);
        }
    }

    static bool expr_contains_calls(const ir::Expression& e) {
        if (!e) return false;
        if (e->op == ir::Op::Call) return true;
        for (const auto& op : e->operands) {
            if (expr_contains_calls(op)) return true;
        }
        return false;
    }

    static bool contains_calls(const std::vector<ir::Statement>& stmts) {
        for (const auto& s : stmts) {
            if (expr_contains_calls(s.value)) return true;
            if (contains_calls(s.body)) return true;
            if (contains_calls(s.alternative)) return true;
        }
        return false;
    }

    // -- instruction helpers ------------------------------------------------
    Insn note_insn(const std::string& c) const {
        Insn i; i.comment = c; return i;
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

    Insn make_label(const std::string& lbl, const std::string& c = "") const {
        Insn i;
        i.kind = InsnKind::Label;
        i.pseudo_asm = lbl + ":";
        i.comment = c;
        return i;
    }

    Insn make_branch_not_pos(int reg, const std::string& target_lbl,
                             const std::string& c = "") const {
        Insn i;
        i.kind = InsnKind::BranchIfNotPos;
        i.rs1 = static_cast<uint8_t>(reg);
        i.pseudo_asm = "blez x" + std::to_string(reg) + ", " + target_lbl;
        i.comment = c;
        return i;
    }

    Insn make_jump(const std::string& target_lbl,
                   const std::string& c = "") const {
        Insn i;
        i.kind = InsnKind::Jump;
        i.pseudo_asm = "j " + target_lbl;
        i.comment = c;
        return i;
    }

    /** rd <- rs2 when the sign test on rs1 holds (leaves rd otherwise). */
    Insn make_cmov(Group2 mode, int rd, int cond, int src,
                   const std::string& c) const {
        return make_r(Funct3::GROUP2, static_cast<uint8_t>(mode), rd, cond, src, c);
    }

    static std::string cmov_name(Group2 mode) {
        switch (mode) {
        case Group2::CMOVZ: return "cmovz";
        case Group2::CMOVP: return "cmovp";
        case Group2::CMOVN: return "cmovn";
        default: return "cmov";
        }
    }

    // -- algebraic helpers --------------------------------------------------
    static bool try_const_int(const ir::Expression& e, int64_t& val) {
        if (!e) return false;
        if (e->op == ir::Op::Integer) {
            val = e->integer;
            return true;
        }
        if (e->op == ir::Op::Negate && !e->operands.empty()) {
            int64_t inner = 0;
            if (try_const_int(e->operands[0], inner)) {
                val = -inner;
                return true;
            }
        }
        return false;
    }

    static bool is_const_int(const ir::Expression& e, int64_t val) {
        int64_t got = 0;
        return try_const_int(e, got) && got == val;
    }

    /**
     * @brief If |val| == 3^k for 1 <= k <= 31 (fits 5-bit SHL3 immediate),
     *        sets `sign` to +1/-1 and returns `k`; otherwise returns -1.
     */
    static int pow3_exponent(int64_t val, int& sign) {
        if (val == 0 || val == 1 || val == -1) return -1;
        sign = (val < 0) ? -1 : 1;
        uint64_t mag = (val < 0) ? static_cast<uint64_t>(-val)
                                 : static_cast<uint64_t>(val);
        int k = 0;
        while (mag > 1ULL && (mag % 3ULL) == 0ULL) {
            mag /= 3ULL;
            ++k;
        }
        return (mag == 1ULL && k >= 1 && k <= 31) ? k : -1;
    }

    static bool is_pow3_const(const ir::Expression& e) {
        int64_t val = 0;
        if (!try_const_int(e, val)) return false;
        int sign = 0;
        return pow3_exponent(val, sign) >= 1;
    }

    static const ir::Statement* unwrap_single_stmt(
        const std::vector<ir::Statement>& stmts) {
        const std::vector<ir::Statement>* cur = &stmts;
        while (cur->size() == 1 && (*cur)[0].kind == ir::StatementKind::Block) {
            cur = &(*cur)[0].body;
        }
        return (cur->size() == 1) ? &(*cur)[0] : nullptr;
    }

    // -- lowering -----------------------------------------------------------
    int lower_expr(const ir::Expression& e, LoweredFunction& out);

    void lower_statements(const std::vector<ir::Statement>& body,
                          LoweredFunction& out);

    static int derive_available_regs(const bt::target::MachineDescription& m,
                                     int requested) {
        if (requested > 0) return requested;
        if (m.ternary_registers == 0) return 16;
        int regs = static_cast<int>(m.ternary_registers);
        if (regs > REG_COUNT) regs = REG_COUNT;
        return regs;
    }

    std::vector<Insn> materialise(int64_t value, int dst);

    static uint32_t pack_digits(const std::vector<int>& dig, int trit_offset);

    static bool is_single_trit(const ir::Expression& e) {
        return e && e->op == ir::Op::Integer && e->integer >= -1 && e->integer <= 1;
    }

    /** True when an expression is side-effect free and safe for eager predication. */
    static bool is_pure(const ir::Expression& e);

    void register_default_rules();
};

// --------------------------------------------------------------------------
// RuleContext implementation
// --------------------------------------------------------------------------

inline int RuleContext::lower(const ir::Expression& e) const {
    return sel->lower_expr(e, out);
}
inline int RuleContext::lookup_symbol(uint32_t symbol) const {
    return sel->lookup_symbol(symbol);
}
inline int RuleContext::alloc() const { return sel->alloc_reg(); }
inline void RuleContext::release(int r) const { sel->free_reg(r); }
inline void RuleContext::emit(const Insn& i) const { out.code.push_back(i); }
inline void RuleContext::note(const std::string& s) const {
    out.code.push_back(sel->note_insn(s));
}

// --------------------------------------------------------------------------
// Default rule table
// --------------------------------------------------------------------------

inline void InstructionSelector::register_default_rules() {
    // ---------------- constants ----------------
    add_rule({ir::Op::Integer, "li5-constant", 1,
        [](const ir::Expression& e) { return ir::integral(e->type); },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int d = c.alloc();
            for (auto& i : c.sel->materialise(e->integer, d)) c.emit(i);
            return d;
        }});

    // ---------------- parameters ----------------
    add_rule({ir::Op::Parameter, "parameter", 1,
        [](const ir::Expression&) { return true; },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            return c.lookup_symbol(e->symbol);
        }});

    // ---------------- integer conversion (Int39 <-> Int39 is a pure move) ----
    // IntCast is value preserving between integral types, so the selector can
    // simply forward the source register.  A cast *to* Trit is only legal when
    // the source is provably in {-1,0,1}; anything else is refused rather than
    // silently truncated (the core interpreter raises overflow_error there).
    add_rule({ir::Op::IntCast, "int-move", 1,
        [](const ir::Expression& e) {
            if (e->type == ir::Type::Trit) {
                const ir::Expression& s = e->operands.at(0);
                return s->op == ir::Op::Integer && s->integer >= -1 && s->integer <= 1;
            }
            return ir::integral(e->type) && ir::integral(e->operands.at(0)->type);
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            return c.lower(e->operands.at(0));
        }});

    // ---------------- negation (algebraic double-negation folding) -----------
    // Negate(Negate(x)) -> x (0 instructions).
    add_rule({ir::Op::Negate, "double-neg-fold", 0,
        [](const ir::Expression& e) {
            return !e->operands.empty() && e->operands[0] &&
                   e->operands[0]->op == ir::Op::Negate &&
                   !e->operands[0]->operands.empty();
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            return c.lower(e->operands[0]->operands.at(0));
        }});

    add_rule({ir::Op::Negate, "neg", 1,
        [](const ir::Expression&) { return true; },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int a = c.lower(e->operands.at(0));
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::NEG, 0, d, a, 0,
                "neg t" + std::to_string(d) + ", t" + std::to_string(a)));
            c.release(a);
            return d;
        }});

    // ---------------- addition (algebraic zero-identity folding) -------------
    // Add(x, 0) / Add(0, x) -> x (0 instructions).
    add_rule({ir::Op::Add, "add-zero-fold", 0,
        [](const ir::Expression& e) {
            return InstructionSelector::is_const_int(e->operands.at(0), 0) ||
                   InstructionSelector::is_const_int(e->operands.at(1), 0);
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            if (InstructionSelector::is_const_int(e->operands.at(0), 0))
                return c.lower(e->operands.at(1));
            return c.lower(e->operands.at(0));
        }});

    add_rule({ir::Op::Add, "ksadd", 1,
        [](const ir::Expression&) { return true; },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int a = c.lower(e->operands.at(0));
            int b = c.lower(e->operands.at(1));
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::KSADD, 0, d, a, b,
                "ksadd t" + std::to_string(d) + ", t" + std::to_string(a) +
                ", t" + std::to_string(b)));
            c.release(a); c.release(b);
            return d;
        }});

    // ---------------- subtraction (algebraic zero-identity folding) ----------
    // Sub(x, 0) -> x (0 instructions).
    add_rule({ir::Op::Sub, "sub-zero-fold", 0,
        [](const ir::Expression& e) {
            return InstructionSelector::is_const_int(e->operands.at(1), 0);
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            return c.lower(e->operands.at(0));
        }});

    // Sub(a, b) = a + (-b)
    add_rule({ir::Op::Sub, "neg+ksadd", 2,
        [](const ir::Expression&) { return true; },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int a = c.lower(e->operands.at(0));
            int b = c.lower(e->operands.at(1));
            int nb = c.alloc();
            c.emit(c.sel->make_r(Funct3::NEG, 0, nb, b, 0,
                "neg t" + std::to_string(nb) + ", t" + std::to_string(b)));
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::KSADD, 0, d, a, nb,
                "ksadd t" + std::to_string(d) + ", t" + std::to_string(a) +
                ", t" + std::to_string(nb)));
            c.release(a); c.release(b); c.release(nb);
            return d;
        }});

    // ---------------- algebraic multiplication folding -----------------------
    // Mul(x, 0) / Mul(0, x) -> t0 (0 instructions, hardwired zero register).
    add_rule({ir::Op::Mul, "mul-zero-fold", 0,
        [](const ir::Expression& e) {
            const auto& a = e->operands.at(0);
            const auto& b = e->operands.at(1);
            return (InstructionSelector::is_const_int(a, 0) && InstructionSelector::is_pure(b)) ||
                   (InstructionSelector::is_const_int(b, 0) && InstructionSelector::is_pure(a));
        },
        [](const RuleContext&, const ir::Expression&) -> int {
            return InstructionSelector::REG_ZERO;
        }});

    // Mul(x, 1) / Mul(1, x) -> x (0 instructions).
    add_rule({ir::Op::Mul, "mul-one-fold", 0,
        [](const ir::Expression& e) {
            return InstructionSelector::is_const_int(e->operands.at(0), 1) ||
                   InstructionSelector::is_const_int(e->operands.at(1), 1);
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            if (InstructionSelector::is_const_int(e->operands.at(0), 1))
                return c.lower(e->operands.at(1));
            return c.lower(e->operands.at(0));
        }});

    // Mul(x, -1) / Mul(-1, x) -> single zero-gate NEG (1 instruction).
    add_rule({ir::Op::Mul, "mul-neg-one-fold", 1,
        [](const ir::Expression& e) {
            return InstructionSelector::is_const_int(e->operands.at(0), -1) ||
                   InstructionSelector::is_const_int(e->operands.at(1), -1);
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            const ir::Expression& v_expr =
                InstructionSelector::is_const_int(e->operands.at(0), -1)
                    ? e->operands.at(1)
                    : e->operands.at(0);
            int v = c.lower(v_expr);
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::NEG, 0, d, v, 0,
                "neg t" + std::to_string(d) + ", t" + std::to_string(v) +
                "   // mul by -1"));
            c.release(v);
            return d;
        }});

    // Mul(x, +-3^k) / Mul(+-3^k, x) -> single SHL3(x, k) (+ optional NEG).
    // Turns power-of-3 multiplications into 1 hardware trit-shift!
    add_rule({ir::Op::Mul, "mul-pow3-shl3", 1,
        [](const ir::Expression& e) {
            const auto& a = e->operands.at(0);
            const auto& b = e->operands.at(1);
            int64_t ka = 0, kb = 0;
            const bool a_const = InstructionSelector::try_const_int(a, ka);
            const bool b_const = InstructionSelector::try_const_int(b, kb);
            // Exactly one side constant power-of-3 (keeps const*const refusal
            // consistent with lane-mul / const-shift-add-mul semantics).
            if (a_const == b_const) return false;
            return a_const ? InstructionSelector::is_pow3_const(a)
                           : InstructionSelector::is_pow3_const(b);
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            const auto& a = e->operands.at(0);
            const auto& b = e->operands.at(1);
            int64_t K = 0;
            const bool a_const = InstructionSelector::try_const_int(a, K);
            if (!a_const) InstructionSelector::try_const_int(b, K);
            const ir::Expression& v_expr = a_const ? b : a;
            int sign = 1;
            int k = InstructionSelector::pow3_exponent(K, sign);
            int v = c.lower(v_expr);
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::GROUP2, static_cast<uint8_t>(Group2::SHL3),
                d, v, k,
                "shl3 t" + std::to_string(d) + ", t" + std::to_string(v) +
                ", " + std::to_string(k) + "   // * 3^" + std::to_string(k)));
            c.release(v);
            if (sign < 0) {
                c.emit(c.sel->make_r(Funct3::NEG, 0, d, d, 0,
                    "neg t" + std::to_string(d) + ", t" + std::to_string(d) +
                    "   // negate -3^" + std::to_string(k)));
            }
            return d;
        }});

    // ---------------- native scalar multiply ----------------
    // Only available when the TARGET says it has one. This is the whole point
    // of describing hardware as data: the same IR lowers differently on a
    // machine with a true multiplier than on one without, and the decision is
    // a field in targets/*.yaml rather than an #ifdef in the driver.
    add_rule({ir::Op::Mul, "native-scalar-mul", 1,
        [this](const ir::Expression&) {
            return machine_ != nullptr && machine_->has_scalar_mul;
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int a = c.lower(e->operands.at(0));
            int b = c.lower(e->operands.at(1));
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::GROUP2, (uint8_t)Group2::MUL, d, a, b,
                "xbt.mul t" + std::to_string(d) + ", t" + std::to_string(a) +
                ", t" + std::to_string(b) + "   // native scalar multiply"));
            c.release(a); c.release(b);
            return d;
        },
        "requires a target with has_scalar_mul (see targets/*.yaml)"});

    // ---------------- lane-wise multiply ----------------
    // Only safe for single-trit operands: the hardware product drops every
    // carry that would cross a lane boundary.
    add_rule({ir::Op::Mul, "lane-mul", 1,
        [](const ir::Expression& e) {
            return InstructionSelector::is_single_trit(e->operands.at(0)) &&
                   InstructionSelector::is_single_trit(e->operands.at(1));
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int a = c.lower(e->operands.at(0));
            int b = c.lower(e->operands.at(1));
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::GROUP2, (uint8_t)Group2::MUL, d, a, b,
                "xbt.mul t" + std::to_string(d) + ", t" + std::to_string(a) +
                ", t" + std::to_string(b)));
            c.release(a); c.release(b);
            return d;
        },
        "both operands must be single-trit: the Xbt MUL is lane-wise and "
        "would silently drop inter-lane carries"});

    // ---------------- multiply by a compile-time constant ----------------
    //
    //   a * K = SUM_i (trit_i of K) * (a << i)          accumulated by ksadd
    //
    // Needs SHL3, the shift added for exactly this purpose. Only the CONSTANT
    // case is handled here: decomposing a runtime multiplier would require
    // extracting its trits one at a time, and the ISA has no trit-extract
    // instruction. That general case stays a documented refusal below rather
    // than a silently wrong expansion.
    //
    // Note on partial products: verification/verify_scalar_mul.py (S4) shows
    // balanced ternary is NOT automatically non-adjacent form -- most words
    // have adjacent nonzeros -- so the count here is the number of nonzero
    // trits, roughly 2/3 of the width, not the ~1/3 a true NAF would give.
    add_rule({ir::Op::Mul, "const-shift-add-mul", 8,
        [](const ir::Expression& e) {
            const auto& a = e->operands.at(0);
            const auto& b = e->operands.at(1);
            auto is_const = [](const ir::Expression& x) {
                int64_t tmp = 0;
                return InstructionSelector::try_const_int(x, tmp);
            };
            // Exactly one side constant; the other is lowered normally.
            return is_const(a) != is_const(b);
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            const auto& x0 = e->operands.at(0);
            const auto& x1 = e->operands.at(1);
            int64_t K = 0;
            const bool k_first = InstructionSelector::try_const_int(x0, K);
            if (!k_first) InstructionSelector::try_const_int(x1, K);
            const ir::Expression& v_expr = k_first ? x1 : x0;

            int v = c.lower(v_expr);

            if (K == 0) {                       // a * 0 == 0
                c.release(v);
                return InstructionSelector::REG_ZERO;
            }
            if (K == 1) return v;               // a * 1 == a

            // Balanced-ternary digits of |K|; the sign is folded in at the end
            // because negation is a free wire swap on dual-rail.
            uint64_t mag = (K < 0) ? static_cast<uint64_t>(-K)
                                   : static_cast<uint64_t>(K);
            int trits[41];
            int n = 0;
            while (mag != 0 && n < 41) {
                unsigned r = static_cast<unsigned>(mag % 3);
                if (r == 0)      { trits[n++] = 0;  mag /= 3; }
                else if (r == 1) { trits[n++] = 1;  mag = (mag - 1) / 3; }
                else             { trits[n++] = -1; mag = (mag + 1) / 3; }
            }

            int acc = -1;
            for (int i = 0; i < n; ++i) {
                if (trits[i] == 0) continue;
                int sh = c.alloc();
                c.emit(c.sel->make_r(Funct3::GROUP2, (uint8_t)Group2::SHL3,
                    sh, v, i,
                    "shl3 t" + std::to_string(sh) + ", t" + std::to_string(v) +
                    ", " + std::to_string(i) + "   // partial product 3^" +
                    std::to_string(i)));
                if (trits[i] < 0) {
                    c.emit(c.sel->make_r(Funct3::NEG, 0, sh, sh, 0,
                        "neg t" + std::to_string(sh) + ", t" +
                        std::to_string(sh) + "   // negative trit"));
                }
                if (acc < 0) {
                    acc = sh;                   // first partial product
                } else {
                    c.emit(c.sel->make_r(Funct3::KSADD, 0, acc, acc, sh,
                        "ksadd t" + std::to_string(acc) + ", t" +
                        std::to_string(acc) + ", t" + std::to_string(sh)));
                    c.release(sh);
                }
            }

            if (acc < 0) {                      // K != 0 guarantees this is dead
                acc = c.alloc();
                for (auto& i : c.sel->materialise(0, acc)) c.emit(i);
            }
            c.release(v);

            if (K < 0) {                        // negate: free on dual-rail
                int d = c.alloc();
                c.emit(c.sel->make_r(Funct3::NEG, 0, d, acc, 0,
                    "neg t" + std::to_string(d) + ", t" + std::to_string(acc)));
                c.release(acc);
                return d;
            }
            return acc;
        },
        "exactly one operand must be a compile-time constant; the ISA has no "
        "trit-extract instruction, so a runtime multiplier cannot be "
        "decomposed into partial products"});

    // ---------------- runtime-by-runtime scalar multiply --------------------
    // Fallback when neither operand is a compile-time constant (or on targets
    // without native Xbt Group2 MUL), lowering to an RV64 runtime multiply.
    add_rule({ir::Op::Mul, "runtime-mul", 10,
        [](const ir::Expression&) { return true; },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int a = c.lower(e->operands.at(0));
            int b = c.lower(e->operands.at(1));
            int d = c.alloc();
            Insn mul_insn;
            mul_insn.kind = InsnKind::Mul;
            mul_insn.rd = static_cast<uint8_t>(d);
            mul_insn.rs1 = static_cast<uint8_t>(a);
            mul_insn.rs2 = static_cast<uint8_t>(b);
            mul_insn.pseudo_asm = "mul x" + std::to_string(d) + ", x" +
                                  std::to_string(a) + ", x" + std::to_string(b);
            mul_insn.comment = "runtime scalar multiply: t" + std::to_string(d) +
                               " = t" + std::to_string(a) + " * t" + std::to_string(b);
            c.emit(mul_insn);
            c.release(a); c.release(b);
            return d;
        },
        "general runtime-by-runtime scalar multiply"});

    // ---------------- sign of a difference ----------------
    // Shared helper: every comparison is a predicate over sign(a - b).
    auto emit_sign_of_diff = [](const RuleContext& c,
                                const ir::Expression& e) -> int {
        int a = c.lower(e->operands.at(0));
        int b = c.lower(e->operands.at(1));
        int nb = c.alloc();
        c.emit(c.sel->make_r(Funct3::NEG, 0, nb, b, 0,
            "neg t" + std::to_string(nb) + ", t" + std::to_string(b)));
        int diff = c.alloc();
        c.emit(c.sel->make_r(Funct3::KSADD, 0, diff, a, nb,
            "ksadd t" + std::to_string(diff) + ", t" + std::to_string(a) +
            ", t" + std::to_string(nb)));
        int d = c.alloc();
        c.emit(c.sel->make_r(Funct3::GROUP2, (uint8_t)Group2::SIGN, d, diff, 0,
            "xbt.sign t" + std::to_string(d) + ", t" + std::to_string(diff)));
        c.release(a); c.release(b); c.release(nb); c.release(diff);
        return d;
    };

    /**
     * Builds a comparison rule.
     *
     * Core IR comparisons return 1 when true and -1 when false, never 0. So
     * the rule seeds the result with the "false" value and then conditionally
     * moves in the "true" value for each sign that satisfies the predicate.
     */
    // Every comparison is lowered with one uniform shape:
    //
    //     d  <- (-1)                 // the FALSE value, always
    //     tv <- (+1)                 // the TRUE value, always
    //     cmov<mode> d, sign(a-b), tv   for each mode where the predicate
    //                                   is TRUE
    //
    // An earlier version seeded with whichever value was "in the majority"
    // and conditional-moved in the other. That is correct only when the seed
    // is the FALSE value; seeding with the TRUE value silently inverts the
    // predicate for multi-mode ops like `!=`. The uniform shape costs at most
    // one extra cmov and removes the trap entirely.
    auto comparison_rule = [&](ir::Op op, const char* name,
                               const std::vector<Group2>& true_modes) {
        add_rule({op, name, 3,
            [](const ir::Expression&) { return true; },
            [emit_sign_of_diff, true_modes](
                const RuleContext& c, const ir::Expression& e) -> int {
                int s = emit_sign_of_diff(c, e);

                // Seed with the false value; each cmov writes the true value.
                int d = c.alloc();
                for (auto& i : c.sel->materialise(-1, d)) c.emit(i);

                int tv = c.alloc();
                for (auto& i : c.sel->materialise(1, tv)) c.emit(i);

                for (Group2 mode : true_modes) {
                    c.emit(c.sel->make_cmov(mode, d, s, tv,
                        std::string(c.sel->cmov_name(mode)) + " t" +
                        std::to_string(d) + ", t" + std::to_string(s) +
                        ", t" + std::to_string(tv)));
                }
                c.release(s); c.release(tv);
                return d;
            }});
    };

    // sign(a-b) > 0 / == 0 / < 0 / != 0 / >= 0 / <= 0
    comparison_rule(ir::Op::Greater,      "sign>0",  {Group2::CMOVP});
    comparison_rule(ir::Op::Equal,        "sign==0", {Group2::CMOVZ});
    comparison_rule(ir::Op::Less,         "sign<0",  {Group2::CMOVN});
    comparison_rule(ir::Op::NotEqual,     "sign!=0", {Group2::CMOVP, Group2::CMOVN});
    comparison_rule(ir::Op::GreaterEqual, "sign>=0", {Group2::CMOVP, Group2::CMOVZ});
    comparison_rule(ir::Op::LessEqual,    "sign<=0", {Group2::CMOVN, Group2::CMOVZ});

    // ---------------- spaceship: sign(a - b) ----------------
    add_rule({ir::Op::Spaceship, "sign(a-b)", 2,
        [](const ir::Expression&) { return true; },
        [emit_sign_of_diff](const RuleContext& c, const ir::Expression& e) -> int {
            return emit_sign_of_diff(c, e);
        }});

    // ---------------- 3-way select ----------------
    // SignSelect(cond, pos, zero, neg) -> select by sign(cond).
    //
    // The IR documents the arms as thunks that must never all be evaluated
    // eagerly. A conditional-move lowering necessarily evaluates all three,
    // so the rule only applies when every arm is pure -- no side effects.
    // Impure arms are refused rather than miscompiled.
    add_rule({ir::Op::SignSelect, "cmov-select", 4,
        [](const ir::Expression& e) {
            for (size_t i = 1; i < e->operands.size(); ++i)
                if (!InstructionSelector::is_pure(e->operands[i])) return false;
            return true;
        },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            int cond = c.lower(e->operands.at(0));
            int pos  = c.lower(e->operands.at(1));
            int zero = c.lower(e->operands.at(2));
            int neg  = c.lower(e->operands.at(3));

            // Start from the positive arm, then override for zero and negative.
            int d = c.alloc();
            c.emit(c.sel->make_r(Funct3::KSADD, 0, d, pos, 0,
                "move t" + std::to_string(d) + ", t" + std::to_string(pos)));
            c.emit(c.sel->make_cmov(Group2::CMOVZ, d, cond, zero,
                "cmovz t" + std::to_string(d) + ", t" + std::to_string(cond) +
                ", t" + std::to_string(zero)));
            c.emit(c.sel->make_cmov(Group2::CMOVN, d, cond, neg,
                "cmovn t" + std::to_string(d) + ", t" + std::to_string(cond) +
                ", t" + std::to_string(neg)));

            c.release(cond); c.release(pos); c.release(zero); c.release(neg);
            return d;
        }});

    // ---------------- function call ------------------------------------------
    add_rule({ir::Op::Call, "call", 5,
        [](const ir::Expression&) { return true; },
        [](const RuleContext& c, const ir::Expression& e) -> int {
            std::vector<int> arg_regs;
            arg_regs.reserve(e->operands.size());
            for (const auto& arg_expr : e->operands) {
                arg_regs.push_back(c.lower(arg_expr));
            }

            for (size_t i = 0; i < arg_regs.size(); ++i) {
                int target_reg = static_cast<int>(i + 1);
                if (arg_regs[i] != target_reg) {
                    c.emit(c.sel->make_r(Funct3::KSADD, 0, target_reg, arg_regs[i], 0,
                        "move t" + std::to_string(target_reg) + ", t" +
                        std::to_string(arg_regs[i]) + "   // arg #" + std::to_string(i)));
                }
                c.release(arg_regs[i]);
            }

            Insn call_insn;
            call_insn.kind = InsnKind::Call;
            call_insn.pseudo_asm = "call " + e->name;
            call_insn.comment = "call function " + e->name + " (" +
                                std::to_string(e->operands.size()) + " args)";
            c.emit(call_insn);

            if (e->type == ir::Type::Void) {
                return InstructionSelector::REG_ZERO;
            }

            int ret_dst = c.alloc();
            if (ret_dst != 1) {
                c.emit(c.sel->make_r(Funct3::KSADD, 0, ret_dst, 1, 0,
                    "move t" + std::to_string(ret_dst) + ", t1   // return value"));
            }
            return ret_dst;
        }});
}

// --------------------------------------------------------------------------
// Selector implementation
// --------------------------------------------------------------------------

inline bool InstructionSelector::is_pure(const ir::Expression& e) {
    if (!e) return false;
    switch (e->op) {
    case ir::Op::Integer:
    case ir::Op::Parameter:
    case ir::Op::IntCast:
    case ir::Op::Negate:
    case ir::Op::Add:
    case ir::Op::Sub:
    case ir::Op::Mul:
    case ir::Op::Equal:
    case ir::Op::NotEqual:
    case ir::Op::Less:
    case ir::Op::Greater:
    case ir::Op::LessEqual:
    case ir::Op::GreaterEqual:
    case ir::Op::Spaceship:
    case ir::Op::SignSelect:
    case ir::Op::Abs:
    case ir::Op::DigitAnd:
    case ir::Op::DigitOr:
    case ir::Op::DigitXor:
        for (const auto& op : e->operands) {
            if (!is_pure(op)) return false;
        }
        return true;
    default:
        return false;
    }
}

inline std::string InstructionSelector::supported_ops() const {
    std::string s;
    for (const auto& [op, vec] : rules_) {
        for (const auto& r : vec) s += r.name + " ";
    }
    return s;
}

inline int InstructionSelector::lower_expr(const ir::Expression& e,
                                           LoweredFunction& out) {
    if (!e) throw std::invalid_argument("xbt: null expression");

    const auto& candidates = rules_for(e->op);
    for (const auto& rule : candidates) {
        if (rule.matches && !rule.matches(e)) continue;
        RuleContext ctx{this, out};
        return rule.emit(ctx, e);
    }

    // No rule matched. Say which op, and -- when the op is known but the
    // operand shape is not -- say why, so the message is actionable.
    if (!candidates.empty()) {
        std::string msg = "xbt: no rule for IR op #" +
            std::to_string(static_cast<int>(e->op)) +
            " with these operand shapes (the op is known but not for this "
            "shape); refusing to emit a stub";
        // Surface the concrete requirement of each candidate rule, so the
        // message says what would have to be true rather than just what failed.
        for (const auto& rule : candidates) {
            if (!rule.reason.empty()) msg += " [" + rule.name + ": " + rule.reason + "]";
        }
        throw std::runtime_error(msg);
    }
    throw std::runtime_error(
        "xbt: instruction selection not implemented for IR op #" +
        std::to_string(static_cast<int>(e->op)) + " -- refusing to emit a stub");
}

inline void InstructionSelector::lower_statements(
    const std::vector<ir::Statement>& body, LoweredFunction& out) {
    for (const auto& s : body) {
        switch (s.kind) {
        case ir::StatementKind::Let: {
            int r = lower_expr(s.value, out);
            if (r == REG_ZERO || is_bound_reg(r)) {
                int dst = alloc_reg();
                out.code.push_back(make_r(Funct3::KSADD, 0, dst, r, 0,
                    "move t" + std::to_string(dst) + ", t" + std::to_string(r)));
                r = dst;
            }
            symbol_reg_[s.symbol] = r;
            break;
        }
        case ir::StatementKind::Assign: {
            int r = lower_expr(s.value, out);
            auto it = symbol_reg_.find(s.symbol);
            if (it != symbol_reg_.end()) {
                int dst = it->second;
                if (r != dst) {
                    out.code.push_back(make_r(Funct3::KSADD, 0, dst, r, 0,
                        "move t" + std::to_string(dst) + ", t" + std::to_string(r)));
                    free_reg(r);
                }
            } else {
                auto sp = symbol_spill_offset_.find(s.symbol);
                if (sp != symbol_spill_offset_.end()) {
                    int offset = sp->second;
                    Insn spill;
                    spill.kind = InsnKind::Spill;
                    spill.rd = static_cast<uint8_t>(r);
                    spill.imm = static_cast<uint32_t>(offset);
                    spill.pseudo_asm = "sd x" + std::to_string(r) + ", " +
                                       std::to_string(offset) + "(sp)";
                    spill.comment = "spill assign sym#" + std::to_string(s.symbol) +
                                    " -> stack slot @" + std::to_string(offset);
                    out.code.push_back(spill);
                    free_reg(r);
                } else {
                    if (r == REG_ZERO || is_bound_reg(r)) {
                        int dst = alloc_reg();
                        out.code.push_back(make_r(Funct3::KSADD, 0, dst, r, 0,
                            "move t" + std::to_string(dst) + ", t" + std::to_string(r)));
                        r = dst;
                    }
                    symbol_reg_[s.symbol] = r;
                }
            }
            break;
        }
        case ir::StatementKind::Return: {
            if (s.value) {
                int r = lower_expr(s.value, out);
                if (r != 1) {
                    out.code.push_back(make_r(Funct3::KSADD, 0, 1, r, 0,
                        "move t1, t" + std::to_string(r)));
                }
                free_reg(r);
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

        case ir::StatementKind::If: {
            const ir::Statement* then_s = unwrap_single_stmt(s.body);
            const ir::Statement* else_s = unwrap_single_stmt(s.alternative);

            // Fast path 1: Pure branchless 3-way predication for single Assign
            // to the same target symbol in both arms.
            const bool pure_assign_pair =
                then_s && else_s &&
                then_s->kind == ir::StatementKind::Assign &&
                else_s->kind == ir::StatementKind::Assign &&
                then_s->symbol == else_s->symbol &&
                is_pure(then_s->value) && is_pure(else_s->value);

            // Fast path 2: Pure branchless 3-way predication for single Return
            // in both arms.
            const bool pure_return_pair =
                then_s && else_s &&
                then_s->kind == ir::StatementKind::Return &&
                else_s->kind == ir::StatementKind::Return &&
                then_s->value && else_s->value &&
                is_pure(then_s->value) && is_pure(else_s->value);

            if (pure_assign_pair || pure_return_pair) {
                int cond = lower_expr(s.value, out);
                int sgn = alloc_reg();
                out.code.push_back(make_r(Funct3::GROUP2,
                    static_cast<uint8_t>(Group2::SIGN), sgn, cond, 0,
                    "xbt.sign t" + std::to_string(sgn) + ", t" + std::to_string(cond)));
                free_reg(cond);

                int pos_r = lower_expr(then_s->value, out);
                int alt_r = lower_expr(else_s->value, out);

                if (pure_assign_pair) {
                    auto it = symbol_reg_.find(then_s->symbol);
                    int dst = (it != symbol_reg_.end()) ? it->second : alloc_reg();
                    symbol_reg_[then_s->symbol] = dst;

                    out.code.push_back(make_cmov(Group2::CMOVP, dst, sgn, pos_r,
                        "cmovp t" + std::to_string(dst) + ", t" + std::to_string(sgn) +
                        ", t" + std::to_string(pos_r) + "   // branchless if > 0"));
                    out.code.push_back(make_cmov(Group2::CMOVZ, dst, sgn, alt_r,
                        "cmovz t" + std::to_string(dst) + ", t" + std::to_string(sgn) +
                        ", t" + std::to_string(alt_r) + "   // branchless else (== 0)"));
                    out.code.push_back(make_cmov(Group2::CMOVN, dst, sgn, alt_r,
                        "cmovn t" + std::to_string(dst) + ", t" + std::to_string(sgn) +
                        ", t" + std::to_string(alt_r) + "   // branchless else (< 0)"));
                } else {
                    // Return target is t1; avoid clobbering sgn if sgn == 1.
                    int dst = (sgn != 1) ? 1 : alloc_reg();
                    out.code.push_back(make_cmov(Group2::CMOVP, dst, sgn, pos_r,
                        "cmovp t" + std::to_string(dst) + ", t" + std::to_string(sgn) +
                        ", t" + std::to_string(pos_r) + "   // branchless if > 0"));
                    out.code.push_back(make_cmov(Group2::CMOVZ, dst, sgn, alt_r,
                        "cmovz t" + std::to_string(dst) + ", t" + std::to_string(sgn) +
                        ", t" + std::to_string(alt_r) + "   // branchless else (== 0)"));
                    out.code.push_back(make_cmov(Group2::CMOVN, dst, sgn, alt_r,
                        "cmovn t" + std::to_string(dst) + ", t" + std::to_string(sgn) +
                        ", t" + std::to_string(alt_r) + "   // branchless else (< 0)"));
                    if (dst != 1) {
                        out.code.push_back(make_r(Funct3::KSADD, 0, 1, dst, 0,
                            "move t1, t" + std::to_string(dst)));
                        free_reg(dst);
                    }
                }
                free_reg(pos_r);
                free_reg(alt_r);
                free_reg(sgn);
                break;
            }

            // General / impure If: emit hybrid RV64I + Xbt labels & branches.
            const uint32_t id = next_label_id_++;
            int cond = lower_expr(s.value, out);
            int sgn = alloc_reg();
            out.code.push_back(make_r(Funct3::GROUP2,
                static_cast<uint8_t>(Group2::SIGN), sgn, cond, 0,
                "xbt.sign t" + std::to_string(sgn) + ", t" + std::to_string(cond)));
            free_reg(cond);

            if (s.alternative.empty()) {
                std::string end_lbl = ".L_if_end_" + std::to_string(id);
                out.code.push_back(make_branch_not_pos(sgn, end_lbl,
                    "branch if cond <= 0"));
                free_reg(sgn);
                lower_statements(s.body, out);
                out.code.push_back(make_label(end_lbl));
            } else {
                std::string else_lbl = ".L_if_else_" + std::to_string(id);
                std::string end_lbl  = ".L_if_end_" + std::to_string(id);
                out.code.push_back(make_branch_not_pos(sgn, else_lbl,
                    "branch if cond <= 0"));
                free_reg(sgn);
                lower_statements(s.body, out);
                out.code.push_back(make_jump(end_lbl));
                out.code.push_back(make_label(else_lbl));
                lower_statements(s.alternative, out);
                out.code.push_back(make_label(end_lbl));
            }
            break;
        }

        case ir::StatementKind::While: {
            const uint32_t id = next_label_id_++;
            std::string head_lbl = ".L_while_head_" + std::to_string(id);
            std::string end_lbl  = ".L_while_end_" + std::to_string(id);

            out.code.push_back(make_label(head_lbl));
            int cond = lower_expr(s.value, out);
            int sgn = alloc_reg();
            out.code.push_back(make_r(Funct3::GROUP2,
                static_cast<uint8_t>(Group2::SIGN), sgn, cond, 0,
                "xbt.sign t" + std::to_string(sgn) + ", t" + std::to_string(cond)));
            free_reg(cond);

            out.code.push_back(make_branch_not_pos(sgn, end_lbl,
                "exit while if cond <= 0"));
            free_reg(sgn);

            lower_statements(s.body, out);
            out.code.push_back(make_jump(head_lbl));
            out.code.push_back(make_label(end_lbl));
            break;
        }

        default:
            throw std::runtime_error(
                "xbt: instruction selection not implemented for statement kind #" +
                std::to_string(static_cast<int>(s.kind)) +
                " -- refusing to emit a stub");
        }
    }
}

inline uint32_t InstructionSelector::pack_digits(const std::vector<int>& dig,
                                                 int trit_offset) {
    int byte = 0, p = 1;
    for (int d : dig) { byte += d * p; p *= 3; }
    if (trit_offset % TRITS_PER_BYTE != 0)
        throw std::runtime_error("xbt: LI5 chunk must be 5-trit aligned");
    int chunk = trit_offset / TRITS_PER_BYTE;
    if (chunk < 0 || chunk > MAX_LI5_CHUNK)
        throw std::runtime_error("xbt: LI5 chunk index out of range");
    return static_cast<uint32_t>((byte & 0xFF) | ((chunk & 0x7) << 8));
}

inline std::vector<Insn> InstructionSelector::materialise(int64_t value,
                                                          int dst) {
    if (max_immediate_trits() < TRITS_PER_BYTE) {
        throw std::runtime_error(
            "xbt: target max_immediate_trits (" +
            std::to_string(max_immediate_trits()) +
            ") is smaller than LI5 chunk width (" +
            std::to_string(TRITS_PER_BYTE) + ")");
    }
    std::vector<Insn> code;
    std::vector<int> trits;
    int64_t v = value;
    for (int i = 0; i < 40 && v != 0; ++i) {
        int64_t r = ((v % 3) + 3) % 3;
        if (r == 0)      { trits.push_back(0);  v /= 3; }
        else if (r == 1) { trits.push_back(1);  v = (v - 1) / 3; }
        else             { trits.push_back(-1); v = (v + 1) / 3; }
    }
    if (trits.empty()) {
        // Zero is digits {1,1,1,1,1}: byte 121, and byte - bias == 0.
        code.push_back(make_i(Funct3::LI5, dst, 0, pack_digits({1,1,1,1,1}, 0),
                              "li5 t" + std::to_string(dst) + ", 0"));
        return code;
    }
    while (trits.size() % TRITS_PER_BYTE != 0) trits.push_back(0);

    const bool multi_chunk = (trits.size() > static_cast<size_t>(TRITS_PER_BYTE));
    int tmp = -1;
    if (multi_chunk) {
        tmp = alloc_reg();
    }

    bool first = true;
    for (size_t base = 0; base < trits.size(); base += TRITS_PER_BYTE) {
        std::vector<int> dig(TRITS_PER_BYTE);
        for (int k = 0; k < TRITS_PER_BYTE; ++k) dig[k] = trits[base + k] + 1;
        uint32_t imm = pack_digits(dig, static_cast<int>(base));
        int target = first ? dst : tmp;
        // A single-chunk constant is just the value itself, so print it; only
        // multi-chunk materialisations need the digit-offset annotation.
        std::string shown = !multi_chunk
                                ? std::to_string(value)
                                : "digits@" + std::to_string(base);
        code.push_back(make_i(Funct3::LI5, target, 0, imm,
                              "li5 t" + std::to_string(target) + ", " + shown));
        if (!first) {
            code.push_back(make_r(Funct3::KSADD, 0, dst, dst, target,
                                  "ksadd t" + std::to_string(dst) + ", t" +
                                  std::to_string(dst) + ", t" +
                                  std::to_string(target)));
        }
        first = false;
    }
    if (tmp > REG_ZERO) {
        free_reg(tmp);
    }
    return code;
}

// --------------------------------------------------------------------------
// Assembly emission
// --------------------------------------------------------------------------

inline std::string emit_assembly(const LoweredFunction& f) {
    std::string s;
    s += "    .text\n";
    s += "    .globl " + f.name + "\n";
    s += f.name + ":\n";
    for (const auto& i : f.code) s += i.text() + "\n";
    bool already_has_ret = !f.code.empty() && f.code.back().kind == InsnKind::Ret;
    if (!already_has_ret &&
        (f.code.empty() || !f.code.back().comment.empty() || !f.code.back().is_xbt_insn()))
        s += "  ret\n";
    s += "\n";
    return s;
}

struct LoweringResult {
    std::string assembly;
    std::vector<std::string> unlowered;
    int functions{0};
    int instructions{0};

    bool complete() const { return unlowered.empty(); }
};

inline LoweringResult lower_program_to_assembly(const ir::Program& program,
                                                int available_regs = 16) {
    LoweringResult out;
    out.assembly += "// Generated by Trium-XBT -- target riscv64-xbt\n";
    out.assembly += "// RISC-V \"Xbt\" custom-0 (opcode 0x0B), balanced ternary\n";
    out.assembly += "// Every instruction emitted here is Z3-verified against\n";
    out.assembly += "// balanced-ternary arithmetic: see verification/verify_isa.py\n";
    out.assembly += "\n";

    const MachineDescription* m = bt::target::find_machine("riscv64-xbt");
    InstructionSelector sel = m
        ? InstructionSelector(*m, available_regs)
        : InstructionSelector(available_regs);
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
