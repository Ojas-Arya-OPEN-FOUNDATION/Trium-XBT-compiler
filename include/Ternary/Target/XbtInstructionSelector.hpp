#ifndef TERNARY_TARGET_XBT_INSTRUCTION_SELECTOR_HPP
#define TERNARY_TARGET_XBT_INSTRUCTION_SELECTOR_HPP

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
 *   v2.1  (proposed)         SIGN CMOVZ CMOVP CMOVN
 *
 * Every instruction emitted here is verified in `verification/verify_isa.py`;
 * see `docs/XBT_ISA_SPEC.md`.
 */

#include "../IR/Core.hpp"
#include "../IR/Program.hpp"

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
};

constexpr int BASE243_BIAS = 121;
constexpr int TRITS_PER_BYTE = 5;
constexpr int MAX_LI5_CHUNK = 7;   // 8 chunks x 5 trits = 40 trits

// --------------------------------------------------------------------------
// Emitted instruction
// --------------------------------------------------------------------------

struct Insn {
    bool is_itype{false};
    uint8_t funct3{0};
    uint8_t funct7{0};
    uint8_t rd{0}, rs1{0}, rs2{0};
    uint32_t imm{0};
    std::string comment;

    /**
     * @brief Binary encoding, bit-compatible with `hw/xbt_alu_model.hpp`.
     *
     * This is what lets a lowered function be handed straight to
     * `RiscvXbtCoprocessor::execute_program`, so the selector's output can be
     * conformance-tested against the IR interpreter instead of only eyeballed.
     */
    uint32_t encode() const {
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
    std::vector<std::string> notes;
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
        if (available_ < 4 || available_ >= REG_COUNT)
            throw std::invalid_argument("xbt: register count out of range");
        register_default_rules();
    }

    LoweredFunction lower_function(const ir::Function& f) {
        LoweredFunction out;
        out.name = f.name;
        reset();

        for (const auto& p : f.parameters) {
            int r = alloc_reg();
            symbol_reg_[p.symbol] = r;
            out.code.push_back(note_insn("param sym#" + std::to_string(p.symbol) +
                                         " -> t" + std::to_string(r)));
            used_registers_.insert(r);
        }

        lower_statements(f.body, out);
        out.ternary_regs_used = next_reg_;
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
    int next_reg_{1};
    std::map<uint32_t, int> symbol_reg_;
    std::set<int> free_regs_;
    std::set<int> used_registers_;
    
    std::map<ir::Op, std::vector<SelectionRule>> rules_;

    void reset() {
        next_reg_ = 1;
        symbol_reg_.clear();
        free_regs_.clear();
        used_registers_.clear();
    }

    // -- register allocation ------------------------------------------------
    int alloc_reg() {
        if (!free_regs_.empty()) {
            int r = *free_regs_.begin();
            free_regs_.erase(free_regs_.begin());
            used_registers_.insert(r);
            return r;
        }
        if (next_reg_ >= available_)
            throw std::runtime_error(
                "xbt: out of ternary registers (limit " + std::to_string(available_) +
                "); spilling is not implemented");
        int r = next_reg_++;
        used_registers_.insert(r);
        return r;
    }

    void free_reg(int r) {
        if (r > REG_ZERO) free_regs_.insert(r);
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

    // -- lowering -----------------------------------------------------------
    int lower_expr(const ir::Expression& e, LoweredFunction& out);

    void lower_statements(const std::vector<ir::Statement>& body,
                          LoweredFunction& out);

    std::vector<Insn> materialise(int64_t value, int dst) const;

    static uint32_t pack_digits(const std::vector<int>& dig, int trit_offset);

    static bool is_single_trit(const ir::Expression& e) {
        return e && e->op == ir::Op::Integer && e->integer >= -1 && e->integer <= 1;
    }

    /** True when an expression is side-effect free and cannot fail. */
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
    auto it = sel->symbol_reg_.find(symbol);
    if (it == sel->symbol_reg_.end())
        throw std::runtime_error("xbt: unknown symbol #" + std::to_string(symbol));
    return it->second;
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

    // ---------------- negation ----------------
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

    // ---------------- addition ----------------
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

    // ---------------- subtraction (a + (-b)) ----------------
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
            [](const ir::Expression& e) { return true; },
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
    // so the rule only applies when every arm is pure -- no side effects and
    // no possible failure. Impure arms are refused rather than miscompiled.
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
}

// --------------------------------------------------------------------------
// Selector implementation
// --------------------------------------------------------------------------

inline bool InstructionSelector::is_pure(const ir::Expression& e) {
    if (!e) return false;
    try {
        return ir::verify(*e, {}, 0, nullptr, nullptr) == ir::Effect::Pure;
    } catch (...) {
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
        case ir::StatementKind::Let:
        case ir::StatementKind::Assign: {
            int r = lower_expr(s.value, out);
            symbol_reg_[s.symbol] = r;
            break;
        }
        case ir::StatementKind::Return: {
            if (s.value) {
                int r = lower_expr(s.value, out);
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
                                                          int dst) const {
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

    bool first = true;
    int tmp = -1;
    for (size_t base = 0; base < trits.size(); base += TRITS_PER_BYTE) {
        std::vector<int> dig(TRITS_PER_BYTE);
        for (int k = 0; k < TRITS_PER_BYTE; ++k) dig[k] = trits[base + k] + 1;
        uint32_t imm = pack_digits(dig, static_cast<int>(base));
        int target = first ? dst : tmp;
        // A single-chunk constant is just the value itself, so print it; only
        // multi-chunk materialisations need the digit-offset annotation.
        std::string shown = (trits.size() == static_cast<size_t>(TRITS_PER_BYTE))
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
    if (f.code.empty() || !f.code.back().comment.empty()) s += "  ret\n";
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
