#ifndef TERNARY_TARGET_TARGET_DESCRIPTION_HPP
#define TERNARY_TARGET_TARGET_DESCRIPTION_HPP

/**
 * @file include/Ternary/Target/TargetDescription.hpp
 * @brief Declarative description of a hardware target.
 *
 * This is the seam that makes the compiler retargetable. A target is *data*:
 * what layouts it stores natively, how wide its ternary words are, which core
 * operations it implements in hardware, and what it falls back to software
 * for. Adding a new machine means writing one of these, not editing the
 * driver.
 *
 * It is deliberately small. Everything here either changes what code gets
 * generated or changes whether generation is legal; anything else belongs in
 * the backend, not the description.
 */

#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace bt::target {

/** Ternary storage layouts a target may support natively. */
enum class Layout {
    DualRail,   // 2 bits per trit   (P/N rails)      -- proven isomorphic
    Base243,    // 1.6 bits per trit (5 trits / byte) -- proven
    TCSR,       // sparse compressed ternary
    BTFP,       // balanced ternary floating point
};

inline const char* layout_name(Layout l) {
    switch (l) {
    case Layout::DualRail: return "dual-rail";
    case Layout::Base243:  return "base-243";
    case Layout::TCSR:     return "tcsr";
    case Layout::BTFP:     return "btfp";
    }
    return "unknown";
}

/**
 * @brief What a piece of hardware is, as data.
 */
struct TargetDescription {
    // ---- identity ---------------------------------------------------------
    std::string name;            // canonical target name, e.g. "riscv64-xbt"
    std::string triple;          // LLVM-style triple, if one applies
    std::vector<std::string> aliases;   // alternative spellings

    // ---- ternary datapath -------------------------------------------------
    unsigned native_trit_width{64};     // trits per native ternary word
    unsigned max_immediate_trits{5};    // trits materialisable by one insn
    bool     has_word_add{true};        // native full-width ternary add
    bool     has_lane_mul{true};        // native per-lane (SIMD) multiply
    bool     has_scalar_mul{false};     // true integer multiply in hardware
    bool     has_popcount_dot{true};    // native planar dot product

    // ---- storage ----------------------------------------------------------
    std::set<Layout> native_layouts;    // layouts held natively
    Layout           preferred_layout{Layout::DualRail};

    // ---- register file ----------------------------------------------------
    unsigned ternary_registers{32};
    unsigned scalar_registers{32};
    bool     zero_register_hardwired{true};   // tr0 == 0, as in RISC-V

    // ---- operation coverage ----------------------------------------------
    // Core-profile operation names (specs/core-profile-v0.1.json) the target
    // implements natively. Anything absent must be lowered to software or
    // refused -- the backend decides which.
    std::set<std::string> native_ops;

    bool has_op(const std::string& op) const {
        return native_ops.empty() || native_ops.count(op) > 0;
    }

    bool supports_layout(Layout l) const {
        return native_layouts.count(l) > 0;
    }

    /**
     * @brief Structured summary, for `--target=... --print-target` and for
     *        diagnostics when a construct cannot be lowered.
     */
    std::string describe() const {
        std::string s;
        s += "target " + name + "\n";
        if (!triple.empty()) s += "  triple: " + triple + "\n";
        s += "  ternary word: " + std::to_string(native_trit_width) + " trits\n";
        s += "  registers: " + std::to_string(ternary_registers) +
             " ternary, " + std::to_string(scalar_registers) + " scalar\n";
        s += "  native: ";
        s += has_word_add ? "add " : "";
        s += has_lane_mul ? "lane-mul " : "";
        s += has_scalar_mul ? "scalar-mul " : "";
        s += has_popcount_dot ? "dot " : "";
        s += "\n";
        s += "  layouts: ";
        for (auto l : native_layouts) {
            s += std::string(layout_name(l)) + " ";
        }
        s += "\n";
        return s;
    }
};

/**
 * @brief The RISC-V Xbt target, v2.
 *
 * Kept here so the description lives with the target it describes and so a new
 * target has a worked example to copy.
 */
inline TargetDescription xbt_target_v2() {
    TargetDescription t;
    t.name = "riscv64-xbt";
    t.triple = "riscv64-unknown-linux-gnu";
    t.aliases = {"riscv-xbt", "xbt"};
    t.native_trit_width = 64;
    t.max_immediate_trits = 5;
    t.has_word_add = true;
    t.has_lane_mul = true;
    t.has_scalar_mul = false;      // lane-wise only; see docs/XBT_ISA_SPEC.md
    t.has_popcount_dot = true;
    t.native_layouts = {Layout::DualRail, Layout::Base243};
    t.preferred_layout = Layout::DualRail;
    t.ternary_registers = 32;
    t.scalar_registers = 32;
    t.zero_register_hardwired = true;
    t.native_ops = {"add", "sub", "negate", "dot", "unpack243", "pack243",
                    "lane-mul"};
    return t;
}

} // namespace bt::target

#endif // TERNARY_TARGET_TARGET_DESCRIPTION_HPP
