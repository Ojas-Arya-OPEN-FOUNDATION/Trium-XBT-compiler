// ==============================================================
//  GeneratedTargets.hpp -- GENERATED FILE, DO NOT EDIT.
//
//  Source of truth:  targets/*.yaml
//  Contract:         schema/target.schema.json
//  Regenerate with:  tools/gen_targets.py
//
//  Generated 2026-11-07T07:44:16Z  schema sha256:1d71032768cd
//  This file is committed so the build never requires Python. A CI
//  job regenerates and fails if the result differs.
// ==============================================================

#ifndef TERNARY_TARGET_GENERATED_TARGETS_HPP
#define TERNARY_TARGET_GENERATED_TARGETS_HPP

#include "TargetDescription.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bt::target {

/** Broad machine class. Determines which backend picks the target up. */
enum class Family { Cpu, Gpu, Accelerator, Simulator };

inline const char* family_name(Family f) {
    switch (f) {
    case Family::Cpu: return "cpu";
    case Family::Gpu: return "gpu";
    case Family::Accelerator: return "accelerator";
    case Family::Simulator: return "simulator";
    }
    return "unknown";
}

/**
 * @brief A hardware target, exactly as described in its YAML file.
 *
 * Superset of TargetDescription: adds the memory hierarchy, the cost
 * model and the simulator hook that Track B and P3 need. Convert to the
 * legacy view with to_target_description().
 */
struct MachineDescription {
    // identity
    const char* name;
    const char* triple;
    const char* const* aliases;  size_t alias_count;
    Family family;

    // ternary datapath
    unsigned native_trit_width;
    unsigned max_immediate_trits;
    bool has_word_add, has_lane_mul, has_scalar_mul, has_popcount_dot;

    // storage
    const Layout* native_layouts;  size_t layout_count;
    Layout preferred_layout;

    // registers
    unsigned ternary_registers, scalar_registers;
    bool zero_register_hardwired;

    // operation coverage (empty == assume everything)
    const char* const* native_ops;  size_t op_count;

    // memory hierarchy (Track A implicit vs Track B explicit)
    uint64_t cache_line_bytes, scratchpad_bytes, global_bytes;
    bool explicit_movement;

    // cost model for the instruction selector
    const char* const* cost_keys;  const unsigned* cost_values;
    size_t cost_count;
    unsigned default_cost;

    // Track B simulator hook; nullptr == real hardware
    const char* simulator_command;
    const char* const* simulator_args;  size_t simulator_arg_count;

    bool has_op(const std::string& op) const {
        if (op_count == 0) return true;   // empty == assume everything
        for (size_t i = 0; i < op_count; ++i)
            if (op == native_ops[i]) return true;
        return false;
    }

    unsigned cost_of(const std::string& op) const {
        for (size_t i = 0; i < cost_count; ++i)
            if (op == cost_keys[i]) return cost_values[i];
        return default_cost;
    }

    bool supports_layout(Layout l) const {
        for (size_t i = 0; i < layout_count; ++i)
            if (native_layouts[i] == l) return true;
        return false;
    }

    bool has_simulator() const { return simulator_command != nullptr; }
};

static const char* const kAliases_generic_cpu[] = {"x86_64", "aarch64", "host", "cpu"};
static const Layout kLayouts_generic_cpu[] = {Layout::DualRail, Layout::Base243};
static const char* const kCostKeys_generic_cpu[] = {"add", "dot", "lane-mul", "negate", "sub"};
static const unsigned kCostVals_generic_cpu[] = {6, 3, 4, 1, 6};

static const char* const kAliases_portable_gpu[] = {"cuda", "vulkan", "gpu"};
static const Layout kLayouts_portable_gpu[] = {Layout::DualRail};
static const char* const kCostKeys_portable_gpu[] = {"add", "dot", "lane-mul", "negate", "sub"};
static const unsigned kCostVals_portable_gpu[] = {8, 4, 4, 1, 8};

static const char* const kAliases_riscv64_xbt[] = {"riscv-xbt", "xbt"};
static const Layout kLayouts_riscv64_xbt[] = {Layout::DualRail, Layout::Base243};
static const char* const kOps_riscv64_xbt[] = {"add", "sub", "negate", "dot", "unpack243", "pack243", "lane-mul"};
static const char* const kCostKeys_riscv64_xbt[] = {"add", "dot", "lane-mul", "negate", "pack243", "sub", "unpack243"};
static const unsigned kCostVals_riscv64_xbt[] = {1, 3, 2, 1, 4, 1, 4};

static const char* const kAliases_ternary_torus[] = {"torus"};
static const Layout kLayouts_ternary_torus[] = {Layout::Base243, Layout::TCSR};
static const char* const kOps_ternary_torus[] = {"add", "sub", "negate", "dot", "mul", "unpack243", "pack243", "lane-mul"};
static const char* const kCostKeys_ternary_torus[] = {"add", "dot", "lane-mul", "mul", "negate", "pack243", "sub", "unpack243"};
static const unsigned kCostVals_ternary_torus[] = {1, 2, 1, 6, 1, 1, 1, 1};
static const char* const kSimCmd_ternary_torus[] = {"bt-torus-sim"};
static const char* const kSimArgs_ternary_torus[] = {"--cycles", "--trace"};

/** Every target compiled into the compiler, sorted by name. */
inline const MachineDescription* machines(size_t* count_out = nullptr) {
    static const MachineDescription table[] = {
        {
            "generic-cpu", "x86_64-unknown-linux-gnu",
            kAliases_generic_cpu, 4,
            Family::Cpu,
            64, 5,
            true, true, true, true,
            kLayouts_generic_cpu, 2, Layout::DualRail,
            0, 16, false,
            nullptr, 0,
            64ULL, 0ULL, 0ULL, false,
            kCostKeys_generic_cpu, kCostVals_generic_cpu, 5, 12,
            nullptr, nullptr, 0,
        },
        {
            "portable-gpu", "nvptx64-nvidia-cuda",
            kAliases_portable_gpu, 3,
            Family::Gpu,
            64, 5,
            true, true, true, true,
            kLayouts_portable_gpu, 1, Layout::DualRail,
            0, 255, false,
            nullptr, 0,
            128ULL, 49152ULL, 0ULL, true,
            kCostKeys_portable_gpu, kCostVals_portable_gpu, 5, 16,
            nullptr, nullptr, 0,
        },
        {
            "riscv64-xbt", "riscv64-unknown-linux-gnu",
            kAliases_riscv64_xbt, 2,
            Family::Accelerator,
            64, 5,
            true, true, false, true,
            kLayouts_riscv64_xbt, 2, Layout::DualRail,
            32, 32, true,
            kOps_riscv64_xbt, 7,
            64ULL, 0ULL, 0ULL, false,
            kCostKeys_riscv64_xbt, kCostVals_riscv64_xbt, 7, 8,
            nullptr, nullptr, 0,
        },
        {
            "ternary-torus", "torus-unknown-none",
            kAliases_ternary_torus, 1,
            Family::Accelerator,
            243, 9,
            true, true, true, true,
            kLayouts_ternary_torus, 2, Layout::Base243,
            64, 16, true,
            kOps_ternary_torus, 8,
            243ULL, 1048576ULL, 4294967296ULL, true,
            kCostKeys_ternary_torus, kCostVals_ternary_torus, 8, 4,
            kSimCmd_ternary_torus[0], kSimArgs_ternary_torus, 2,
        },
    };
    if (count_out) *count_out = sizeof(table) / sizeof(table[0]);
    return table;
}

/** Number of compiled-in targets. */
inline size_t machine_count() { size_t n = 0; machines(&n); return n; }

/** Look up a target by canonical name or alias; nullptr if unknown. */
inline const MachineDescription* find_machine(const std::string& name) {
    size_t n = 0; const MachineDescription* t = machines(&n);
    for (size_t i = 0; i < n; ++i) {
        if (name == t[i].name) return &t[i];
        for (size_t j = 0; j < t[i].alias_count; ++j)
            if (name == t[i].aliases[j]) return &t[i];
    }
    return nullptr;
}

/** All names a target may be selected by. */
inline std::vector<std::string> machine_names() {
    std::vector<std::string> out;
    size_t n = 0; const MachineDescription* t = machines(&n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(t[i].name);
        for (size_t j = 0; j < t[i].alias_count; ++j)
            out.push_back(t[i].aliases[j]);
    }
    return out;
}

/**
 * Project onto the legacy TargetDescription. Fields the older struct
 * does not model (memory, costs, simulator) are dropped, so this is
 * lossy -- prefer MachineDescription in new code.
 */
inline TargetDescription to_target_description(const MachineDescription& m) {
    TargetDescription t;
    t.name = m.name;
    t.triple = m.triple ? m.triple : "";
    for (size_t i = 0; i < m.alias_count; ++i)
        t.aliases.push_back(m.aliases[i]);
    t.native_trit_width = m.native_trit_width;
    t.max_immediate_trits = m.max_immediate_trits;
    t.has_word_add = m.has_word_add;
    t.has_lane_mul = m.has_lane_mul;
    t.has_scalar_mul = m.has_scalar_mul;
    t.has_popcount_dot = m.has_popcount_dot;
    for (size_t i = 0; i < m.layout_count; ++i)
        t.native_layouts.insert(m.native_layouts[i]);
    t.preferred_layout = m.preferred_layout;
    t.ternary_registers = m.ternary_registers;
    t.scalar_registers = m.scalar_registers;
    t.zero_register_hardwired = m.zero_register_hardwired;
    for (size_t i = 0; i < m.op_count; ++i)
        t.native_ops.insert(m.native_ops[i]);
    return t;
}

} // namespace bt::target

#endif // TERNARY_TARGET_GENERATED_TARGETS_HPP
