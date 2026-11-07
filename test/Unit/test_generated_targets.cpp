/**
 * @file test_generated_targets.cpp
 * @brief Conformance tests for the YAML -> JSON Schema -> C++ target pipeline.
 *
 * The point of this test is the VENDOR CONTRACT (P3): adding a machine means
 * adding a targets/*.yaml file and nothing else. So we assert the things a
 * vendor could get wrong that the schema cannot catch on its own, plus that
 * the generated table is faithful to the legacy hand-written description.
 */

#include "Ternary/Target/GeneratedTargets.hpp"
#include "Ternary/Target/TargetDescription.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace bt::target;

static int g_checks = 0;
static int g_fails = 0;

static void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  [FAIL] %s\n", what);
    }
}

// ---------------------------------------------------------------------------

static void test_table_is_populated() {
    size_t n = machine_count();
    check(n >= 4, "at least the four shipped targets are compiled in");
    check(machines() != nullptr, "machine table is non-null");

    // Names sorted and unique -- the generator promises this.
    size_t count = 0;
    const MachineDescription* t = machines(&count);
    std::set<std::string> seen;
    std::string prev;
    bool sorted = true;
    for (size_t i = 0; i < count; ++i) {
        check(seen.insert(t[i].name).second, "target name is unique");
        if (i && t[i].name < prev) sorted = false;
        prev = t[i].name;
    }
    check(sorted, "target table is sorted by name");
}

static void test_lookup() {
    // Every alias resolves to its canonical target.
    size_t n = 0;
    const MachineDescription* t = machines(&n);
    for (size_t i = 0; i < n; ++i) {
        const MachineDescription* by_name = find_machine(t[i].name);
        check(by_name != nullptr && by_name->name == t[i].name,
              "canonical name resolves to itself");
        for (size_t j = 0; j < t[i].alias_count; ++j) {
            const MachineDescription* by_alias = find_machine(t[i].aliases[j]);
            check(by_alias != nullptr && by_alias->name == t[i].name,
                  "alias resolves to its canonical target");
        }
    }
    check(find_machine("definitely-not-a-target") == nullptr,
          "unknown target returns nullptr");
    check(find_machine("") == nullptr, "empty name returns nullptr");
}

static void test_invariants() {
    size_t n = 0;
    const MachineDescription* t = machines(&n);
    for (size_t i = 0; i < n; ++i) {
        // The generator rejects a preferred layout that is not native, but the
        // compiled table must hold too -- a stale hand-edit must not slip by.
        check(t[i].supports_layout(t[i].preferred_layout),
              "preferred layout is one of the native layouts");
        check(t[i].native_trit_width > 0, "native trit width is positive");
        check(t[i].layout_count > 0, "every target declares a storage layout");

        // Cost model: listed ops return their listed cost, unlisted return the
        // default, and the default itself is always reachable.
        for (size_t j = 0; j < t[i].cost_count; ++j)
            check(t[i].cost_of(t[i].cost_keys[j]) == t[i].cost_values[j],
                  "listed op returns its listed cost");
        check(t[i].cost_of("no-such-op") == t[i].default_cost,
              "unlisted op falls back to the default cost");
    }
}

/**
 * The generated description of riscv64-xbt must agree with the hand-written
 * xbt_target_v2() it replaces. This is the regression guard for the migration:
 * if someone edits one and not the other, this fails.
 */
static void test_matches_legacy_xbt() {
    const MachineDescription* m = find_machine("riscv64-xbt");
    check(m != nullptr, "riscv64-xbt exists");
    if (!m) return;

    TargetDescription g = to_target_description(*m);
    TargetDescription h = xbt_target_v2();

    check(g.name == h.name, "name matches legacy");
    check(g.triple == h.triple, "triple matches legacy");
    check(g.aliases == h.aliases, "aliases match legacy");
    check(g.native_trit_width == h.native_trit_width, "trit width matches");
    check(g.max_immediate_trits == h.max_immediate_trits, "immediates match");
    check(g.has_word_add == h.has_word_add, "has_word_add matches");
    check(g.has_lane_mul == h.has_lane_mul, "has_lane_mul matches");
    check(g.has_scalar_mul == h.has_scalar_mul, "has_scalar_mul matches");
    check(g.has_popcount_dot == h.has_popcount_dot, "has_popcount_dot matches");
    check(g.native_layouts == h.native_layouts, "layouts match");
    check(g.preferred_layout == h.preferred_layout, "preferred layout matches");
    check(g.ternary_registers == h.ternary_registers, "ternary regs match");
    check(g.scalar_registers == h.scalar_registers, "scalar regs match");
    check(g.zero_register_hardwired == h.zero_register_hardwired,
          "zero register matches");
    check(g.native_ops == h.native_ops, "native ops match");
}

static void test_capability_differences() {
    // The whole point of describing targets as data: capability differences
    // are visible to the compiler as values, not as #ifdefs.
    const MachineDescription* xbt = find_machine("riscv64-xbt");
    const MachineDescription* cpu = find_machine("generic-cpu");
    const MachineDescription* gpu = find_machine("portable-gpu");
    const MachineDescription* torus = find_machine("ternary-torus");
    check(xbt && cpu && gpu && torus, "all four named targets exist");
    if (!(xbt && cpu && gpu && torus)) return;

    // Scalar multiply: refused on Xbt, available on the binary targets.
    check(!xbt->has_scalar_mul, "Xbt has no scalar multiply");
    check(cpu->has_scalar_mul, "generic CPU has scalar multiply");
    check(torus->has_scalar_mul, "ternary torus has scalar multiply");
    check(!xbt->has_op("mul"), "Xbt's op set omits mul");
    check(cpu->has_op("mul"), "empty op set means 'assume everything'");

    // Track A vs Track B: who moves the data.
    check(!cpu->explicit_movement, "CPU cache movement is implicit (Track A)");
    check(gpu->explicit_movement, "GPU shared memory is explicit");
    check(torus->explicit_movement, "accelerator movement is explicit (Track B)");
    check(torus->scratchpad_bytes > 0, "accelerator has a scratchpad");
    check(cpu->scratchpad_bytes == 0, "CPU has no scratchpad");

    // Layout preference follows the density argument in
    // docs/TENSOR_MEMORY_MATHEMATICS.md: base-243 fits 25% more elements.
    check(torus->preferred_layout == Layout::Base243,
          "wide-word target prefers base-243");
    check(xbt->preferred_layout == Layout::DualRail, "Xbt prefers dual-rail");

    // Track B: the simulator hook exists only where there is no silicon.
    check(torus->has_simulator(), "torus is simulated");
    check(!xbt->has_simulator() && !cpu->has_simulator(),
          "real hardware declares no simulator");
    check(std::string(torus->simulator_command) == "bt-torus-sim",
          "simulator command comes through");
    check(torus->simulator_arg_count == 2, "simulator args come through");

    // Cost model differentiates the same op across targets.
    check(xbt->cost_of("add") < cpu->cost_of("add"),
          "native ternary add is cheaper than emulated add");
    check(xbt->cost_of("negate") == 1, "dual-rail negate is free (wire swap)");
}

int main() {
    std::printf("generated target descriptions\n");
    test_table_is_populated();
    test_lookup();
    test_invariants();
    test_matches_legacy_xbt();
    test_capability_differences();
    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
