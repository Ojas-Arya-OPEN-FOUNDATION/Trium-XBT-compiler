/**
 * @file test_unit_simulator.cpp
 * @brief The simulator plug-in interface (docs/SCALING_PLAN.md 6.3).
 *
 * The contract: one small interface -- load / exec / read_state / read_memory
 * -- so that the SAME conformance vectors drive the host C++ model today, an
 * instruction-set simulator next, and RTL simulation or real silicon later.
 *
 * The distinction that matters is Failed vs Skipped. A simulator that has not
 * implemented an instruction yet must NOT be reported as incorrect; it must be
 * reported as not having run. Conflating the two is how "nothing has run on
 * real hardware" quietly turns into "everything passes".
 */

#include "Ternary/Target/Simulator.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace bt::target;

namespace {

int g_checks = 0;
int g_fails = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

/** A reference "device": doubles its inputs. Stands in for the host model. */
std::vector<int64_t> doubler(const std::vector<uint8_t>&,
                             const std::vector<int64_t>& in) {
    std::vector<int64_t> out;
    for (int64_t v : in) out.push_back(v * 2);
    return out;
}

/** A device that only handles non-negative inputs; throws otherwise. */
std::vector<int64_t> picky(const std::vector<uint8_t>&,
                           const std::vector<int64_t>& in) {
    for (int64_t v : in)
        if (v < 0) throw std::runtime_error("negative input not implemented");
    std::vector<int64_t> out;
    for (int64_t v : in) out.push_back(v + 1);
    return out;
}

std::vector<ConformanceVector> build_vectors() {
    std::vector<ConformanceVector> vs;
    for (int i = 0; i < 8; ++i) {
        ConformanceVector v;
        v.name = "double." + std::to_string(i);
        v.program = {0x01, 0x02};
        v.inputs = {static_cast<int64_t>(i)};
        v.expected_state = {static_cast<int64_t>(i) * 2};
        vs.push_back(v);
    }
    return vs;
}

}  // namespace

int main() {
    std::printf("simulator plug-in interface\n");

    // ---- the host-model-shaped simulator passes its own vectors ----------
    {
        std::printf("  reference simulator\n");
        CallbackSimulator sim("host-model", doubler);
        ConformanceRunner runner(sim);
        auto results = runner.run(build_vectors());
        auto s = ConformanceRunner::summarise(results);
        check(s.passed == 8, "all 8 vectors pass: " +
                                 std::to_string(s.passed));
        check(s.failed == 0 && s.skipped == 0, "nothing failed or skipped");
        check(s.clean(), "summary is clean");
        check(s.total() == 8, "total counts every vector");
    }

    // ---- a wrong answer is Failed, not Skipped ---------------------------
    {
        std::printf("  wrong answers are Failed\n");
        CallbackSimulator sim("host-model", doubler);
        ConformanceRunner runner(sim);
        std::vector<ConformanceVector> vs = build_vectors();
        vs[0].expected_state = {999};          // deliberately wrong
        auto results = runner.run(vs);
        auto s = ConformanceRunner::summarise(results);
        check(s.failed == 1, "the wrong vector fails: " +
                                 std::to_string(s.failed));
        check(results[0].detail.find("state mismatch") != std::string::npos,
              "the failure explains the mismatch: " + results[0].detail);
        check(!s.clean(), "summary is not clean");
    }

    // ---- an unimplemented instruction is Skipped, NOT Failed -------------
    {
        std::printf("  unimplemented is Skipped, not Failed\n");
        CallbackSimulator sim("picky-device", picky);
        ConformanceRunner runner(sim);
        std::vector<ConformanceVector> vs;
        for (int64_t v : {1LL, -1LL, 2LL}) {
            ConformanceVector c;
            c.name = "picky." + std::to_string(v);
            c.inputs = {v};
            c.expected_state = {v + 1};
            vs.push_back(c);
        }
        auto results = runner.run(vs);
        auto s = ConformanceRunner::summarise(results);
        check(s.passed == 2, "the two it implements pass");
        check(s.skipped == 1, "the unimplemented one is SKIPPED: " +
                                  std::to_string(s.skipped));
        check(s.failed == 0,
              "it is NOT reported as a correctness failure");
        check(results[1].detail.find("could not run") != std::string::npos,
              "the skip says why: " + results[1].detail);
        // Skipped still blocks "clean" -- otherwise "nothing ran" would read
        // as "everything passes".
        check(!s.clean(), "skipped vectors prevent a clean summary");
    }

    // ---- external simulators come from the YAML --------------------------
    {
        std::printf("  external simulators from targets/*.yaml\n");
        const MachineDescription* torus = find_machine("ternary-torus");
        const MachineDescription* xbt = find_machine("riscv64-xbt");
        check(torus && xbt, "both targets exist");
        if (torus && xbt) {
            auto sim_torus = ExternalSimulator::from_machine(*torus);
            auto sim_xbt = ExternalSimulator::from_machine(*xbt);
            check(sim_torus != nullptr,
                  "a target with a simulator block yields a simulator");
            check(sim_xbt == nullptr,
                  "real hardware yields none -- there is nothing to simulate");
            if (sim_torus) {
                check(sim_torus->command_line() == "bt-torus-sim --cycles --trace",
                      "command line comes from the YAML: " +
                          sim_torus->command_line());
                check(sim_torus->name() == "ternary-torus-sim",
                      "simulator is named after its target");

                // It is a Simulator, so the same driver runs it.
                ConformanceRunner runner(*sim_torus);
                ConformanceVector v;
                v.name = "torus.smoke";
                v.inputs = {1, 2, 3};
                v.expected_state = {1, 2, 3};
                auto r = runner.run_one(v);
                check(r.status == VectorResult::Status::Passed,
                      "the external simulator is driven by the same driver");
            }
        }
    }

    // ---- interface guards -----------------------------------------------
    {
        std::printf("  interface guards\n");
        bool threw = false;
        try { ExternalSimulator bad("", {}); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "empty command is refused");

        threw = false;
        try { CallbackSimulator bad("x", nullptr); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "null callback is refused");

        ExternalSimulator sim("some-sim", {});
        threw = false;
        try { sim.exec({1}); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "exec before load is refused");

        threw = false;
        try { sim.read_memory(0, 4); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "read_memory without a transport says so");
    }

    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
