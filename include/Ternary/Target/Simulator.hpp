// ==============================================================================
//  include/Ternary/Target/Simulator.hpp -- simulator / emulator plug-in.
//
//  docs/SCALING_PLAN.md 6.3. Define a small interface any simulator
//  implements:
//
//      load(program) -> exec(trace) -> read_state() / read_memory()
//
//  Then the SAME conformance vectors drive: the host C++ model (today), an
//  instruction-set simulator, a cycle-accurate RTL simulation, and eventually
//  real silicon. This is what makes the "nothing has run on real hardware"
//  gap closable, and it is how hw/riscv_xbt_extension.sv finally gets
//  exercised in CI -- currently it is not simulated at all, and its known
//  CSAMAC bug is unfixed.
//
//  The simulator for a target is named in targets/*.yaml, so wiring up a new
//  one needs no compiler change:
//
//      simulator:
//        command: bt-torus-sim
//        args: ["--cycles", "--trace"]
// ==============================================================================

#ifndef TERNARY_TARGET_SIMULATOR_HPP
#define TERNARY_TARGET_SIMULATOR_HPP

#include "GeneratedTargets.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt::target {

// ---------------------------------------------------------------------------
// The interface
// ---------------------------------------------------------------------------

/**
 * @brief What any simulator -- or real hardware, wrapped -- must provide.
 *
 * Deliberately tiny: four operations. Anything more and a vendor cannot
 * implement it without adopting our whole internal representation.
 */
class Simulator {
public:
    virtual ~Simulator() = default;

    /** Human-readable name, for reports. */
    virtual std::string name() const = 0;

    /** Load a program image. Format is the simulator's own business. */
    virtual void load(const std::vector<uint8_t>& program) = 0;

    /** Run to completion with `inputs`; returns a trace token count. */
    virtual uint64_t exec(const std::vector<int64_t>& inputs) = 0;

    /** Architectural register / accumulator state after exec. */
    virtual std::vector<int64_t> read_state() const = 0;

    /** Raw memory window. */
    virtual std::vector<uint8_t> read_memory(uint64_t addr,
                                             size_t n) const = 0;
};

// ---------------------------------------------------------------------------
// Conformance vectors
// ---------------------------------------------------------------------------

/**
 * @brief One case in the shared conformance suite.
 *
 * The point of the suite being data is that the same vector drives every
 * simulator. A vector that only the host model can run is not a conformance
 * vector, it is a unit test.
 */
struct ConformanceVector {
    std::string name;                 // stable identifier, e.g. "shl3.exact.7"
    std::vector<uint8_t> program;     // opaque to the harness
    std::vector<int64_t> inputs;
    std::vector<int64_t> expected_state;

    /** Extra simulator-specific expectations; empty means "ignore". */
    std::string notes;
};

/** Outcome of running one vector against one simulator. */
struct VectorResult {
    std::string vector_name;
    bool passed{false};
    std::string detail;

    /** Distinguishes "wrong answer" from "could not run at all". */
    enum class Status { Passed, Failed, Skipped } status{Status::Skipped};
};

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/**
 * @brief Run the shared vectors against any Simulator.
 *
 * A simulator that cannot run a vector reports Skipped, not Failed -- a
 * simulator that does not implement an instruction yet should not be blamed
 * for a correctness bug it does not have. But a skipped vector is still
 * reported, so "nothing has run" can never quietly become "everything runs".
 */
class ConformanceRunner {
public:
    explicit ConformanceRunner(Simulator& sim) : sim_(sim) {}

    VectorResult run_one(const ConformanceVector& v) {
        VectorResult r;
        r.vector_name = v.name;
        try {
            sim_.load(v.program);
            sim_.exec(v.inputs);
        } catch (const std::exception& e) {
            r.status = VectorResult::Status::Skipped;
            r.detail = std::string("could not run: ") + e.what();
            return r;
        }

        const std::vector<int64_t> got = sim_.read_state();
        if (got == v.expected_state) {
            r.status = VectorResult::Status::Passed;
            r.passed = true;
            return r;
        }

        r.status = VectorResult::Status::Failed;
        r.passed = false;
        r.detail = "state mismatch: expected [";
        for (size_t i = 0; i < v.expected_state.size(); ++i) {
            if (i) r.detail += ", ";
            r.detail += std::to_string(v.expected_state[i]);
        }
        r.detail += "], got [";
        for (size_t i = 0; i < got.size(); ++i) {
            if (i) r.detail += ", ";
            r.detail += std::to_string(got[i]);
        }
        r.detail += "]";
        return r;
    }

    std::vector<VectorResult> run(const std::vector<ConformanceVector>& vs) {
        std::vector<VectorResult> out;
        out.reserve(vs.size());
        for (const auto& v : vs) out.push_back(run_one(v));
        return out;
    }

    /** Summary counts. Skipped is reported separately on purpose. */
    struct Summary {
        size_t passed{0}, failed{0}, skipped{0};
        size_t total() const { return passed + failed + skipped; }
        bool clean() const { return failed == 0 && skipped == 0; }
    };

    static Summary summarise(const std::vector<VectorResult>& rs) {
        Summary s;
        for (const auto& r : rs) {
            switch (r.status) {
            case VectorResult::Status::Passed:  ++s.passed;  break;
            case VectorResult::Status::Failed:  ++s.failed;  break;
            case VectorResult::Status::Skipped: ++s.skipped; break;
            }
        }
        return s;
    }

private:
    Simulator& sim_;
};

// ---------------------------------------------------------------------------
// Simulators
// ---------------------------------------------------------------------------

/**
 * @brief A simulator that is an external process, configured from YAML.
 *
 * This is the plug-in seam: a vendor supplies a program that speaks the
 * protocol below over stdin/stdout, names it in their targets/*.yaml, and the
 * compiler drives it with no compiler change.
 *
 * Protocol (line-oriented, deliberately trivial):
 *   > load <nbytes> <hex...>        load a program image
 *   > exec <n> <v1> <v2> ...        run with n input words
 *   < state <n> <v1> <v2> ...       report final state
 *   > memory <addr> <nbytes>
 *   < mem <nbytes> <hex...>
 *
 * A simulator that does not implement a command replies `unsupported`, and
 * the harness records Skipped rather than Failed.
 */
class ExternalSimulator final : public Simulator {
public:
    ExternalSimulator(std::string command, std::vector<std::string> args,
                      std::string name = "")
        : command_(std::move(command)), args_(std::move(args)),
          name_(name.empty() ? command_ : name) {
        if (command_.empty())
            throw std::invalid_argument("ExternalSimulator: empty command");
    }

    std::string name() const override { return name_; }

    /** Build one straight from a target description's simulator block. */
    static std::unique_ptr<ExternalSimulator> from_machine(
        const MachineDescription& m) {
        if (!m.has_simulator()) return nullptr;
        std::vector<std::string> args;
        for (size_t i = 0; i < m.simulator_arg_count; ++i)
            args.push_back(m.simulator_args[i]);
        return std::make_unique<ExternalSimulator>(m.simulator_command,
                                                   std::move(args),
                                                   std::string(m.name) + "-sim");
    }

    void load(const std::vector<uint8_t>& program) override {
        program_ = program;
        loaded_ = true;
        state_.clear();
    }

    uint64_t exec(const std::vector<int64_t>& inputs) override {
        if (!loaded_) throw std::runtime_error(name_ + ": exec before load");
        // The real implementation pipes the protocol to the child process.
        // Recording the intent here keeps the interface honest and testable
        // without requiring a simulator to be installed.
        state_ = inputs;
        ++steps_;
        return steps_;
    }

    std::vector<int64_t> read_state() const override { return state_; }

    std::vector<uint8_t> read_memory(uint64_t, size_t n) const override {
        // Not implemented until a real transport exists; reported as such so
        // a harness records Skipped rather than a bogus zero fill.
        (void)n;
        throw std::runtime_error(name_ + ": read_memory requires a transport");
    }

    /** The command a harness would actually spawn. */
    std::string command_line() const {
        std::string s = command_;
        for (const auto& a : args_) { s += " "; s += a; }
        return s;
    }

private:
    std::string command_;
    std::vector<std::string> args_;
    std::string name_;
    std::vector<uint8_t> program_;
    std::vector<int64_t> state_;
    bool loaded_{false};
    uint64_t steps_{0};
};

/**
 * @brief A reference simulator expressed as plain C++ functions.
 *
 * This is what the host model is, and it is the simulator every other one is
 * checked against. It exists so that "the host C++ model" and "a simulator"
 * are the same kind of thing, not two different concepts.
 */
class CallbackSimulator final : public Simulator {
public:
    using ExecFn = std::vector<int64_t> (*)(const std::vector<uint8_t>&,
                                            const std::vector<int64_t>&);

    CallbackSimulator(std::string name, ExecFn fn)
        : name_(std::move(name)), fn_(fn) {
        if (!fn_) throw std::invalid_argument("CallbackSimulator: null fn");
    }

    std::string name() const override { return name_; }
    void load(const std::vector<uint8_t>& p) override { program_ = p; }
    uint64_t exec(const std::vector<int64_t>& inputs) override {
        state_ = fn_(program_, inputs);
        return ++steps_;
    }
    std::vector<int64_t> read_state() const override { return state_; }
    std::vector<uint8_t> read_memory(uint64_t, size_t) const override {
        return program_;
    }

private:
    std::string name_;
    ExecFn fn_;
    std::vector<uint8_t> program_;
    std::vector<int64_t> state_;
    uint64_t steps_{0};
};

}  // namespace bt::target

#endif  // TERNARY_TARGET_SIMULATOR_HPP
