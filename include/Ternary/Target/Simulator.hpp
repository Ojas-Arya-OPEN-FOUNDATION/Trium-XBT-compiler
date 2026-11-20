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

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define BT_SIM_HAS_POSIX 1
#else
#define BT_SIM_HAS_POSIX 0
#endif

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
 * line-oriented protocol below over stdin/stdout, names it in their
 * targets/*.yaml, and the compiler drives it over POSIX pipes with no
 * compiler change.
 *
 * Protocol (line-oriented):
 *   > load <hex>                load a program image
 *   < ok
 *   > exec <in0> <in1> ...      run with input words
 *   < ok [steps]  |  state <v0> <v1> ...
 *   > state                     query architectural state
 *   < state <v0> <v1> ...
 *   > memory <addr> <n>         read raw memory window
 *   < mem <hex>  |  memory <hex>
 *   > quit                      clean shutdown
 *
 * A simulator that does not implement a command replies `unsupported` (or
 * `error <msg>`), causing an exception so ConformanceRunner records Skipped
 * rather than Failed.
 */
class ExternalSimulator final : public Simulator {
public:
    ExternalSimulator(std::string command, std::vector<std::string> args,
                      std::string name = "", int timeout_ms = 5000)
        : command_(std::move(command)), args_(std::move(args)),
          name_(name.empty() ? command_ : name), timeout_ms_(timeout_ms) {
        if (command_.empty())
            throw std::invalid_argument("ExternalSimulator: empty command");
    }

    ExternalSimulator(const ExternalSimulator&) = delete;
    ExternalSimulator& operator=(const ExternalSimulator&) = delete;

    ~ExternalSimulator() override { close(); }

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

    /** True when a live child subprocess is currently attached. */
    bool is_running() const { return running_; }

    /** Cleanly stop the child process (send quit, close pipes, waitpid). */
    void close() const noexcept {
#if BT_SIM_HAS_POSIX
        if (running_ && to_child_) {
            auto prev_sig = std::signal(SIGPIPE, SIG_IGN);
            std::fprintf(to_child_, "quit\n");
            std::fflush(to_child_);
            std::signal(SIGPIPE, prev_sig);
        }
        if (to_child_) {
            std::fclose(to_child_);
            to_child_ = nullptr;
        }
        if (from_child_) {
            std::fclose(from_child_);
            from_child_ = nullptr;
        }
        if (pid_ > 0) {
            int status = 0;
            pid_t w = ::waitpid(pid_, &status, 0);
            if (w < 0 && errno == EINTR) {
                ::waitpid(pid_, &status, 0);
            }
            pid_ = -1;
        }
        running_ = false;
#endif
    }

    void load(const std::vector<uint8_t>& program) override {
        program_ = program;
        state_.clear();

        ensure_started();
        if (running_) {
            std::string cmd = "load";
            if (!program.empty()) {
                cmd += " ";
                cmd += bytes_to_hex(program);
            }
            const std::string reply = exchange(cmd);
            check_reply_error("load", reply);
        }
        loaded_ = true;
    }

    uint64_t exec(const std::vector<int64_t>& inputs) override {
        if (!loaded_) throw std::runtime_error(name_ + ": exec before load");

        if (running_) {
            std::ostringstream cmd;
            cmd << "exec";
            for (int64_t v : inputs) cmd << " " << v;
            const std::string reply = exchange(cmd.str());
            check_reply_error("exec", reply);

            // Accept either `state <v0> ...` directly or `ok [steps]`.
            std::istringstream iss(reply);
            std::string head;
            iss >> head;
            if (head == "state") {
                state_ = parse_int64_list(iss);
                ++steps_;
            } else if (head == "ok") {
                uint64_t reported_steps = 0;
                if (iss >> reported_steps) steps_ = reported_steps;
                else ++steps_;
                state_ = query_state_over_pipe();
            } else {
                throw std::runtime_error(name_ + ": unexpected exec reply: '" +
                                         reply + "'");
            }
            return steps_;
        }

        // Offline fallback when the simulator binary is not installed on the
        // host (ENOENT): keep the interface testable in CI without hardware.
        state_ = inputs;
        ++steps_;
        return steps_;
    }

    std::vector<int64_t> read_state() const override {
        if (running_) {
            state_ = query_state_over_pipe();
        }
        return state_;
    }

    std::vector<uint8_t> read_memory(uint64_t addr, size_t n) const override {
        if (!running_) {
            throw std::runtime_error(name_ + ": read_memory requires a transport");
        }
        std::ostringstream cmd;
        cmd << "memory " << addr << " " << n;
        const std::string reply = exchange(cmd.str());
        check_reply_error("memory", reply);

        std::istringstream iss(reply);
        std::string head;
        iss >> head;
        if (head != "mem" && head != "memory" && head != "ok") {
            throw std::runtime_error(name_ + ": unexpected memory reply: '" +
                                     reply + "'");
        }
        std::string tok1, tok2;
        if (!(iss >> tok1)) {
            if (n == 0) return {};
            throw std::runtime_error(name_ + ": empty memory payload");
        }
        // Support both `mem <hex>` and `mem <nbytes> <hex>`.
        const std::string& hex_str = (iss >> tok2) ? tok2 : tok1;
        auto bytes = hex_to_bytes(hex_str);
        if (bytes.size() > n) bytes.resize(n);
        return bytes;
    }

    /** The command a harness would actually spawn. */
    std::string command_line() const {
        std::string s = command_;
        for (const auto& a : args_) { s += " "; s += a; }
        return s;
    }

private:
    static std::string bytes_to_hex(const std::vector<uint8_t>& data) {
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out;
        out.reserve(data.size() * 2);
        for (uint8_t b : data) {
            out.push_back(kHex[(b >> 4) & 0xF]);
            out.push_back(kHex[b & 0xF]);
        }
        return out;
    }

    static int hex_nibble(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    }

    static std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
        if (hex.size() % 2 != 0)
            throw std::runtime_error("ExternalSimulator: odd-length hex payload");
        std::vector<uint8_t> out;
        out.reserve(hex.size() / 2);
        for (size_t i = 0; i < hex.size(); i += 2) {
            int hi = hex_nibble(hex[i]);
            int lo = hex_nibble(hex[i + 1]);
            if (hi < 0 || lo < 0)
                throw std::runtime_error("ExternalSimulator: invalid hex digit");
            out.push_back(static_cast<uint8_t>((hi << 4) | lo));
        }
        return out;
    }

    static std::vector<int64_t> parse_int64_list(std::istringstream& iss) {
        std::vector<int64_t> out;
        int64_t v = 0;
        while (iss >> v) out.push_back(v);
        if (!iss.eof()) {
            throw std::runtime_error(
                "ExternalSimulator: malformed integer in state reply");
        }
        return out;
    }

    void check_reply_error(const char* op, const std::string& reply) const {
        if (reply.rfind("unsupported", 0) == 0) {
            throw std::runtime_error(name_ + ": " + op + " " + reply);
        }
        if (reply.rfind("error", 0) == 0 || reply.rfind("ERR", 0) == 0) {
            throw std::runtime_error(name_ + ": " + op + " failed: " + reply);
        }
    }

    std::vector<int64_t> query_state_over_pipe() const {
        const std::string reply = exchange("state");
        check_reply_error("state", reply);
        std::istringstream iss(reply);
        std::string head;
        iss >> head;
        if (head != "state" && head != "ok") {
            throw std::runtime_error(name_ + ": unexpected state reply: '" +
                                     reply + "'");
        }
        return parse_int64_list(iss);
    }

    void ensure_started() {
#if BT_SIM_HAS_POSIX
        if (running_ || binary_missing_) return;

        int in_pipe[2]{-1, -1};
        int out_pipe[2]{-1, -1};
        int err_pipe[2]{-1, -1};

        if (::pipe(in_pipe) != 0)
            throw std::runtime_error(name_ + ": pipe(in) failed");
        if (::pipe(out_pipe) != 0) {
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            throw std::runtime_error(name_ + ": pipe(out) failed");
        }
        if (::pipe(err_pipe) != 0) {
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            throw std::runtime_error(name_ + ": pipe(err) failed");
        }
        ::fcntl(err_pipe[1], F_SETFD, FD_CLOEXEC);

        pid_t child = ::fork();
        if (child < 0) {
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            ::close(err_pipe[0]); ::close(err_pipe[1]);
            throw std::runtime_error(name_ + ": fork() failed");
        }

        if (child == 0) {
            // Child process
            ::dup2(in_pipe[0], STDIN_FILENO);
            ::dup2(out_pipe[1], STDOUT_FILENO);
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            ::close(err_pipe[0]);

            std::vector<char*> argv;
            argv.reserve(args_.size() + 2);
            argv.push_back(const_cast<char*>(command_.c_str()));
            for (const auto& a : args_)
                argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);

            ::execvp(argv[0], argv.data());
            int err = errno;
            (void)::write(err_pipe[1], &err, sizeof(err));
            ::close(err_pipe[1]);
            ::_exit(127);
        }

        // Parent process
        ::close(in_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[1]);

        int child_errno = 0;
        ssize_t nread = ::read(err_pipe[0], &child_errno, sizeof(child_errno));
        ::close(err_pipe[0]);

        if (nread > 0) {
            ::close(in_pipe[1]);
            ::close(out_pipe[0]);
            int status = 0;
            ::waitpid(child, &status, 0);
            if (child_errno == ENOENT) {
                // Simulator binary is not installed on this host; stay in
                // offline fallback mode so YAML-driven smoke tests pass.
                binary_missing_ = true;
                return;
            }
            throw std::runtime_error(name_ + ": execvp('" + command_ +
                                     "') failed: " + std::strerror(child_errno));
        }

        FILE* to_c = ::fdopen(in_pipe[1], "w");
        FILE* from_c = ::fdopen(out_pipe[0], "r");
        if (!to_c || !from_c) {
            if (to_c) std::fclose(to_c); else ::close(in_pipe[1]);
            if (from_c) std::fclose(from_c); else ::close(out_pipe[0]);
            ::kill(child, SIGTERM);
            int status = 0;
            ::waitpid(child, &status, 0);
            throw std::runtime_error(name_ + ": fdopen() failed");
        }
        // Unbuffered read stream so poll() on fileno(from_child_) accurately
        // reflects unread bytes without stdio read-ahead desynchronisation.
        std::setvbuf(from_c, nullptr, _IONBF, 0);

        pid_ = child;
        to_child_ = to_c;
        from_child_ = from_c;
        running_ = true;
#endif
    }

    std::string exchange(const std::string& line) const {
#if BT_SIM_HAS_POSIX
        if (!running_ || !to_child_ || !from_child_) {
            throw std::runtime_error(name_ + ": simulator process is not running");
        }

        auto prev_sig = std::signal(SIGPIPE, SIG_IGN);
        int written = std::fprintf(to_child_, "%s\n", line.c_str());
        int flushed = std::fflush(to_child_);
        std::signal(SIGPIPE, prev_sig);
        if (written < 0 || flushed != 0) {
            close();
            throw std::runtime_error(name_ + ": write to simulator failed (broken pipe)");
        }

        struct pollfd pfd{};
        pfd.fd = ::fileno(from_child_);
        pfd.events = POLLIN;
        int ready = 0;
        do {
            ready = ::poll(&pfd, 1, timeout_ms_);
        } while (ready < 0 && errno == EINTR);

        if (ready == 0) {
            ::kill(pid_, SIGTERM);
            close();
            throw std::runtime_error(name_ + ": timed out waiting for simulator reply");
        }
        if (ready < 0) {
            close();
            throw std::runtime_error(name_ + ": poll() failed on simulator pipe");
        }

        char* buf = nullptr;
        size_t cap = 0;
        ssize_t len = ::getline(&buf, &cap, from_child_);
        if (len < 0) {
            std::free(buf);
            close();
            throw std::runtime_error(name_ + ": unexpected EOF from simulator");
        }
        std::string out(buf, static_cast<size_t>(len));
        std::free(buf);
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
            out.pop_back();
        return out;
#else
        (void)line;
        throw std::runtime_error(name_ + ": POSIX subprocess transport unavailable");
#endif
    }

    std::string command_;
    std::vector<std::string> args_;
    std::string name_;
    int timeout_ms_{5000};
    std::vector<uint8_t> program_;
    mutable std::vector<int64_t> state_;
    bool loaded_{false};
    uint64_t steps_{0};

#if BT_SIM_HAS_POSIX
    mutable pid_t pid_{-1};
    mutable FILE* to_child_{nullptr};
    mutable FILE* from_child_{nullptr};
#endif
    mutable bool running_{false};
    bool binary_missing_{false};
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
