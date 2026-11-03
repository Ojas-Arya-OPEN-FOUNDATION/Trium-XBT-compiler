/**
 * @file bench_harness.hpp
 * @brief A benchmark harness that cannot lie.
 *
 * Every performance bug found in this repository shared one cause: a number
 * was reported that had not been earned. The specific ways that happens:
 *
 *   1. Measuring something you never verified. A fast wrong kernel looks
 *      identical to a fast right one. This harness verifies bit-exactness
 *      against a reference BEFORE it is willing to time anything, and prints
 *      "UNVERIFIED" instead of a number when it fails.
 *   2. Letting the compiler delete the work. Consuming one output element
 *      through `volatile` is not enough - GCC will happily compute just that
 *      element. The sink here is a full-array comparison, performed OUTSIDE
 *      the timed region.
 *   3. Reporting a single sample. Best, median and worst are all reported, so
 *      variance is visible instead of hidden.
 *   4. Hiding a failure. Any unverified case sets a nonzero exit code. A
 *      benchmark that passes while producing no numbers has failed.
 *   5. Omitting the machine. Every run records CPU, ISA features, thread
 *      count and build flags, because "167 G MAC/s" is meaningless without
 *      them and worse than meaningless when quoted from a different machine.
 *
 * Usage:
 *   btbench::Runner r;
 *   r.add({"name", m, n, k, compute_fn, output_fn, reference_fn});
 *   int rc = r.run_all(reps);      // 0 = every case verified and timed
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace btbench {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Machine fingerprint
// ---------------------------------------------------------------------------
struct MachineInfo {
    std::string cpu{"unknown"};
    int threads{1};
    bool avx2{false}, avx512f{false}, avx512bw{false}, avx512vpopcntdq{false};
    bool bmi2{false}, neon{false};

    std::string isa_summary() const {
        std::string s;
        auto add = [&](bool b, const char* n) {
            if (b) { if (!s.empty()) s += "+"; s += n; }
        };
        add(avx512vpopcntdq, "AVX512VPOPCNTDQ");
        add(avx512bw, "AVX512BW");
        add(avx512f, "AVX512F");
        add(avx2, "AVX2");
        add(bmi2, "BMI2");
        add(neon, "NEON");
        return s.empty() ? "scalar" : s;
    }
};

/** Probe the machine. Uses CPUID via __builtin_cpu_supports, NOT /proc/cpuinfo:
 *  on this class of VM /proc/cpuinfo masks AVX-512 VPOPCNTDQ, and trusting it
 *  previously led to a wrong conclusion about what the hardware supports. */
inline MachineInfo probe_machine() {
    MachineInfo m;
    m.threads = static_cast<int>(std::thread::hardware_concurrency());
    if (m.threads < 1) m.threads = 1;
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    m.avx2            = __builtin_cpu_supports("avx2");
    m.avx512f         = __builtin_cpu_supports("avx512f");
    m.avx512bw        = __builtin_cpu_supports("avx512bw");
    m.avx512vpopcntdq = __builtin_cpu_supports("avx512vpopcntdq");
    m.bmi2            = __builtin_cpu_supports("bmi2");
#elif defined(__aarch64__)
    m.neon = true;
#endif
    if (FILE* f = std::fopen("/proc/cpuinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, f)) {
            if (std::strncmp(line, "model name", 10) == 0) {
                const char* c = std::strchr(line, ':');
                if (c) {
                    m.cpu = std::string(c + 1);
                    while (!m.cpu.empty() && (m.cpu.front() == ' ' || m.cpu.front() == '\t'))
                        m.cpu.erase(m.cpu.begin());
                    while (!m.cpu.empty() && (m.cpu.back() == '\n' || m.cpu.back() == '\r'))
                        m.cpu.pop_back();
                }
                break;
            }
        }
        std::fclose(f);
    }
    return m;
}

// ---------------------------------------------------------------------------
// Cases and results
// ---------------------------------------------------------------------------
struct BenchCase {
    std::string name;
    size_t m{0}, n{0}, k{0};
    std::function<void()>                      compute;    // the thing under test
    std::function<const std::vector<int64_t>&()> output;   // what it produced
    std::function<std::vector<int64_t>()>      reference;  // ground truth
    bool skip_verify{false};   // only for references themselves
};

struct BenchResult {
    std::string name;
    size_t m{0}, n{0}, k{0};
    bool   verified{false};
    double best_ms{0}, median_ms{0}, worst_ms{0};
    double gmacs{0};           // 0 unless verified
    std::string note;
};

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
class Runner {
public:
    explicit Runner(int reps = 5) : reps_(reps < 3 ? 3 : reps) {}

    void add(BenchCase c) { cases_.push_back(std::move(c)); }

    /** Run every case. Returns 0 only if every case verified. */
    int run_all(bool quiet = false) {
        const MachineInfo mi = probe_machine();
        if (!quiet) {
            std::printf("machine   : %s\n", mi.cpu.c_str());
            std::printf("isa       : %s\n", mi.isa_summary().c_str());
            std::printf("threads   : %d   build: %s\n", mi.threads,
#if defined(__OPTIMIZE__)
                        "optimised"
#else
                        "UNOPTIMISED (numbers are meaningless)"
#endif
            );
            std::printf("reps      : %d (best / median / worst reported)\n\n", reps_);
        }
        std::printf("%-34s %6s %10s %10s %10s %12s\n",
                    "kernel", "n", "best ms", "median ms", "worst ms", "G MAC/s");
        std::printf("%s\n", std::string(90, '-').c_str());

        for (auto& c : cases_) {
            BenchResult r = run_one(c);
            results_.push_back(r);
            if (!r.verified) {
                std::printf("%-34s %6zu %10s %10s %10s %12s  %s\n",
                            c.name.c_str(), c.m, "-", "-", "-", "UNVERIFIED",
                            r.note.c_str());
                ++failures_;
            } else {
                std::printf("%-34s %6zu %10.3f %10.3f %10.3f %12.2f\n",
                            r.name.c_str(), r.m, r.best_ms, r.median_ms,
                            r.worst_ms, r.gmacs);
            }
        }
        std::printf("\n");
        if (failures_)
            std::printf("FAILED: %d case(s) did not verify. No number is "
                        "reported for them, and this run exits nonzero.\n", failures_);
        else
            std::printf("All %zu case(s) verified bit-exact against their "
                        "reference before timing.\n", results_.size());
        return failures_ ? 1 : 0;
    }

    const std::vector<BenchResult>& results() const { return results_; }

    /** Machine-readable output. Unverified cases carry verified=false and no
     *  throughput, so a consumer cannot accidentally plot a wrong number. */
    std::string to_json(const MachineInfo& mi) const {
        std::string s = "{\n  \"machine\": {\"cpu\": \"" + mi.cpu +
                        "\", \"isa\": \"" + mi.isa_summary() +
                        "\", \"threads\": " + std::to_string(mi.threads) + "},\n";
        s += "  \"results\": [\n";
        for (size_t i = 0; i < results_.size(); ++i) {
            const BenchResult& r = results_[i];
            s += "    {\"name\": \"" + r.name + "\", \"m\": " + std::to_string(r.m) +
                 ", \"n\": " + std::to_string(r.n) + ", \"k\": " + std::to_string(r.k) +
                 ", \"verified\": " + (r.verified ? "true" : "false") +
                 ", \"best_ms\": " + std::to_string(r.best_ms) +
                 ", \"median_ms\": " + std::to_string(r.median_ms) +
                 ", \"worst_ms\": " + std::to_string(r.worst_ms) +
                 ", \"gmacs\": " + std::to_string(r.gmacs) +
                 ", \"note\": \"" + r.note + "\"}" +
                 (i + 1 < results_.size() ? "," : "") + "\n";
        }
        s += "  ]\n}\n";
        return s;
    }

private:
    BenchResult run_one(const BenchCase& c) {
        BenchResult r;
        r.name = c.name; r.m = c.m; r.n = c.n; r.k = c.k;

        // --- 1. Correctness first. No verification, no number. ---
        if (!c.skip_verify) {
            std::vector<int64_t> ref = c.reference();
            c.compute();                       // warm-up also populates output
            const std::vector<int64_t>& got = c.output();
            if (got.size() != ref.size()) {
                r.note = "output size " + std::to_string(got.size()) +
                         " != reference " + std::to_string(ref.size());
                return r;
            }
            size_t first_bad = size_t(-1);
            for (size_t i = 0; i < ref.size(); ++i)
                if (ref[i] != got[i]) { first_bad = i; break; }
            if (first_bad != size_t(-1)) {
                r.note = "mismatch at element " + std::to_string(first_bad) +
                         ": got " + std::to_string(got[first_bad]) +
                         " want " + std::to_string(ref[first_bad]);
                return r;
            }
            r.verified = true;
        } else {
            r.verified = true;
            r.note = "reference kernel (not verified against itself)";
        }

        // --- 2. Time it. The correctness check above is the sink, and it runs
        //        outside the timed region, so the compiler cannot elide the
        //        work - it must materialise the whole array for the comparison. ---
        std::vector<double> ts;
        ts.reserve(reps_);
        for (int i = 0; i < reps_; ++i) {
            const auto t0 = Clock::now();
            c.compute();
            const auto t1 = Clock::now();
            ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        r.best_ms   = ts.front();
        r.median_ms = ts[ts.size() / 2];
        r.worst_ms  = ts.back();

        // --- 3. Re-verify after timing: catches kernels that corrupt state or
        //        are only correct on the first call. ---
        if (!c.skip_verify) {
            std::vector<int64_t> ref = c.reference();
            const std::vector<int64_t>& got = c.output();
            for (size_t i = 0; i < ref.size(); ++i)
                if (ref[i] != got[i]) {
                    r.verified = false; r.gmacs = 0;
                    r.note = "result changed after repeated runs (element " +
                             std::to_string(i) + ")";
                    return r;
                }
        }

        if (r.best_ms > 0 && r.verified)
            r.gmacs = (double(c.m) * double(c.n) * double(c.k)) /
                      (r.best_ms * 1e-3) / 1e9;
        return r;
    }

    int reps_;
    std::vector<BenchCase> cases_;
    std::vector<BenchResult> results_;
    int failures_{0};
};

} // namespace btbench
