/**
 * @file benchmark_wide_math.cpp
 * @brief Rigorous Scientific Benchmark & Verification Harness for Wide Math Operations
 *        Compares optimized times_pow3 and shifted() vs baseline serial loops in ternary_wide.hpp.
 */

#include "ternary_wide.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace bt::fp_detail;

template <typename T>
inline void do_not_optimize(T const& val) {
    asm volatile("" : : "g"(val) : "memory");
}

template <size_t N>
Wide<N> baseline_times_pow3(Wide<N> v, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        v = v.times(3);
    }
    return v;
}

template <size_t N>
Wide<N> baseline_shifted(Wide<N> v, unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        v = v.times(2);
    }
    return v;
}

struct StatResult {
    double mean = 0.0;
    double stddev = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
};

StatResult compute_stats(const std::vector<double>& samples) {
    if (samples.empty()) return {};
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / samples.size();
    double var_sum = 0.0;
    double min_v = samples[0];
    double max_v = samples[0];
    for (double x : samples) {
        var_sum += (x - mean) * (x - mean);
        if (x < min_v) min_v = x;
        if (x > max_v) max_v = x;
    }
    double stddev = (samples.size() > 1) ? std::sqrt(var_sum / (samples.size() - 1)) : 0.0;
    return {mean, stddev, min_v, max_v};
}

int main(int argc, char** argv) {
    int num_runs = 10;
    if (argc > 1) {
        num_runs = std::max(1, std::stoi(argv[1]));
    }

    std::cout << "========================================================================\n";
    std::cout << "  Trium-XBT Wide Math Benchmark & Arithmetic Verification Suite\n";
    std::cout << "  Operations: times_pow3() and shifted() vs Baseline Serial Loops\n";
    std::cout << "========================================================================\n\n";

    constexpr size_t NUM_OPS = 100000;
    using W = Wide<8>;

    // Generate deterministic test vectors
    std::cout << "[1/3] Generating deterministic test vectors (" << NUM_OPS << " items)...\n";
    std::mt19937_64 rng(0xDEADBEEFCAFEULL);
    std::vector<W> inputs_pow3(NUM_OPS);
    std::vector<uint64_t> deltas(NUM_OPS);
    std::vector<W> inputs_shift(NUM_OPS);
    std::vector<unsigned> shifts(NUM_OPS);

    for (size_t i = 0; i < NUM_OPS; ++i) {
        // Significand for exponent alignment: 64-bit mantissa (well within 512-bit Wide<8> for 3^160)
        uint64_t w0 = (rng() & 0x7FFFFFFFFFFFFFFFULL) | 1ULL;
        inputs_pow3[i] = W(w0);
        deltas[i] = (i % 160) + 1; // Delta in [1, 160]

        // Input for shift: 128-bit value (fits in Wide<8> when shifted by up to 128 bits)
        uint64_t s0 = rng() | 1ULL;
        uint64_t s1 = rng() & 0x0000FFFFFFFFFFFFULL;
        unsigned __int128 init_s = (static_cast<unsigned __int128>(s1) << 64) | s0;
        inputs_shift[i] = W(init_s);
        shifts[i] = (i % 128) + 1; // Shift k in [1, 128]
    }

    // Exact Arithmetic Verification Phase
    std::cout << "[2/3] Verifying exact arithmetic equality against serial baseline...\n";
    
    // Benchmark 1 Verification (Exponent Alignment: times_pow3 vs repeated times(3))
    size_t verify_pow3_count = 10000; // Deep exhaustive check on 10,000 representative cases
    for (size_t i = 0; i < verify_pow3_count; ++i) {
        W acc1 = inputs_pow3[i].times_pow3(deltas[i]);
        W acc2 = baseline_times_pow3(inputs_pow3[i], deltas[i]);
        if (acc1.compare(acc2) != 0) {
            std::cerr << "FATAL: times_pow3 verification failed at index " << i
                      << " (delta=" << deltas[i] << ")\n";
            return 1;
        }
    }
    // Also explicitly verify all boundary deltas 1..160
    for (uint64_t d = 1; d <= 160; ++d) {
        W base(0xCAFE12345678ULL);
        W acc1 = base.times_pow3(d);
        W acc2 = baseline_times_pow3(base, d);
        assert(acc1.compare(acc2) == 0);
    }
    std::cout << "  [PASS] Benchmark 1: acc1.compare(acc2) == 0 across all deltas [1, 160]\n";

    // Benchmark 2 Verification (Bit Shift: shifted(k) vs repeated times(2))
    size_t verify_shift_count = 10000;
    for (size_t i = 0; i < verify_shift_count; ++i) {
        W s1 = inputs_shift[i].shifted(shifts[i]);
        W s2 = baseline_shifted(inputs_shift[i], shifts[i]);
        if (s1.compare(s2) != 0) {
            std::cerr << "FATAL: shifted(k) verification failed at index " << i
                      << " (k=" << shifts[i] << ")\n";
            return 1;
        }
    }
    // Also explicitly verify all shift boundary values 1..128
    for (unsigned k = 1; k <= 128; ++k) {
        W base(0xFEEDFACE12345678ULL);
        W s1 = base.shifted(k);
        W s2 = baseline_shifted(base, k);
        assert(s1.compare(s2) == 0);
    }
    std::cout << "  [PASS] Benchmark 2: s1.compare(s2) == 0 across all shifts [1, 128]\n\n";

    // Benchmark Execution Phase (10 runs)
    std::cout << "[3/3] Running " << num_runs << " benchmark trials (" << NUM_OPS << " ops/trial)...\n\n";

    std::vector<double> b1_base_times(num_runs);
    std::vector<double> b1_opt_times(num_runs);
    std::vector<double> b2_base_times(num_runs);
    std::vector<double> b2_opt_times(num_runs);

    // Warm-up
    {
        uint64_t dummy = 0;
        for (size_t i = 0; i < 1000; ++i) {
            W r1 = inputs_pow3[i].times_pow3(deltas[i]);
            W r2 = inputs_shift[i].shifted(shifts[i]);
            dummy ^= r1.words[0] + r2.words[0];
            do_not_optimize(dummy);
        }
    }

    for (int run = 0; run < num_runs; ++run) {
        // --- Benchmark 1: Baseline (repeated times(3)) ---
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t chk_b1_base = 0;
        for (size_t i = 0; i < NUM_OPS; ++i) {
            W r = baseline_times_pow3(inputs_pow3[i], deltas[i]);
            chk_b1_base ^= r.words[0];
            do_not_optimize(r);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        b1_base_times[run] = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- Benchmark 1: Optimized (times_pow3(delta)) ---
        auto t2 = std::chrono::high_resolution_clock::now();
        uint64_t chk_b1_opt = 0;
        for (size_t i = 0; i < NUM_OPS; ++i) {
            W r = inputs_pow3[i].times_pow3(deltas[i]);
            chk_b1_opt ^= r.words[0];
            do_not_optimize(r);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        b1_opt_times[run] = std::chrono::duration<double, std::milli>(t3 - t2).count();
        assert(chk_b1_base == chk_b1_opt);

        // --- Benchmark 2: Baseline (repeated times(2)) ---
        auto t4 = std::chrono::high_resolution_clock::now();
        uint64_t chk_b2_base = 0;
        for (size_t i = 0; i < NUM_OPS; ++i) {
            W r = baseline_shifted(inputs_shift[i], shifts[i]);
            chk_b2_base ^= r.words[0];
            do_not_optimize(r);
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        b2_base_times[run] = std::chrono::duration<double, std::milli>(t5 - t4).count();

        // --- Benchmark 2: Optimized (shifted(k)) ---
        auto t6 = std::chrono::high_resolution_clock::now();
        uint64_t chk_b2_opt = 0;
        for (size_t i = 0; i < NUM_OPS; ++i) {
            W r = inputs_shift[i].shifted(shifts[i]);
            chk_b2_opt ^= r.words[0];
            do_not_optimize(r);
        }
        auto t7 = std::chrono::high_resolution_clock::now();
        b2_opt_times[run] = std::chrono::duration<double, std::milli>(t7 - t6).count();
        assert(chk_b2_base == chk_b2_opt);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Run " << std::setw(2) << (run + 1) << " | "
                  << "B1 Base: " << std::setw(7) << b1_base_times[run] << " ms, "
                  << "B1 Opt: " << std::setw(6) << b1_opt_times[run] << " ms ("
                  << std::setw(5) << (b1_base_times[run] / b1_opt_times[run]) << "x) | "
                  << "B2 Base: " << std::setw(7) << b2_base_times[run] << " ms, "
                  << "B2 Opt: " << std::setw(6) << b2_opt_times[run] << " ms ("
                  << std::setw(5) << (b2_base_times[run] / b2_opt_times[run]) << "x)\n";
    }

    StatResult s_b1_base = compute_stats(b1_base_times);
    StatResult s_b1_opt = compute_stats(b1_opt_times);
    StatResult s_b2_base = compute_stats(b2_base_times);
    StatResult s_b2_opt = compute_stats(b2_opt_times);

    double speedup_b1 = s_b1_base.mean / s_b1_opt.mean;
    double speedup_b2 = s_b2_base.mean / s_b2_opt.mean;

    std::cout << "\n========================================================================\n";
    std::cout << "                        STATISTICAL SUMMARY (" << num_runs << " Runs)\n";
    std::cout << "========================================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Benchmark 1: Exponent Alignment (3^Delta scaling for Delta in [1, 160])\n";
    std::cout << "  Baseline (times(3) loop):  " << std::setw(7) << s_b1_base.mean << " +/- " << s_b1_base.stddev << " ms (range: " << s_b1_base.min_val << " - " << s_b1_base.max_val << " ms)\n";
    std::cout << "  Optimized (times_pow3):    " << std::setw(7) << s_b1_opt.mean << " +/- " << s_b1_opt.stddev << " ms (range: " << s_b1_opt.min_val << " - " << s_b1_opt.max_val << " ms)\n";
    std::cout << "  SPEEDUP FACTOR:            " << std::setw(7) << speedup_b1 << "x\n";
    std::cout << "  Arithmetic Verification:   VERIFIED EXACT (acc1.compare(acc2) == 0)\n\n";

    std::cout << "Benchmark 2: Big-Int Bit Shift (2^k scaling for k in [1, 128])\n";
    std::cout << "  Baseline (times(2) loop):  " << std::setw(7) << s_b2_base.mean << " +/- " << s_b2_base.stddev << " ms (range: " << s_b2_base.min_val << " - " << s_b2_base.max_val << " ms)\n";
    std::cout << "  Optimized (shifted(k)):    " << std::setw(7) << s_b2_opt.mean << " +/- " << s_b2_opt.stddev << " ms (range: " << s_b2_opt.min_val << " - " << s_b2_opt.max_val << " ms)\n";
    std::cout << "  SPEEDUP FACTOR:            " << std::setw(7) << speedup_b2 << "x\n";
    std::cout << "  Arithmetic Verification:   VERIFIED EXACT (s1.compare(s2) == 0)\n";
    std::cout << "========================================================================\n";

    return 0;
}
