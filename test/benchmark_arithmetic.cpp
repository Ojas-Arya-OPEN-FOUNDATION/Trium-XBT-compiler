/**
 * @file benchmark_arithmetic.cpp
 * @brief Empirical Benchmark & Contract Verification: Native Checked 64-bit vs i128 Extension for Int39
 *
 * Verifies and benchmarks 10,000,000 Int39 additions under the documented language contract:
 *   1) Strict overflow detection: arithmetic outside [-H, +H] must trigger an error, never clamp.
 *   2) Baseline checked lowering via i128 widening and range check.
 *   3) Optimized checked native 64-bit integer lowering (native add i64 + range check guard).
 *   4) Unchecked fast-path native i64 addition for interval-analyzed provably-safe operations.
 */

#include "Ternary/IR/LLVM.hpp"
#include "Ternary/Math/Integer.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace bt::ir;
using namespace bt::math;

constexpr int64_t MAX_INT39 = ScalarInteger::maximum; // (3^39 - 1) / 2 = 2026277576509488133LL
constexpr int64_t MIN_INT39 = ScalarInteger::minimum; // -2026277576509488133LL

// Baseline: i128 widening with range check (contract-compliant overflow reporting)
__attribute__((noinline))
void add_int39_checked_i128(const int64_t* __restrict__ a,
                           const int64_t* __restrict__ b,
                           int64_t* __restrict__ out,
                           uint8_t* __restrict__ overflow,
                           size_t n) {
    constexpr __int128 MAX_W = static_cast<__int128>(MAX_INT39);
    constexpr __int128 MIN_W = static_cast<__int128>(MIN_INT39);
    for (size_t i = 0; i < n; ++i) {
        __int128 wide_a = static_cast<__int128>(a[i]);
        __int128 wide_b = static_cast<__int128>(b[i]);
        __int128 sum = wide_a + wide_b;
        bool bad = (sum < MIN_W) || (sum > MAX_W);
        overflow[i] = bad ? 1 : 0;
        out[i] = bad ? 0 : static_cast<int64_t>(sum);
    }
}

// Optimized: Native 64-bit checked addition with branchless range verification
__attribute__((noinline))
void add_int39_checked_i64(const int64_t* __restrict__ a,
                          const int64_t* __restrict__ b,
                          int64_t* __restrict__ out,
                          uint8_t* __restrict__ overflow,
                          size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int64_t sum = a[i] + b[i];
        bool bad = (sum < MIN_INT39) || (sum > MAX_INT39);
        overflow[i] = bad ? 1 : 0;
        out[i] = bad ? 0 : sum;
    }
}

// Fast path: Unchecked native i64 addition (for interval-verified in-domain values)
__attribute__((noinline))
void add_int39_fast_i64(const int64_t* __restrict__ a,
                        const int64_t* __restrict__ b,
                        int64_t* __restrict__ out,
                        size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
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
    constexpr size_t TOTAL_OPS = 10000000; // 10,000,000 operations
    int num_runs = 10;
    if (argc > 1) {
        num_runs = std::max(1, std::stoi(argv[1]));
    }

    std::cout << "========================================================================\n";
    std::cout << "  Int39 Arithmetic Benchmark: Native 64-bit Checked vs i128 Extension   \n";
    std::cout << "  Operations per Run: " << TOTAL_OPS << " additions\n";
    std::cout << "  Number of Benchmark Iterations: " << num_runs << "\n";
    std::cout << "  Int39 Domain: [" << MIN_INT39 << ", " << MAX_INT39 << "]\n";
    std::cout << "========================================================================\n\n";

    // 1. Verify Compiler IR Lowering Output
    std::cout << "[1/4] Verifying LLVM IR Lowering for Op::Add, Op::Sub, Op::Mul...\n";
    {
        auto param = [](uint32_t s) {
            auto e = std::make_shared<Expr>(Expr{Op::Parameter, Type::Int39});
            e->symbol = s;
            return e;
        };
        auto ret = [](Expression e) {
            Statement s{StatementKind::Return};
            s.value = e;
            return s;
        };

        // Function with add: x + y
        Program p_add;
        Function f_add{"add_test", Type::Int39, {{1, Type::Int39}, {2, Type::Int39}},
                       {ret(std::make_shared<Expr>(Expr{Op::Add, Type::Int39, {param(1), param(2)}}))}};
        p_add.functions.push_back(f_add);

        LLVMEmitter emitter;
        std::string llvm_add = emitter.generate(p_add, "x86_64-unknown-linux-gnu", "");
        assert(llvm_add.find("add i64") != std::string::npos && "Op::Add must use native add i64!");
        assert(llvm_add.find("icmp slt i64") != std::string::npos && "Op::Add must check lower bound!");
        assert(llvm_add.find("icmp sgt i64") != std::string::npos && "Op::Add must check upper bound!");
        assert(llvm_add.find("br i1") != std::string::npos && "Op::Add must guard with conditional branch!");
        assert(llvm_add.find("label %error") != std::string::npos && "Op::Add must branch to error on overflow!");

        // Function with sub: x - y
        Program p_sub;
        Function f_sub{"sub_test", Type::Int39, {{1, Type::Int39}, {2, Type::Int39}},
                       {ret(std::make_shared<Expr>(Expr{Op::Sub, Type::Int39, {param(1), param(2)}}))}};
        p_sub.functions.push_back(f_sub);
        std::string llvm_sub = emitter.generate(p_sub, "x86_64-unknown-linux-gnu", "");
        assert(llvm_sub.find("sub i64") != std::string::npos && "Op::Sub must use native sub i64!");
        assert(llvm_sub.find("label %error") != std::string::npos && "Op::Sub must branch to error on overflow!");

        // Function with mul: x * y
        Program p_mul;
        Function f_mul{"mul_test", Type::Int39, {{1, Type::Int39}, {2, Type::Int39}},
                       {ret(std::make_shared<Expr>(Expr{Op::Mul, Type::Int39, {param(1), param(2)}}))}};
        p_mul.functions.push_back(f_mul);
        std::string llvm_mul = emitter.generate(p_mul, "x86_64-unknown-linux-gnu", "");
        assert(llvm_mul.find("mul i128") != std::string::npos && "Op::Mul must reserve i128 extension!");

        std::cout << "      -> Op::Add: lowers to native add i64 + checked range guard branching to error.\n";
        std::cout << "      -> Op::Sub: lowers to native sub i64 + checked range guard branching to error.\n";
        std::cout << "      -> Op::Mul: strictly reserves i128 extension (verified i128 present).\n";
    }

    // 2. Generating Deterministic Test Vectors (10,000,000 pairs)
    std::cout << "\n[2/4] Generating " << TOTAL_OPS << " deterministic Int39 test vectors...\n";
    std::mt19937_64 rng(0x139ACCA7);
    std::vector<int64_t> a(TOTAL_OPS);
    std::vector<int64_t> b(TOTAL_OPS);
    std::vector<int64_t> res_i128(TOTAL_OPS);
    std::vector<int64_t> res_i64(TOTAL_OPS);
    std::vector<int64_t> res_fast(TOTAL_OPS);
    std::vector<uint8_t> ovf_i128(TOTAL_OPS);
    std::vector<uint8_t> ovf_i64(TOTAL_OPS);

    size_t expected_overflow_count = 0;
    for (size_t i = 0; i < TOTAL_OPS; ++i) {
        if (i % 100 == 0) {
            // Intentional boundary vectors that exceed MAX_INT39
            a[i] = MAX_INT39 - (i % 500);
            b[i] = static_cast<int64_t>((i % 500) + 1); // sum = MAX_INT39 + 1 (overflow!)
            expected_overflow_count++;
        } else if (i % 100 == 1) {
            // Intentional boundary vectors that exceed MIN_INT39 negatively
            a[i] = MIN_INT39 + (i % 500);
            b[i] = -static_cast<int64_t>((i % 500) + 1); // sum = MIN_INT39 - 1 (overflow!)
            expected_overflow_count++;
        } else {
            // Valid in-domain pairs with sum guaranteed in [-MAX_INT39, MAX_INT39]
            int64_t half_max = MAX_INT39 / 2;
            a[i] = static_cast<int64_t>((rng() % (2 * static_cast<uint64_t>(half_max) + 1))) - half_max;
            b[i] = static_cast<int64_t>((rng() % (2 * static_cast<uint64_t>(half_max) + 1))) - half_max;
        }
    }

    // 3. Functional Equivalence and Overflow Contract Verification
    std::cout << "[3/4] Verifying contract compliance (overflow detection and in-domain identity)...\n";
    add_int39_checked_i128(a.data(), b.data(), res_i128.data(), ovf_i128.data(), TOTAL_OPS);
    add_int39_checked_i64(a.data(), b.data(), res_i64.data(), ovf_i64.data(), TOTAL_OPS);
    add_int39_fast_i64(a.data(), b.data(), res_fast.data(), TOTAL_OPS);

    size_t detected_overflows = 0;
    for (size_t i = 0; i < TOTAL_OPS; ++i) {
        assert(ovf_i128[i] == ovf_i64[i] && "Overflow detection mismatch between i128 and i64!");
        if (ovf_i128[i]) {
            detected_overflows++;
        } else {
            assert(res_i128[i] == res_i64[i] && "Arithmetic mismatch on in-domain inputs!");
            assert(res_i64[i] == res_fast[i] && "Fast-path mismatch on in-domain inputs!");
        }
    }
    assert(detected_overflows == expected_overflow_count && "All boundary overflows must be detected!");
    std::cout << "      -> " << detected_overflows << " overflows correctly detected and rejected (0 silent saturation).\n";
    std::cout << "      -> " << (TOTAL_OPS - detected_overflows) << " in-domain results match bit-for-bit (100% equivalence verified).\n";

    // 4. Benchmark Execution
    std::cout << "\n[4/4] Executing " << num_runs << " benchmark trials (" << TOTAL_OPS << " adds per trial)...\n";
    std::vector<double> times_i128_us;
    std::vector<double> times_i64_us;
    std::vector<double> times_fast_us;
    times_i128_us.reserve(num_runs);
    times_i64_us.reserve(num_runs);
    times_fast_us.reserve(num_runs);

    // Warm-up caches
    add_int39_checked_i128(a.data(), b.data(), res_i128.data(), ovf_i128.data(), 1000000);
    add_int39_checked_i64(a.data(), b.data(), res_i64.data(), ovf_i64.data(), 1000000);
    add_int39_fast_i64(a.data(), b.data(), res_fast.data(), 1000000);

    for (int run = 0; run < num_runs; ++run) {
        // Measure i128 extension
        auto start_128 = std::chrono::high_resolution_clock::now();
        add_int39_checked_i128(a.data(), b.data(), res_i128.data(), ovf_i128.data(), TOTAL_OPS);
        auto end_128 = std::chrono::high_resolution_clock::now();
        double elapsed_128 = std::chrono::duration<double, std::micro>(end_128 - start_128).count();
        times_i128_us.push_back(elapsed_128);

        // Measure native i64 checked
        auto start_64 = std::chrono::high_resolution_clock::now();
        add_int39_checked_i64(a.data(), b.data(), res_i64.data(), ovf_i64.data(), TOTAL_OPS);
        auto end_64 = std::chrono::high_resolution_clock::now();
        double elapsed_64 = std::chrono::duration<double, std::micro>(end_64 - start_64).count();
        times_i64_us.push_back(elapsed_64);

        // Measure native i64 fast-path
        auto start_fast = std::chrono::high_resolution_clock::now();
        add_int39_fast_i64(a.data(), b.data(), res_fast.data(), TOTAL_OPS);
        auto end_fast = std::chrono::high_resolution_clock::now();
        double elapsed_fast = std::chrono::duration<double, std::micro>(end_fast - start_fast).count();
        times_fast_us.push_back(elapsed_fast);

        std::cout << "  Trial " << std::setw(2) << (run + 1) << ": "
                  << "checked i128 = " << std::fixed << std::setprecision(1) << elapsed_128 << " us | "
                  << "checked i64 = " << std::setprecision(1) << elapsed_64 << " us | "
                  << "fast i64 = " << std::setprecision(1) << elapsed_fast << " us\n";
    }

    StatResult stats_128 = compute_stats(times_i128_us);
    StatResult stats_64 = compute_stats(times_i64_us);
    StatResult stats_fast = compute_stats(times_fast_us);
    double speedup_checked = stats_128.mean / stats_64.mean;
    double speedup_fast = stats_128.mean / stats_fast.mean;

    std::cout << "\n========================================================================\n";
    std::cout << "  EMPIRICAL BENCHMARK RESULTS (10,000,000 Int39 Additions)\n";
    std::cout << "========================================================================\n";
    std::cout << "  [Baseline] Checked i128 Extension Lowering:\n";
    std::cout << "    Mean Wall-Clock Time:  " << std::fixed << std::setprecision(2) << stats_128.mean << " us (" << (stats_128.mean / 1000.0) << " ms)\n";
    std::cout << "    Throughput:            " << std::setprecision(2) << (TOTAL_OPS / stats_128.mean) << " MOps/sec\n\n";

    std::cout << "  [Optimized] Checked Native i64 Lowering:\n";
    std::cout << "    Mean Wall-Clock Time:  " << std::fixed << std::setprecision(2) << stats_64.mean << " us (" << (stats_64.mean / 1000.0) << " ms)\n";
    std::cout << "    Throughput:            " << std::setprecision(2) << (TOTAL_OPS / stats_64.mean) << " MOps/sec\n";
    std::cout << "    Speedup vs i128:       " << std::setprecision(2) << speedup_checked << "x\n\n";

    std::cout << "  [Fast Path] Unchecked Native i64 Lowering (Interval-Proven):\n";
    std::cout << "    Mean Wall-Clock Time:  " << std::fixed << std::setprecision(2) << stats_fast.mean << " us (" << (stats_fast.mean / 1000.0) << " ms)\n";
    std::cout << "    Throughput:            " << std::setprecision(2) << (TOTAL_OPS / stats_fast.mean) << " MOps/sec\n";
    std::cout << "    Speedup vs i128:       " << std::setprecision(2) << speedup_fast << "x\n";
    std::cout << "========================================================================\n";

    return 0;
}
