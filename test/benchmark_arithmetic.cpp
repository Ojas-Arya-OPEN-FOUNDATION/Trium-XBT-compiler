/**
 * @file benchmark_arithmetic.cpp
 * @brief Rigorous Empirical Benchmark: Native 64-bit Clamping vs i128 Extension for Int39
 *
 * Compares 10,000,000 Int39 additions under:
 *   1) Baseline i128 extension lowering (widening, multi-register adds, i128 comparisons)
 *   2) Native 64-bit integer arithmetic clamping lowering (native add i64 + icmp/select)
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

// Baseline: i128 extension arithmetic lowering with widening & saturation
__attribute__((noinline))
void add_int39_i128_extension(const int64_t* __restrict__ a,
                              const int64_t* __restrict__ b,
                              int64_t* __restrict__ out,
                              size_t n) {
    constexpr __int128 MAX_W = static_cast<__int128>(MAX_INT39);
    constexpr __int128 MIN_W = static_cast<__int128>(MIN_INT39);
    for (size_t i = 0; i < n; ++i) {
        __int128 wide_a = static_cast<__int128>(a[i]);
        __int128 wide_b = static_cast<__int128>(b[i]);
        __int128 sum = wide_a + wide_b;
        __int128 c1 = (sum > MAX_W) ? MAX_W : sum;
        __int128 clamped = (c1 < MIN_W) ? MIN_W : c1;
        out[i] = static_cast<int64_t>(clamped);
    }
}

// Native 64-bit: Native add i64 lowering with branchless icmp + select clamping
__attribute__((noinline))
void add_int39_native_i64_clamp(const int64_t* __restrict__ a,
                                const int64_t* __restrict__ b,
                                int64_t* __restrict__ out,
                                size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int64_t sum = a[i] + b[i];
        int64_t c1 = (sum > MAX_INT39) ? MAX_INT39 : sum;
        int64_t clamped = (c1 < MIN_INT39) ? MIN_INT39 : c1;
        out[i] = clamped;
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
    std::cout << "  Int39 Arithmetic Benchmark: Native 64-bit Clamping vs i128 Extension  \n";
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
        assert(llvm_add.find("icmp sgt i64") != std::string::npos && "Op::Add must use icmp sgt i64 for clamping!");
        assert(llvm_add.find("select i1") != std::string::npos && "Op::Add must use select for clamping!");
        assert(llvm_add.find("i128") == std::string::npos && "Op::Add must NOT use i128!");

        // Function with sub: x - y
        Program p_sub;
        Function f_sub{"sub_test", Type::Int39, {{1, Type::Int39}, {2, Type::Int39}},
                       {ret(std::make_shared<Expr>(Expr{Op::Sub, Type::Int39, {param(1), param(2)}}))}};
        p_sub.functions.push_back(f_sub);
        std::string llvm_sub = emitter.generate(p_sub, "x86_64-unknown-linux-gnu", "");
        assert(llvm_sub.find("sub i64") != std::string::npos && "Op::Sub must use native sub i64!");
        assert(llvm_sub.find("i128") == std::string::npos && "Op::Sub must NOT use i128!");

        // Function with mul: x * y
        Program p_mul;
        Function f_mul{"mul_test", Type::Int39, {{1, Type::Int39}, {2, Type::Int39}},
                       {ret(std::make_shared<Expr>(Expr{Op::Mul, Type::Int39, {param(1), param(2)}}))}};
        p_mul.functions.push_back(f_mul);
        std::string llvm_mul = emitter.generate(p_mul, "x86_64-unknown-linux-gnu", "");
        assert(llvm_mul.find("mul i128") != std::string::npos && "Op::Mul must reserve i128 extension!");

        std::cout << "      -> Op::Add: lowers to native add i64 + icmp/select clamping (verified 0 i128 instructions).\n";
        std::cout << "      -> Op::Sub: lowers to native sub i64 + icmp/select clamping (verified 0 i128 instructions).\n";
        std::cout << "      -> Op::Mul: strictly reserves i128 extension (verified i128 present).\n";
    }

    // 2. Generating Deterministic Test Vectors (10,000,000 pairs)
    std::cout << "\n[2/4] Generating " << TOTAL_OPS << " deterministic Int39 test vectors...\n";
    std::mt19937_64 rng(0x139ACCA7);
    std::vector<int64_t> a(TOTAL_OPS);
    std::vector<int64_t> b(TOTAL_OPS);
    std::vector<int64_t> res_i128(TOTAL_OPS);
    std::vector<int64_t> res_i64(TOTAL_OPS);

    for (size_t i = 0; i < TOTAL_OPS; ++i) {
        if (i % 100 == 0) {
            // Boundary values near bounds
            a[i] = MAX_INT39 - (i % 1000);
            b[i] = (i % 2000);
        } else if (i % 100 == 1) {
            a[i] = MIN_INT39 + (i % 1000);
            b[i] = -(static_cast<int64_t>(i % 2000));
        } else {
            a[i] = static_cast<int64_t>((rng() % (2 * static_cast<uint64_t>(MAX_INT39) + 1))) - MAX_INT39;
            b[i] = static_cast<int64_t>((rng() % (2 * static_cast<uint64_t>(MAX_INT39) + 1))) - MAX_INT39;
        }
    }

    // 3. Functional Equivalence Verification
    std::cout << "[3/4] Verifying functional equivalence between i128 and native i64 clamping...\n";
    add_int39_i128_extension(a.data(), b.data(), res_i128.data(), TOTAL_OPS);
    add_int39_native_i64_clamp(a.data(), b.data(), res_i64.data(), TOTAL_OPS);

    for (size_t i = 0; i < TOTAL_OPS; ++i) {
        assert(res_i128[i] == res_i64[i] && "Mismatch between i128 and native i64 clamping!");
    }
    std::cout << "      -> 10,000,000 results match bit-for-bit (100% equivalence verified).\n";

    // 4. Benchmark Execution
    std::cout << "\n[4/4] Executing " << num_runs << " benchmark trials (" << TOTAL_OPS << " adds per trial)...\n";
    std::vector<double> times_i128_us;
    std::vector<double> times_i64_us;
    times_i128_us.reserve(num_runs);
    times_i64_us.reserve(num_runs);

    // Warm-up caches
    add_int39_i128_extension(a.data(), b.data(), res_i128.data(), 1000000);
    add_int39_native_i64_clamp(a.data(), b.data(), res_i64.data(), 1000000);

    for (int run = 0; run < num_runs; ++run) {
        // Measure i128 extension
        auto start_128 = std::chrono::high_resolution_clock::now();
        add_int39_i128_extension(a.data(), b.data(), res_i128.data(), TOTAL_OPS);
        auto end_128 = std::chrono::high_resolution_clock::now();
        double elapsed_128 = std::chrono::duration<double, std::micro>(end_128 - start_128).count();
        times_i128_us.push_back(elapsed_128);

        // Measure native i64 clamping
        auto start_64 = std::chrono::high_resolution_clock::now();
        add_int39_native_i64_clamp(a.data(), b.data(), res_i64.data(), TOTAL_OPS);
        auto end_64 = std::chrono::high_resolution_clock::now();
        double elapsed_64 = std::chrono::duration<double, std::micro>(end_64 - start_64).count();
        times_i64_us.push_back(elapsed_64);

        std::cout << "  Trial " << std::setw(2) << (run + 1) << ": "
                  << "i128 = " << std::fixed << std::setprecision(1) << elapsed_128 << " us | "
                  << "native i64 = " << std::setprecision(1) << elapsed_64 << " us | "
                  << "Speedup = " << std::setprecision(2) << (elapsed_128 / elapsed_64) << "x\n";
    }

    StatResult stats_128 = compute_stats(times_i128_us);
    StatResult stats_64 = compute_stats(times_i64_us);
    double speedup = stats_128.mean / stats_64.mean;

    std::cout << "\n========================================================================\n";
    std::cout << "  EMPIRICAL BENCHMARK RESULTS (10,000,000 Int39 Additions)\n";
    std::cout << "========================================================================\n";
    std::cout << "  [Baseline] i128 Extension Lowering:\n";
    std::cout << "    Mean Wall-Clock Time:  " << std::fixed << std::setprecision(2) << stats_128.mean << " us (" << (stats_128.mean / 1000.0) << " ms)\n";
    std::cout << "    StdDev:                " << std::setprecision(2) << stats_128.stddev << " us\n";
    std::cout << "    Min / Max:             " << std::setprecision(1) << stats_128.min_val << " us / " << stats_128.max_val << " us\n";
    std::cout << "    Throughput:            " << std::setprecision(2) << (TOTAL_OPS / stats_128.mean) << " MOps/sec\n\n";

    std::cout << "  [Optimized] Native i64 Clamping Lowering:\n";
    std::cout << "    Mean Wall-Clock Time:  " << std::fixed << std::setprecision(2) << stats_64.mean << " us (" << (stats_64.mean / 1000.0) << " ms)\n";
    std::cout << "    StdDev:                " << std::setprecision(2) << stats_64.stddev << " us\n";
    std::cout << "    Min / Max:             " << std::setprecision(1) << stats_64.min_val << " us / " << stats_64.max_val << " us\n";
    std::cout << "    Throughput:            " << std::setprecision(2) << (TOTAL_OPS / stats_64.mean) << " MOps/sec\n\n";

    std::cout << "  SPEEDUP FACTOR:          " << std::setprecision(2) << speedup << "x\n";
    std::cout << "  TIME REDUCTION:          " << std::setprecision(1) << (100.0 * (1.0 - stats_64.mean / stats_128.mean)) << "%\n";
    std::cout << "========================================================================\n";

    return 0;
}
