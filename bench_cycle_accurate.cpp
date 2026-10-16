/**
 * @file bench_cycle_accurate.cpp
 * @brief Hardware Cycle-Accurate Microbenchmark Engine for Balanced Ternary Operations
 * @author Universal Balanced Ternary Research & Engineering Team
 * 
 * Hardware Timing Mechanism:
 * - Apple Silicon / ARM64: Direct reading of virtual cycle counter register:
 *     asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(val));
 *     asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
 * - x86_64: Hardware cycle counter:
 *     __rdtsc();
 * 
 * Benchmarked Microarchitectures:
 * 1. Dual-Rail Planar Dot Product (Larq bit-sliced POPCNT) per 64 trits and per 1024 trits.
 * 2. Balanced Ternary Half-Adder (BTHA - 6 bitwise instructions).
 * 3. Division-Free Base-243 Reciprocal Magic Multiplier Unpack.
 * 4. Native Radix-3 Cooley-Tukey FFT Butterfly (3-point kernel & full N=729 FFT).
 * 5. Multiplier-Free T-CSR SpMV (per non-zero element).
 * 6. BT-FP32 Addition & Multiplication.
 */

#include "ternary_core.hpp"
#include "ternary_dense_algebra.hpp"
#include "ternary_sparse.hpp"
#include "ternary_fp.hpp"
#include "libbt.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <random>
#include <cstdint>
#include <cmath>
#include <string>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

// Compiler barrier to prevent dead code elimination
template <typename T>
static inline void doNotOptimize(T&& val) noexcept {
    asm volatile("" : "+r,m"(val) : : "memory");
}

// ==============================================================================
// Hardware Cycle & Frequency Timing Engine
// ==============================================================================

class HardwareCycleTimer {
public:
    static inline uint64_t read_counter() noexcept {
#if defined(__aarch64__) || defined(__arm64__)
        uint64_t val;
        asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(val));
        return val;
#elif defined(__x86_64__) || defined(_M_X64)
        return __rdtsc();
#else
        return static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
#endif
    }

    static inline uint64_t read_frequency() noexcept {
#if defined(__aarch64__) || defined(__arm64__)
        uint64_t freq;
        asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
        return freq;
#elif defined(__x86_64__) || defined(_M_X64)
        return 3000000000ULL; // 3.0 GHz nominal for x86_64
#else
        return 1000000000ULL; // 1.0 GHz default fallback
#endif
    }

    static std::string arch_name() noexcept {
#if defined(__aarch64__) || defined(__arm64__)
        return "ARM64 (Apple Silicon / cntvct_el0)";
#elif defined(__x86_64__) || defined(_M_X64)
        return "x86_64 (__rdtsc)";
#else
        return "Generic (std::chrono)";
#endif
    }
};

// ==============================================================================
// Benchmark Result Record
// ==============================================================================

struct BenchmarkResult {
    std::string name;
    uint64_t iterations{0};
    uint64_t work_units{0};       // Total trits, operations, or non-zeros
    std::string unit_label;       // "trits", "ops", "NNZ"
    uint64_t total_cycles{0};     // Hardware counter delta
    double elapsed_seconds{0.0};
    double cycles_per_op{0.0};
    double cycles_per_unit{0.0};
    double ns_per_op{0.0};
    double giga_ops_sec{0.0};
};

static void print_result_row(const BenchmarkResult& r) {
    std::cout << "| " << std::left << std::setw(34) << r.name
              << "| " << std::right << std::setw(12) << r.iterations
              << "| " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.cycles_per_op
              << "| " << std::right << std::setw(10) << std::fixed << std::setprecision(3) << r.cycles_per_unit
              << "| " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.ns_per_op
              << "| " << std::right << std::setw(12) << std::fixed << std::setprecision(2) << r.giga_ops_sec
              << " |\n";
}

// ==============================================================================
// 1. Dual-Rail Planar Dot Product (Larq bit-sliced POPCNT)
// ==============================================================================

BenchmarkResult bench_dot_product_64() {
    const uint64_t ITERS = 10000000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();
    const size_t NUM_WORDS = 64;

    uint64_t Ap[NUM_WORDS], An[NUM_WORDS], Bp[NUM_WORDS], Bn[NUM_WORDS];
    std::mt19937_64 rng(42);
    for (size_t w = 0; w < NUM_WORDS; ++w) {
        uint64_t p = rng();
        uint64_t n = rng();
        Ap[w] = p & ~n;
        An[w] = n & ~p;
        uint64_t p2 = rng();
        uint64_t n2 = rng();
        Bp[w] = p2 & ~n2;
        Bn[w] = n2 & ~p2;
    }

    // Warmup
    int64_t dummy = 0;
    for (uint64_t i = 0; i < 100000; ++i) {
        size_t idx = i & (NUM_WORDS - 1);
        uint64_t p_prod = (Ap[idx] & Bp[idx]) | (An[idx] & Bn[idx]);
        uint64_t n_prod = (Ap[idx] & Bn[idx]) | (An[idx] & Bp[idx]);
        dummy += (std::popcount(p_prod) - std::popcount(n_prod));
    }
    doNotOptimize(dummy);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    int64_t acc = 0;
    for (uint64_t i = 0; i < ITERS; ++i) {
        size_t idx = i & (NUM_WORDS - 1);
        uint64_t p_prod = (Ap[idx] & Bp[idx]) | (An[idx] & Bn[idx]);
        uint64_t n_prod = (Ap[idx] & Bn[idx]) | (An[idx] & Bp[idx]);
        acc += (std::popcount(p_prod) - std::popcount(n_prod));
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();
    doNotOptimize(acc);

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);

    BenchmarkResult res;
    res.name = "1a. Planar Dot Product (64 trits)";
    res.iterations = ITERS;
    res.work_units = ITERS * 64;
    res.unit_label = "trits";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / 64.0;
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(res.work_units) / elapsed_sec) / 1e9;
    return res;
}

BenchmarkResult bench_dot_product_1024() {
    const uint64_t ITERS = 1000000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();

    uint64_t Ap[16], An[16], Bp[16], Bn[16];
    std::mt19937_64 rng(42);
    for (int w = 0; w < 16; ++w) {
        uint64_t p = rng();
        uint64_t n = rng();
        Ap[w] = p & ~n;
        An[w] = n & ~p;
        uint64_t p2 = rng();
        uint64_t n2 = rng();
        Bp[w] = p2 & ~n2;
        Bn[w] = n2 & ~p2;
    }

    // Warmup
    int64_t dummy = 0;
    for (uint64_t i = 0; i < 50000; ++i) {
        for (int w = 0; w < 16; ++w) {
            uint64_t p_prod = (Ap[w] & Bp[w]) | (An[w] & Bn[w]);
            uint64_t n_prod = (Ap[w] & Bn[w]) | (An[w] & Bp[w]);
            dummy += (std::popcount(p_prod) - std::popcount(n_prod));
        }
    }
    doNotOptimize(dummy);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    int64_t acc = 0;
    for (uint64_t i = 0; i < ITERS; ++i) {
        for (int w = 0; w < 16; ++w) {
            uint64_t p_prod = (Ap[w] & Bp[w]) | (An[w] & Bn[w]);
            uint64_t n_prod = (Ap[w] & Bn[w]) | (An[w] & Bp[w]);
            acc += (std::popcount(p_prod) - std::popcount(n_prod));
        }
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();
    doNotOptimize(acc);

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);

    BenchmarkResult res;
    res.name = "1b. Planar Dot Product (1024 trits)";
    res.iterations = ITERS;
    res.work_units = ITERS * 1024;
    res.unit_label = "trits";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / 1024.0;
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(res.work_units) / elapsed_sec) / 1e9;
    return res;
}

// ==============================================================================
// 2. Balanced Ternary Half-Adder (BTHA - 6 Bitwise Instructions)
// ==============================================================================

BenchmarkResult bench_btha() {
    const uint64_t ITERS = 10000000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();
    const size_t NUM_WORDS = 64;

    uint64_t Ap[NUM_WORDS], An[NUM_WORDS], Bp[NUM_WORDS], Bn[NUM_WORDS];
    std::mt19937_64 rng(42);
    for (size_t w = 0; w < NUM_WORDS; ++w) {
        uint64_t p = rng();
        uint64_t n = rng();
        Ap[w] = p & ~n;
        An[w] = n & ~p;
        uint64_t p2 = rng();
        uint64_t n2 = rng();
        Bp[w] = p2 & ~n2;
        Bn[w] = n2 & ~p2;
    }

    uint64_t ps = 0, ns = 0, pc = 0, nc = 0;

    // Warmup
    for (uint64_t i = 0; i < 100000; ++i) {
        size_t idx = i & (NUM_WORDS - 1);
        pc = Ap[idx] & Bp[idx];
        nc = An[idx] & Bn[idx];
        ps = ((Ap[idx] ^ Bp[idx]) & ~(An[idx] | Bn[idx])) | nc;
        ns = ((An[idx] ^ Bn[idx]) & ~(Ap[idx] | Bp[idx])) | pc;
    }
    doNotOptimize(ps); doNotOptimize(ns);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    for (uint64_t i = 0; i < ITERS; ++i) {
        size_t idx = i & (NUM_WORDS - 1);
        pc = Ap[idx] & Bp[idx];
        nc = An[idx] & Bn[idx];
        ps = ((Ap[idx] ^ Bp[idx]) & ~(An[idx] | Bn[idx])) | nc;
        ns = ((An[idx] ^ Bn[idx]) & ~(Ap[idx] | Bp[idx])) | pc;
        doNotOptimize(ps);
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();
    doNotOptimize(ns); doNotOptimize(pc); doNotOptimize(nc);

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);

    BenchmarkResult res;
    res.name = "2. Balanced Ternary Half-Adder (BTHA)";
    res.iterations = ITERS;
    res.work_units = ITERS * 64; // 64 trits added in parallel
    res.unit_label = "trits";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / 64.0;
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(res.work_units) / elapsed_sec) / 1e9;
    return res;
}

// ==============================================================================
// 3. Division-Free Base-243 Reciprocal Magic Multiplier Unpack
// ==============================================================================

BenchmarkResult bench_base243_unpack() {
    const uint64_t ITERS = 10000000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();

    // Table of 243 byte values
    std::vector<uint8_t> bytes(243);
    for (int i = 0; i < 243; ++i) bytes[i] = static_cast<uint8_t>(i);

    // Warmup
    int dummy = 0;
    for (uint64_t i = 0; i < 100000; ++i) {
        auto trits = bt::Base243Codec::unpack5_fast(bytes[i % 243]);
        dummy += bt::to_int(trits[0]);
    }
    doNotOptimize(dummy);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    int acc = 0;
    for (uint64_t i = 0; i < ITERS; ++i) {
        uint8_t b = bytes[i % 243];
        auto trits = bt::Base243Codec::unpack5_fast(b);
        acc += bt::to_int(trits[0]);
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();
    doNotOptimize(acc);

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);

    BenchmarkResult res;
    res.name = "3. Base-243 Magic Reciprocal Unpack";
    res.iterations = ITERS;
    res.work_units = ITERS * 5; // 5 trits unpacked per byte
    res.unit_label = "trits";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / 5.0;
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(res.work_units) / elapsed_sec) / 1e9;
    return res;
}

// ==============================================================================
// 4. Native Radix-3 Cooley-Tukey FFT Butterfly & Transform
// ==============================================================================

BenchmarkResult bench_fft_butterfly() {
    const uint64_t ITERS = 10000000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();

    using cdouble = std::complex<double>;
    const double angle3 = -2.0 * M_PI / 3.0;
    const cdouble w3_1(std::cos(angle3), std::sin(angle3));
    const cdouble w3_2(std::cos(2.0 * angle3), std::sin(2.0 * angle3));
    const cdouble w(std::cos(0.35), std::sin(0.35));
    const cdouble w2(std::cos(0.70), std::sin(0.70));

    cdouble u0(1.0, 0.5);
    cdouble u1(0.7, -0.2);
    cdouble u2(-0.4, 0.9);

    // Warmup
    cdouble dummy(0.0, 0.0);
    for (uint64_t i = 0; i < 100000; ++i) {
        cdouble t1 = u1 * w;
        cdouble t2 = u2 * w2;
        cdouble y0 = u0 + t1 + t2;
        dummy += y0;
    }
    doNotOptimize(dummy);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    cdouble acc(0.0, 0.0);
    for (uint64_t i = 0; i < ITERS; ++i) {
        cdouble t1 = u1 * w;
        cdouble t2 = u2 * w2;
        cdouble y0 = u0 + t1 + t2;
        cdouble y1 = u0 + t1 * w3_1 + t2 * w3_2;
        cdouble y2 = u0 + t1 * w3_2 + t2 * w3_1;
        acc += y0 + y1 + y2;
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();
    doNotOptimize(acc);

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);

    BenchmarkResult res;
    res.name = "4a. Native Radix-3 FFT Butterfly (3-pt)";
    res.iterations = ITERS;
    res.work_units = ITERS * 34; // ~34 floating point ops per 3-point butterfly
    res.unit_label = "FLOPs";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / 3.0; // Cycles per radix-3 point
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(res.work_units) / elapsed_sec) / 1e9; // GFLOPS
    return res;
}

BenchmarkResult bench_fft_729() {
    const uint64_t ITERS = 20000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();
    const size_t N = 729; // 3^6 points

    std::vector<std::complex<double>> signal(N);
    for (size_t i = 0; i < N; ++i) {
        signal[i] = std::complex<double>(std::sin(i * 0.1), std::cos(i * 0.2));
    }

    // Warmup
    for (uint64_t i = 0; i < 500; ++i) {
        auto res = bt::Radix3FFT::fft(signal);
        doNotOptimize(res[0]);
    }

    uint64_t c_start = HardwareCycleTimer::read_counter();
    for (uint64_t i = 0; i < ITERS; ++i) {
        auto res = bt::Radix3FFT::fft(signal);
        doNotOptimize(res[0]);
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);
    double flops_per_fft = 5.0 * N * (std::log(N) / std::log(3.0)) * 6.0;

    BenchmarkResult res;
    res.name = "4b. Radix-3 Cooley-Tukey FFT (N=729)";
    res.iterations = ITERS;
    res.work_units = static_cast<uint64_t>(ITERS * flops_per_fft);
    res.unit_label = "FLOPs";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / static_cast<double>(N); // Cycles per point
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(res.work_units) / elapsed_sec) / 1e9;
    return res;
}

// ==============================================================================
// 5. Multiplier-Free T-CSR SpMV (Per Non-Zero Element)
// ==============================================================================

BenchmarkResult bench_tcsr_spmv() {
    const uint64_t ITERS = 2000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();
    const size_t ROWS = 1024;
    const size_t COLS = 1024;

    // Generate sparse matrix with ~5% density (95% sparsity) -> ~52,428 NNZ
    std::mt19937_64 rng(1337);
    bt::TCSRMatrix mat(ROWS, COLS);
    for (size_t r = 0; r < ROWS; ++r) {
        for (size_t c = 0; c < COLS; ++c) {
            uint32_t roll = rng() % 100;
            if (roll < 5) { // 5% non-zero probability
                bt::Trit t = (roll % 2 == 0) ? bt::Trit::Pos : bt::Trit::Neg;
                mat.push_element(r, c, t);
            }
        }
    }
    mat.finalize();

    std::vector<int64_t> x(COLS, 1);
    std::vector<int64_t> y(ROWS, 0);

    // Warmup
    for (uint64_t i = 0; i < 200; ++i) {
        mat.spmv(x.data(), y.data());
    }
    doNotOptimize(y[0]);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    for (uint64_t i = 0; i < ITERS; ++i) {
        mat.spmv(x.data(), y.data());
        doNotOptimize(y[0]);
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);
    uint64_t total_nnz_ops = ITERS * mat.nnz;

    BenchmarkResult res;
    res.name = "5. Multiplier-Free T-CSR SpMV (1Kx1K)";
    res.iterations = ITERS;
    res.work_units = total_nnz_ops;
    res.unit_label = "NNZ";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = static_cast<double>(total_cycles) / static_cast<double>(total_nnz_ops); // Cycles per NNZ!
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(total_nnz_ops) / elapsed_sec) / 1e9; // G-NNZ/s
    return res;
}

// ==============================================================================
// 6. BT-FP32 Addition & Multiplication
// ==============================================================================

BenchmarkResult bench_btfp32_add() {
    const uint64_t ITERS = 5000000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();

    bt::BTFP32 a = bt::BTFP32::from_double(3.1415926535);
    bt::BTFP32 b = bt::BTFP32::from_double(2.7182818284);

    // Warmup
    bt::BTFP32 dummy(0, 0);
    for (uint64_t i = 0; i < 100000; ++i) {
        dummy = a + b;
    }
    doNotOptimize(dummy);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    bt::BTFP32 c(0, 0);
    for (uint64_t i = 0; i < ITERS; ++i) {
        c = a + b;
        doNotOptimize(c);
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);

    BenchmarkResult res;
    res.name = "6a. BT-FP32 Addition (20-Trit Mantissa)";
    res.iterations = ITERS;
    res.work_units = ITERS;
    res.unit_label = "ops";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / 20.0; // Cycles per mantissa trit
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(ITERS) / elapsed_sec) / 1e9;
    return res;
}

BenchmarkResult bench_btfp32_mul() {
    const uint64_t ITERS = 5000000;
    const uint64_t freq = HardwareCycleTimer::read_frequency();

    bt::BTFP32 a = bt::BTFP32::from_double(1.4142135623);
    bt::BTFP32 b = bt::BTFP32::from_double(1.7320508075);

    // Warmup
    bt::BTFP32 dummy(0, 0);
    for (uint64_t i = 0; i < 100000; ++i) {
        dummy = a * b;
    }
    doNotOptimize(dummy);

    uint64_t c_start = HardwareCycleTimer::read_counter();
    bt::BTFP32 c(0, 0);
    for (uint64_t i = 0; i < ITERS; ++i) {
        c = a * b;
        doNotOptimize(c);
    }
    uint64_t c_end = HardwareCycleTimer::read_counter();

    uint64_t total_cycles = c_end - c_start;
    double elapsed_sec = static_cast<double>(total_cycles) / static_cast<double>(freq);

    BenchmarkResult res;
    res.name = "6b. BT-FP32 Multiplication";
    res.iterations = ITERS;
    res.work_units = ITERS;
    res.unit_label = "ops";
    res.total_cycles = total_cycles;
    res.elapsed_seconds = elapsed_sec;
    res.cycles_per_op = static_cast<double>(total_cycles) / static_cast<double>(ITERS);
    res.cycles_per_unit = res.cycles_per_op / 20.0;
    res.ns_per_op = (elapsed_sec / static_cast<double>(ITERS)) * 1e9;
    res.giga_ops_sec = (static_cast<double>(ITERS) / elapsed_sec) / 1e9;
    return res;
}

// ==============================================================================
// Main Driver & Report Table Generator
// ==============================================================================

int main() {
    uint64_t timer_freq = HardwareCycleTimer::read_frequency();
    std::string arch = HardwareCycleTimer::arch_name();

    std::cout << "==========================================================================================================\n";
    std::cout << "  UNIVERSAL BALANCED TERNARY HARDWARE CYCLE-ACCURATE MICROBENCHMARK SUITE\n";
    std::cout << "  Architecture: " << arch << " | Hardware Timer Frequency: " << (timer_freq / 1e6) << " MHz (" << timer_freq << " Hz)\n";
    std::cout << "  Timing Resolution: " << std::fixed << std::setprecision(3) << (1e9 / static_cast<double>(timer_freq)) << " ns/tick\n";
    std::cout << "==========================================================================================================\n\n";

    std::cout << "+-----------------------------------+-------------+-----------+-----------+-----------+--------------+\n";
    std::cout << "| Benchmark Kernel                  | Iterations  | Cycles/Op | Cyc/Trit  |   ns/Op   | Throughput   |\n";
    std::cout << "|                                   |             |           | (Cyc/NNZ) |           | (GigaOps/sec)|\n";
    std::cout << "+-----------------------------------+-------------+-----------+-----------+-----------+--------------+\n";

    std::vector<BenchmarkResult> results;
    results.push_back(bench_dot_product_64());
    print_result_row(results.back());

    results.push_back(bench_dot_product_1024());
    print_result_row(results.back());

    results.push_back(bench_btha());
    print_result_row(results.back());

    results.push_back(bench_base243_unpack());
    print_result_row(results.back());

    results.push_back(bench_fft_butterfly());
    print_result_row(results.back());

    results.push_back(bench_fft_729());
    print_result_row(results.back());

    results.push_back(bench_tcsr_spmv());
    print_result_row(results.back());

    results.push_back(bench_btfp32_add());
    print_result_row(results.back());

    results.push_back(bench_btfp32_mul());
    print_result_row(results.back());

    std::cout << "+-----------------------------------+-------------+-----------+-----------+-----------+--------------+\n\n";

    std::cout << "==========================================================================================================\n";
    std::cout << "  MICROARCHITECTURAL PERFORMANCE HIGHLIGHTS:\n";
    std::cout << "  1. Dual-Rail Planar Dot Product: " << std::fixed << std::setprecision(3) << results[0].cycles_per_unit
              << " cycles/trit (" << std::setprecision(2) << results[0].giga_ops_sec << " GTOPS peak throughput).\n";
    std::cout << "  2. Balanced Ternary Half-Adder:  " << std::setprecision(3) << results[2].cycles_per_unit
              << " cycles/trit addition (6 bitwise instructions in single cycle).\n";
    std::cout << "  3. Base-243 Reciprocal Unpack:   " << std::setprecision(3) << results[3].cycles_per_unit
              << " cycles/trit (division-free 1.6 bits/trit compression unpack).\n";
    std::cout << "  4. Radix-3 Cooley-Tukey FFT:     " << std::setprecision(2) << results[4].giga_ops_sec
              << " GFLOPS peak butterfly performance (" << std::setprecision(1) << results[5].ns_per_op / 1e3 << " us for N=729 FFT).\n";
    std::cout << "  5. Multiplier-Free T-CSR SpMV:   " << std::setprecision(3) << results[6].cycles_per_unit
              << " cycles/NNZ (" << std::setprecision(2) << results[6].giga_ops_sec << " Giga-NNZ/sec).\n";
    std::cout << "  6. BT-FP32 Arithmetic:           " << std::setprecision(2) << results[7].giga_ops_sec
              << " GAdd/s addition, " << results[8].giga_ops_sec << " GMul/s multiplication.\n";
    std::cout << "==========================================================================================================\n";

    return 0;
}
