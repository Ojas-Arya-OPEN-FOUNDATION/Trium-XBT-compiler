/**
 * @file bench_dense_algebra.cpp
 * @brief High-Resolution Microbenchmark Suite for Balanced Ternary Dense Algebra
 * @author Antigravity Prime Engineering Engine
 */

#include "ternary_dense_algebra.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

using namespace bt;
using namespace std::chrono;

TritWord64 randomTritWord(std::mt19937_64& rng) {
    uint64_t raw_p = rng();
    uint64_t raw_n = rng();
    return TritWord64(raw_p & ~raw_n, raw_n & ~raw_p);
}

void bench_planar_gemm() {
    std::cout << "--- 1. Bit-Sliced Planar GEMM Benchmark (Hardware POPCNT) ---\n";
    size_t M = 128;
    size_t K = 512;
    size_t N = 128;

    std::mt19937_64 rng(42);
    TernaryMatrix A(M, K);
    TernaryMatrix B(K, N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int r = rng() % 3;
            A.set(i, k, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            int r = rng() % 3;
            B.set(k, j, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
        }
    }

    DenseMatrixInt64 C(M, N);

    // Warmup
    gemm(A, B, C);

    int iters = 200;
    auto start = high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        gemm(A, B, C);
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration<double, std::milli>(end - start).count();
    double total_macs = static_cast<double>(iters) * M * K * N; // 200 * 128 * 512 * 128 = 1.677 Billion MACs
    double gtops = (total_macs / (elapsed_ms / 1000.0)) / 1e9;

    std::cout << "  Matrix size: (" << M << " x " << K << ") * (" << K << " x " << N << ")\n";
    std::cout << "  Total Workload: " << std::fixed << std::setprecision(2) << (total_macs / 1e6) << " Million Trit-MACs\n";
    std::cout << "  Elapsed Time:   " << elapsed_ms << " ms (" << (elapsed_ms / iters) << " ms/iter)\n";
    std::cout << "  Throughput:     " << "\033[32m\033[1m" << gtops << " GTOPS\033[0m (Giga-Trit Operations/Sec)\n\n";
}

void bench_tiled_gemm() {
    std::cout << "--- 2. Tiled Block GEMM Benchmark (64x64 Cache Tile) ---\n";
    size_t M = 256;
    size_t K = 512;
    size_t N = 256;

    std::mt19937_64 rng(101);
    TernaryMatrix A(M, K);
    TernaryMatrix B(K, N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int r = rng() % 3;
            A.set(i, k, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            int r = rng() % 3;
            B.set(k, j, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
        }
    }

    DenseMatrixInt64 C(M, N);
    gemm_tiled(A, B, C, 64); // Warmup

    int iters = 50;
    auto start = high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        gemm_tiled(A, B, C, 64);
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration<double, std::milli>(end - start).count();
    double total_macs = static_cast<double>(iters) * M * K * N; // 50 * 256 * 512 * 256 = 1.677 Billion MACs
    double gtops = (total_macs / (elapsed_ms / 1000.0)) / 1e9;

    std::cout << "  Matrix size: (" << M << " x " << K << ") * (" << K << " x " << N << ")\n";
    std::cout << "  Throughput:     " << "\033[32m\033[1m" << gtops << " GTOPS\033[0m\n\n";
}

void bench_cacheline_tiled_l1_l2() {
    std::cout << "--- 2b. Cache-Line Tiled L1/L2 GEMM Engine (gemm_tiled_l1_l2 4x2 Microkernel) ---\n";
    size_t M = 256;
    size_t K = 512;
    size_t N = 256;

    std::mt19937_64 rng(101);
    TernaryMatrix A(M, K);
    TernaryMatrix B(K, N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int r = rng() % 3;
            A.set(i, k, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            int r = rng() % 3;
            B.set(k, j, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
        }
    }

    DenseMatrixInt64 C(M, N);
    gemm_tiled_l1_l2(A, B, C, 64, 64, 16); // Warmup

    int iters = 50;
    auto start = high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        gemm_tiled_l1_l2(A, B, C, 64, 64, 16);
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration<double, std::milli>(end - start).count();
    double total_macs = static_cast<double>(iters) * M * K * N;
    double gtops = (total_macs / (elapsed_ms / 1000.0)) / 1e9;

    std::cout << "  Matrix size: (" << M << " x " << K << ") * (" << K << " x " << N << ")\n";
    std::cout << "  Throughput:     " << "\033[32m\033[1m" << gtops << " GTOPS\033[0m\n\n";
}

void bench_csa_compressor() {
    std::cout << "--- 3. 4-to-2 BT-CSA Compressor Tree Benchmark ---\n";
    std::mt19937_64 rng(2026);
    TritWord64 a = randomTritWord(rng);
    TritWord64 b = randomTritWord(rng);
    TritWord64 c = randomTritWord(rng);
    TritWord64 d = randomTritWord(rng);

    AdderOutput out = BT_Compress4to2(a, b, c, d);

    int iters = 20'000'000;
    auto start = high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        out = BT_Compress4to2(a, b, c, d);
        a.P ^= out.Sum.P & 1; // prevent compiler from optimizing loop away
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration<double, std::milli>(end - start).count();
    double mtops = (static_cast<double>(iters) * 64 * 4 / (elapsed_ms / 1000.0)) / 1e6;

    std::cout << "  Iterations: " << iters << " (4 x 64 trits per call)\n";
    std::cout << "  Elapsed:    " << elapsed_ms << " ms\n";
    std::cout << "  Rate:       " << "\033[32m\033[1m" << mtops << " MTOPS\033[0m (" << (mtops / 1000.0) << " GTOPS)\n\n";
}

void bench_branchless_select() {
    std::cout << "--- 4. Branchless 3-Way Conditional Select Benchmark ---\n";
    std::mt19937_64 rng(777);
    TritWord64 cond = randomTritWord(rng);
    TritWord64 p = randomTritWord(rng);
    TritWord64 z = randomTritWord(rng);
    TritWord64 n = randomTritWord(rng);

    TritWord64 res = ternary_select(cond, p, z, n);

    int iters = 50'000'000;
    auto start = high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        res = ternary_select(cond, p, z, n);
        cond.P ^= res.P & 1;
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration<double, std::milli>(end - start).count();
    double mtops = (static_cast<double>(iters) * 64 / (elapsed_ms / 1000.0)) / 1e6;

    std::cout << "  Iterations: " << iters << " (64 trits per select)\n";
    std::cout << "  Rate:       " << "\033[32m\033[1m" << mtops << " M-Selections/sec\033[0m (" << (mtops / 1000.0) << " Billion/sec)\n\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  Hardware Microbenchmark: Balanced Ternary Dense Algebra Engine\n";
    std::cout << "  Target: Apple Silicon / ARM64 (Single Core, 100% Vectorized Bitwise)\n";
    std::cout << "======================================================================\n\n";

    bench_planar_gemm();
    bench_tiled_gemm();
    bench_cacheline_tiled_l1_l2();
    bench_csa_compressor();
    bench_branchless_select();

    std::cout << "======================================================================\n";
    std::cout << "  BENCHMARKS COMPLETE.\n";
    std::cout << "======================================================================\n";
    return 0;
}
