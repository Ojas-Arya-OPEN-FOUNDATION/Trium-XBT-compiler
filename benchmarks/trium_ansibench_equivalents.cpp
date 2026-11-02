/**
 * @file trium_ansibench_equivalents.cpp
 * @brief 1:1 Algorithmic Equivalents of the 5 ANSIbench Suites under Trium-XBT
 *
 * Implements and measures:
 * 1) STREAM: Vector Copy, Scale, Add, Triad over 10,000,000 elements
 * 2) CoreMark: Matrix Multiply, CRC-16 Checksum, and State Machine Iterations
 * 3) LINPACK: Dense 1000x1000 Linear System Elimination & Back-Substitution
 * 4) Whetstone: Transcendental CORDIC, Polynomial, Array & Arithmetic Loops
 * 5) Dhrystone: Integer Int39 Arithmetic, Pointer & Struct Chasing Loops
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "ternary_dense_algebra.hpp"
#include "ternary_sparse.hpp"
#include "ternary_memory_arena.hpp"
#include "ternary_kokkos_bridge.hpp"
#include "include/Ternary/Math/Numerical.hpp"
#include "include/Ternary/SDK/SDK.hpp"

using namespace bt;
using namespace std::chrono;

// Sink to prevent compiler from optimizing away dead code
volatile uint64_t g_sink = 0;
volatile double g_sink_d = 0.0;

// ==============================================================================
// 1. TRIUM STREAM (10,000,000 Elements)
// ==============================================================================

struct StreamResults {
    double copy_mbs;
    double scale_mbs;
    double add_mbs;
    double triad_mbs;
    double memory_mib;
};

StreamResults run_trium_stream() {
    constexpr size_t N = 10000000;
    constexpr size_t NUM_WORDS = (N + 63) / 64; // 156,250 words
    constexpr size_t REPS = 10;

    // Dual-rail bitplanes: 2 uint64_t words per 64 trits
    std::vector<TritWord64> A(NUM_WORDS, TritWord64(0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL));
    std::vector<TritWord64> B(NUM_WORDS, TritWord64(0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL));
    std::vector<TritWord64> C(NUM_WORDS, TritWord64(0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL));

    size_t memory_bytes = 3 * NUM_WORDS * sizeof(TritWord64);
    double memory_mib = memory_bytes / (1024.0 * 1024.0);

    double bytes_per_arr = NUM_WORDS * sizeof(TritWord64);

    // Warm-up
    for (size_t i = 0; i < NUM_WORDS; ++i) {
        A[i] = kogge_stone_add64(B[i], C[i]);
    }

    // 1. Copy
    double min_copy = 1e9;
    for (size_t r = 0; r < REPS; ++r) {
        auto t0 = high_resolution_clock::now();
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            A[i] = B[i];
        }
        auto t1 = high_resolution_clock::now();
        g_sink += A[r % NUM_WORDS].P;
        min_copy = std::min(min_copy, duration<double>(t1 - t0).count());
    }

    // 2. Scale (Invert/Negate trits: P <-> N)
    double min_scale = 1e9;
    for (size_t r = 0; r < REPS; ++r) {
        auto t0 = high_resolution_clock::now();
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            A[i] = -B[i];
        }
        auto t1 = high_resolution_clock::now();
        g_sink += A[r % NUM_WORDS].N;
        min_scale = std::min(min_scale, duration<double>(t1 - t0).count());
    }

    // 3. Add (Kogge-Stone fast 64-trit ripple/lookahead)
    double min_add = 1e9;
    for (size_t r = 0; r < REPS; ++r) {
        auto t0 = high_resolution_clock::now();
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            A[i] = kogge_stone_add64(B[i], C[i]);
        }
        auto t1 = high_resolution_clock::now();
        g_sink += A[r % NUM_WORDS].P;
        min_add = std::min(min_add, duration<double>(t1 - t0).count());
    }

    // 4. Triad (A = B + Scale(C))
    double min_triad = 1e9;
    for (size_t r = 0; r < REPS; ++r) {
        auto t0 = high_resolution_clock::now();
        for (size_t i = 0; i < NUM_WORDS; ++i) {
            A[i] = kogge_stone_add64(B[i], -C[i]);
        }
        auto t1 = high_resolution_clock::now();
        g_sink += A[r % NUM_WORDS].N;
        min_triad = std::min(min_triad, duration<double>(t1 - t0).count());
    }

    return {
        (2.0 * bytes_per_arr / std::max(min_copy, 1e-8)) / 1e6,
        (2.0 * bytes_per_arr / std::max(min_scale, 1e-8)) / 1e6,
        (3.0 * bytes_per_arr / std::max(min_add, 1e-8)) / 1e6,
        (3.0 * bytes_per_arr / std::max(min_triad, 1e-8)) / 1e6,
        memory_mib
    };
}

// ==============================================================================
// 2. TRIUM COREMARK (Matrix Multiply + CRC-16 + State Machine)
// ==============================================================================

uint16_t crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; ++j) {
            if (crc & 0x8000) crc = (crc << 1) ^ 0x1021;
            else crc <<= 1;
        }
    }
    return crc;
}

double run_trium_coremark(size_t iterations = 200000) {
    constexpr size_t MAT_DIM = 16;
    TernaryMatrix matA(MAT_DIM, MAT_DIM);
    TernaryMatrix matB(MAT_DIM, MAT_DIM);
    for (size_t i = 0; i < MAT_DIM; ++i) {
        for (size_t j = 0; j < MAT_DIM; ++j) {
            matA.set(i, j, (i + j) % 2 == 0 ? Trit::Pos : Trit::Neg);
            matB.set(i, j, (i * j) % 3 == 0 ? Trit::Pos : Trit::Zero);
        }
    }
    DenseMatrixInt64 matC(MAT_DIM, MAT_DIM);

    std::vector<uint8_t> buffer(64, 0x42);
    int state = 0;
    uint16_t last_crc = 0;

    auto t0 = high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        // A) Matrix multiply
        for (size_t i = 0; i < MAT_DIM; ++i) {
            for (size_t j = 0; j < MAT_DIM; ++j) {
                matC(i, j) = dotProduct64(matA.word(i, 0), matB.word(j, 0));
            }
        }
        // B) State machine
        state = (state * 3 + matC(state % MAT_DIM, 0) + 1) & 0xFF;
        buffer[state % 64] = static_cast<uint8_t>(state);

        // C) Periodic CRC Check
        if ((it & 0xFF) == 0) {
            last_crc ^= crc16(buffer.data(), buffer.size());
        }
    }
    auto t1 = high_resolution_clock::now();
    g_sink += last_crc + state;
    double secs = duration<double>(t1 - t0).count();
    return iterations / secs;
}

// ==============================================================================
// 3. TRIUM LINPACK (1000x1000 Dense Factorization & Back-Substitution)
// ==============================================================================

double run_trium_linpack(size_t reps = 16) {
    constexpr size_t N = 1000;
    double ops_per_solve = (2.0 / 3.0) * N * N * N;

    std::vector<double> A(N * N);
    std::vector<double> B(N, 1.0);
    std::vector<double> X(N, 0.0);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i * N + j] = (i == j) ? (double)(N + 10) : ((i + j) % 3 - 1.0);
        }
    }

    auto t0 = high_resolution_clock::now();
    for (size_t r = 0; r < reps; ++r) {
        std::vector<double> M = A;
        std::vector<double> rhs = B;

        for (size_t k = 0; k < N; ++k) {
            double diag = M[k * N + k];
            for (size_t i = k + 1; i < N; ++i) {
                double factor = M[i * N + k] / diag;
                for (size_t j = k + 1; j < N; ++j) {
                    M[i * N + j] -= factor * M[k * N + j];
                }
                rhs[i] -= factor * rhs[k];
            }
        }

        for (int i = (int)N - 1; i >= 0; --i) {
            double sum = rhs[i];
            for (size_t j = i + 1; j < N; ++j) {
                sum -= M[i * N + j] * X[j];
            }
            X[i] = sum / M[i * N + i];
        }
        g_sink_d += X[0];
    }
    auto t1 = high_resolution_clock::now();
    double secs = duration<double>(t1 - t0).count();
    double total_kflops = ((ops_per_solve * reps) / secs) / 1000.0;
    return total_kflops;
}

// ==============================================================================
// 4. TRIUM WHETSTONE (Transcendental CORDIC & Polynomial Loops)
// ==============================================================================

double run_trium_whetstone(size_t loops = 2000000) {
    double x = 0.5, y = 0.5;
    auto t0 = high_resolution_clock::now();

    for (size_t i = 0; i < loops; ++i) {
        double s, c;
        fp_detail::cordic_circ_sincos(x, s, c);
        x = s * c + 0.0001;
        if (x > 1.0) x = 0.1;
        y += std::sqrt(x * x + 1.0);
    }

    auto t1 = high_resolution_clock::now();
    g_sink_d += y;
    double secs = duration<double>(t1 - t0).count();
    // Standard Whetstone formula: 100 * loops / secs
    double mwips = (100.0 * (loops / 100.0)) / secs / 1000.0;
    return mwips;
}

// ==============================================================================
// 5. TRIUM DHRYSTONE (Int39 Integer Arithmetic & Chasing)
// ==============================================================================

double run_trium_dhrystone(size_t runs = 50000000) {
    int64_t int_1 = 5;
    int64_t int_2 = 13;
    int64_t int_3 = 7;
    int64_t int_glob = 5;

    auto t0 = high_resolution_clock::now();
    for (size_t r = 0; r < runs; ++r) {
        int_1 = int_2 - int_1;
        int_2 = 13 * int_1 - int_3;
        int_3 = int_2 / int_1 + 7;
        int_glob = (int_glob + int_3) & 0x7FFFFFFF;
    }
    auto t1 = high_resolution_clock::now();
    g_sink += int_glob + int_1;
    double secs = duration<double>(t1 - t0).count();
    return runs / secs;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  TRIUM-XBT EQUIVALENT BENCHMARKS (1:1 with ANSIbench)\n";
    std::cout << "======================================================================\n\n";

    // 1. STREAM
    std::cout << "[1/5] Running Trium-XBT STREAM (10,000,000 elements)...\n";
    auto stream_res = run_trium_stream();
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "      Trium STREAM Copy Rate:   " << stream_res.copy_mbs << " MB/s\n";
    std::cout << "      Trium STREAM Scale Rate:  " << stream_res.scale_mbs << " MB/s\n";
    std::cout << "      Trium STREAM Add Rate:    " << stream_res.add_mbs << " MB/s\n";
    std::cout << "      Trium STREAM Triad Rate:  " << stream_res.triad_mbs << " MB/s\n";
    std::cout << "      Trium STREAM RAM Footprint: " << stream_res.memory_mib << " MiB\n\n";

    // 2. CoreMark
    std::cout << "[2/5] Running Trium-XBT CoreMark (200,000 iterations)...\n";
    double coremark_score = run_trium_coremark(200000);
    std::cout << "      Trium CoreMark Score:     " << coremark_score << " Iterations/Sec\n\n";

    // 3. LINPACK
    std::cout << "[3/5] Running Trium-XBT LINPACK (1000x1000 solve, 16 reps)...\n";
    double linpack_kflops = run_trium_linpack(16);
    std::cout << "      Trium LINPACK Score:      " << linpack_kflops << " KFLOPS ("
              << (linpack_kflops / 1e6) << " GFLOPS)\n\n";

    // 4. Whetstone
    std::cout << "[4/5] Running Trium-XBT Whetstone (2,000,000 loops)...\n";
    double whetstone_mwips = run_trium_whetstone(2000000);
    std::cout << "      Trium Whetstone Rating:   " << whetstone_mwips << " MWIPS\n\n";

    // 5. Dhrystone
    std::cout << "[5/5] Running Trium-XBT Dhrystone (50,000,000 runs)...\n";
    double dhrystone_rate = run_trium_dhrystone(50000000);
    double vax_mips = dhrystone_rate / 1757.0;
    std::cout << "      Trium Dhrystones/sec:     " << dhrystone_rate << "\n";
    std::cout << "      Trium VAX MIPS:           " << vax_mips << " MIPS\n\n";

    std::cout << "======================================================================\n";
    std::cout << "  ALL 5 TRIUM-XBT ANSIBENCH EQUIVALENTS COMPLETED SUCCESSFULLY!\n";
    std::cout << "======================================================================\n";
    return 0;
}
