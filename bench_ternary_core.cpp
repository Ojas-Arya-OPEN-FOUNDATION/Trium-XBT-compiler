/**
 * @file bench_ternary_core.cpp
 * @brief High-Resolution Performance Benchmarking Harness for ternary_core.hpp
 * @author Antigravity Prime Research Engine
 */

#include "ternary_core.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  Production-Grade Ternary Core Hardware Benchmark (Apple Silicon)\n";
    std::cout << "=================================================================\n";

    using namespace bt;
    std::mt19937_64 rng(1337);

    // 1. Benchmark Bit-Sliced POPCNT Dot Product (64 trits per vector)
    const size_t NUM_DOTS = 5000000; // 5 Million dot products = 320 Million trit operations
    std::vector<TritWord64> vecA(NUM_DOTS);
    std::vector<TritWord64> vecB(NUM_DOTS);
    for (size_t i = 0; i < NUM_DOTS; ++i) {
        uint64_t m1 = rng(), m2 = rng();
        vecA[i] = {m1 & ~m2, ~m1 & m2};
        uint64_t m3 = rng(), m4 = rng();
        vecB[i] = {m3 & ~m4, ~m3 & m4};
    }

    auto start_dot = std::chrono::high_resolution_clock::now();
    int64_t accum_dot = 0;
    for (size_t i = 0; i < NUM_DOTS; ++i) {
        accum_dot += dotProduct(vecA[i], vecB[i]);
    }
    auto end_dot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_dot = end_dot - start_dot;

    double tops_dot = (NUM_DOTS * 64.0) / (ms_dot.count() / 1000.0) / 1e9;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[BENCHMARK 1] Bit-Sliced POPCNT GEMM Dot-Product (64 trits/op):\n";
    std::cout << "  Iterations: " << NUM_DOTS << " operations\n";
    std::cout << "  Time:       " << ms_dot.count() << " ms\n";
    std::cout << "  Throughput: " << tops_dot << " Giga-Trit-Ops/sec (GTOPS) on 1 CPU Core!\n";
    std::cout << "  Accumulated Checksum: " << accum_dot << "\n\n";

    // 2. Benchmark BT-CSA Carry-Save Accumulator (3-to-2 Compression in O(1))
    const size_t NUM_CSA = 10000000; // 10 Million words accumulated
    BT_CSA csa_acc;
    TritWord64 sample_input(0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL);

    auto start_csa = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_CSA; ++i) {
        csa_acc.accumulate(sample_input);
    }
    TritWord64 canon = csa_acc.canonicalize();
    auto end_csa = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_csa = end_csa - start_csa;

    double mops_csa = (NUM_CSA * 64.0) / (ms_csa.count() / 1000.0) / 1e6;
    std::cout << "[BENCHMARK 2] BT-CSA Carry-Save Accumulator (O(1) Ripple-Free Accumulation):\n";
    std::cout << "  Iterations: " << NUM_CSA << " 64-trit words\n";
    std::cout << "  Time:       " << ms_csa.count() << " ms\n";
    std::cout << "  Throughput: " << mops_csa << " Million Trit-Additions/sec (MTOPS)\n";
    std::cout << "  Canonical P: 0x" << std::hex << canon.P << " N: 0x" << canon.N << std::dec << "\n\n";

    // 3. Benchmark Base-243 Division-Free Unpacking (Magic Reciprocal Multipliers)
    const size_t NUM_BYTES = 10000000; // 10 Million bytes unpacked = 50 Million trits
    std::vector<uint8_t> packed_bytes(NUM_BYTES);
    for (size_t i = 0; i < NUM_BYTES; ++i) {
        packed_bytes[i] = static_cast<uint8_t>(rng() % 243);
    }

    auto start_unpack = std::chrono::high_resolution_clock::now();
    int64_t dummy_sum = 0;
    for (size_t i = 0; i < NUM_BYTES; ++i) {
        auto trits = Base243Codec::unpack5_fast(packed_bytes[i]);
        dummy_sum += to_int(trits[0]);
    }
    auto end_unpack = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_unpack = end_unpack - start_unpack;

    double mtrits_sec = (NUM_BYTES * 5.0) / (ms_unpack.count() / 1000.0) / 1e6;
    std::cout << "[BENCHMARK 3] Base-243 Division-Free Magic Reciprocal Unpacking:\n";
    std::cout << "  Processed:  " << NUM_BYTES << " bytes (" << (NUM_BYTES * 5) << " trits)\n";
    std::cout << "  Time:       " << ms_unpack.count() << " ms\n";
    std::cout << "  Throughput: " << mtrits_sec << " Million Trits Decompressed/sec\n";
    std::cout << "  Checksum:   " << dummy_sum << "\n";
    std::cout << "=================================================================\n";

    return 0;
}
