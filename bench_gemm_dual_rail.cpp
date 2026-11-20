/**
 * @file bench_gemm_dual_rail.cpp
 * @brief Controlled GEMM Benchmark: NEON vdotq_s32 INT8 vs Dual-Rail (P, N) + Harley-Seal CSA
 *
 * Compares on M=256, K=512, N=256 (pre-packed weights B^T, dynamic int8_t activations A):
 *   1. INT8 NEON vdotq_s32 GEMM (competent hardware baseline)
 *   2. Dual-Rail (P, N) Kernel-Only GEMM (POPCNT)
 *   3. Dual-Rail (P, N) End-to-End GEMM (NEON SIMD pack of A + POPCNT)
 *   4. Dual-Rail (P, N) Harley-Seal Carry-Save GEMM (Kernel-Only & End-to-End)
 */

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

template <typename T>
static inline void doNotOptimize(T& val) noexcept {
    asm volatile("" : "+r,m"(val) : : "memory");
}

template <typename T>
static inline void clobberMemory(T* ptr) noexcept {
    asm volatile("" : : "r"(ptr) : "memory");
}

struct TritWord64 {
    uint64_t P{0};
    uint64_t N{0};
};

// ==============================================================================
// 1. NEON SIMD Packer: int8_t[64] -> TritWord64 (P, N)
// ==============================================================================
static inline void pack_64_neon(const int8_t* __restrict src, TritWord64& dst) noexcept {
#if defined(__aarch64__) || defined(__arm64__)
    static const uint8x16_t kBitMask = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    };
    const int8x16_t v_pos1 = vdupq_n_s8(1);
    const int8x16_t v_neg1 = vdupq_n_s8(-1);

    uint64_t p_out = 0;
    uint64_t n_out = 0;

    for (int chunk = 0; chunk < 4; ++chunk) {
        int8x16_t v = vld1q_s8(src + chunk * 16);
        uint8x16_t mp = vandq_u8(vceqq_s8(v, v_pos1), kBitMask);
        uint8x16_t mn = vandq_u8(vceqq_s8(v, v_neg1), kBitMask);

        uint8x8_t p_lo = vget_low_u8(mp);
        uint8x8_t p_hi = vget_high_u8(mp);
        uint8x8_t n_lo = vget_low_u8(mn);
        uint8x8_t n_hi = vget_high_u8(mn);

        uint16_t p16 = static_cast<uint16_t>(vaddv_u8(p_lo)) |
                       (static_cast<uint16_t>(vaddv_u8(p_hi)) << 8);
        uint16_t n16 = static_cast<uint16_t>(vaddv_u8(n_lo)) |
                       (static_cast<uint16_t>(vaddv_u8(n_hi)) << 8);

        p_out |= (static_cast<uint64_t>(p16) << (chunk * 16));
        n_out |= (static_cast<uint64_t>(n16) << (chunk * 16));
    }
    dst.P = p_out;
    dst.N = n_out;
#else
    uint64_t p = 0, n = 0;
    for (int b = 0; b < 64; ++b) {
        if (src[b] == 1) p |= (1ULL << b);
        else if (src[b] == -1) n |= (1ULL << b);
    }
    dst.P = p;
    dst.N = n;
#endif
}

void pack_matrix_neon(const int8_t* __restrict src,
                      TritWord64* __restrict dst,
                      size_t rows, size_t K) noexcept {
    const size_t words = K / 64;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t w = 0; w < words; ++w) {
            pack_64_neon(&src[r * K + w * 64], dst[r * words + w]);
        }
    }
}

// ==============================================================================
// 2. GEMM Kernels
// ==============================================================================

void gemm_int8_neon(const int8_t* __restrict A,
                    const int8_t* __restrict BT,
                    int32_t* __restrict C,
                    size_t M, size_t K, size_t N) noexcept {
    for (size_t i = 0; i < M; ++i) {
        const int8_t* a_ptr = &A[i * K];
        for (size_t j = 0; j < N; ++j) {
            const int8_t* b_ptr = &BT[j * K];
#if defined(__aarch64__) || defined(__arm64__)
            int32x4_t v0 = vdupq_n_s32(0);
            int32x4_t v1 = vdupq_n_s32(0);
            size_t k = 0;
#if defined(__ARM_FEATURE_DOTPROD)
            for (; k + 32 <= K; k += 32) {
                v0 = vdotq_s32(v0, vld1q_s8(a_ptr + k), vld1q_s8(b_ptr + k));
                v1 = vdotq_s32(v1, vld1q_s8(a_ptr + k + 16), vld1q_s8(b_ptr + k + 16));
            }
#endif
            for (; k + 16 <= K; k += 16) {
                int8x16_t va = vld1q_s8(a_ptr + k);
                int8x16_t vb = vld1q_s8(b_ptr + k);
                v0 = vpadalq_s16(v0, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
                v1 = vpadalq_s16(v1, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
            }
            int32_t sum = vaddvq_s32(vaddq_s32(v0, v1));
            for (; k < K; ++k) sum += static_cast<int32_t>(a_ptr[k]) * b_ptr[k];
            C[i * N + j] = sum;
#else
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) sum += static_cast<int32_t>(a_ptr[k]) * b_ptr[k];
            C[i * N + j] = sum;
#endif
        }
    }
}

void gemm_dual_rail_popcnt(const TritWord64* __restrict A,
                           const TritWord64* __restrict BT,
                           int64_t* __restrict C,
                           size_t M, size_t K, size_t N) noexcept {
    const size_t words = K / 64;
    for (size_t i = 0; i < M; ++i) {
        const TritWord64* a_ptr = &A[i * words];
        for (size_t j = 0; j < N; ++j) {
            const TritWord64* b_ptr = &BT[j * words];
            int64_t sum = 0;
            for (size_t w = 0; w < words; ++w) {
                uint64_t pp = (a_ptr[w].P & b_ptr[w].P) | (a_ptr[w].N & b_ptr[w].N);
                uint64_t np = (a_ptr[w].P & b_ptr[w].N) | (a_ptr[w].N & b_ptr[w].P);
                sum += static_cast<int64_t>(std::popcount(pp)) -
                       static_cast<int64_t>(std::popcount(np));
            }
            C[i * N + j] = sum;
        }
    }
}

// Harley-Seal 3:2 Carry-Save Adder (CSA) reduction across word triples
static inline void csa3(uint64_t a, uint64_t b, uint64_t c,
                        uint64_t& ones, uint64_t& twos) noexcept {
    uint64_t u = a ^ b;
    ones = u ^ c;
    twos = (a & b) | (u & c);
}

void gemm_dual_rail_harley_seal(const TritWord64* __restrict A,
                                const TritWord64* __restrict BT,
                                int64_t* __restrict C,
                                size_t M, size_t K, size_t N) noexcept {
    const size_t words = K / 64;
    for (size_t i = 0; i < M; ++i) {
        const TritWord64* a_ptr = &A[i * words];
        for (size_t j = 0; j < N; ++j) {
            const TritWord64* b_ptr = &BT[j * words];
            int64_t sum = 0;
            size_t w = 0;
            for (; w + 3 <= words; w += 3) {
                uint64_t p0 = (a_ptr[w].P & b_ptr[w].P) | (a_ptr[w].N & b_ptr[w].N);
                uint64_t p1 = (a_ptr[w+1].P & b_ptr[w+1].P) | (a_ptr[w+1].N & b_ptr[w+1].N);
                uint64_t p2 = (a_ptr[w+2].P & b_ptr[w+2].P) | (a_ptr[w+2].N & b_ptr[w+2].N);

                uint64_t n0 = (a_ptr[w].P & b_ptr[w].N) | (a_ptr[w].N & b_ptr[w].P);
                uint64_t n1 = (a_ptr[w+1].P & b_ptr[w+1].N) | (a_ptr[w+1].N & b_ptr[w+1].P);
                uint64_t n2 = (a_ptr[w+2].P & b_ptr[w+2].N) | (a_ptr[w+2].N & b_ptr[w+2].P);

                uint64_t p_ones, p_twos, n_ones, n_twos;
                csa3(p0, p1, p2, p_ones, p_twos);
                csa3(n0, n1, n2, n_ones, n_twos);

                sum += static_cast<int64_t>(std::popcount(p_ones)) +
                       2LL * static_cast<int64_t>(std::popcount(p_twos)) -
                       static_cast<int64_t>(std::popcount(n_ones)) -
                       2LL * static_cast<int64_t>(std::popcount(n_twos));
            }
            for (; w < words; ++w) {
                uint64_t pp = (a_ptr[w].P & b_ptr[w].P) | (a_ptr[w].N & b_ptr[w].N);
                uint64_t np = (a_ptr[w].P & b_ptr[w].N) | (a_ptr[w].N & b_ptr[w].P);
                sum += static_cast<int64_t>(std::popcount(pp)) -
                       static_cast<int64_t>(std::popcount(np));
            }
            C[i * N + j] = sum;
        }
    }
}

template <typename Fn>
double bench_median_ms(size_t warmup, size_t iters, Fn&& fn) {
    for (size_t i = 0; i < warmup; ++i) fn();
    std::vector<double> samples;
    samples.reserve(iters);
    for (size_t i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[iters / 2];
}

int main() {
    constexpr size_t M = 256, K = 512, N = 256;
    const double total_ops = 2.0 * M * K * N;

    std::mt19937_64 rng(1337);
    std::vector<int8_t> A_i8(M * K), BT_i8(N * K);
    for (auto& v : A_i8) v = static_cast<int8_t>((rng() % 3) - 1);
    for (auto& v : BT_i8) v = static_cast<int8_t>((rng() % 3) - 1);

    std::vector<TritWord64> A_dr(M * (K / 64)), BT_dr(N * (K / 64));
    pack_matrix_neon(A_i8.data(), A_dr.data(), M, K);
    pack_matrix_neon(BT_i8.data(), BT_dr.data(), N, K);

    std::vector<int32_t> C_i8(M * N, 0);
    std::vector<int64_t> C_pop(M * N, 0), C_hs(M * N, 0);

    gemm_int8_neon(A_i8.data(), BT_i8.data(), C_i8.data(), M, K, N);
    gemm_dual_rail_popcnt(A_dr.data(), BT_dr.data(), C_pop.data(), M, K, N);
    gemm_dual_rail_harley_seal(A_dr.data(), BT_dr.data(), C_hs.data(), M, K, N);

    for (size_t i = 0; i < M * N; ++i) {
        if (C_i8[i] != C_pop[i] || C_i8[i] != C_hs[i]) {
            std::cerr << "[FAIL] Mismatch at " << i << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] 100% Exact Equivalence Verified (M=" << M << ", K=" << K << ", N=" << N << ")\n\n";

    double ms_i8 = bench_median_ms(5, 50, [&]() {
        gemm_int8_neon(A_i8.data(), BT_i8.data(), C_i8.data(), M, K, N);
        clobberMemory(C_i8.data());
    });
    double ms_pop_k = bench_median_ms(5, 50, [&]() {
        gemm_dual_rail_popcnt(A_dr.data(), BT_dr.data(), C_pop.data(), M, K, N);
        clobberMemory(C_pop.data());
    });
    double ms_pop_e2e = bench_median_ms(5, 50, [&]() {
        pack_matrix_neon(A_i8.data(), A_dr.data(), M, K);
        gemm_dual_rail_popcnt(A_dr.data(), BT_dr.data(), C_pop.data(), M, K, N);
        clobberMemory(C_pop.data());
    });
    double ms_hs_k = bench_median_ms(5, 50, [&]() {
        gemm_dual_rail_harley_seal(A_dr.data(), BT_dr.data(), C_hs.data(), M, K, N);
        clobberMemory(C_hs.data());
    });
    double ms_hs_e2e = bench_median_ms(5, 50, [&]() {
        pack_matrix_neon(A_i8.data(), A_dr.data(), M, K);
        gemm_dual_rail_harley_seal(A_dr.data(), BT_dr.data(), C_hs.data(), M, K, N);
        clobberMemory(C_hs.data());
    });

    auto print_row = [&](const std::string& name, double ms) {
        double gops = (total_ops / (ms / 1000.0)) / 1e9;
        double speedup = ms_i8 / ms;
        std::cout << std::left << std::setw(42) << name
                  << std::right << std::fixed << std::setprecision(4) << std::setw(10) << ms << " ms"
                  << std::setprecision(2) << std::setw(12) << gops << " GOPS"
                  << std::setw(10) << speedup << "x\n";
    };

    print_row("1. INT8 NEON vdotq_s32 (Baseline)", ms_i8);
    print_row("2. Dual-Rail POPCNT (Kernel-Only)", ms_pop_k);
    print_row("3. Dual-Rail POPCNT (End-to-End Pack+GEMM)", ms_pop_e2e);
    print_row("4. Dual-Rail Harley-Seal CSA (Kernel-Only)", ms_hs_k);
    print_row("5. Dual-Rail Harley-Seal (End-to-End)", ms_hs_e2e);

    return 0;
}
