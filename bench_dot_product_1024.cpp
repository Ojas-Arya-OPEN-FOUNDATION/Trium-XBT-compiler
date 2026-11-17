/**
 * @file bench_dot_product_1024.cpp
 * @brief Controlled 1,024-Element Dot-Product Benchmark & Independent Integer Oracle
 *
 * Designed for reproducible cross-architecture comparison between:
 *   1. Dual-Rail (P, N) bitplane kernel-only (pre-converted tensors)
 *   2. Dual-Rail (P, N) end-to-end (allocation + packing + kernel + output)
 *   3. Packed 2-bit ternary kernel (4 trits/byte)
 *   4. INT8 kernel (auto-vectorized C++ and explicit ARM64 NEON SIMD)
 *
 * Timing & Operation Counting Transparency:
 *   - Uses std::chrono::steady_clock (wall-clock nanoseconds).
 *   - On ARM64, cntvct_el0 / cntfrq_el0 is a fixed-frequency system timer
 *     (e.g. 1 GHz = 1 ns/tick on Apple Silicon), NOT CPU core clock cycles.
 *     We report wall-clock ns/call directly rather than mislabeling timer ticks
 *     as CPU cycles.
 *   - Reports BOTH operation conventions explicitly:
 *       * GElems/s = N / time_sec              (1 element-pair per op)
 *       * GTOPS    = (2 * N) / time_sec        (2 ops per MAC: mul + add)
 */

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

// Compiler barrier to prevent dead-code elimination or loop hoisting
template <typename T>
static inline void doNotOptimize(T& val) noexcept {
    asm volatile("" : "+r,m"(val) : : "memory");
}

template <typename T>
static inline void clobberMemory(T* ptr) noexcept {
    asm volatile("" : : "r"(ptr) : "memory");
}

// ==============================================================================
// 1. Dual-Rail (P, N) & Packed 2-Bit Data Structures
// ==============================================================================

struct TritWord64 {
    uint64_t P{0}; // bit i = 1 iff trit i == +1
    uint64_t N{0}; // bit i = 1 iff trit i == -1
    // Invariant: (P & N) == 0 at all valid bit positions.
};

static inline size_t words_for_trits(size_t n_elems) noexcept {
    return (n_elems + 63) / 64;
}

static inline size_t packed2bit_bytes(size_t n_elems) noexcept {
    return (n_elems + 3) / 4;
}

static inline uint64_t tail_mask_for(size_t n_elems) noexcept {
    size_t rem = n_elems % 64;
    return (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
}

/**
 * @brief Validates Dual-Rail invariants:
 *   1. Disjoint rails: (P & N) == 0 on every word.
 *   2. Tail zeroing: bits >= (n_elems % 64) in the final word are zero.
 */
bool validate_dual_rail(const TritWord64* words, size_t n_elems) noexcept {
    if (n_elems == 0) return true;
    const size_t n_words = words_for_trits(n_elems);
    for (size_t w = 0; w < n_words; ++w) {
        if ((words[w].P & words[w].N) != 0ULL) {
            return false;
        }
    }
    const uint64_t tmask = tail_mask_for(n_elems);
    if (((words[n_words - 1].P | words[n_words - 1].N) & ~tmask) != 0ULL) {
        return false;
    }
    return true;
}

void pack_dual_rail(const int8_t* src, TritWord64* dst, size_t n_elems) {
    const size_t n_words = words_for_trits(n_elems);
    for (size_t w = 0; w < n_words; ++w) {
        uint64_t p = 0;
        uint64_t n = 0;
        const size_t base = w * 64;
        const size_t limit = std::min<size_t>(64, n_elems - base);
        for (size_t b = 0; b < limit; ++b) {
            const int8_t v = src[base + b];
            if (v == 1) {
                p |= (1ULL << b);
            } else if (v == -1) {
                n |= (1ULL << b);
            } else if (v != 0) {
                throw std::invalid_argument("pack_dual_rail: input must be in {-1, 0, +1}");
            }
        }
        dst[w] = TritWord64{p, n};
    }
}

void pack_2bit_ternary(const int8_t* src, uint8_t* dst, size_t n_elems) {
    const size_t n_bytes = packed2bit_bytes(n_elems);
    for (size_t idx = 0; idx < n_bytes; ++idx) {
        uint8_t byte_val = 0;
        const size_t base = idx * 4;
        const size_t limit = std::min<size_t>(4, n_elems - base);
        for (size_t k = 0; k < limit; ++k) {
            const int8_t v = src[base + k];
            uint8_t code = 0; // 00 = 0, 01 = +1, 11 = -1
            if (v == 1) code = 0x1;
            else if (v == -1) code = 0x3;
            else if (v != 0) throw std::invalid_argument("pack_2bit_ternary: invalid trit");
            byte_val |= static_cast<uint8_t>(code << (2 * k));
        }
        dst[idx] = byte_val;
    }
}

// ==============================================================================
// 2. Independent Exact Integer Oracle & Dot-Product Kernels
// ==============================================================================

int64_t dot_oracle_i128(const int8_t* a, const int8_t* b, size_t n_elems) {
    __int128 acc = 0;
    for (size_t i = 0; i < n_elems; ++i) {
        if (a[i] < -1 || a[i] > 1 || b[i] < -1 || b[i] > 1) {
            throw std::invalid_argument("dot_oracle_i128: input out of {-1, 0, +1}");
        }
        acc += static_cast<__int128>(a[i]) * static_cast<__int128>(b[i]);
    }
    const __int128 bound = static_cast<__int128>(n_elems);
    if (acc < -bound || acc > bound) {
        throw std::logic_error("dot_oracle_i128: accumulator bound violated");
    }
    return static_cast<int64_t>(acc);
}

int64_t dot_dual_rail(const TritWord64* __restrict a,
                      const TritWord64* __restrict b,
                      size_t n_elems) noexcept {
    if (n_elems == 0) return 0;
    const size_t full_words = n_elems / 64;
    int64_t acc = 0;

    for (size_t w = 0; w < full_words; ++w) {
        const uint64_t p_prod = (a[w].P & b[w].P) | (a[w].N & b[w].N);
        const uint64_t n_prod = (a[w].P & b[w].N) | (a[w].N & b[w].P);
        acc += static_cast<int64_t>(std::popcount(p_prod)) -
               static_cast<int64_t>(std::popcount(n_prod));
    }

    if ((n_elems % 64) != 0) {
        const uint64_t mask = tail_mask_for(n_elems);
        const uint64_t ap = a[full_words].P & mask;
        const uint64_t an = a[full_words].N & mask;
        const uint64_t bp = b[full_words].P & mask;
        const uint64_t bn = b[full_words].N & mask;
        const uint64_t p_prod = (ap & bp) | (an & bn);
        const uint64_t n_prod = (ap & bn) | (an & bp);
        acc += static_cast<int64_t>(std::popcount(p_prod)) -
               static_cast<int64_t>(std::popcount(n_prod));
    }
    return acc;
}

int64_t dot_packed_2bit(const uint8_t* __restrict a,
                        const uint8_t* __restrict b,
                        size_t n_elems) noexcept {
    static constexpr int8_t kDecode[4] = {0, 1, 0, -1};
    int64_t acc = 0;
    const size_t full_bytes = n_elems / 4;

    for (size_t idx = 0; idx < full_bytes; ++idx) {
        const uint8_t av = a[idx];
        const uint8_t bv = b[idx];
        acc += static_cast<int64_t>(kDecode[av & 3U]) * kDecode[bv & 3U];
        acc += static_cast<int64_t>(kDecode[(av >> 2) & 3U]) * kDecode[(bv >> 2) & 3U];
        acc += static_cast<int64_t>(kDecode[(av >> 4) & 3U]) * kDecode[(bv >> 4) & 3U];
        acc += static_cast<int64_t>(kDecode[(av >> 6) & 3U]) * kDecode[(bv >> 6) & 3U];
    }

    const size_t tail = n_elems % 4;
    if (tail != 0) {
        const uint8_t av = a[full_bytes];
        const uint8_t bv = b[full_bytes];
        for (size_t k = 0; k < tail; ++k) {
            acc += static_cast<int64_t>(kDecode[(av >> (2 * k)) & 3U]) *
                   kDecode[(bv >> (2 * k)) & 3U];
        }
    }
    return acc;
}

int64_t dot_int8_autovec(const int8_t* __restrict a,
                         const int8_t* __restrict b,
                         size_t n_elems) noexcept {
    int32_t acc = 0;
#if defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
    for (size_t i = 0; i < n_elems; ++i) {
        acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return static_cast<int64_t>(acc);
}

#if defined(__aarch64__) || defined(__arm64__)
int64_t dot_int8_neon(const int8_t* __restrict a,
                      const int8_t* __restrict b,
                      size_t n_elems) noexcept {
    size_t i = 0;
    int32x4_t vacc0 = vdupq_n_s32(0);
    int32x4_t vacc1 = vdupq_n_s32(0);

#if defined(__ARM_FEATURE_DOTPROD)
    for (; i + 32 <= n_elems; i += 32) {
        int8x16_t va0 = vld1q_s8(a + i);
        int8x16_t vb0 = vld1q_s8(b + i);
        int8x16_t va1 = vld1q_s8(a + i + 16);
        int8x16_t vb1 = vld1q_s8(b + i + 16);
        vacc0 = vdotq_s32(vacc0, va0, vb0);
        vacc1 = vdotq_s32(vacc1, va1, vb1);
    }
    for (; i + 16 <= n_elems; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        vacc0 = vdotq_s32(vacc0, va, vb);
    }
#else
    for (; i + 16 <= n_elems; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        int16x8_t p_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t p_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        vacc0 = vpadalq_s16(vacc0, p_lo);
        vacc1 = vpadalq_s16(vacc1, p_hi);
    }
#endif

    int64_t sum = static_cast<int64_t>(vaddvq_s32(vaddq_s32(vacc0, vacc1)));
    for (; i < n_elems; ++i) {
        sum += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
    }
    return sum;
}
#endif

// ==============================================================================
// 3. Correctness & Oracle Verification Suite (Tail + Bounds + Invalid State)
// ==============================================================================

bool run_oracle_verification() {
    std::mt19937_64 rng(20261117ULL);
    const std::vector<size_t> test_lengths = {
        0, 1, 2, 7, 15, 63, 64, 65, 127, 128, 129, 511, 512, 1000, 1023, 1024, 1025, 2051
    };

    for (size_t n : test_lengths) {
        std::vector<int8_t> a(n), b(n);
        std::vector<TritWord64> a_dr(words_for_trits(n)), b_dr(words_for_trits(n));
        std::vector<uint8_t> a_2b(packed2bit_bytes(n)), b_2b(packed2bit_bytes(n));

        // Directed patterns + random dense + random sparse
        for (int mode = 0; mode < 6; ++mode) {
            for (size_t i = 0; i < n; ++i) {
                if (mode == 0) { a[i] = 1; b[i] = 1; }
                else if (mode == 1) { a[i] = 1; b[i] = -1; }
                else if (mode == 2) { a[i] = 0; b[i] = (i % 2 == 0) ? 1 : -1; }
                else if (mode == 3) { a[i] = (i % 2 == 0) ? 1 : -1; b[i] = (i % 2 == 0) ? 1 : -1; }
                else if (mode == 4) { // 90% sparse
                    a[i] = (rng() % 10 == 0) ? ((rng() % 2 == 0) ? 1 : -1) : 0;
                    b[i] = (rng() % 10 == 0) ? ((rng() % 2 == 0) ? 1 : -1) : 0;
                } else { // uniform dense
                    a[i] = static_cast<int8_t>((rng() % 3) - 1);
                    b[i] = static_cast<int8_t>((rng() % 3) - 1);
                }
            }

            pack_dual_rail(a.data(), a_dr.data(), n);
            pack_dual_rail(b.data(), b_dr.data(), n);
            pack_2bit_ternary(a.data(), a_2b.data(), n);
            pack_2bit_ternary(b.data(), b_2b.data(), n);

            if (!validate_dual_rail(a_dr.data(), n) || !validate_dual_rail(b_dr.data(), n)) {
                std::cerr << "[FAIL] Dual-rail invariant check failed for n=" << n << "\n";
                return false;
            }

            const int64_t expected = dot_oracle_i128(a.data(), b.data(), n);
            const int64_t got_dr = dot_dual_rail(a_dr.data(), b_dr.data(), n);
            const int64_t got_2b = dot_packed_2bit(a_2b.data(), b_2b.data(), n);
            const int64_t got_i8 = dot_int8_autovec(a.data(), b.data(), n);

            if (got_dr != expected || got_2b != expected || got_i8 != expected) {
                std::cerr << "[FAIL] Oracle mismatch at n=" << n << " mode=" << mode
                          << " expected=" << expected << " dr=" << got_dr
                          << " 2bit=" << got_2b << " i8=" << got_i8 << "\n";
                return false;
            }
#if defined(__aarch64__) || defined(__arm64__)
            const int64_t got_neon = dot_int8_neon(a.data(), b.data(), n);
            if (got_neon != expected) {
                std::cerr << "[FAIL] NEON INT8 mismatch at n=" << n << " got=" << got_neon
                          << " expected=" << expected << "\n";
                return false;
            }
#endif
        }
    }

    // Verify invalid P/N state and dirty tail bit detection
    {
        TritWord64 bad_disjoint{0x1ULL, 0x1ULL};
        if (validate_dual_rail(&bad_disjoint, 64)) {
            std::cerr << "[FAIL] Failed to catch (P & N) != 0 invalid state\n";
            return false;
        }
        TritWord64 bad_tail{1ULL << 63, 0ULL};
        if (validate_dual_rail(&bad_tail, 63)) {
            std::cerr << "[FAIL] Failed to catch dirty tail bit\n";
            return false;
        }
    }
    return true;
}

// ==============================================================================
// 4. Benchmark Harness
// ==============================================================================

struct BenchSummary {
    std::string label;
    double min_ns;
    double median_ns;
    double mean_ns;
    double g_elems_sec;
    double g_mac_ops_sec;
    int64_t checksum;
};

template <typename Fn>
BenchSummary measure_kernel(const std::string& label,
                            size_t n_elems,
                            size_t inner_iters,
                            size_t trials,
                            Fn&& fn) {
    // Warmup
    int64_t sink = 0;
    for (size_t i = 0; i < 50000; ++i) {
        sink += fn();
    }
    doNotOptimize(sink);

    std::vector<double> ns_samples;
    ns_samples.reserve(trials);
    int64_t checksum = 0;

    for (size_t t = 0; t < trials; ++t) {
        int64_t acc = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < inner_iters; ++i) {
            acc += fn();
        }
        doNotOptimize(acc);
        const auto t1 = std::chrono::steady_clock::now();
        const double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        ns_samples.push_back(total_ns / static_cast<double>(inner_iters));
        checksum = acc;
    }

    std::sort(ns_samples.begin(), ns_samples.end());
    double sum_ns = 0.0;
    for (double v : ns_samples) sum_ns += v;

    const double min_ns = ns_samples.front();
    const double median_ns = ns_samples[trials / 2];
    const double mean_ns = sum_ns / static_cast<double>(trials);

    const double g_elems = static_cast<double>(n_elems) / median_ns;        // 1e9 elems/s
    const double g_mac_ops = (2.0 * static_cast<double>(n_elems)) / median_ns; // 2*N ops/s

    return BenchSummary{label, min_ns, median_ns, mean_ns, g_elems, g_mac_ops, checksum};
}

int main() {
    std::cout << "============================================================================================\n";
    std::cout << "  Trium-XBT Controlled 1,024-Element Dot-Product Benchmark & Oracle Verification\n";
    std::cout << "============================================================================================\n";

#if defined(__aarch64__) || defined(__arm64__)
    uint64_t cntfrq = 0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(cntfrq));
    std::cout << "  Arch: ARM64 | System Timer (cntfrq_el0): " << cntfrq << " Hz ("
              << (1e9 / static_cast<double>(cntfrq)) << " ns/tick, wall-clock timer)\n";
#if defined(__ARM_FEATURE_DOTPROD)
    std::cout << "  INT8 SIMD: ARMv8.2 DotProd (vdotq_s32) ENABLED\n";
#else
    std::cout << "  INT8 SIMD: Baseline NEON (vmull_s8 + vpadalq_s16)\n";
#endif
#elif defined(__x86_64__) || defined(_M_X64)
    std::cout << "  Arch: x86_64\n";
#endif

    std::cout << "  Clock: std::chrono::steady_clock (nanoseconds)\n\n";

    // --- Step 1: Run Independent Oracle Verification ---
    if (!run_oracle_verification()) {
        std::cerr << "ABORT: Oracle verification failed.\n";
        return 1;
    }
    std::cout << "  [PASS] Independent __int128 oracle verified across 18 lengths (0..2051, with tails),\n"
              << "         6 data patterns, accumulator bounds (|dot| <= N), and invalid P/N checks.\n\n";

    // --- Step 2: Benchmark 1,024-Element Dot Product ---
    constexpr size_t N = 1024;
    constexpr size_t INNER_ITERS = 200000;
    constexpr size_t TRIALS = 15;

    std::mt19937_64 rng(1337);
    std::vector<int8_t> a_raw(N), b_raw(N);
    for (size_t i = 0; i < N; ++i) {
        a_raw[i] = static_cast<int8_t>((rng() % 3) - 1);
        b_raw[i] = static_cast<int8_t>((rng() % 3) - 1);
    }

    std::vector<TritWord64> a_dr(words_for_trits(N)), b_dr(words_for_trits(N));
    pack_dual_rail(a_raw.data(), a_dr.data(), N);
    pack_dual_rail(b_raw.data(), b_dr.data(), N);

    std::vector<uint8_t> a_2b(packed2bit_bytes(N)), b_2b(packed2bit_bytes(N));
    pack_2bit_ternary(a_raw.data(), a_2b.data(), N);
    pack_2bit_ternary(b_raw.data(), b_2b.data(), N);

    const int64_t expected_single = dot_oracle_i128(a_raw.data(), b_raw.data(), N);

    std::vector<BenchSummary> rows;

    // 1. Dual-Rail Kernel-Only
    rows.push_back(measure_kernel("1. Dual-Rail (P,N) Kernel-Only", N, INNER_ITERS, TRIALS, [&]() {
        clobberMemory(a_dr.data());
        clobberMemory(b_dr.data());
        return dot_dual_rail(a_dr.data(), b_dr.data(), N);
    }));

    // 2. Dual-Rail End-to-End (pack raw int8 -> TritWord64 + dot)
    rows.push_back(measure_kernel("2. Dual-Rail (P,N) End-to-End (Pack+Dot)", N, 20000, TRIALS, [&]() {
        TritWord64 tmp_a[16];
        TritWord64 tmp_b[16];
        pack_dual_rail(a_raw.data(), tmp_a, N);
        pack_dual_rail(b_raw.data(), tmp_b, N);
        clobberMemory(tmp_a);
        clobberMemory(tmp_b);
        return dot_dual_rail(tmp_a, tmp_b, N);
    }));

    // 3. Packed 2-Bit Ternary Kernel-Only
    rows.push_back(measure_kernel("3. Packed 2-Bit Ternary Kernel-Only", N, 50000, TRIALS, [&]() {
        clobberMemory(a_2b.data());
        clobberMemory(b_2b.data());
        return dot_packed_2bit(a_2b.data(), b_2b.data(), N);
    }));

    // 4. INT8 Auto-Vectorized Kernel-Only
    rows.push_back(measure_kernel("4. INT8 Auto-Vectorized Kernel-Only", N, INNER_ITERS, TRIALS, [&]() {
        clobberMemory(a_raw.data());
        clobberMemory(b_raw.data());
        return dot_int8_autovec(a_raw.data(), b_raw.data(), N);
    }));

#if defined(__aarch64__) || defined(__arm64__)
    // 5. INT8 Explicit NEON SIMD Kernel-Only
    rows.push_back(measure_kernel("5. INT8 NEON SIMD Kernel-Only", N, INNER_ITERS, TRIALS, [&]() {
        clobberMemory(a_raw.data());
        clobberMemory(b_raw.data());
        return dot_int8_neon(a_raw.data(), b_raw.data(), N);
    }));
#endif

    std::cout << "  Single dot(A, B) exact oracle = " << expected_single << " (N=" << N << ")\n\n";
    std::cout << std::left << std::setw(42) << "  Mode"
              << std::right << std::setw(11) << "Median(ns)"
              << std::setw(11) << "Min(ns)"
              << std::setw(11) << "Mean(ns)"
              << std::setw(13) << "GElems/s"
              << std::setw(13) << "2N-Ops(GOPS)" << "\n";
    std::cout << "  -----------------------------------------------------------------------------------------------------\n";

    for (const auto& r : rows) {
        std::cout << "  " << std::left << std::setw(40) << r.label
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(11) << r.median_ns
                  << std::setw(11) << r.min_ns
                  << std::setw(11) << r.mean_ns
                  << std::setw(13) << r.g_elems_sec
                  << std::setw(13) << r.g_mac_ops_sec << "\n";
    }
    std::cout << "============================================================================================\n";
    return 0;
}
