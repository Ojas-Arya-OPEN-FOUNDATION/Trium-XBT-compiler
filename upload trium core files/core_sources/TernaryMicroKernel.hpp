/**
 * @file TernaryMicroKernel.hpp
 * @brief Unified ternary GEMM micro-kernel: register-blocked, ISA-dispatched,
 *        packed-storage. One kernel body, many backends (CPU SIMD today, GPU seam).
 *
 * WHY THIS EXISTS
 * ---------------
 * Measured on a 2-core Xeon @2.6 GHz (AVX-512F/BW, no AVX512VPOPCNTDQ):
 *
 *   dot-product primitive (streaming) ....... 156 G trit/s   <- what the metal can do
 *   gemm_autotuned (best in-tree) ...........  45.6 G MAC/s  <- what we actually got
 *   hand-written 4x4 blocked (no intrinsics)   68.5 G MAC/s  <- +50% from blocking alone
 *
 * The GEMM was reaching only ~44% of the rate its own dot-product primitive
 * achieves. The gap is register blocking and operand reuse, not the arithmetic.
 * This header closes that gap and adds an ISA dispatch layer on top, so the same
 * kernel scales from a portable scalar fallback up to AVX-512 VPOPCNTDQ / NEON.
 *
 * DESIGN
 * ------
 *   Lane      - abstracts "W machine words of dual-rail data processed per step".
 *               ScalarLane W=1, Avx2Lane W=4, Avx512VpopcntLane W=8, NeonLane W=2.
 *               Every specialisation implements the SAME primitive:
 *                   sum_k a_k * b_k  =  popcount(PP|NN) - popcount(PN|NP)
 *   MicroKernel<Lane,MR,NR>
 *             - MRxNR register tile. Accumulators stay in registers across the
 *               whole K loop, so each loaded A/B word is reused NR/MR times.
 *   PackedTensor
 *             - bulk storage at 5 trits/byte (1.6 bits/trit) via the existing
 *               Z3-verified Base243Codec, vs 2 bits/trit for dual-rail: 20% less
 *               memory, and ~20x less than FP32 weights.
 *   ternary_gemm()
 *             - one entry point. Picks ISA at runtime, packs panels, runs the
 *               kernel, accumulates in int64 (exact - no rounding anywhere).
 *
 * ACCURACY
 * --------
 * Accumulation is exact integer arithmetic. A ternary MAC is a sum of {-1,0,+1}
 * products, so the result is exact in int64 for any K < 2^63. There is no
 * rounding error to accumulate, unlike FP32/FP16 GEMM. `checked` mode bounds the
 * accumulator so a pathological K cannot wrap silently.
 *
 * PORTABILITY
 * -----------
 * ScalarLane is always available and is the reference implementation. SIMD lanes
 * compile only when the compiler targets the ISA, and are selected only when
 * runtime CPUID confirms it. Unknown hardware silently uses ScalarLane.
 */

#ifndef TERNARY_BLAS_TERNARY_MICRO_KERNEL_HPP
#define TERNARY_BLAS_TERNARY_MICRO_KERNEL_HPP

#include "ternary_core.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// ISA availability. These say what the COMPILER targeted; runtime CPUID in
// detect_isa() says what the MACHINE actually has. Both must agree.
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
#  define TERNARY_BLAS_X86 1
#  if defined(__AVX2__)
#    include <immintrin.h>
#    define TERNARY_BLAS_HAVE_AVX2 1
#  endif
#  if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512F__)
#    include <immintrin.h>
#    define TERNARY_BLAS_HAVE_AVX512VPOPCNT 1
#  endif
#else
#  define TERNARY_BLAS_X86 0
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#  include <arm_neon.h>
#  define TERNARY_BLAS_HAVE_NEON 1
#else
#  define TERNARY_BLAS_HAVE_NEON 0
#endif

namespace bt {
namespace blas {

// ===========================================================================
// 1. ISA enumeration and runtime detection
// ===========================================================================

enum class Isa : int {
    Auto           = -1, // probe the machine at runtime (default)
    Scalar         = 0,  // portable uint64_t, always available
    Avx2           = 1,  // 256-bit, pshufb popcount,  4 words/step
    Avx512Vpopcnt  = 2,  // 512-bit vpopcntq,          8 words/step
    Neon           = 3   // ARM, vcntq_u8 horizontal,  2 words/step
};

inline const char* isa_name(Isa i) noexcept {
    switch (i) {
        case Isa::Avx512Vpopcnt: return "AVX-512 + VPOPCNTDQ";
        case Isa::Avx2:          return "AVX2 (pshufb popcount)";
        case Isa::Neon:          return "ARM NEON";
        case Isa::Auto:          return "auto (runtime probe)";
        case Isa::Scalar:
        default:                 return "scalar (portable)";
    }
}

/** Runtime CPUID-based selection. Returns the best ISA the *machine* supports
 *  among those this translation unit was compiled for. */
inline Isa detect_isa() noexcept {
#if TERNARY_BLAS_X86
    __builtin_cpu_init();
#  if TERNARY_BLAS_HAVE_AVX512VPOPCNT
    if (__builtin_cpu_supports("avx512vpopcntdq") &&
        __builtin_cpu_supports("avx512f"))
        return Isa::Avx512Vpopcnt;
#  endif
#  if TERNARY_BLAS_HAVE_AVX2
    if (__builtin_cpu_supports("avx2")) return Isa::Avx2;
#  endif
#elif TERNARY_BLAS_HAVE_NEON
    return Isa::Neon;
#endif
    return Isa::Scalar;
}

// ===========================================================================
// 2. Lane abstractions
//    Each lane carries W dual-rail words and computes the popcount dot product.
//    All lanes implement the identical mathematical primitive.
// ===========================================================================

// ---- Scalar: the portable reference. W = 1 word (64 trits) per step. -------
struct ScalarLane {
    static constexpr int W = 1;
    struct Vec { uint64_t P, N; };

    static inline Vec load(const uint64_t* p, const uint64_t* n) noexcept {
        return Vec{p[0], n[0]};
    }
    /** sum over the W words of (a_k * b_k) with a_k,b_k in {-1,0,+1}. */
    static inline int64_t dot(Vec a, Vec b) noexcept {
        const uint64_t pos = (a.P & b.P) | (a.N & b.N);   // k where a*b = +1
        const uint64_t neg = (a.P & b.N) | (a.N & b.P);   // k where a*b = -1
        return static_cast<int64_t>(TritWord64::popcount64(pos))
             - static_cast<int64_t>(TritWord64::popcount64(neg));
    }
};

#if TERNARY_BLAS_HAVE_AVX2
// ---- AVX2: 256-bit. Popcount via the pshufb nibble lookup, 4 words/step. ---
struct Avx2Lane {
    static constexpr int W = 4;
    struct Vec { __m256i P, N; };

    static inline Vec load(const uint64_t* p, const uint64_t* n) noexcept {
        return Vec{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)),
                   _mm256_loadu_si256(reinterpret_cast<const __m256i*>(n))};
    }
    /** Per-64-bit-lane popcount. Result: 4x uint64, each = popcount of a lane. */
    static inline __m256i popcnt_epi64(__m256i v) noexcept {
        const __m256i lookup = _mm256_setr_epi8(
            0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4,
            0,1,1,2, 1,2,2,3, 1,2,2,3, 2,3,3,4);
        const __m256i mask = _mm256_set1_epi8(0x0f);
        __m256i lo = _mm256_and_si256(v, mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
        __m256i s  = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                                     _mm256_shuffle_epi8(lookup, hi));
        // sad_epu8 horizontally sums each 8-byte group into a 64-bit lane
        return _mm256_sad_epu8(s, _mm256_setzero_si256());
    }
    static inline int64_t hsum_epi64(__m256i v) noexcept {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i s  = _mm_add_epi64(lo, hi);
        // swap the two 64-bit halves and add: both lanes now hold the total
        __m128i sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
        s = _mm_add_epi64(s, sh);
        return static_cast<int64_t>(_mm_cvtsi128_si64(s));
    }
    static inline int64_t dot(Vec a, Vec b) noexcept {
        __m256i pos = _mm256_or_si256(_mm256_and_si256(a.P, b.P),
                                      _mm256_and_si256(a.N, b.N));
        __m256i neg = _mm256_or_si256(_mm256_and_si256(a.P, b.N),
                                      _mm256_and_si256(a.N, b.P));
        return hsum_epi64(_mm256_sub_epi64(popcnt_epi64(pos), popcnt_epi64(neg)));
    }
};
#endif

#if TERNARY_BLAS_HAVE_AVX512VPOPCNT
// ---- AVX-512 + VPOPCNTDQ: hardware vector popcount, 8 words/step. ----------
struct Avx512VpopcntLane {
    static constexpr int W = 8;
    struct Vec { __m512i P, N; };

    static inline Vec load(const uint64_t* p, const uint64_t* n) noexcept {
        return Vec{_mm512_loadu_si512(p), _mm512_loadu_si512(n)};
    }
    static inline int64_t dot(Vec a, Vec b) noexcept {
        __m512i pos = _mm512_or_si512(_mm512_and_si512(a.P, b.P),
                                      _mm512_and_si512(a.N, b.N));
        __m512i neg = _mm512_or_si512(_mm512_and_si512(a.P, b.N),
                                      _mm512_and_si512(a.N, b.P));
        __m512i d = _mm512_sub_epi64(_mm512_popcnt_epi64(pos),
                                     _mm512_popcnt_epi64(neg));
        return _mm512_reduce_add_epi64(d);
    }
};
#endif

#if TERNARY_BLAS_HAVE_NEON
// ---- ARM NEON: byte-wise vcntq_u8 then widen-accumulate. 2 words/step. -----
struct NeonLane {
    static constexpr int W = 2;
    struct Vec { uint64x2_t P, N; };

    static inline Vec load(const uint64_t* p, const uint64_t* n) noexcept {
        return Vec{vld1q_u64(p), vld1q_u64(n)};
    }
    static inline uint64x2_t popcnt_epi64(uint64x2_t v) noexcept {
        uint8x16_t bytes = vreinterpretq_u8_u64(v);
        uint8x16_t c = vcntq_u8(bytes);          // per-byte popcount
        uint16x8_t w = vpaddlq_u8(c);            // pairwise widen to u16
        uint32x4_t q = vpaddlq_u16(w);           // widen to u32
        return vpaddlq_u32(q);                   // widen to u64
    }
    static inline int64_t dot(Vec a, Vec b) noexcept {
        uint64x2_t pos = vorrq_u64(vandq_u64(a.P, b.P), vandq_u64(a.N, b.N));
        uint64x2_t neg = vorrq_u64(vandq_u64(a.P, b.N), vandq_u64(a.N, b.P));
        uint64x2_t d = vsubq_u64(popcnt_epi64(pos), popcnt_epi64(neg));
        return static_cast<int64_t>(vgetq_lane_u64(d, 0))
             + static_cast<int64_t>(vgetq_lane_u64(d, 1));
    }
};
#endif

// ===========================================================================
// 3. Packed panel storage
//    Panels are dual-rail (fast to consume) and laid out so the micro-kernel
//    streams contiguous memory: A row-panels of MR rows, B col-panels of NR.
// ===========================================================================

struct DualRailPanel {
    std::vector<uint64_t> P, N;   // each: rows * k_words (or cols * k_words)
    size_t rows{0};               // rows for A, cols for B
    size_t k_words{0};

    inline const uint64_t* p(size_t r, size_t w) const noexcept { return &P[r * k_words + w]; }
    inline const uint64_t* n(size_t r, size_t w) const noexcept { return &N[r * k_words + w]; }
};

/** Pack an MxK dual-rail matrix (row-major words) into an A panel: MR rows at a
 *  time, K contiguous, so the micro-kernel reads straight through. */
inline DualRailPanel pack_a(const uint64_t* Ap, const uint64_t* An,
                            size_t m, size_t k_words, int MR) noexcept {
    const size_t mp = ((m + MR - 1) / MR) * MR;
    DualRailPanel pan; pan.rows = mp; pan.k_words = k_words;
    pan.P.assign(mp * k_words, 0); pan.N.assign(mp * k_words, 0);
    for (size_t i = 0; i < m; ++i)
        for (size_t w = 0; w < k_words; ++w) {
            pan.P[i * k_words + w] = Ap[i * k_words + w];
            pan.N[i * k_words + w] = An[i * k_words + w];
        }
    // zero-fill the padded tail so it contributes nothing
    return pan;
}

/** Pack B transposed into a B panel: NR columns at a time, K contiguous.
 *  Taking B^T once converts the inner loop from strided to sequential. */
inline DualRailPanel pack_bt(const uint64_t* Bp, const uint64_t* Bn,
                             size_t k_trits, size_t n, size_t k_words, int NR) noexcept {
    (void)k_trits;
    const size_t np = ((n + NR - 1) / NR) * NR;
    DualRailPanel pan; pan.rows = np; pan.k_words = k_words;
    pan.P.assign(np * k_words, 0); pan.N.assign(np * k_words, 0);
    for (size_t j = 0; j < n; ++j)
        for (size_t w = 0; w < k_words; ++w) {
            pan.P[j * k_words + w] = Bp[w * n + j];
            pan.N[j * k_words + w] = Bn[w * n + j];
        }
    return pan;
}

// ===========================================================================
// 4. The micro-kernel
//    MRxNR accumulators live in registers for the entire K loop. Every A word
//    loaded is reused NR times and every B word MR times, which is where the
//    1.5x over the previous autotuner comes from.
// ===========================================================================

template <class Lane, int MR, int NR>
struct MicroKernel {
    /** Accumulate one MRxNR tile over kw words into C (row-major, ldc stride).
     *  `m`/`n` bound the stores: panels are padded up to a whole tile, so the
     *  trailing tile must not write past the end of C. */
    static void run(const DualRailPanel& A, const DualRailPanel& B,
                    size_t i0, size_t j0, size_t kw,
                    int64_t* C, size_t ldc, size_t m, size_t n) noexcept {
        int64_t acc[MR][NR];
        for (int a = 0; a < MR; ++a)
            for (int b = 0; b < NR; ++b) acc[a][b] = 0;

        constexpr int W = Lane::W;
        size_t w = 0;
        for (; w + W <= kw; w += W) {
            // Hoist A out of the inner loop: it is reused NR times.
            typename Lane::Vec va[MR];
            for (int a = 0; a < MR; ++a) va[a] = Lane::load(A.p(i0 + a, w), A.n(i0 + a, w));
            for (int b = 0; b < NR; ++b) {
                typename Lane::Vec vb = Lane::load(B.p(j0 + b, w), B.n(j0 + b, w));
                for (int a = 0; a < MR; ++a) acc[a][b] += Lane::dot(va[a], vb);
            }
        }
        // Scalar tail: kw is not always a multiple of the lane width.
        for (; w < kw; ++w) {
            for (int a = 0; a < MR; ++a) {
                typename ScalarLane::Vec va = ScalarLane::load(A.p(i0 + a, w), A.n(i0 + a, w));
                for (int b = 0; b < NR; ++b) {
                    typename ScalarLane::Vec vb = ScalarLane::load(B.p(j0 + b, w), B.n(j0 + b, w));
                    acc[a][b] += ScalarLane::dot(va, vb);
                }
            }
        }
        for (int a = 0; a < MR; ++a) {
            if (i0 + a >= m) break;
            for (int b = 0; b < NR; ++b) {
                if (j0 + b >= n) break;
                C[(i0 + a) * ldc + (j0 + b)] = acc[a][b];
            }
        }
    }
};

// ===========================================================================
// 5. Packed bulk storage - 5 trits/byte via the Z3-verified Base243Codec
//    1.6 bits/trit vs 2.0 for dual-rail: 20% smaller, and ~20x smaller
//    than the FP32 weights a binary pipeline would carry.
// ===========================================================================

class PackedTensor {
public:
    PackedTensor() = default;
    PackedTensor(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        bytes_per_row_ = (cols + 4) / 5;                 // 5 trits per byte
        bytes_.assign(rows * bytes_per_row_, 0);
    }

    inline size_t rows() const noexcept { return rows_; }
    inline size_t cols() const noexcept { return cols_; }
    inline size_t bytes() const noexcept { return bytes_.size(); }
    /** Bits per trit actually used, for reporting. */
    inline double bits_per_trit() const noexcept {
        return rows_ && cols_ ? (8.0 * bytes_.size()) / double(rows_ * cols_) : 0.0;
    }

    inline void set(size_t r, size_t c, Trit t) {
        if (r >= rows_ || c >= cols_) throw std::out_of_range("PackedTensor::set");
        const size_t base = r * bytes_per_row_ + c / 5;
        const int     slot = static_cast<int>(c % 5);
        std::array<Trit, 5> g = Base243Codec::unpack5_fast(bytes_[base]);
        g[slot] = t;
        bytes_[base] = Base243Codec::pack5(g);
    }

    inline Trit get(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) throw std::out_of_range("PackedTensor::get");
        return Base243Codec::unpack5_fast(bytes_[r * bytes_per_row_ + c / 5])[c % 5];
    }

    /** Bytes per row of the compressed stream. A host->device upload strides
     *  by exactly this, so it is part of the public contract. */
    inline size_t bytes_per_row() const noexcept { return bytes_per_row_; }

    /** Raw compressed byte `b` of row `r`. Lets a caller upload the packed
     *  stream without inflating it back to dual-rail first. */
    inline uint8_t row_byte(size_t r, size_t b) const {
        if (r >= rows_ || b >= bytes_per_row_)
            throw std::out_of_range("PackedTensor::row_byte");
        return bytes_[r * bytes_per_row_ + b];
    }

    /** Materialise dual-rail words for one row (K contiguous). This is the
     *  unpack step; it happens per row-block, so the working set stays in cache. */
    inline void unpack_row(size_t r, uint64_t* P, uint64_t* N, size_t k_words) const {
        const uint8_t* row = bytes_.data() + r * bytes_per_row_;
        for (size_t w = 0; w < k_words; ++w) {
            const size_t trit0 = w * 64;
            if (trit0 >= cols_) { P[w] = 0; N[w] = 0; continue; }
            const size_t ntri = std::min<size_t>(64, cols_ - trit0);
            uint64_t pp = 0, nn = 0;
            // Decode each 5-trit byte ONCE and consume all five trits it
            // yields. The original loop called get() once per trit, re-running
            // the base-243 decode and both bounds checks five times per group;
            // that made decompression the dominant cost of the packed GEMM.
            // A word begins at trit 64*w, which is not a multiple of 5, so
            // groups straddle word boundaries - `slot` tracks the offset.
            size_t b = 0;
            while (b < ntri) {
                const size_t abs  = trit0 + b;
                const size_t slot = abs % 5;
                const std::array<Trit, 5> g = Base243Codec::unpack5_fast(row[abs / 5]);
                const size_t lim = std::min<size_t>(5 - slot, ntri - b);
                for (size_t s = 0; s < lim; ++s) {
                    const uint64_t bit = 1ULL << (b + s);
                    if      (g[slot + s] == Trit::Pos) pp |= bit;
                    else if (g[slot + s] == Trit::Neg) nn |= bit;
                }
                b += lim;   // lim >= 1, so this always advances
            }
            P[w] = pp; N[w] = nn;
        }
    }

private:
    size_t rows_{0}, cols_{0}, bytes_per_row_{0};
    std::vector<uint8_t> bytes_;
};

// ===========================================================================
// 6. Public entry point
// ===========================================================================

/** Where the kernel should run. Host today; the Device tag is the seam a GPU
 *  backend plugs into - the MRxNR decomposition maps directly onto a GPU
 *  workgroup, and the Lane primitive (AND/OR/popcount) is identical. */
enum class Device : int { Host = 0, Vulkan = 1, Cuda = 2, Metal = 3 };

struct GemmOptions {
    Device device{Device::Host};
    Isa    isa{Isa::Auto};          // Auto = probe the machine; else force an ISA
    int    mr{4};                   // rows per register tile
    int    nr{4};                   // cols per register tile
    bool   checked{false};          // bounds-check the accumulator
    int64_t acc_limit{0};           // 0 = no limit; else |acc| must stay <= limit
};

struct GemmStats {
    Isa    isa{Isa::Scalar};
    int    mr{0}, nr{0};
    double gmacs{0.0};
    double ms{0.0};
};

namespace detail {

/** Dispatch table over (ISA x tile). Only tiles 4x4 and 8x4 are instantiated;
 *  both are parameterised on the Lane so there is one kernel body. */
template <class Lane, int MR, int NR>
void gemm_tiled(const uint64_t* Ap, const uint64_t* An,
                const uint64_t* Bp, const uint64_t* Bn,
                int64_t* C, size_t m, size_t n, size_t k,
                const GemmOptions& o, GemmStats& st);

template <class Lane>
void dispatch_mr(const uint64_t* Ap, const uint64_t* An,
                 const uint64_t* Bp, const uint64_t* Bn,
                 int64_t* C, size_t m, size_t n, size_t k,
                 const GemmOptions& o, GemmStats& st) {
    if (o.mr == 8) gemm_tiled<Lane, 8, 4>(Ap, An, Bp, Bn, C, m, n, k, o, st);
    else           gemm_tiled<Lane, 4, 4>(Ap, An, Bp, Bn, C, m, n, k, o, st);
}

} // namespace detail

/**
 * @brief Unified ternary GEMM: C[m,n] = A[m,k] * B[k,n], exact integer result.
 *
 * Operands are dual-rail word arrays, kw = ceil(k/64) words per row, but the
 * two are NOT stored the same way:
 *
 *   Ap/An : A, row-major.      word w of row i  ->  Ap[i*kw + w]
 *   Bp/Bn : B TRANSPOSED, then word-PLANE-major.
 *           word w of B^T row j  ->  Bp[w*n + j]
 *
 * The two B orderings coincide when kw == 1 (k <= 64), so a row-major B looks
 * correct on small tests and silently returns wrong values for larger K.
 * See test/Unit/test_unit_microkernel.cpp::make_operands for the reference
 * encoding. C is row-major int64 with stride n.
 *
 * The result is mathematically exact: every partial product is in {-1,0,+1}
 * and the accumulator is int64, so there is no rounding error at any K that
 * fits. This is a strict accuracy advantage over FP32/FP16 GEMM.
 */
inline GemmStats ternary_gemm(const uint64_t* Ap, const uint64_t* An,
                              const uint64_t* Bp, const uint64_t* Bn,
                              int64_t* C, size_t m, size_t n, size_t k,
                              const GemmOptions& opts = {}) {
    if (!Ap || !An || !Bp || !Bn || !C)
        throw std::invalid_argument("ternary_gemm: null operand");
    if (m == 0 || n == 0 || k == 0)
        throw std::invalid_argument("ternary_gemm: empty dimensions");

    GemmOptions o = opts;
    if (o.device != Device::Host)
        throw std::runtime_error("ternary_gemm: only Device::Host is implemented "
                                 "so far; GPU backends plug into this entry point");
    if (o.isa == Isa::Auto) o.isa = detect_isa();

    GemmStats st; st.isa = o.isa; st.mr = o.mr; st.nr = o.nr;

#if TERNARY_BLAS_HAVE_AVX512VPOPCNT
    if (o.isa == Isa::Avx512Vpopcnt) {
        detail::dispatch_mr<Avx512VpopcntLane>(Ap, An, Bp, Bn, C, m, n, k, o, st); return st;
    }
#endif
#if TERNARY_BLAS_HAVE_AVX2
    if (o.isa == Isa::Avx2) {
        detail::dispatch_mr<Avx2Lane>(Ap, An, Bp, Bn, C, m, n, k, o, st); return st;
    }
#endif
#if TERNARY_BLAS_HAVE_NEON
    if (o.isa == Isa::Neon) {
        detail::dispatch_mr<NeonLane>(Ap, An, Bp, Bn, C, m, n, k, o, st); return st;
    }
#endif
    detail::dispatch_mr<ScalarLane>(Ap, An, Bp, Bn, C, m, n, k, o, st);
    return st;
}

// ---------------------------------------------------------------------------
// Tiled driver: pack once, then stream MRxNR tiles.
// ---------------------------------------------------------------------------
namespace detail {

template <class Lane, int MR, int NR>
void gemm_tiled(const uint64_t* Ap, const uint64_t* An,
                const uint64_t* Bp, const uint64_t* Bn,
                int64_t* C, size_t m, size_t n, size_t k,
                const GemmOptions& o, GemmStats& st) {
    const size_t kw = (k + 63) / 64;

    DualRailPanel A  = pack_a (Ap, An, m, kw, MR);
    DualRailPanel BT = pack_bt(Bp, Bn, k, n, kw, NR);

    const int64_t limit = (o.checked && o.acc_limit > 0)
                        ? o.acc_limit : std::numeric_limits<int64_t>::max();

    for (size_t i0 = 0; i0 < A.rows; i0 += MR)
        for (size_t j0 = 0; j0 < BT.rows; j0 += NR)
            MicroKernel<Lane, MR, NR>::run(A, BT, i0, j0, kw, C, n, m, n);

    if (o.checked) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                int64_t v = C[i * n + j];
                if (v > limit || v < -limit)
                    throw std::overflow_error("ternary_gemm: accumulator exceeded "
                                              "the configured checked bound");
            }
    }
    (void)st;
}

} // namespace detail


// ===========================================================================
// 7. GEMM straight from 1.6 bits/trit packed storage
//    This is the path that makes the compression usable rather than merely
//    representable: store weights packed, decompress into panels once, then
//    run the identical tiled micro-kernel.
// ===========================================================================

/** Build a PackedTensor from dual-rail planes laid out as rows of k trits. */
inline PackedTensor packed_from_dual_rail(const uint64_t* P, const uint64_t* N,
                                          size_t rows, size_t k, size_t k_words) {
    PackedTensor T(rows, k);
    for (size_t r = 0; r < rows; ++r)
        for (size_t t = 0; t < k; ++t) {
            const bool p = (P[r * k_words + t / 64] >> (t % 64)) & 1ULL;
            const bool n = (N[r * k_words + t / 64] >> (t % 64)) & 1ULL;
            T.set(r, t, p ? Trit::Pos : n ? Trit::Neg : Trit::Zero);
        }
    return T;
}

/** Unpack `count` consecutive rows into a dual-rail panel the kernel consumes. */
inline void unpack_row_block(const PackedTensor& T, size_t r0, size_t count,
                             uint64_t* P, uint64_t* N, size_t k_words) noexcept {
    for (size_t i = 0; i < count; ++i)
        T.unpack_row(r0 + i, P + i * k_words, N + i * k_words, k_words);
}

namespace detail {

template <class Lane, int MR, int NR>
void gemm_packed_tiled(const PackedTensor& A, const PackedTensor& BT,
                       int64_t* C, size_t m, size_t n, size_t k,
                       const GemmOptions& o, GemmStats& st) {
    const size_t kw = (k + 63) / 64;
    const size_t mp = ((m + MR - 1) / MR) * MR;
    const size_t np = ((n + NR - 1) / NR) * NR;

    // Decompress each operand EXACTLY ONCE into the dual-rail panels the
    // kernel consumes, then run the ordinary tiled loop over those panels.
    //
    // Decompressing inside the tile loops instead - the obvious first
    // implementation - re-decodes B m/MR times and A n/NR times over. At
    // n=512 that measured 0.32 G MAC/s against 155 for the dual-rail path:
    // ~500x slower, and slower than the naive O(n^3) reference it exists to
    // accelerate. Hoisting the decompression is what makes packed storage
    // usable at all.
    DualRailPanel pa; pa.rows = mp; pa.k_words = kw;
    pa.P.assign(mp * kw, 0); pa.N.assign(mp * kw, 0);
    for (size_t i = 0; i < m; ++i)
        unpack_row_block(A, i, 1, pa.P.data() + i * kw, pa.N.data() + i * kw, kw);

    DualRailPanel pb; pb.rows = np; pb.k_words = kw;
    pb.P.assign(np * kw, 0); pb.N.assign(np * kw, 0);
    for (size_t j = 0; j < n; ++j)
        unpack_row_block(BT, j, 1, pb.P.data() + j * kw, pb.N.data() + j * kw, kw);

    for (size_t i0 = 0; i0 < m; i0 += MR)
        for (size_t j0 = 0; j0 < n; j0 += NR)
            MicroKernel<Lane, MR, NR>::run(pa, pb, i0, j0, kw, C, n, m, n);
    (void)o; (void)st;
}

template <class Lane>
void dispatch_mr_packed(const PackedTensor& A, const PackedTensor& BT,
                        int64_t* C, size_t m, size_t n, size_t k,
                        const GemmOptions& o, GemmStats& st) {
    if (o.mr == 8) gemm_packed_tiled<Lane, 8, 4>(A, BT, C, m, n, k, o, st);
    else           gemm_packed_tiled<Lane, 4, 4>(A, BT, C, m, n, k, o, st);
}

} // namespace detail

/**
 * @brief GEMM directly from 1.6 bits/trit packed storage.
 *
 * A is an M x K PackedTensor; BT is the N x K PackedTensor holding B
 * transposed (row j of BT is column j of B). C is row-major int64, stride n.
 *
 * The arithmetic is the same exact integer MAC as ternary_gemm, so the two
 * produce bit-identical results. Packed storage is a *storage and transfer*
 * format: at 1.6 bits/trit it is ~20x smaller than FP32 weights, but it costs
 * a decompression pass per call. Decompress once and GEMM many times if the
 * same weights are reused.
 */
inline GemmStats ternary_gemm_packed(const PackedTensor& A, const PackedTensor& BT,
                                     int64_t* C, const GemmOptions& opts = {}) {
    if (!C) throw std::invalid_argument("ternary_gemm_packed: null output");
    const size_t m = A.rows(), n = BT.rows(), k = A.cols();
    if (!m || !n || !k) throw std::invalid_argument("ternary_gemm_packed: empty dimensions");
    if (BT.cols() != k)
        throw std::invalid_argument("ternary_gemm_packed: inner dimensions differ");

    GemmOptions o = opts;
    if (o.isa == Isa::Auto) o.isa = detect_isa();
    GemmStats st; st.isa = o.isa; st.mr = o.mr; st.nr = o.nr;

#if TERNARY_BLAS_HAVE_AVX512VPOPCNTDQ
    if (o.isa == Isa::Avx512Vpopcnt) {
        detail::dispatch_mr_packed<Avx512VpopcntLane>(A, BT, C, m, n, k, o, st); return st;
    }
#endif
#if TERNARY_BLAS_HAVE_AVX2
    if (o.isa == Isa::Avx2) {
        detail::dispatch_mr_packed<Avx2Lane>(A, BT, C, m, n, k, o, st); return st;
    }
#endif
#if TERNARY_BLAS_HAVE_NEON
    if (o.isa == Isa::Neon) {
        detail::dispatch_mr_packed<NeonLane>(A, BT, C, m, n, k, o, st); return st;
    }
#endif
    detail::dispatch_mr_packed<ScalarLane>(A, BT, C, m, n, k, o, st);
    return st;
}

} // namespace blas
} // namespace bt

#endif // TERNARY_BLAS_TERNARY_MICRO_KERNEL_HPP
