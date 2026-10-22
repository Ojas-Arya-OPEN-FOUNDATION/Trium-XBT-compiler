/**
 * @file ternary_universal_codec.hpp
 * @brief Universal Tri-Radix Number System (Q_{2,3,5}) & Unmodified-Float-to-Ternary-Plane Codec
 *
 * Allows ANY modern data type (IEEE-754 FP32/FP16/BF16, INT8/INT16, and exact Decimal 10^-d)
 * to be represented 100% losslessly in Balanced Ternary bitplanes without modifying weights!
 */

#ifndef TERNARY_UNIVERSAL_CODEC_HPP
#define TERNARY_UNIVERSAL_CODEC_HPP

#include "ternary_core.hpp"
#include "ternary_dense_algebra.hpp"
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

namespace bt::universal {

// ==============================================================================
// 1. Universal Tri-Radix Scalar: Value = M_bal3 * 2^(e2) * 3^(e3) * 5^(e5)
//    Represents Binary Floats (2^e2), Decimal Finance (10^d = 2^d * 5^d),
//    and Ternary Numbers (3^e3) with 100% exactness and ZERO repeating fractions!
// ==============================================================================
struct TriRadixScalar {
    TritWord64 mantissa{0, 0}; // 39-trit exact balanced ternary integer mantissa
    int16_t exp2{0};           // Exact binary exponent (2^exp2)
    int16_t exp3{0};           // Exact ternary exponent (3^exp3)
    int16_t exp5{0};           // Exact decimal-quinary exponent (5^exp5)

    constexpr TriRadixScalar() noexcept = default;
    constexpr TriRadixScalar(TritWord64 m, int16_t e2, int16_t e3 = 0, int16_t e5 = 0) noexcept
        : mantissa(m), exp2(e2), exp3(e3), exp5(e5) {}

    // 0-Cycle Negation: swaps P and N rails of the mantissa!
    constexpr TriRadixScalar operator-() const noexcept {
        return TriRadixScalar(-mantissa, exp2, exp3, exp5);
    }

    // Exact lossless construction from any IEEE-754 float (0 precision loss!)
    static inline TriRadixScalar from_float_lossless(float val) noexcept {
        if (val == 0.0f || !std::isfinite(val)) {
            return TriRadixScalar(TritWord64(0, 0), 0, 0, 0);
        }
        int e2 = 0;
        float norm = std::frexp(val, &e2); // val = norm * 2^e2, |norm| in [0.5, 1.0)
        // Scale by 2^24 so significand is an exact 24-bit integer -> fits in 16 trits!
        int64_t exact_sig = static_cast<int64_t>(std::ldexp(norm, 24));
        int16_t final_e2 = static_cast<int16_t>(e2 - 24);

        // Strip factors of 3 into exp3 for canonical compression
        int16_t e3 = 0;
        while (exact_sig != 0 && (exact_sig % 3 == 0)) {
            exact_sig /= 3;
            e3++;
        }
        return TriRadixScalar(Codec::from_int64(exact_sig, 16), final_e2, e3, 0);
    }

    // Exact lossless construction from a decimal number: coeff * 10^(exp10)
    // e.g., $0.10 = 1 * 10^(-1) -> mantissa = 1, exp2 = -1, exp5 = -1 (No repeating fraction!)
    static inline TriRadixScalar from_decimal_exact(int64_t coeff, int16_t exp10) noexcept {
        return TriRadixScalar(Codec::from_int64(coeff, 39), exp10, 0, exp10);
    }

    inline double to_double() const noexcept {
        int64_t m = Codec::to_int64(mantissa, 39);
        if (m == 0) return 0.0;
        double v = std::ldexp(static_cast<double>(m), exp2);
        if (exp3 != 0) v *= std::pow(3.0, static_cast<double>(exp3));
        if (exp5 != 0) v *= std::pow(5.0, static_cast<double>(exp5));
        return v;
    }
};

// ==============================================================================
// 2. Unmodified Float/BF16 Tensor -> Block-Scaled Balanced Ternary Bitplanes
//    Decomposes any standard float matrix [rows x cols] into P balanced ternary
//    bitplanes per 64-element block without modifying the original weights!
// ==============================================================================
template <size_t NumPlanes = 8>
struct BlockTernaryTensor {
    static_assert(NumPlanes >= 1 && NumPlanes <= 16, "NumPlanes must be in [1, 16]");

    // Maximum integer representable by NumPlanes balanced trits: (3^NumPlanes - 1) / 2
    static constexpr int64_t MAX_TRIT_INT = []() constexpr {
        int64_t p = 1;
        for (size_t i = 0; i < NumPlanes; ++i) p *= 3;
        return (p - 1) / 2;
    }();

    size_t rows{0};
    size_t cols{0};
    size_t blocks_per_row{0}; // ceil(cols / 64)

    // Layout: [row * blocks_per_row + blk]
    std::vector<float> block_scales; // Shared scale per 64-element block
    // Layout: [(row * blocks_per_row + blk) * NumPlanes + plane]
    std::vector<TritWord64> planes;
    // Bitmask of non-zero planes in each block: bit p is 1 iff planes[p] != 0
    std::vector<uint16_t> active_plane_mask;

    BlockTernaryTensor() = default;
    BlockTernaryTensor(size_t r, size_t c)
        : rows(r), cols(c), blocks_per_row((c + 63) / 64),
          block_scales(r * blocks_per_row, 0.0f),
          planes(r * blocks_per_row * NumPlanes, TritWord64(0, 0)),
          active_plane_mask(r * blocks_per_row, 0) {}

    /**
     * @brief Decomposes an unmodified row-major float matrix [rows x cols]
     * into Balanced Ternary bitplanes with 0 bias and automatic zero-plane masking.
     */
    static BlockTernaryTensor from_float_matrix(const float* src, size_t r, size_t c) {
        BlockTernaryTensor out(r, c);
        const size_t bpr = out.blocks_per_row;

        for (size_t i = 0; i < r; ++i) {
            const float* row_ptr = src + i * c;
            for (size_t b = 0; b < bpr; ++b) {
                const size_t base_col = b * 64;
                const size_t lim = std::min<size_t>(64, c - base_col);
                const size_t blk_idx = i * bpr + b;

                // 1. Find block maximum absolute value
                float max_abs = 0.0f;
                for (size_t k = 0; k < lim; ++k) {
                    float a = std::fabs(row_ptr[base_col + k]);
                    if (a > max_abs) max_abs = a;
                }
                if (max_abs == 0.0f) {
                    out.block_scales[blk_idx] = 0.0f;
                    out.active_plane_mask[blk_idx] = 0;
                    continue;
                }

                // 2. Compute block scale so max_abs maps to MAX_TRIT_INT = (3^P - 1)/2
                const float scale = max_abs / static_cast<float>(MAX_TRIT_INT);
                const float inv_scale = static_cast<float>(MAX_TRIT_INT) / max_abs;
                out.block_scales[blk_idx] = scale;

                // 3. Decompose all 64 integers into NumPlanes Dual-Rail TritWord64s
                TritWord64* blk_planes = &out.planes[blk_idx * NumPlanes];
                for (size_t k = 0; k < lim; ++k) {
                    int64_t q = std::llround(row_ptr[base_col + k] * inv_scale);
                    q = std::clamp<int64_t>(q, -MAX_TRIT_INT, MAX_TRIT_INT);
                    uint64_t bit = 1ULL << k;
                    for (size_t p = 0; p < NumPlanes && q != 0; ++p) {
                        int64_t rem = q % 3;
                        q /= 3;
                        if (rem == 2)       { rem = -1; q += 1; }
                        else if (rem == -2) { rem =  1; q -= 1; }

                        if (rem > 0)      blk_planes[p].P |= bit;
                        else if (rem < 0) blk_planes[p].N |= bit;
                    }
                }

                // 4. Record which planes are non-zero (for 0-cycle sparsity skipping)
                uint16_t mask = 0;
                for (size_t p = 0; p < NumPlanes; ++p) {
                    if ((blk_planes[p].P | blk_planes[p].N) != 0ULL) {
                        mask |= static_cast<uint16_t>(1U << p);
                    }
                }
                out.active_plane_mask[blk_idx] = mask;
            }
        }
        return out;
    }

    /**
     * @brief Reconstructs the float matrix from the Balanced Ternary planes.
     */
    void to_float_matrix(float* dst) const noexcept {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t b = 0; b < blocks_per_row; ++b) {
                const size_t base_col = b * 64;
                const size_t lim = std::min<size_t>(64, cols - base_col);
                const size_t blk_idx = i * blocks_per_row + b;
                const float scale = block_scales[blk_idx];
                if (scale == 0.0f) {
                    for (size_t k = 0; k < lim; ++k) dst[i * cols + base_col + k] = 0.0f;
                    continue;
                }
                const TritWord64* blk_planes = &planes[blk_idx * NumPlanes];
                for (size_t k = 0; k < lim; ++k) {
                    int64_t acc = 0;
                    int64_t p3 = 1;
                    for (size_t p = 0; p < NumPlanes; ++p) {
                        int64_t t = static_cast<int64_t>((blk_planes[p].P >> k) & 1ULL) -
                                    static_cast<int64_t>((blk_planes[p].N >> k) & 1ULL);
                        acc += t * p3;
                        p3 *= 3;
                    }
                    dst[i * cols + base_col + k] = static_cast<float>(acc) * scale;
                }
            }
        }
    }
};

} // namespace bt::universal

#endif // TERNARY_UNIVERSAL_CODEC_HPP
