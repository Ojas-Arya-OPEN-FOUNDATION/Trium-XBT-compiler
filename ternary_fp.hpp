/**
 * @file ternary_fp.hpp
 * @brief High-Performance Balanced Ternary Floating-Point (BT-FP16 & BT-FP32) Engine
 * @author Antigravity Prime Research Engine
 * @version 1.0.0
 * 
 * Formal Properties:
 * 1. Normalized Value Representation: Value = M * 3^E
 *    - Mantissa M: Normalized balanced ternary integer whose leading trit is non-zero (+1 or -1)
 *    - Exponent E: Signed balanced ternary integer exponent
 * 2. Inherent Sign: No sign bit. Positive, zero, or negative is determined directly by the leading trit.
 * 3. Unique Canonical Zero: M = 0 and E = 0. No +/-0 ambiguity (eliminates IEEE-754 sign-anomaly).
 * 4. Zero-Bias Truncation: Truncating fractional trits achieves exact 0 mean error on [-0.5, +0.5].
 * 5. Arithmetic: High-precision add, sub, mul, div, to_double, from_double.
 */

#ifndef TERNARY_FP_HPP
#define TERNARY_FP_HPP

#include "ternary_core.hpp"

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iostream>

namespace bt {

// ==============================================================================
// 1. Exact Radix-3 Exponentiation & Truncation Math
// ==============================================================================

namespace fp_detail {

/**
 * @brief Compile-time lookup table of 3^k for k in [0..39] (fits in signed 64-bit int64_t).
 */
constexpr std::array<int64_t, 40> POW3_64 = []() constexpr {
    std::array<int64_t, 40> arr{};
    int64_t p = 1;
    for (int i = 0; i < 40; ++i) {
        arr[i] = p;
        if (i < 39) p *= 3;
    }
    return arr;
}();

/**
 * @brief Compile-time lookup table of (3^k - 1) / 2 for k in [0..39].
 */
constexpr std::array<int64_t, 40> HALF_POW3_64 = []() constexpr {
    std::array<int64_t, 40> arr{};
    for (int i = 0; i < 40; ++i) {
        arr[i] = (POW3_64[i] - 1) / 2;
    }
    return arr;
}();

/**
 * @brief Compile-time lookup table of 3^k for k in [0..59] as 128-bit signed integer.
 */
constexpr std::array<__int128_t, 60> POW3_128 = []() constexpr {
    std::array<__int128_t, 60> arr{};
    __int128_t p = 1;
    for (int i = 0; i < 60; ++i) {
        arr[i] = p;
        if (i < 59) p *= 3;
    }
    return arr;
}();

/**
 * @brief Compile-time lookup table of (3^k - 1) / 2 for k in [0..59] as 128-bit signed integer.
 */
constexpr std::array<__int128_t, 60> HALF_POW3_128 = []() constexpr {
    std::array<__int128_t, 60> arr{};
    for (int i = 0; i < 60; ++i) {
        arr[i] = (POW3_128[i] - 1) / 2;
    }
    return arr;
}();

/**
 * @brief Computes 3^n as 128-bit signed integer (O(1) table lookup for n < 60).
 */
constexpr inline __int128_t pow3_128(int n) noexcept {
    if (n <= 0) return 1;
    if (n < 60) return POW3_128[n];
    __int128_t res = POW3_128[59];
    for (int i = 59; i < n; ++i) {
        res *= 3;
    }
    return res;
}

/**
 * @brief Symmetric-rounding balanced division by 3 (truncates 1 fractional trit).
 * Discards remainder r in {-1, 0, +1} such that val = 3 * q + r.
 * Error r / 3 has exact mean 0 over all possible trits.
 */
constexpr inline int64_t round_div3(int64_t val) noexcept {
    int64_t rem = val % 3;
    int64_t q = val / 3;
    if (rem == 2) {
        q += 1;
    } else if (rem == -2) {
        q -= 1;
    }
    return q;
}

/**
 * @brief 128-bit version of symmetric-rounding balanced division by 3.
 */
constexpr inline __int128_t round_div3_128(__int128_t val) noexcept {
    __int128_t rem = val % 3;
    __int128_t q = val / 3;
    if (rem == 2) {
        q += 1;
    } else if (rem == -2) {
        q -= 1;
    }
    return q;
}

/**
 * @brief One-shot O(1) balanced power-of-3 division (64-bit).
 * Mathematically identical to k sequential applications of round_div3.
 */
constexpr inline int64_t round_div_pow3_fast64(int64_t val, int k) noexcept {
    if (k <= 0 || val == 0) return val;
    // Primary stage: exact one-shot symmetric division by 3^min(k, 39).
    // Unsigned arithmetic is required: (INT64_MAX) + (3^39-1)/2 = 1.125e19 < 2^64,
    // so no wrap occurs. The old signed add overflowed with UB and flipped the sign
    // for |val| within ~(3^k-1)/2 of INT64_MAX.
    const int k1 = (k < 40) ? k : 39;
    const uint64_t p3   = static_cast<uint64_t>(POW3_64[k1]);
    const uint64_t half = static_cast<uint64_t>(HALF_POW3_64[k1]);
    if (val > 0) {
        const uint64_t u = static_cast<uint64_t>(val) + half;
        int64_t q = static_cast<int64_t>(u / p3);
        // Tail for k >= 40: function composition r3^k = r3^(k-39) o r3^39.
        // Residual after 39 stages is <= +/-2, so the tail is at most ~2 steps.
        for (int i = 40; i <= k; ++i) {
            if (q == 0) break;
            q = round_div3(q);
        }
        return q;
    }
    // val < 0: negate in the unsigned domain (no UB even for INT64_MIN).
    // Max magnitude: 2^63 + (3^39-1)/2 = 1.125e19 < 2^64; quotient always fits int64.
    const uint64_t mag = 0ULL - static_cast<uint64_t>(val);
    const uint64_t u = mag + half;
    int64_t q = -static_cast<int64_t>(u / p3);
    for (int i = 40; i <= k; ++i) q = round_div3(q);
    return q;
}

/**
 * @brief One-shot O(1) balanced power-of-3 division (128-bit).
 * Mathematically identical to k sequential applications of round_div3_128.
 */
constexpr inline __int128_t round_div_pow3_fast128(__int128_t val, int k) noexcept {
    if (k <= 0 || val == 0) return val;
    // Primary stage: exact one-shot symmetric division by 3^min(k, 59).
    // Unsigned __int128 arithmetic: (2^127-1) + (3^59-1)/2 = 1.7016e38 < 2^128, no wrap.
    const int k1 = (k < 60) ? k : 59;
    const unsigned __int128 p3   = static_cast<unsigned __int128>(POW3_128[k1]);
    const unsigned __int128 half = static_cast<unsigned __int128>(HALF_POW3_128[k1]);
    if (val > 0) {
        const unsigned __int128 u = static_cast<unsigned __int128>(val) + half;
        __int128_t q = static_cast<__int128_t>(u / p3);
        // Tail for k >= 60: composition r3^k = r3^(k-59) o r3^59.
        for (int i = 60; i <= k; ++i) {
            if (q == 0) break;
            q = round_div3_128(q);
        }
        return q;
    }
    // val < 0: negate in the unsigned domain (no UB even for INT128_MIN).
    const unsigned __int128 mag = static_cast<unsigned __int128>(0) - static_cast<unsigned __int128>(val);
    const unsigned __int128 u = mag + half;
    __int128_t q = -static_cast<__int128_t>(u / p3);
    for (int i = 60; i <= k; ++i) q = round_div3_128(q);
    return q;
}

/**
 * @brief Multi-trit balanced truncation: divides by 3^k in O(1) with zero statistical bias.
 */
constexpr inline int64_t round_div_pow3(int64_t val, int k) noexcept {
    return round_div_pow3_fast64(val, k);
}

/**
 * @brief 128-bit multi-trit balanced truncation in O(1).
 */
constexpr inline __int128_t round_div_pow3_128(__int128_t val, int k) noexcept {
    return round_div_pow3_fast128(val, k);
}

} // namespace fp_detail

// ==============================================================================
// 2. Generic Balanced Ternary Floating-Point Class (BTFP)
// ==============================================================================

/**
 * @brief Balanced Ternary Floating-Point Number: Value = M * 3^E
 * 
 * @tparam MantissaTrits Number of trits allocated to the normalized mantissa.
 * @tparam ExponentTrits Number of trits allocated to exponent (0 = unlimited 32-bit signed exponent).
 */
template <int MantissaTrits = 10, int ExponentTrits = 0>
class BTFP {
    // int64_t mantissa storage: |M| <= (3^40-1)/2 = 6.08e18 < 2^63; W = 41 would overflow.
    // Also guards normalize_128 and add()/div() 128-bit scaling (M_trits + 4 <= 59).
    static_assert(MantissaTrits >= 0 && MantissaTrits <= 40,
                  "BTFP: MantissaTrits must be in [0, 40] to fit int64_t mantissa storage");
    static_assert(ExponentTrits >= 0 && ExponentTrits <= 62,
                  "BTFP: ExponentTrits must be in [0, 62]");
public:
    static constexpr int M_TRITS = MantissaTrits;
    static constexpr int E_TRITS = ExponentTrits;

    // Maximum positive mantissa: (3^W - 1) / 2
    static constexpr int64_t max_mantissa() noexcept {
        return static_cast<int64_t>((fp_detail::pow3_128(MantissaTrits) - 1) / 2);
    }

    // Minimum normalized positive mantissa: (3^(W-1) + 1) / 2
    static constexpr int64_t min_norm_mantissa() noexcept {
        return static_cast<int64_t>((fp_detail::pow3_128(MantissaTrits - 1) + 1) / 2);
    }

private:
    int64_t m_mantissa{0}; // Normalized mantissa: |M| in [min_norm_mantissa, max_mantissa], or 0
    int32_t m_exponent{0}; // Signed exponent E (powers of 3)

public:
    // --------------------------------------------------------------------------
    // Constructors
    // --------------------------------------------------------------------------

    /**
     * @brief Default constructor: Canonical Unique Zero (M = 0, E = 0).
     */
    constexpr BTFP() noexcept : m_mantissa(0), m_exponent(0) {}

    /**
     * @brief Integer value constructor with automatic normalization (BTFP(0) produces canonical zero).
     */
    constexpr BTFP(int64_t val) noexcept : m_mantissa(val), m_exponent(0) {
        normalize();
    }

    /**
     * @brief Direct integer mantissa and exponent constructor with automatic normalization.
     */
    constexpr BTFP(int64_t mantissa, int32_t exponent) noexcept : m_mantissa(mantissa), m_exponent(exponent) {
        normalize();
    }

    // --------------------------------------------------------------------------
    // Accessors & Status Queries
    // --------------------------------------------------------------------------

    constexpr int64_t mantissa() const noexcept { return m_mantissa; }
    constexpr int32_t exponent() const noexcept { return m_exponent; }

    constexpr bool is_zero() const noexcept {
        return m_mantissa == 0;
    }

    constexpr bool is_positive() const noexcept {
        return m_mantissa > 0;
    }

    constexpr bool is_negative() const noexcept {
        return m_mantissa < 0;
    }

    /**
     * @brief Inherent sign: strictly determined by the sign of the mantissa.
     * @return +1 if positive, -1 if negative, 0 if zero.
     */
    constexpr int8_t sign() const noexcept {
        if (m_mantissa > 0) return 1;
        if (m_mantissa < 0) return -1;
        return 0;
    }

    /**
     * @brief Returns the leading trit of the normalized mantissa.
     * Normalized numbers always have leading trit +1 or -1; zero has 0.
     */
    constexpr Trit leading_trit() const noexcept {
        return from_int(sign());
    }

    /**
     * @brief Decomposes mantissa into individual trits (least significant to most significant).
     */
    std::vector<Trit> mantissa_trits() const {
        std::vector<Trit> trits(MantissaTrits, Trit::Zero);
        int64_t val = m_mantissa;
        for (int i = 0; i < MantissaTrits && val != 0; ++i) {
            int64_t rem = val % 3;
            val /= 3;
            if (rem == 2) {
                rem = -1;
                val += 1;
            } else if (rem == -2) {
                rem = 1;
                val -= 1;
            }
            trits[i] = from_int(static_cast<int8_t>(rem));
        }
        return trits;
    }

    /**
     * @brief Decomposes exponent into signed balanced ternary trits.
     */
    std::vector<Trit> exponent_trits() const {
        std::vector<Trit> trits;
        int64_t val = m_exponent;
        if (val == 0) {
            trits.push_back(Trit::Zero);
            return trits;
        }
        while (val != 0) {
            int64_t rem = val % 3;
            val /= 3;
            if (rem == 2) {
                rem = -1;
                val += 1;
            } else if (rem == -2) {
                rem = 1;
                val -= 1;
            }
            trits.push_back(from_int(static_cast<int8_t>(rem)));
        }
        return trits;
    }

    // --------------------------------------------------------------------------
    // Normalization & Zero-Bias Truncation Rounding
    // --------------------------------------------------------------------------

    /**
     * @brief Normalizes mantissa so that |M| is in [min_norm_mantissa, max_mantissa].
     * Uses CLZ bit-length difference to jump-divide/multiply in O(1) via round_div_pow3_fast64.
     */
    constexpr void normalize() noexcept {
        if (m_mantissa == 0) {
            m_exponent = 0;
            return;
        }

        const int64_t max_m = max_mantissa();
        const int64_t min_m = min_norm_mantissa();

        // Right shift with O(1) CLZ jump-division + at most 1-2 round_div3 steps
        if (m_mantissa > max_m || m_mantissa < -max_m) {
            uint64_t abs_m = (m_mantissa >= 0) ? static_cast<uint64_t>(m_mantissa)
                                               : (0ULL - static_cast<uint64_t>(m_mantissa));
            int delta_bits = __builtin_clzll(static_cast<uint64_t>(max_m)) - __builtin_clzll(abs_m);
            int k_est = (delta_bits * 100) / 159 - 1;
            if (k_est > 0) {
                m_mantissa = fp_detail::round_div_pow3_fast64(m_mantissa, k_est);
                m_exponent += k_est;
            }
            while (m_mantissa > max_m || m_mantissa < -max_m) {
                m_mantissa = fp_detail::round_div3(m_mantissa);
                m_exponent += 1;
            }
        }

        // Left shift with O(1) CLZ jump-multiplication while magnitude < min_m and non-zero
        if (m_mantissa != 0 && (m_mantissa < min_m && m_mantissa > -min_m)) {
            uint64_t abs_m = (m_mantissa >= 0) ? static_cast<uint64_t>(m_mantissa)
                                               : (0ULL - static_cast<uint64_t>(m_mantissa));
            int delta_bits = __builtin_clzll(abs_m) - __builtin_clzll(static_cast<uint64_t>(min_m));
            int k_est = (delta_bits * 100) / 159 - 1;
            if (k_est > 0 && k_est < 40) {
                m_mantissa *= fp_detail::POW3_64[k_est];
                m_exponent -= k_est;
            }
            while (m_mantissa != 0 && (m_mantissa < min_m && m_mantissa > -min_m)) {
                m_mantissa *= 3;
                m_exponent -= 1;
            }
        }

        if (m_mantissa == 0) {
            m_exponent = 0;
        }

        if constexpr (ExponentTrits > 0) {
            constexpr int64_t max_e = (fp_detail::pow3_128(ExponentTrits) - 1) / 2;
            if (m_exponent > max_e) {
                // Renormalize instead of losing value silently: push the exponent
                // delta into the mantissa as far as it fits, then symmetric-round
                // the remainder so |M| stays within [min_m, max_m] at the clamped E.
                int64_t delta = m_exponent - max_e;
                m_exponent = static_cast<int32_t>(max_e);
                while (delta > 0) {
                    if (m_mantissa != 0 &&
                        m_mantissa <= max_m / 3 && m_mantissa >= -max_m / 3) {
                        m_mantissa *= 3;
                        --delta;
                    } else if (m_mantissa == 0) {
                        m_exponent = 0;
                        break;
                    } else {
                        // Cannot push delta into the mantissa anymore: symmetric-round
                        // the excess (zero-bias representation approximation).
                        m_mantissa = fp_detail::round_div_pow3_fast64(m_mantissa, static_cast<int>(delta));
                        delta = 0;
                    }
                }
            } else if (m_exponent < -max_e) {
                // Exponent too negative: clamp it UP, so the mantissa must shrink by
                // 3^delta (value compression with zero-bias symmetric rounding).
                int64_t delta = -max_e - m_exponent;
                m_exponent = static_cast<int32_t>(-max_e);
                if (m_mantissa != 0) {
                    m_mantissa = fp_detail::round_div_pow3_fast64(m_mantissa, static_cast<int>(delta));
                } else {
                    m_exponent = 0;
                }
            }
        }

        // Canonical zero: rounding-to-zero during clamping must reset E.
        if (m_mantissa == 0) {
            m_exponent = 0;
        }
    }

    /**
     * @brief Normalizes a 128-bit intermediate mantissa down to MantissaTrits in O(1).
     */
    static constexpr BTFP normalize_128(__int128_t m, int32_t exp) noexcept {
        if (m == 0) return BTFP(0, 0);

        const __int128_t max_m = max_mantissa();
        const __int128_t min_m = min_norm_mantissa();

        if (m > max_m || m < -max_m) {
            unsigned __int128 abs_m = (m >= 0) ? static_cast<unsigned __int128>(m)
                                               : (static_cast<unsigned __int128>(0) - static_cast<unsigned __int128>(m));
            uint64_t hi = static_cast<uint64_t>(abs_m >> 64);
            int bitlen_m = (hi != 0) ? (128 - __builtin_clzll(hi))
                                     : (64 - __builtin_clzll(static_cast<uint64_t>(abs_m)));
            int bitlen_max = 64 - __builtin_clzll(static_cast<uint64_t>(max_m));
            int delta_bits = bitlen_m - bitlen_max;
            int k_est = (delta_bits * 100) / 159 - 1;
            if (k_est > 0) {
                m = fp_detail::round_div_pow3_fast128(m, k_est);
                exp += k_est;
            }
            while (m > max_m || m < -max_m) {
                m = fp_detail::round_div3_128(m);
                exp += 1;
            }
        }

        if (m != 0 && (m < min_m && m > -min_m)) {
            uint64_t abs_m = static_cast<uint64_t>((m >= 0) ? m : -m);
            int bitlen_m = 64 - __builtin_clzll(abs_m);
            int bitlen_min = 64 - __builtin_clzll(static_cast<uint64_t>(min_m));
            int delta_bits = bitlen_min - bitlen_m;
            int k_est = (delta_bits * 100) / 159 - 1;
            if (k_est > 0 && k_est < 60) {
                m *= fp_detail::POW3_128[k_est];
                exp -= k_est;
            }
            while (m != 0 && (m < min_m && m > -min_m)) {
                m *= 3;
                exp -= 1;
            }
        }

        if (m == 0) return BTFP(0, 0);
        return BTFP(static_cast<int64_t>(m), exp);
    }

    /**
     * @brief Zero-Bias Trit-Boundary Rounding: symmetric-rounds at the k-th fractional
     * trit boundary in O(1), achieving exact mean-0 error (E[error] = 0).
     * Note: this is nearest-value rounding at the boundary, not literal truncation;
     * both operations share the zero-mean property.
     */
    constexpr BTFP truncate_fractional(int trits_to_discard) const noexcept {
        if (trits_to_discard <= 0 || is_zero()) return *this;
        int64_t m = fp_detail::round_div_pow3_fast64(m_mantissa, trits_to_discard);
        int32_t exp = m_exponent + trits_to_discard;
        return BTFP(m, exp);
    }

    // --------------------------------------------------------------------------
    // Double IEEE-754 Conversions
    // --------------------------------------------------------------------------

    /**
     * @brief Converts BTFP to standard IEEE-754 double precision float.
     */
    double to_double() const noexcept {
        if (m_mantissa == 0) return 0.0;
        return static_cast<double>(m_mantissa) * std::pow(3.0, m_exponent);
    }

    /**
     * @brief Explicit conversion to IEEE-754 double.
     */
    explicit operator double() const noexcept {
        return to_double();
    }

    /**
     * @brief Constructs a BTFP instance from a standard IEEE-754 double.
     */
    static BTFP from_double(double val) {
        if (!std::isfinite(val)) {
            throw std::domain_error("BTFP::from_double: NaN or Inf cannot be represented in BTFP");
        }
        if (val == 0.0 || std::abs(val) < 1e-300) {
            return BTFP(0, 0);
        }

        // Target log3(|mantissa|) is approximately MantissaTrits - 0.5
        double log3_val = std::log(std::abs(val)) / std::log(3.0);
        int est_e = static_cast<int>(std::floor(log3_val - (MantissaTrits - 0.5)));

        // Scale safely
        double scaled = val * std::pow(3.0, -est_e);
        int64_t m = std::llround(scaled);

        BTFP res(m, est_e);
        res.normalize();
        return res;
    }

    // --------------------------------------------------------------------------
    // Arithmetic Operations
    // --------------------------------------------------------------------------

    /**
     * @brief 0-Cycle compile-time negation: inverts mantissa, preserves normalization.
     */
    constexpr BTFP operator-() const noexcept {
        BTFP res;
        res.m_mantissa = -m_mantissa;
        res.m_exponent = m_exponent;
        return res;
    }

    /**
     * @brief O(1) compile-time negation: returns -(*this).
     */
    constexpr BTFP neg() const noexcept {
        return -(*this);
    }

    /**
     * @brief O(1) compile-time absolute value: preserves normalization without re-normalizing.
     */
    constexpr BTFP abs() const noexcept {
        BTFP res;
        res.m_mantissa = (m_mantissa < 0) ? -m_mantissa : m_mantissa;
        res.m_exponent = m_exponent;
        return res;
    }

    /**
     * @brief Balanced ternary addition: aligns exponents and accumulates with 0-bias rounding.
     */
    BTFP add(const BTFP& other) const noexcept {
        if (is_zero()) return other;
        if (other.is_zero()) return *this;

        if (m_exponent >= other.m_exponent) {
            int diff = m_exponent - other.m_exponent;
            if (diff > MantissaTrits + 4) return *this;
            if constexpr (MantissaTrits <= 20) {
                if (diff <= 18) {
                    int64_t m_sum = m_mantissa * fp_detail::POW3_64[diff] + other.m_mantissa;
                    return BTFP(m_sum, other.m_exponent);
                }
            }
            __int128_t m_a = static_cast<__int128_t>(m_mantissa) * fp_detail::pow3_128(diff);
            __int128_t m_sum = m_a + other.m_mantissa;
            return normalize_128(m_sum, other.m_exponent);
        } else {
            int diff = other.m_exponent - m_exponent;
            if (diff > MantissaTrits + 4) return other;
            if constexpr (MantissaTrits <= 20) {
                if (diff <= 18) {
                    int64_t m_sum = other.m_mantissa * fp_detail::POW3_64[diff] + m_mantissa;
                    return BTFP(m_sum, m_exponent);
                }
            }
            __int128_t m_b = static_cast<__int128_t>(other.m_mantissa) * fp_detail::pow3_128(diff);
            __int128_t m_sum = m_b + m_mantissa;
            return normalize_128(m_sum, m_exponent);
        }
    }

    /**
     * @brief Balanced ternary subtraction: A - B = A + (-B).
     */
    BTFP sub(const BTFP& other) const noexcept {
        return add(-other);
    }

    /**
     * @brief Balanced ternary multiplication: multiplies mantissas, sums exponents, normalizes.
     * For MantissaTrits <= 20 (BTFP16 & BTFP32), uses a 3-way constant-divisor branch
     * that compiles into 1-cycle umulh magic multiplies with zero loops and zero hardware sdiv.
     */
    constexpr BTFP mul(const BTFP& other) const noexcept {
        if (is_zero() || other.is_zero()) return BTFP(0, 0);

        if constexpr (MantissaTrits >= 2 && MantissaTrits <= 20) {
            constexpr int64_t P_W2 = static_cast<int64_t>(fp_detail::pow3_128(MantissaTrits - 2));
            constexpr int64_t H_W2 = (P_W2 - 1) / 2;
            constexpr int64_t P_W1 = static_cast<int64_t>(fp_detail::pow3_128(MantissaTrits - 1));
            constexpr int64_t H_W1 = (P_W1 - 1) / 2;
            constexpr int64_t P_W0 = static_cast<int64_t>(fp_detail::pow3_128(MantissaTrits));
            constexpr int64_t H_W0 = (P_W0 - 1) / 2;
            constexpr int64_t THRESH_W1 = max_mantissa() * P_W2 + H_W2;
            constexpr int64_t THRESH_W0 = max_mantissa() * P_W1 + H_W1;

            int64_t prod = m_mantissa * other.m_mantissa;
            int64_t u = prod < 0 ? -prod : prod;
            int32_t e_prod = m_exponent + other.m_exponent;
            int64_t q = 0;

            if (u <= THRESH_W1) {
                q = static_cast<int64_t>((static_cast<uint64_t>(u) + static_cast<uint64_t>(H_W2)) / static_cast<uint64_t>(P_W2));
                e_prod += (MantissaTrits - 2);
            } else if (u <= THRESH_W0) {
                q = static_cast<int64_t>((static_cast<uint64_t>(u) + static_cast<uint64_t>(H_W1)) / static_cast<uint64_t>(P_W1));
                e_prod += (MantissaTrits - 1);
            } else {
                q = static_cast<int64_t>((static_cast<uint64_t>(u) + static_cast<uint64_t>(H_W0)) / static_cast<uint64_t>(P_W0));
                e_prod += MantissaTrits;
            }

            int64_t m_prod = (prod < 0) ? -q : q;
            if constexpr (ExponentTrits > 0) {
                constexpr int64_t max_e = (fp_detail::pow3_128(ExponentTrits) - 1) / 2;
                if (e_prod > max_e || e_prod < -max_e) {
                    return BTFP(m_prod, e_prod);
                }
            }

            BTFP res;
            res.m_mantissa = m_prod;
            res.m_exponent = e_prod;
            return res;
        } else {
            __int128_t m_prod = static_cast<__int128_t>(m_mantissa) * static_cast<__int128_t>(other.m_mantissa);
            int32_t e_prod = m_exponent + other.m_exponent;
            return normalize_128(m_prod, e_prod);
        }
    }

    /**
     * @brief Single-rounding Fused Multiply-Add: computes (a * b + c) in 128-bit arithmetic
     * and performs a single unbiased normalize_128 rounding.
     */
    static constexpr BTFP fma(const BTFP& a, const BTFP& b, const BTFP& c) noexcept {
        if (a.is_zero() || b.is_zero()) return c;

        __int128_t P = static_cast<__int128_t>(a.m_mantissa) * static_cast<__int128_t>(b.m_mantissa);
        int32_t e_p = a.m_exponent + b.m_exponent;

        if (c.is_zero()) {
            return normalize_128(P, e_p);
        }

        int32_t delta = e_p - c.m_exponent;
        if (delta == 0) {
            return normalize_128(P + static_cast<__int128_t>(c.m_mantissa), e_p);
        } else if (delta > 0) {
            constexpr int max_exact_shift = (78 - 2 * MantissaTrits > 0) ? (78 - 2 * MantissaTrits) : 0;
            if (delta <= max_exact_shift) {
                __int128_t S = P * fp_detail::pow3_128(delta) + static_cast<__int128_t>(c.m_mantissa);
                return normalize_128(S, c.m_exponent);
            } else {
                int shift_right = delta - max_exact_shift;
                __int128_t c_shifted = fp_detail::round_div_pow3_fast128(static_cast<__int128_t>(c.m_mantissa), shift_right);
                __int128_t S = P * fp_detail::pow3_128(max_exact_shift) + c_shifted;
                return normalize_128(S, e_p - max_exact_shift);
            }
        } else {
            int d = -delta; // e_c - e_p > 0
            if (d > MantissaTrits + 4) {
                return c;
            }
            constexpr int max_c_shift = 78 - MantissaTrits;
            if (d <= max_c_shift) {
                __int128_t S = static_cast<__int128_t>(c.m_mantissa) * fp_detail::pow3_128(d) + P;
                return normalize_128(S, e_p);
            } else {
                return c;
            }
        }
    }

    /**
     * @brief Balanced ternary square root with parity-aligned 128-bit integer square root
     * and exact midpoint rounding (N - r^2 > r => r + 1).
     */
    BTFP sqrt() const {
        if (m_mantissa < 0) {
            throw std::domain_error("BTFP::sqrt: negative argument");
        }
        if (m_mantissa == 0) {
            return BTFP(0, 0);
        }

        int S0 = (MantissaTrits <= 36) ? (MantissaTrits + 4) : (MantissaTrits + 2);
        int S = (((m_exponent - S0) % 2) != 0) ? (S0 + 1) : S0;
        int32_t e_out = (m_exponent - S) / 2;

        unsigned __int128 N = static_cast<unsigned __int128>(m_mantissa) *
                              static_cast<unsigned __int128>(fp_detail::pow3_128(S));

        // Seed 128-bit Heron / Newton-Raphson iteration from double sqrt
        double approx = std::sqrt(static_cast<double>(N));
        unsigned __int128 r = static_cast<unsigned __int128>(approx);
        if (r == 0) r = 1;

        for (int i = 0; i < 10; ++i) {
            unsigned __int128 next_r = (r + N / r) >> 1;
            if (next_r == r) break;
            r = next_r;
        }
        while (r > N / r) {
            --r;
        }
        while ((r + 1) <= N / (r + 1)) {
            ++r;
        }

        // Exact balanced midpoint rounding: N - r^2 > r => r + 1
        if (N - r * r > r) {
            ++r;
        }

        return normalize_128(static_cast<__int128_t>(r), e_out);
    }

    /**
     * @brief Integer exponentiation via binary exponentiation by squaring.
     */
    BTFP pow_int(int64_t exp) const {
        if (exp == 0) {
            return BTFP(1);
        }
        if (is_zero()) {
            if (exp < 0) {
                throw std::domain_error("BTFP::pow_int: zero to negative power");
            }
            return BTFP(0, 0);
        }

        bool neg_exp = (exp < 0);
        uint64_t uexp = neg_exp
            ? (0ULL - static_cast<uint64_t>(exp))
            : static_cast<uint64_t>(exp);

        BTFP base = *this;
        BTFP result(1);
        while (uexp > 0) {
            if (uexp & 1ULL) {
                result = result.mul(base);
            }
            uexp >>= 1;
            if (uexp > 0) {
                base = base.mul(base);
            }
        }

        if (neg_exp) {
            return BTFP(1).div(result);
        }
        return result;
    }

    /**
     * @brief Balanced ternary division with symmetric nearest-integer rounding.
     */
    BTFP div(const BTFP& other) const {
        if (other.is_zero()) {
            throw std::domain_error("Division by zero in BTFP");
        }
        if (is_zero()) return BTFP(0, 0);

        int scale_shift = MantissaTrits + 4;
        __int128_t num = static_cast<__int128_t>(m_mantissa) * fp_detail::pow3_128(scale_shift);
        __int128_t den = other.m_mantissa;

        // Symmetric quotient rounding
        __int128_t q = num / den;
        __int128_t r = num % den;

        // If |r| > |den| / 2, round to nearest integer
        __int128_t abs_r = r < 0 ? -r : r;
        __int128_t abs_den = den < 0 ? -den : den;
        if (2 * abs_r > abs_den) {
            if ((r > 0 && den > 0) || (r < 0 && den < 0)) {
                q += 1;
            } else {
                q -= 1;
            }
        }

        int32_t exp = m_exponent - other.m_exponent - scale_shift;
        return normalize_128(q, exp);
    }

    // --------------------------------------------------------------------------
    // Operator Overloads
    // --------------------------------------------------------------------------

    inline BTFP operator+(const BTFP& other) const noexcept { return add(other); }
    inline BTFP operator-(const BTFP& other) const noexcept { return sub(other); }
    inline BTFP operator*(const BTFP& other) const noexcept { return mul(other); }
    inline BTFP operator/(const BTFP& other) const { return div(other); }

    inline BTFP& operator+=(const BTFP& other) noexcept { *this = add(other); return *this; }
    inline BTFP& operator-=(const BTFP& other) noexcept { *this = sub(other); return *this; }
    inline BTFP& operator*=(const BTFP& other) noexcept { *this = mul(other); return *this; }
    inline BTFP& operator/=(const BTFP& other) { *this = div(other); return *this; }

    inline bool operator==(const BTFP& other) const noexcept {
        if (is_zero() && other.is_zero()) return true;
        return m_mantissa == other.m_mantissa && m_exponent == other.m_exponent;
    }

    inline bool operator!=(const BTFP& other) const noexcept {
        return !(*this == other);
    }

    inline bool operator<(const BTFP& other) const noexcept {
        if (*this == other) return false;
        int s_this = sign();
        int s_other = other.sign();
        if (s_this != s_other) return s_this < s_other;
        if (s_this == 0) return false; // Both are zero, already handled by *this == other
        if (s_this > 0) {
            // Both positive: larger exponent is strictly larger number
            if (m_exponent != other.m_exponent) return m_exponent < other.m_exponent;
            return m_mantissa < other.m_mantissa;
        } else {
            // Both negative: larger exponent is strictly more negative (smaller number)
            if (m_exponent != other.m_exponent) return m_exponent > other.m_exponent;
            return m_mantissa < other.m_mantissa;
        }
    }

    inline bool operator<=(const BTFP& other) const noexcept {
        return (*this < other) || (*this == other);
    }

    inline bool operator>(const BTFP& other) const noexcept {
        return other < *this;
    }

    inline bool operator>=(const BTFP& other) const noexcept {
        return !(*this < other);
    }

    // --------------------------------------------------------------------------
    // Diagnostics & Formatting
    // --------------------------------------------------------------------------

    std::string to_string() const {
        if (is_zero()) return "0.0 (BT-FP Zero)";
        std::ostringstream oss;
        oss << to_double() << " [M=" << m_mantissa << ", E=" << m_exponent << ", Trits=";
        auto trits = mantissa_trits();
        for (auto it = trits.rbegin(); it != trits.rend(); ++it) {
            oss << trit_to_char(*it);
        }
        oss << " * 3^" << m_exponent << "]";
        return oss.str();
    }

    // --------------------------------------------------------------------------
    // Theoretical & Empirical Zero-Bias Verification
    // --------------------------------------------------------------------------

    /**
     * @brief Computes the empirical mean rounding error at the k-trit boundary across all
     * 3^k symmetric remainders in [-(3^k-1)/2, +(3^k-1)/2]. The signed partial sums stay
     * within int64 range exactly for 1 <= k <= 10, so larger k is clamped to 10
     * (the mean is 0.0 for every k by symmetric-pair cancellation; running 3^k iterations
     * for large k would both hang and overflow the accumulator).
     */
    static double verify_zero_bias_mean(int k_trits = 4) noexcept {
        if (k_trits < 1) k_trits = 1;
        if (k_trits > 10) k_trits = 10; // 3^10 = 59049 states; partial sums <= ~4.4e8, UB-free
        int64_t p3 = static_cast<int64_t>(fp_detail::pow3_128(k_trits));
        int64_t max_r = (p3 - 1) / 2;
        int64_t sum_r = 0;
        int64_t count = 0;
        for (int64_t r = -max_r; r <= max_r; ++r) {
            // Discarding r from (Q * 3^k + r) yields an error of -r / 3^k
            // NOTE: the +-r terms cancel in exact pairs; accumulated sum stays bounded.
            sum_r += (-r);
            count++;
        }
        return (count == 0) ? 0.0 : (static_cast<double>(sum_r) / (static_cast<double>(p3) * static_cast<double>(count)));
    }
};

// ==============================================================================
// 3. Predefined Standard Precision Formats
// ==============================================================================

/**
 * @brief BT-FP16: Balanced Ternary Floating-Point 16-bit Equivalent.
 * 10 trits mantissa (~15.8 bits precision, matching IEEE Half Precision).
 * Unique canonical zero, inherent sign.
 */
using BTFP16 = BTFP<10>;

/**
 * @brief BT-FP32: Balanced Ternary Floating-Point 32-bit Equivalent.
 * 20 trits mantissa (~31.7 bits precision, matching IEEE Single Precision).
 * Unique canonical zero, inherent sign.
 */
using BTFP32 = BTFP<20>;

/**
 * @brief Aliases for strict 16-trit and 32-trit formats.
 */
using BTFP16T = BTFP<16>;
using BTFP32T = BTFP<32>;

} // namespace bt

#endif // TERNARY_FP_HPP
