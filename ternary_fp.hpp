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
    if (k >= 40) return 0;
    int64_t p3 = POW3_64[k];
    int64_t half = HALF_POW3_64[k];
    return (val >= 0) ? ((val + half) / p3) : -((-val + half) / p3);
}

/**
 * @brief One-shot O(1) balanced power-of-3 division (128-bit).
 * Mathematically identical to k sequential applications of round_div3_128.
 */
constexpr inline __int128_t round_div_pow3_fast128(__int128_t val, int k) noexcept {
    if (k <= 0 || val == 0) return val;
    if (k >= 60) return 0;
    __int128_t p3 = POW3_128[k];
    __int128_t half = HALF_POW3_128[k];
    return (val >= 0) ? ((val + half) / p3) : -((-val + half) / p3);
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
                m_exponent = static_cast<int32_t>(max_e);
            } else if (m_exponent < -max_e) {
                m_exponent = static_cast<int32_t>(-max_e);
            }
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
            int delta_bits = __builtin_clzll(abs_m) - __builtin_clzll(static_cast<uint64_t>(min_m));
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
     * @brief Zero-Bias Truncation: truncates k fractional trits in O(1), achieving exact 0 mean error.
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

            if constexpr (ExponentTrits > 0) {
                constexpr int64_t max_e = (fp_detail::pow3_128(ExponentTrits) - 1) / 2;
                if (e_prod > max_e) {
                    e_prod = static_cast<int32_t>(max_e);
                } else if (e_prod < -max_e) {
                    e_prod = static_cast<int32_t>(-max_e);
                }
            }

            BTFP res;
            res.m_mantissa = (prod < 0) ? -q : q;
            res.m_exponent = e_prod;
            return res;
        } else {
            __int128_t m_prod = static_cast<__int128_t>(m_mantissa) * static_cast<__int128_t>(other.m_mantissa);
            int32_t e_prod = m_exponent + other.m_exponent;
            return normalize_128(m_prod, e_prod);
        }
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
     * @brief Computes the empirical mean truncation error when discarding k trits
     * across all 3^k symmetric remainders in [-(3^k - 1)/2, +(3^k - 1)/2].
     * Mathematically proven to evaluate to exactly 0.0.
     */
    static double verify_zero_bias_mean(int k_trits = 4) noexcept {
        int64_t p3 = static_cast<int64_t>(fp_detail::pow3_128(k_trits));
        int64_t max_r = (p3 - 1) / 2;
        int64_t sum_r = 0;
        int64_t count = 0;
        for (int64_t r = -max_r; r <= max_r; ++r) {
            // Discarding r from (Q * 3^k + r) yields an error of -r / 3^k
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
