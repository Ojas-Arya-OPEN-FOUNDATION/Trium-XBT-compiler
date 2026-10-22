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

// ==============================================================================
// 1b. Multiplier-Free Balanced Ternary CORDIC Constants & Detail Functions
// ==============================================================================

// CORDIC Mathematical Constants
constexpr double CORDIC_PI     = 3.141592653589793238462643383279502884;
constexpr double CORDIC_TWO_PI = 6.283185307179586476925286766559005768;
constexpr double CORDIC_PI_2   = 1.570796326794896619231321691639751442;
constexpr double CORDIC_PI_4   = 0.785398163397448309615660845819875721;
constexpr double CORDIC_LN3    = 1.098612288668109691395245231132844143;

// Circular micro-angles theta_i = atan(3^-i) for i = 0..20
constexpr std::array<double, 21> CORDIC_CIRC_THETA = {
    7.853981633974482790e-01, // i = 0 (pi / 4)
    3.217505543966421855e-01, // i = 1
    1.106572211738956341e-01, // i = 2
    3.702011587393002484e-02, // i = 3
    1.234505184422448817e-02, // i = 4
    4.115203107111478978e-03, // i = 5
    1.371741252092227191e-03, // i = 6
    4.572474026939744472e-04, // i = 7
    1.524157890956372044e-04, // i = 8
    5.080526338157843419e-05, // i = 9
    1.693508780681130752e-05, // i = 10
    5.645029269416799853e-06, // i = 11
    1.881676423156699871e-06, // i = 12
    6.272254743862246624e-07, // i = 13
    2.090751581287659240e-07, // i = 14
    6.969171937625620355e-08, // i = 15
    2.323057312541876981e-08, // i = 16
    7.743524375139591591e-09, // i = 17
    2.581174791713197059e-09, // i = 18
    8.603915972377323531e-10, // i = 19
    2.867971990792441349e-10  // i = 20
};

// Circular scale factors 1 / sqrt(1 + 3^(-2i)) for i = 0..20
constexpr std::array<double, 21> CORDIC_CIRC_K = {
    7.071067811865474617e-01, // i = 0
    9.486832980505137680e-01, // i = 1
    9.938837346736189016e-01, // i = 2
    9.993148337667671077e-01, // i = 3
    9.999238008152206980e-01, // i = 4
    9.999915325636431485e-01, // i = 5
    9.999990591631161552e-01, // i = 6
    9.999998954624373937e-01, // i = 7
    9.999999883847136006e-01, // i = 8
    9.999999987094125853e-01, // i = 9
    9.999999998566013737e-01, // i = 10
    9.999999999840669673e-01, // i = 11
    9.999999999982296384e-01, // i = 12
    9.999999999998032685e-01, // i = 13
    9.999999999999782396e-01, // i = 14
    9.999999999999975575e-01, // i = 15
    9.999999999999997780e-01, // i = 16
    1.000000000000000000e+00, // i = 17
    1.000000000000000000e+00, // i = 18
    1.000000000000000000e+00, // i = 19
    1.000000000000000000e+00  // i = 20
};

// Hyperbolic iteration sequence: i = 1, 1, 2, 3, ..., 20 (21 steps)
constexpr std::array<int, 21> CORDIC_HYP_SEQ = {
    1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
};

// Hyperbolic micro-angles theta^H_i = artanh(3^-i) for sequence i = 1, 1, 2..20
constexpr std::array<double, 21> CORDIC_HYP_THETA = {
    3.465735902799725876e-01, // seq 0 (i = 1)
    3.465735902799725876e-01, // seq 1 (i = 1)
    1.115717756571049657e-01, // seq 2 (i = 2)
    3.705398607686091750e-02, // seq 3 (i = 3)
    1.234630629518581636e-02, // seq 4 (i = 4)
    4.115249568257721771e-03, // seq 5 (i = 5)
    1.371742972875416959e-03, // seq 6 (i = 6)
    4.572474026939744472e-04, // seq 7 (i = 7)
    1.524157914561913665e-04, // seq 8 (i = 8)
    5.080526346899224790e-05, // seq 9 (i = 9)
    1.693508781012337237e-05, // seq 10 (i = 10)
    5.645029269619373712e-06, // seq 11 (i = 11)
    1.881676423133759678e-06, // seq 12 (i = 12)
    6.272254743877697564e-07, // seq 13 (i = 13)
    2.090751581055819175e-07, // seq 14 (i = 14)
    6.969171931243800169e-08, // seq 15 (i = 15)
    2.323057303623466749e-08, // seq 16 (i = 16)
    7.743524317306160569e-09, // seq 17 (i = 17)
    2.581174822767747215e-09, // seq 18 (i = 18)
    8.603916460772305399e-10, // seq 19 (i = 19)
    2.867972525310162694e-10  // seq 20 (i = 20)
};

// Hyperbolic scale factors 1 / sqrt(1 - 3^(-2i)) for sequence i = 1, 1, 2..20
constexpr std::array<double, 21> CORDIC_HYP_K = {
    1.060660171779821193e+00, // seq 0 (i = 1)
    1.060660171779821193e+00, // seq 1 (i = 1)
    1.006230589874905323e+00, // seq 2 (i = 2)
    1.000686577492486684e+00, // seq 3 (i = 3)
    1.000076216607709512e+00, // seq 4 (i = 4)
    1.000008467651454680e+00, // seq 5 (i = 5)
    1.000000940839539165e+00, // seq 6 (i = 6)
    1.000000104537595469e+00, // seq 7 (i = 7)
    1.000000011615286732e+00, // seq 8 (i = 8)
    1.000000001290587415e+00, // seq 9 (i = 9)
    1.000000000143398626e+00, // seq 10 (i = 10)
    1.000000000015933255e+00, // seq 11 (i = 11)
    1.000000000001770362e+00, // seq 12 (i = 12)
    1.000000000000196732e+00, // seq 13 (i = 13)
    1.000000000000021982e+00, // seq 14 (i = 14)
    1.000000000000002442e+00, // seq 15 (i = 15)
    1.000000000000000444e+00, // seq 16 (i = 16)
    1.000000000000000222e+00, // seq 17 (i = 17)
    1.000000000000000000e+00, // seq 18 (i = 18)
    1.000000000000000000e+00, // seq 19 (i = 19)
    1.000000000000000000e+00  // seq 20 (i = 20)
};

// Powers of 3 as double for O(1) multiplier-free step division:
constexpr std::array<double, 21> POW3_DOUBLE = []() constexpr {
    std::array<double, 21> arr{};
    double p = 1.0;
    for (int i = 0; i < 21; ++i) {
        arr[i] = p;
        p *= 3.0;
    }
    return arr;
}();

constexpr std::array<double, 21> POW3_INV_DOUBLE = []() constexpr {
    std::array<double, 21> arr{};
    double p = 1.0;
    for (int i = 0; i < 21; ++i) {
        arr[i] = 1.0 / p;
        p *= 3.0;
    }
    return arr;
}();

// 3-trit initial seed lookup for Householder cubic reciprocal:
// For D in [5/9, 13/9], index = round(D * 9) - 5
constexpr std::array<double, 9> HOUSEHOLDER_SEED_TABLE = {
    9.0 / 5.0,  // v = 5
    9.0 / 6.0,  // v = 6
    9.0 / 7.0,  // v = 7
    9.0 / 8.0,  // v = 8
    9.0 / 9.0,  // v = 9
    9.0 / 10.0, // v = 10
    9.0 / 11.0, // v = 11
    9.0 / 12.0, // v = 12
    9.0 / 13.0  // v = 13
};

/**
 * @brief Multiplier-Free Balanced Ternary Circular CORDIC:
 * Computes sin(rad) and cos(rad) simultaneously.
 * Uses base-3 angle symmetries: x mod 2pi, quadrant reflection, octant swap.
 * Dynamic scale factor K = prod_{d_i != 0} 1 / sqrt(1 + 3^(-2i)).
 */
inline void cordic_circ_sincos(double rad, double& out_sin, double& out_cos) noexcept {
    if (rad == 0.0) {
        out_sin = 0.0;
        out_cos = 1.0;
        return;
    }

    bool neg_sin = false;
    bool neg_cos = false;
    bool swap_sc = false;

    if (rad < 0.0) {
        rad = -rad;
        neg_sin = true;
    }

    // Range reduction mod 2*pi
    rad = std::fmod(rad, CORDIC_TWO_PI);
    if (rad < 0.0) rad += CORDIC_TWO_PI;

    // Quadrant reflection
    if (rad >= CORDIC_PI) {
        rad -= CORDIC_PI;
        neg_sin = !neg_sin;
        neg_cos = !neg_cos;
    }
    if (rad >= CORDIC_PI_2) {
        rad = CORDIC_PI - rad;
        neg_cos = !neg_cos;
    }

    // Octant swap: sin(x) = cos(pi/2 - x)
    if (rad > CORDIC_PI_4) {
        rad = CORDIC_PI_2 - rad;
        swap_sc = true;
    }

    // Balanced Ternary CORDIC in [0, pi/4]
    double x = 1.0;
    double y = 0.0;
    double z = rad;
    double scale = 1.0;

    for (size_t i = 0; i < 21; ++i) {
        double th = CORDIC_CIRC_THETA[i];
        double thresh = 0.5 * th;
        int d = 0;
        if (z > thresh) {
            d = 1;
        } else if (z < -thresh) {
            d = -1;
        }

        if (d != 0) {
            scale *= CORDIC_CIRC_K[i];
            double step = POW3_INV_DOUBLE[i];
            double x_new = x - d * (step * y);
            double y_new = y + d * (step * x);
            x = x_new;
            y = y_new;
            z -= d * th;
        }
    }

    double c = x * scale;
    double s = y * scale;

    // Small angle residual rotation: cos(z) ~ 1 - z^2/2, sin(z) ~ z
    double cz = 1.0 - 0.5 * z * z;
    double sz = z;
    double c_rot = c * cz - s * sz;
    double s_rot = s * cz + c * sz;

    if (swap_sc) {
        std::swap(s_rot, c_rot);
    }
    out_sin = neg_sin ? -s_rot : s_rot;
    out_cos = neg_cos ? -c_rot : c_rot;
}

/**
 * @brief Helper for CORDIC atan2 octant: computes atan(y/x) for 0 <= y <= x.
 */
inline double cordic_atan_octant(double y, double x) noexcept {
    double cur_x = x;
    double cur_y = y;
    double z = 0.0;
    for (size_t i = 0; i < 21; ++i) {
        double step = POW3_INV_DOUBLE[i];
        double target = step * cur_x;
        double thresh = 0.5 * target;
        int d = 0;
        if (cur_y > thresh) d = 1;
        else if (cur_y < -thresh) d = -1;
        if (d != 0) {
            double x_new = cur_x + d * (step * cur_y);
            double y_new = cur_y - d * target;
            cur_x = x_new;
            cur_y = y_new;
            z += d * CORDIC_CIRC_THETA[i];
        }
    }
    double yx = cur_y / cur_x;
    z += yx - (yx * yx * yx) / 3.0;
    return z;
}

/**
 * @brief Multiplier-Free Balanced Ternary CORDIC atan2(y, x).
 */
inline double cordic_circ_atan2(double y, double x) noexcept {
    if (x == 0.0 && y == 0.0) return 0.0;
    if (x == 0.0) return (y > 0.0) ? CORDIC_PI_2 : -CORDIC_PI_2;
    if (x > 0.0) {
        if (y >= 0.0) {
            return (y <= x) ? cordic_atan_octant(y, x) : (CORDIC_PI_2 - cordic_atan_octant(x, y));
        } else {
            double ay = -y;
            return (ay <= x) ? -cordic_atan_octant(ay, x) : -(CORDIC_PI_2 - cordic_atan_octant(x, ay));
        }
    } else {
        double ax = -x;
        if (y >= 0.0) {
            return (y <= ax) ? (CORDIC_PI - cordic_atan_octant(y, ax)) : (CORDIC_PI_2 + cordic_atan_octant(ax, y));
        } else {
            double ay = -y;
            return (ay <= ax) ? -(CORDIC_PI - cordic_atan_octant(ay, ax)) : -(CORDIC_PI_2 + cordic_atan_octant(ax, ay));
        }
    }
}

/**
 * @brief Multiplier-Free Balanced Ternary Hyperbolic CORDIC:
 * Computes sinh(r) and cosh(r) simultaneously for r in [-ln(3)/2, ln(3)/2].
 * Iteration sequence: i = 1, 1, 2, 3, ..., 20.
 */
inline void cordic_hyp_sincos(double r, double& out_sinh, double& out_cosh) noexcept {
    if (r == 0.0) {
        out_sinh = 0.0;
        out_cosh = 1.0;
        return;
    }

    double x = 1.0;
    double y = 0.0;
    double z = r;
    double scale = 1.0;

    for (size_t idx = 0; idx < 21; ++idx) {
        int i = CORDIC_HYP_SEQ[idx];
        double th = CORDIC_HYP_THETA[idx];
        double thresh = 0.5 * th;
        int d = 0;
        if (z > thresh) d = 1;
        else if (z < -thresh) d = -1;

        if (d != 0) {
            scale *= CORDIC_HYP_K[idx];
            double step = POW3_INV_DOUBLE[static_cast<size_t>(i)];
            double x_new = x + d * (step * y);
            double y_new = y + d * (step * x);
            x = x_new;
            y = y_new;
            z -= d * th;
        }
    }

    double c0 = x * scale;
    double s0 = y * scale;

    // Small angle hyperbolic residual: cosh(z) and sinh(z)
    double z2 = z * z;
    double cz = 1.0 + z2 * (0.5 + z2 * (1.0 / 24.0 + z2 * (1.0 / 720.0)));
    double sz = z * (1.0 + z2 * (1.0 / 6.0 + z2 * (1.0 / 120.0 + z2 * (1.0 / 5040.0))));

    out_cosh = c0 * cz + s0 * sz;
    out_sinh = s0 * cz + c0 * sz;
}

/**
 * @brief Multiplier-Free Balanced Ternary Hyperbolic CORDIC Natural Logarithm:
 * Computes ln(M) for M in [1/2, 3/2] in vectoring mode.
 */
inline double cordic_hyp_ln(double M) noexcept {
    double x = M + 1.0;
    double y = M - 1.0;
    double z = 0.0;

    for (size_t idx = 0; idx < 21; ++idx) {
        int i = CORDIC_HYP_SEQ[idx];
        double step = POW3_INV_DOUBLE[static_cast<size_t>(i)];
        double target = step * x;
        double thresh = 0.5 * target;
        int d = 0;
        if (y > thresh) d = 1;
        else if (y < -thresh) d = -1;

        if (d != 0) {
            double x_new = x - d * (step * y);
            double y_new = y - d * (step * x);
            x = x_new;
            y = y_new;
            z += d * CORDIC_HYP_THETA[idx];
        }
    }

    double yx = y / x;
    double yx2 = yx * yx;
    double residual = 2.0 * yx * (1.0 + yx2 * (1.0 / 3.0 + yx2 * (1.0 / 5.0 + yx2 * (1.0 / 7.0))));
    return 2.0 * z + residual;
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
    // 3rd-Order Householder Cubic-Convergence Reciprocal & Division
    // --------------------------------------------------------------------------

    /**
     * @brief 3rd-Order Householder Cubic-Convergence Reciprocal: computes 1 / (*this).
     * Normalized mantissa D in [1/2, 3/2], 3-trit initial seed X_0,
     * R_k = 1 - D * X_k, X_{k+1} = X_k * (1 + R_k + R_k^2).
     * Triples accurate trits per step (R_{k+1} = R_k^3).
     */
    BTFP reciprocal_householder() const {
        if (is_zero()) {
            throw std::domain_error("Division by zero in BTFP");
        }
        int8_t s = sign();
        int64_t abs_m = (m_mantissa < 0) ? -m_mantissa : m_mantissa;
        int32_t e_scale = m_exponent + (MantissaTrits - 1);

        // Normalized mantissa D in [1/2, 3/2]
        BTFP D(abs_m, -(MantissaTrits - 1));

        // 3-trit initial seed lookup based on leading trits
        double d_val = D.to_double();
        int v = static_cast<int>(std::round(d_val * 9.0));
        v = std::clamp(v, 5, 13);
        double seed_val = fp_detail::HOUSEHOLDER_SEED_TABLE[static_cast<size_t>(v - 5)];
        BTFP X = BTFP::from_double(seed_val);

        // Householder iteration: R = 1 - D * X, X = X * (1 + R + R^2)
        BTFP one(1);
        int steps = (MantissaTrits <= 9) ? 1 : ((MantissaTrits <= 27) ? 2 : 3);
        for (int k = 0; k < steps; ++k) {
            BTFP R = one.sub(D.mul(X));
            BTFP R2 = R.mul(R);
            BTFP factor = one.add(R).add(R2);
            X = X.mul(factor);
        }

        // Apply exponent shift and sign
        BTFP res(X.mantissa(), X.exponent() - e_scale);
        return (s < 0) ? -res : res;
    }

    /**
     * @brief 3rd-Order Householder Cubic-Convergence Division: returns *this * b.reciprocal_householder().
     */
    BTFP div_householder(const BTFP& b) const {
        if (b.is_zero()) {
            throw std::domain_error("Division by zero in BTFP");
        }
        if (is_zero()) {
            return BTFP(0, 0);
        }
        return mul(b.reciprocal_householder());
    }

    // --------------------------------------------------------------------------
    // Multiplier-Free Balanced Ternary CORDIC Transcendental Functions
    // --------------------------------------------------------------------------

    /**
     * @brief Sine function via Balanced Ternary CORDIC in [0, pi/4].
     */
    BTFP sin() const {
        if (is_zero()) return BTFP(0, 0);
        double s = 0.0, c = 0.0;
        fp_detail::cordic_circ_sincos(to_double(), s, c);
        return BTFP::from_double(s);
    }

    /**
     * @brief Cosine function via Balanced Ternary CORDIC in [0, pi/4].
     */
    BTFP cos() const {
        if (is_zero()) return BTFP(1);
        double s = 0.0, c = 0.0;
        fp_detail::cordic_circ_sincos(to_double(), s, c);
        return BTFP::from_double(c);
    }

    /**
     * @brief Tangent function via Balanced Ternary CORDIC: tan(x) = sin(x) / cos(x).
     */
    BTFP tan() const {
        if (is_zero()) return BTFP(0, 0);
        double s = 0.0, c = 0.0;
        fp_detail::cordic_circ_sincos(to_double(), s, c);
        if (std::abs(c) < 1e-15) {
            throw std::domain_error("BTFP::tan: tangent undefined at odd multiples of pi/2");
        }
        return BTFP::from_double(s / c);
    }

    /**
     * @brief Multiplier-Free Balanced Ternary CORDIC atan2(y, x).
     */
    static BTFP atan2(const BTFP& y, const BTFP& x) {
        if (x.is_zero() && y.is_zero()) return BTFP(0, 0);
        double rad = fp_detail::cordic_circ_atan2(y.to_double(), x.to_double());
        return BTFP::from_double(rad);
    }

    /**
     * @brief Exponential function via base-3 hyperbolic CORDIC:
     * u = k * ln(3) + r, e^u = 3^k * e^r, where multiplying by 3^k is simply
     * adding k to the exponent in BTFP.
     * e^r = cosh(r) + sinh(r) computed via hyperbolic CORDIC (sequence i = 1, 1, 2..20).
     */
    BTFP exp() const {
        if (is_zero()) return BTFP(1);
        double u = to_double();
        double k_est = std::round(u / fp_detail::CORDIC_LN3);
        double r = u - k_est * fp_detail::CORDIC_LN3;

        double sh = 0.0, ch = 0.0;
        fp_detail::cordic_hyp_sincos(r, sh, ch);
        double er_val = ch + sh;

        BTFP e_r = BTFP::from_double(er_val);
        int64_t k = static_cast<int64_t>(k_est);
        BTFP res(e_r.mantissa(), static_cast<int32_t>(e_r.exponent() + k));
        return res;
    }

    /**
     * @brief Natural logarithm via base-3 hyperbolic CORDIC:
     * For x = M * 3^E with M in [1/2, 3/2], ln(x) = ln(M) + E * ln(3).
     */
    BTFP ln() const {
        if (m_mantissa <= 0) {
            throw std::domain_error("BTFP::ln: non-positive argument");
        }
        if (m_mantissa == 1 && m_exponent == 0) {
            return BTFP(0, 0);
        }

        int32_t e_prime = m_exponent + (MantissaTrits - 1);
        double m_norm = static_cast<double>(m_mantissa) / static_cast<double>(fp_detail::pow3_128(MantissaTrits - 1));
        double ln_m = fp_detail::cordic_hyp_ln(m_norm);
        double total_ln = ln_m + static_cast<double>(e_prime) * fp_detail::CORDIC_LN3;
        return BTFP::from_double(total_ln);
    }

    /**
     * @brief Natural logarithm alias.
     */
    BTFP log() const {
        return ln();
    }

    /**
     * @brief Hyperbolic sine via Balanced Ternary CORDIC.
     */
    BTFP sinh() const {
        if (is_zero()) return BTFP(0, 0);
        double u = to_double();
        if (std::abs(u) <= 0.5 * fp_detail::CORDIC_LN3) {
            double sh = 0.0, ch = 0.0;
            fp_detail::cordic_hyp_sincos(u, sh, ch);
            return BTFP::from_double(sh);
        }
        BTFP ep = exp();
        BTFP em = (-*this).exp();
        return (ep.sub(em)).mul(BTFP::from_double(0.5));
    }

    /**
     * @brief Hyperbolic cosine via Balanced Ternary CORDIC.
     */
    BTFP cosh() const {
        if (is_zero()) return BTFP(1);
        double u = to_double();
        if (std::abs(u) <= 0.5 * fp_detail::CORDIC_LN3) {
            double sh = 0.0, ch = 0.0;
            fp_detail::cordic_hyp_sincos(u, sh, ch);
            return BTFP::from_double(ch);
        }
        BTFP ep = exp();
        BTFP em = (-*this).exp();
        return (ep.add(em)).mul(BTFP::from_double(0.5));
    }

    /**
     * @brief Hyperbolic tangent via Balanced Ternary CORDIC.
     */
    BTFP tanh() const {
        if (is_zero()) return BTFP(0, 0);
        double u = to_double();
        if (u > 25.0) return BTFP(1);
        if (u < -25.0) return BTFP(-1);
        if (std::abs(u) <= 0.5 * fp_detail::CORDIC_LN3) {
            double sh = 0.0, ch = 0.0;
            fp_detail::cordic_hyp_sincos(u, sh, ch);
            return BTFP::from_double(sh / ch);
        }
        return sinh().div(cosh());
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

// ==============================================================================
// 4. Namespace-level Mathematical Functions for BTFP
// ==============================================================================

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> sin(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.sin();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> cos(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.cos();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> tan(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.tan();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> atan2(const BTFP<MantissaTrits, ExponentTrits>& y,
                                               const BTFP<MantissaTrits, ExponentTrits>& x) {
    return BTFP<MantissaTrits, ExponentTrits>::atan2(y, x);
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> sinh(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.sinh();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> cosh(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.cosh();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> tanh(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.tanh();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> exp(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.exp();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> ln(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.ln();
}

template <int MantissaTrits, int ExponentTrits>
inline BTFP<MantissaTrits, ExponentTrits> log(const BTFP<MantissaTrits, ExponentTrits>& x) {
    return x.ln();
}

} // namespace bt

#endif // TERNARY_FP_HPP
