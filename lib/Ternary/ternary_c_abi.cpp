/**
 * @file ternary_c_abi.cpp
 * @brief Implementation of the Balanced Ternary Dynamic Shared Runtime Library C ABI
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "include/Ternary/ternary_c_abi.h"
#include "ternary_core.hpp"
#include "ternary_fp.hpp"
#include "ternary_sparse.hpp"
#include "libbt.hpp"

#include <cmath>
#include <vector>

extern "C" {

int64_t bt_swar_word64_to_int64(bt_word64_t w, size_t num_trits) {
    if (num_trits == 0) {
        return 0;
    }
    if (num_trits > 39) {
        num_trits = 39;
    }
    bt::TritWord64 tw(w.p, w.n);
    return bt::swar_to_int64(tw, num_trits);
}

int64_t bt_dot_product_array(const bt_word64_t* a, const bt_word64_t* b, size_t num_words) {
    if (!a || !b || num_words == 0) {
        return 0;
    }
    int64_t total = 0;
    for (size_t i = 0; i < num_words; ++i) {
        total += bt_dot_product_64(a[i], b[i]);
    }
    return total;
}

bt_fp32_t bt_fp32_from_double(double val) {
    if (std::isnan(val) || std::isinf(val)) {
        return bt_fp32_t{ 0, 0 };
    }
    bt::BTFP32 fp = bt::BTFP32::from_double(val);
    return bt_fp32_t{ fp.mantissa(), fp.exponent() };
}

double bt_fp32_to_double(bt_fp32_t a) {
    bt::BTFP32 fp(a.mantissa, a.exponent);
    return fp.to_double();
}

bt_fp32_t bt_fp32_add(bt_fp32_t a, bt_fp32_t b) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 fpb(b.mantissa, b.exponent);
    bt::BTFP32 res = fpa.add(fpb);
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_sub(bt_fp32_t a, bt_fp32_t b) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 fpb(b.mantissa, b.exponent);
    bt::BTFP32 res = fpa.sub(fpb);
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_mul(bt_fp32_t a, bt_fp32_t b) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 fpb(b.mantissa, b.exponent);
    bt::BTFP32 res = fpa.mul(fpb);
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_div(bt_fp32_t a, bt_fp32_t b) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 fpb(b.mantissa, b.exponent);
    if (fpb.is_zero()) {
        return bt_fp32_t{ 0, 0 };
    }
    bt::BTFP32 res = fpa.div(fpb);
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_fma(bt_fp32_t a, bt_fp32_t b, bt_fp32_t c) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 fpb(b.mantissa, b.exponent);
    bt::BTFP32 fpc(c.mantissa, c.exponent);
    bt::BTFP32 res = bt::BTFP32::fma(fpa, fpb, fpc);
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_sqrt(bt_fp32_t a) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 res = fpa.sqrt();
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_abs(bt_fp32_t a) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 res = fpa.abs();
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_neg(bt_fp32_t a) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 res = fpa.neg();
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

bt_fp32_t bt_fp32_pow_int(bt_fp32_t a, int64_t exp) {
    bt::BTFP32 fpa(a.mantissa, a.exponent);
    bt::BTFP32 res = fpa.pow_int(exp);
    return bt_fp32_t{ res.mantissa(), res.exponent() };
}

void bt_csr_spmv_exec(const bt_csr_t* mat, const int64_t* x, int64_t* y) {
    if (!mat || !x || !y) {
        return;
    }
    for (uint64_t r = 0; r < mat->rows; ++r) {
        int64_t sum = 0;
        if (mat->row_ptr) {
            uint32_t start = mat->row_ptr[r];
            uint32_t end = mat->row_ptr[r + 1];
            for (uint32_t idx = start; idx < end; ++idx) {
                uint32_t c = mat->col_indices[idx];
                bool is_pos = (mat->signs[idx / 64] >> (idx % 64)) & 1ULL;
                sum += is_pos ? x[c] : -x[c];
            }
        }
        y[r] = sum;
    }
}

int bt_radix3_fft_forward(const double* real_in, const double* imag_in, double* real_out, double* imag_out, size_t n) {
    if (!real_in || !real_out || !imag_out || n == 0) {
        return -1;
    }
    if (!bt::Radix3FFT::is_power_of_3(n)) {
        return -2;
    }
    std::vector<bt::Radix3FFT::cdouble> vec(n);
    for (size_t i = 0; i < n; ++i) {
        double re = real_in[i];
        double im = imag_in ? imag_in[i] : 0.0;
        vec[i] = bt::Radix3FFT::cdouble(re, im);
    }
    bt::Radix3FFT::transform_inplace(vec, false);
    for (size_t i = 0; i < n; ++i) {
        real_out[i] = vec[i].real();
        imag_out[i] = vec[i].imag();
    }
    return 0;
}

int bt_radix3_fft_inverse(const double* real_in, const double* imag_in, double* real_out, double* imag_out, size_t n) {
    if (!real_in || !real_out || !imag_out || n == 0) {
        return -1;
    }
    if (!bt::Radix3FFT::is_power_of_3(n)) {
        return -2;
    }
    std::vector<bt::Radix3FFT::cdouble> vec(n);
    for (size_t i = 0; i < n; ++i) {
        double re = real_in[i];
        double im = imag_in ? imag_in[i] : 0.0;
        vec[i] = bt::Radix3FFT::cdouble(re, im);
    }
    bt::Radix3FFT::transform_inplace(vec, true);
    for (size_t i = 0; i < n; ++i) {
        real_out[i] = vec[i].real();
        imag_out[i] = vec[i].imag();
    }
    return 0;
}

int bt_radix3_convolve(const double* a, size_t na, const double* b, size_t nb, double* out) {
    if (!a || !b || !out || na == 0 || nb == 0) {
        return -1;
    }
    std::vector<double> va(a, a + na);
    std::vector<double> vb(b, b + nb);
    std::vector<double> res = bt::Radix3FFT::convolve_real_radix3(va, vb);
    size_t out_len = na + nb - 1;
    for (size_t i = 0; i < out_len && i < res.size(); ++i) {
        out[i] = res[i];
    }
    return 0;
}

bt_word64_t bt_kogge_stone_add64_val(bt_word64_t a, bt_word64_t b, int8_t cin, int8_t* cout_msb) {
    bt::TritWord64 wa{a.p, a.n};
    bt::TritWord64 wb{b.p, b.n};
    bt::TritWord64 wsum = bt::kogge_stone_add64(wa, wb, cin, cout_msb);
    return bt_word64_t{wsum.P, wsum.N};
}

int64_t bt_dot_product_64_val(bt_word64_t a, bt_word64_t b) {
    return bt_dot_product_64(a, b);
}

bt_btha_result_t bt_btha_step_val(bt_word64_t a, bt_word64_t b) {
    return bt_btha_step(a, b);
}

uint8_t bt_base243_pack_val(const int8_t trits[5]) {
    return bt_base243_pack(trits);
}

void bt_base243_unpack_val(uint8_t byte, int8_t out_trits[5]) {
    bt_base243_unpack(byte, out_trits);
}

uint32_t bt_hamming_distance_64_val(bt_word64_t a, bt_word64_t b) {
    return bt_hamming_distance_64(a, b);
}

bool bt_verify_disjointness_val(bt_word64_t word) {
    return bt_verify_disjointness(word);
}

int64_t bt_kleene_and(int64_t a, int64_t b) {
    if (a >= -1 && a <= 1 && b >= -1 && b <= 1) {
        return (a < b) ? a : b;
    }
    auto wa = bt::Codec::from_int64(a);
    auto wb = bt::Codec::from_int64(b);
    auto wr = bt::kleene_and(wa, wb);
    return bt::swar_to_int64(wr);
}

int64_t bt_kleene_or(int64_t a, int64_t b) {
    if (a >= -1 && a <= 1 && b >= -1 && b <= 1) {
        return (a > b) ? a : b;
    }
    auto wa = bt::Codec::from_int64(a);
    auto wb = bt::Codec::from_int64(b);
    auto wr = bt::kleene_or(wa, wb);
    return bt::swar_to_int64(wr);
}

int64_t bt_kleene_xor(int64_t a, int64_t b) {
    if (a >= -1 && a <= 1 && b >= -1 && b <= 1) {
        return a * b;
    }
    auto wa = bt::Codec::from_int64(a);
    auto wb = bt::Codec::from_int64(b);
    auto wr = bt::kleene_xor(wa, wb);
    return bt::swar_to_int64(wr);
}

} // extern "C"
