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
#include "ternary_xbt_blas.hpp"
#include "ternary_nonlinear_engine.hpp"

#include <cmath>
#include <vector>
#include <cstdio>
#include <limits>
#include <algorithm>
#include <type_traits>

namespace {
thread_local int abi_error_code = BT_STATUS_OK;
thread_local char abi_error_message[256] = {};
void abi_error(int code, const char* message) noexcept {
    abi_error_code = code;
    std::snprintf(abi_error_message, sizeof abi_error_message, "%s", message);
}
void validate_word(bt_word64_t w) {
    if(w.p&w.n) throw std::invalid_argument("invalid dual-rail word: overlapping positive/negative rails");
}
template <class F> auto abi_call(F&& function) noexcept -> decltype(function()) {
    using Result = decltype(function());
    abi_error(BT_STATUS_OK, "");
    try { return function(); }
    catch (const std::overflow_error& e) { abi_error(BT_STATUS_OVERFLOW, e.what()); }
    catch (const std::invalid_argument& e) { abi_error(BT_STATUS_INVALID_ARGUMENT, e.what()); }
    catch (const std::domain_error& e) { abi_error(BT_STATUS_INVALID_ARGUMENT, e.what()); }
    catch (const std::exception& e) { abi_error(BT_STATUS_INTERNAL_ERROR, e.what()); }
    catch (...) { abi_error(BT_STATUS_INTERNAL_ERROR, "unknown runtime failure"); }
    if constexpr (std::is_void_v<Result>) return;
    else if constexpr (std::is_same_v<Result, double>) return std::numeric_limits<double>::quiet_NaN();
    else if constexpr (std::is_same_v<Result, int>) return -1;
    else return Result{};
}

}

extern "C" {
uint32_t bt_runtime_abi_version(void) { return 1; }
int bt_last_error_code(void) { return abi_error_code; }
const char* bt_last_error_message(void) { return abi_error_message; }
void bt_clear_error(void) { abi_error(BT_STATUS_OK, ""); }

bt_status_t bt_csr_spmv_checked(const bt_csr_t* mat, const int64_t* x,
    size_t x_len, int64_t* y, size_t y_len, size_t row_ptr_len,
    size_t col_indices_len, size_t signs_len) {
    bt_clear_error();
    try {
        if (!mat || mat->rows == UINT64_MAX || row_ptr_len <= mat->rows ||
            x_len < mat->cols || y_len < mat->rows || col_indices_len < mat->nnz ||
            signs_len < mat->nnz / 64 + (mat->nnz % 64 != 0) || !mat->row_ptr ||
            (mat->cols && !x) || (mat->rows && !y) ||
            (mat->nnz && (!mat->col_indices || !mat->signs)))
            throw std::invalid_argument("invalid CSR buffers or lengths");
        if (mat->nnz > UINT32_MAX || mat->row_ptr[0] != 0 || mat->row_ptr[mat->rows] != mat->nnz)
            throw std::invalid_argument("invalid CSR row offsets");
        for (uint64_t r = 0; r < mat->rows; ++r)
            if (mat->row_ptr[r] > mat->row_ptr[r+1] || mat->row_ptr[r+1] > mat->nnz)
                throw std::invalid_argument("CSR row offsets must be monotone and bounded");
        for (uint64_t i = 0; i < mat->nnz; ++i)
            if (mat->col_indices[i] >= mat->cols) throw std::invalid_argument("CSR column out of range");
        std::vector<int64_t> result(mat->rows);
        for (uint64_t r = 0; r < mat->rows; ++r) {
            __int128 sum = 0;
            for (uint32_t i = mat->row_ptr[r]; i < mat->row_ptr[r+1]; ++i) {
                __int128 value = x[mat->col_indices[i]];
                sum += ((mat->signs[i/64] >> (i%64)) & 1) ? value : -value;
            }
            if (sum < INT64_MIN || sum > INT64_MAX) throw std::overflow_error("CSR result exceeds int64 range");
            result[r] = static_cast<int64_t>(sum);
        }
        if (!result.empty()) std::copy(result.begin(), result.end(), y);
        return BT_STATUS_OK;
    } catch (const std::invalid_argument& e) { abi_error(BT_STATUS_INVALID_ARGUMENT, e.what()); }
      catch (const std::overflow_error& e) { abi_error(BT_STATUS_OVERFLOW, e.what()); }
      catch (const std::exception& e) { abi_error(BT_STATUS_INTERNAL_ERROR, e.what()); }
      catch (...) { abi_error(BT_STATUS_INTERNAL_ERROR, "unknown CSR failure"); }
    return static_cast<bt_status_t>(abi_error_code);
}


int64_t bt_swar_word64_to_int64(bt_word64_t w, size_t num_trits) {
    return abi_call([&]() -> int64_t {
        validate_word(w);
        if (num_trits == 0) {
            return 0;
        }
        if (num_trits > 39) {
            throw std::invalid_argument("positional conversion supports at most 39 trits");
        }
        bt::TritWord64 tw(w.p, w.n);
        return bt::swar_to_int64(tw, num_trits);
    });
}

int64_t bt_dot_product_array(const bt_word64_t* a, const bt_word64_t* b, size_t num_words) {
    return abi_call([&]() -> int64_t {
        if(!num_words) return 0;
        if(!a || !b) throw std::invalid_argument("null dot-product input");
        __int128 total=0;
        for(size_t i=0;i<num_words;++i) {validate_word(a[i]);validate_word(b[i]);total+=bt_dot_product_64(a[i],b[i]);}
        if(total<INT64_MIN || total>INT64_MAX) throw std::overflow_error("dot-product result outside int64");
        return static_cast<int64_t>(total);
    });
}

bt_fp32_t bt_fp32_from_double(double val) {
    return abi_call([&]() -> bt_fp32_t {
        if (std::isnan(val) || std::isinf(val)) {
            throw std::domain_error("BTFP32 conversion requires a finite input");
        }
        bt::BTFP32 fp = bt::BTFP32::from_double(val);
        return bt_fp32_t{ fp.mantissa(), fp.exponent() };
    });
}

double bt_fp32_to_double(bt_fp32_t a) {
    return abi_call([&]() -> double {
        bt::BTFP32 fp(a.mantissa, a.exponent);
        return fp.to_double();
    });
}

bt_fp32_t bt_fp32_add(bt_fp32_t a, bt_fp32_t b) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 fpb(b.mantissa, b.exponent);
        bt::BTFP32 res = fpa.add(fpb);
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_sub(bt_fp32_t a, bt_fp32_t b) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 fpb(b.mantissa, b.exponent);
        bt::BTFP32 res = fpa.sub(fpb);
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_mul(bt_fp32_t a, bt_fp32_t b) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 fpb(b.mantissa, b.exponent);
        bt::BTFP32 res = fpa.mul(fpb);
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_div(bt_fp32_t a, bt_fp32_t b) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 fpb(b.mantissa, b.exponent);
        if (fpb.is_zero()) throw std::domain_error("BTFP32 division by zero");
        bt::BTFP32 res = fpa.div(fpb);
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_fma(bt_fp32_t a, bt_fp32_t b, bt_fp32_t c) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 fpb(b.mantissa, b.exponent);
        bt::BTFP32 fpc(c.mantissa, c.exponent);
        bt::BTFP32 res = bt::BTFP32::fma(fpa, fpb, fpc);
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_sqrt(bt_fp32_t a) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 res = fpa.sqrt();
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_abs(bt_fp32_t a) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 res = fpa.abs();
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_neg(bt_fp32_t a) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 res = fpa.neg();
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

bt_fp32_t bt_fp32_pow_int(bt_fp32_t a, int64_t exp) {
    return abi_call([&]() -> bt_fp32_t {
        bt::BTFP32 fpa(a.mantissa, a.exponent);
        bt::BTFP32 res = fpa.pow_int(exp);
        return bt_fp32_t{ res.mantissa(), res.exponent() };
    });
}

void bt_csr_spmv_exec(const bt_csr_t* mat, const int64_t* x, int64_t* y) {
    if (!mat) { abi_error(BT_STATUS_INVALID_ARGUMENT, "null CSR descriptor"); return; }
    // Legacy API: buffer capacities are a documented caller precondition.
    bt_csr_spmv_checked(mat, x, mat->cols, y, mat->rows,
        mat->rows == UINT64_MAX ? 0 : mat->rows + 1, mat->nnz,
        mat->nnz / 64 + (mat->nnz % 64 != 0));
}

int bt_radix3_fft_forward(const double* real_in, const double* imag_in, double* real_out, double* imag_out, size_t n) {
    return abi_call([&]() -> int {
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
    });
}

int bt_radix3_fft_inverse(const double* real_in, const double* imag_in, double* real_out, double* imag_out, size_t n) {
    return abi_call([&]() -> int {
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
    });
}

int bt_radix3_convolve(const double* a, size_t na, const double* b, size_t nb, double* out) {
    return abi_call([&]() -> int {
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
    });
}

bt_word64_t bt_kogge_stone_add64_val(bt_word64_t a, bt_word64_t b, int8_t cin, int8_t* cout_msb) {
    return abi_call([&]() -> bt_word64_t {
        validate_word(a);validate_word(b);
        if(cin < -1 || cin > 1) throw std::invalid_argument("carry-in must be a trit");
        bt::TritWord64 wa{a.p, a.n};
        bt::TritWord64 wb{b.p, b.n};
        bt::TritWord64 wsum = bt::kogge_stone_add64(wa, wb, cin, cout_msb);
        return bt_word64_t{wsum.P, wsum.N};
    });
}

int64_t bt_dot_product_64_val(bt_word64_t a, bt_word64_t b) {
    return abi_call([&]() -> int64_t {
        validate_word(a);validate_word(b);
        return bt_dot_product_64(a, b);
    });
}

bt_btha_result_t bt_btha_step_val(bt_word64_t a, bt_word64_t b) {
    return abi_call([&]() -> bt_btha_result_t {
        validate_word(a);validate_word(b);
        return bt_btha_step(a, b);
    });
}

uint8_t bt_base243_pack_val(const int8_t trits[5]) {
    return abi_call([&]() -> uint8_t {
        if(!trits) throw std::invalid_argument("null base243 input");
        for(size_t i=0;i<5;++i) if(trits[i]<-1 || trits[i]>1) throw std::invalid_argument("base243 input must contain trits");
        return bt_base243_pack(trits);
    });
}

void bt_base243_unpack_val(uint8_t byte, int8_t out_trits[5]) {
    return abi_call([&]() -> void {
        if(!out_trits || byte>=243) throw std::invalid_argument("invalid base243 byte/output");
        bt_base243_unpack(byte, out_trits);
    });
}

uint32_t bt_hamming_distance_64_val(bt_word64_t a, bt_word64_t b) {
    return abi_call([&]() -> uint32_t {
        validate_word(a);validate_word(b);
        return bt_hamming_distance_64(a, b);
    });
}

bool bt_verify_disjointness_val(bt_word64_t word) {
    return abi_call([&]() -> bool {
        return bt_verify_disjointness(word);
    });
}

int64_t bt_kleene_and(int64_t a, int64_t b) {
    return abi_call([&]() -> int64_t {
        if (a >= -1 && a <= 1 && b >= -1 && b <= 1) {
            return (a < b) ? a : b;
        }
        auto wa = bt::Codec::from_int64(a);
        auto wb = bt::Codec::from_int64(b);
        auto wr = bt::kleene_and(wa, wb);
        return bt::swar_to_int64(wr);
    });
}

int64_t bt_kleene_or(int64_t a, int64_t b) {
    return abi_call([&]() -> int64_t {
        if (a >= -1 && a <= 1 && b >= -1 && b <= 1) {
            return (a > b) ? a : b;
        }
        auto wa = bt::Codec::from_int64(a);
        auto wb = bt::Codec::from_int64(b);
        auto wr = bt::kleene_or(wa, wb);
        return bt::swar_to_int64(wr);
    });
}

int64_t bt_kleene_xor(int64_t a, int64_t b) {
    return abi_call([&]() -> int64_t {
        if (a >= -1 && a <= 1 && b >= -1 && b <= 1) {
            return a * b;
        }
        auto wa = bt::Codec::from_int64(a);
        auto wb = bt::Codec::from_int64(b);
        auto wr = bt::kleene_xor(wa, wb);
        return bt::swar_to_int64(wr);
    });
}

void xbt_sgemm(const float* a, const float* b, float* c, size_t m, size_t k, size_t n, int min_plane_sum_cutoff) {
    return abi_call([&]() -> void {
        bt::blas::xbt_sgemm_unmodified<8>(a, b, c, m, k, n, min_plane_sum_cutoff);
    });
}

void xbt_rmsnorm(const float* x, const float* weight, float* out, size_t n, float eps) {
    return abi_call([&]() -> void {
        bt::nonlinear::xbt_rmsnorm(x, weight, out, n, eps);
    });
}

void xbt_softmax(const float* x, float* out, size_t n) {
    return abi_call([&]() -> void {
        bt::nonlinear::xbt_softmax(x, out, n);
    });
}

double xbt_exp(double x) {
    return abi_call([&]() -> double {
        return bt::nonlinear::xbt_exp_fast(x);
    });
}

double xbt_tanh(double x) {
    return abi_call([&]() -> double {
        return bt::nonlinear::xbt_tanh_fast(x);
    });
}

double xbt_sigmoid(double x) {
    return abi_call([&]() -> double {
        return bt::nonlinear::xbt_sigmoid_fast(x);
    });
}

double xbt_silu(double x) {
    return abi_call([&]() -> double {
        return bt::nonlinear::xbt_silu_fast(x);
    });
}

double xbt_gelu(double x) {
    return abi_call([&]() -> double {
        return bt::nonlinear::xbt_gelu_fast(x);
    });
}

double xbt_rsqrt(double x) {
    return abi_call([&]() -> double {
        return bt::nonlinear::xbt_rsqrt_householder(x);
    });
}

} // extern "C"
