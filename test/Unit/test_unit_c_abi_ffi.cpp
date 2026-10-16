/**
 * @file test_unit_c_abi_ffi.cpp
 * @brief Unit tests for the C ABI FFI Shared Runtime Library (libbt_runtime)
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "include/Ternary/ternary_c_abi.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <iostream>

static void test_swar_conversion() {
    std::cout << "  [C ABI 1/6] Testing SWAR 64-trit Tree Reduction..." << std::endl;

    // Zero
    bt_word64_t w0 = bt_make_word64(0, 0);
    assert(bt_swar_word64_to_int64(w0, 0) == 0);
    assert(bt_swar_word64_to_int64(w0, 16) == 0);
    assert(bt_swar_word64_to_int64(w0, 39) == 0);

    // Single trits
    bt_word64_t w_pos1 = bt_make_word64(1, 0);
    assert(bt_swar_word64_to_int64(w_pos1, 1) == 1);

    bt_word64_t w_neg1 = bt_make_word64(0, 1);
    assert(bt_swar_word64_to_int64(w_neg1, 1) == -1);

    // Trit 0 = 0, Trit 1 = +1 -> 3
    bt_word64_t w_3 = bt_make_word64(2, 0);
    assert(bt_swar_word64_to_int64(w_3, 2) == 3);

    // Trit 0 = +1, Trit 1 = +1 -> 4
    bt_word64_t w_4 = bt_make_word64(3, 0);
    assert(bt_swar_word64_to_int64(w_4, 2) == 4);

    // Trit 0 = -1, Trit 1 = -1 -> -4
    bt_word64_t w_neg4 = bt_make_word64(0, 3);
    assert(bt_swar_word64_to_int64(w_neg4, 2) == -4);

    // 13 = 1 * 3^2 + 1 * 3^1 + 1 * 3^0
    bt_word64_t w_13 = bt_make_word64(0b111, 0);
    assert(bt_swar_word64_to_int64(w_13, 3) == 13);

    std::cout << "    -> PASSED: SWAR parallel tree reduction verified." << std::endl;
}

static void test_dot_product_array() {
    std::cout << "  [C ABI 2/6] Testing Bit-Sliced Dot Product Array..." << std::endl;

    // Single word test
    bt_word64_t a = bt_make_word64(0b0011ULL, 0b1100ULL);
    bt_word64_t b = bt_make_word64(0b0101ULL, 0b1010ULL);
    assert(bt_dot_product_64(a, b) == 0);
    assert(bt_dot_product_64_val(a, b) == 0);

    // Multi-word test
    bt_word64_t arr_a[4] = {
        bt_make_word64(0b0011ULL, 0b1100ULL),
        bt_make_word64(0b1111ULL, 0ULL),
        bt_make_word64(0ULL, 0b1111ULL),
        bt_make_word64(0b0101ULL, 0b1010ULL)
    };

    bt_word64_t arr_b[4] = {
        bt_make_word64(0b0101ULL, 0b1010ULL),
        bt_make_word64(0b1111ULL, 0ULL),
        bt_make_word64(0ULL, 0b1111ULL),
        bt_make_word64(0b0101ULL, 0b1010ULL)
    };

    // arr_a[0] . arr_b[0] = 0
    // arr_a[1] . arr_b[1] = 4
    // arr_a[2] . arr_b[2] = 4
    // arr_a[3] . arr_b[3] = 4
    // Total = 12
    int64_t total = bt_dot_product_array(arr_a, arr_b, 4);
    assert(total == 12);

    // Edge cases
    assert(bt_dot_product_array(nullptr, arr_b, 4) == 0);
    assert(bt_dot_product_array(arr_a, nullptr, 4) == 0);
    assert(bt_dot_product_array(arr_a, arr_b, 0) == 0);

    std::cout << "    -> PASSED: Array dot product popcount contraction verified." << std::endl;
}

static void test_fp32_arithmetic() {
    std::cout << "  [C ABI 3/6] Testing BT-FP32 Floating-Point Arithmetic..." << std::endl;

    // Double conversion roundtrip
    double test_vals[] = { 0.0, 1.0, -1.0, 0.5, -0.5, 13.5, -27.0, 100.25, 0.125 };
    for (double val : test_vals) {
        bt_fp32_t fp = bt_fp32_from_double(val);
        double roundtrip = bt_fp32_to_double(fp);
        assert(std::abs(roundtrip - val) < 1e-4);
    }

    // Arithmetic tests
    bt_fp32_t a = bt_fp32_from_double(12.5);
    bt_fp32_t b = bt_fp32_from_double(2.5);

    // Addition
    bt_fp32_t sum = bt_fp32_add(a, b);
    assert(std::abs(bt_fp32_to_double(sum) - 15.0) < 1e-4);

    // Subtraction
    bt_fp32_t diff = bt_fp32_sub(a, b);
    assert(std::abs(bt_fp32_to_double(diff) - 10.0) < 1e-4);

    // Multiplication
    bt_fp32_t prod = bt_fp32_mul(a, b);
    assert(std::abs(bt_fp32_to_double(prod) - 31.25) < 1e-3);

    // Division
    bt_fp32_t quot = bt_fp32_div(a, b);
    assert(std::abs(bt_fp32_to_double(quot) - 5.0) < 1e-4);

    // Division by zero edge case
    bt_fp32_t zero = bt_fp32_from_double(0.0);
    bt_fp32_t div_zero = bt_fp32_div(a, zero);
    assert(div_zero.mantissa == 0 && div_zero.exponent == 0);

    std::cout << "    -> PASSED: BT-FP32 0-bias truncation and rounding arithmetic verified." << std::endl;
}

static void test_csr_spmv() {
    std::cout << "  [C ABI 4/6] Testing Multiplier-Free T-CSR SpMV..." << std::endl;

    // 4x4 matrix:
    // Row 0: (+1, col 0), (-1, col 2)
    // Row 1: (+1, col 1), (+1, col 3)
    // Row 2: (0)
    // Row 3: (-1, col 0), (+1, col 1), (-1, col 3)
    uint32_t row_ptr[5] = { 0, 2, 4, 4, 7 };
    uint32_t col_indices[7] = { 0, 2, 1, 3, 0, 1, 3 };
    // signs bit-packed:
    // idx 0=+1(1), 1=-1(0)
    // idx 2=+1(1), 3=+1(1)
    // idx 4=-1(0), 5=+1(1), 6=-1(0)
    // binary: 0 1 0 1 1 0 1 = 0b0101101 = 0x2D
    uint64_t signs[1] = { 0x2D };

    bt_csr_t mat;
    mat.rows = 4;
    mat.cols = 4;
    mat.nnz = 7;
    mat.row_ptr = row_ptr;
    mat.col_indices = col_indices;
    mat.signs = signs;

    int64_t x[4] = { 10, 20, 30, 40 };
    int64_t y[4] = { 0, 0, 0, 0 };

    bt_csr_spmv_exec(&mat, x, y);

    // Expected y:
    // y[0] = 10 - 30 = -20
    // y[1] = 20 + 40 = 60
    // y[2] = 0
    // y[3] = -10 + 20 - 40 = -30
    assert(y[0] == -20);
    assert(y[1] == 60);
    assert(y[2] == 0);
    assert(y[3] == -30);

    // Null pointer resilience
    bt_csr_spmv_exec(nullptr, x, y);
    bt_csr_spmv_exec(&mat, nullptr, y);
    bt_csr_spmv_exec(&mat, x, nullptr);

    std::cout << "    -> PASSED: Multiplier-free T-CSR SpMV verified." << std::endl;
}

static void test_radix3_fft() {
    std::cout << "  [C ABI 5/6] Testing Radix-3 Cooley-Tukey FFT & IFFT..." << std::endl;

    // 1. Error handling
    assert(bt_radix3_fft_forward(nullptr, nullptr, nullptr, nullptr, 0) == -1);
    double dummy = 0.0;
    assert(bt_radix3_fft_forward(&dummy, nullptr, &dummy, &dummy, 4) == -2); // 4 is not power of 3

    // 2. N = 9 transform test
    const size_t N = 9;
    std::vector<double> real_in = { 1.0, -1.0, 0.5, 2.0, -0.5, 3.0, 1.5, -2.0, 0.0 };
    std::vector<double> imag_in(N, 0.0);
    std::vector<double> real_fft(N, 0.0);
    std::vector<double> imag_fft(N, 0.0);
    std::vector<double> real_ifft(N, 0.0);
    std::vector<double> imag_ifft(N, 0.0);

    int status_fwd = bt_radix3_fft_forward(real_in.data(), imag_in.data(),
                                           real_fft.data(), imag_fft.data(), N);
    assert(status_fwd == 0);

    int status_inv = bt_radix3_fft_inverse(real_fft.data(), imag_fft.data(),
                                           real_ifft.data(), imag_ifft.data(), N);
    assert(status_inv == 0);

    // Roundtrip verification: IFFT(FFT(x)) == x
    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(real_ifft[i] - real_in[i]) < 1e-10);
        assert(std::abs(imag_ifft[i]) < 1e-10);
    }

    // 3. N = 27 (3^3) Impulse test
    const size_t N27 = 27;
    std::vector<double> impulse_in(N27, 0.0);
    impulse_in[0] = 1.0; // delta impulse at origin -> flat spectrum in frequency
    std::vector<double> f_re(N27, 0.0), f_im(N27, 0.0);
    assert(bt_radix3_fft_forward(impulse_in.data(), nullptr, f_re.data(), f_im.data(), N27) == 0);

    for (size_t i = 0; i < N27; ++i) {
        assert(std::abs(f_re[i] - 1.0) < 1e-10);
        assert(std::abs(f_im[i]) < 1e-10);
    }

    std::cout << "    -> PASSED: Radix-3 forward and inverse transforms verified." << std::endl;
}

static void test_exported_helpers() {
    std::cout << "  [C ABI 6/6] Testing Exported Helpers & Codecs..." << std::endl;

    // Disjointness
    bt_word64_t w_ok = bt_make_word64(0x12345678ULL, 0x87654321ULL);
    assert(bt_verify_disjointness_val(w_ok) == ((0x12345678ULL & 0x87654321ULL) == 0));

    // BTHA
    bt_btha_result_t r = bt_btha_step_val(bt_make_word64(1, 0), bt_make_word64(1, 0));
    assert(r.sum.p == 0 && r.sum.n == 1);
    assert(r.carry.p == 1 && r.carry.n == 0);

    // Hamming distance
    bt_word64_t h1 = bt_make_word64(0xF, 0x0);
    bt_word64_t h2 = bt_make_word64(0x0, 0xF);
    assert(bt_hamming_distance_64_val(h1, h2) == 4);

    // Base-243 Codec
    int8_t trits_in[5] = { 1, 0, -1, 1, 0 };
    uint8_t byte = bt_base243_pack_val(trits_in);
    int8_t trits_out[5];
    bt_base243_unpack_val(byte, trits_out);
    for (int i = 0; i < 5; ++i) {
        assert(trits_out[i] == trits_in[i]);
    }

    std::cout << "    -> PASSED: Exported helpers, BTHA, and Base-243 codec verified." << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  [C++ Unit Test Layer] Production C ABI & Shared Runtime (libbt_runtime)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    test_swar_conversion();
    test_dot_product_array();
    test_fp32_arithmetic();
    test_csr_spmv();
    test_radix3_fft();
    test_exported_helpers();

    std::cout << "======================================================================" << std::endl;
    std::cout << "  -> PASSED: All C ABI FFI Unit Tests Verified Successfully (0 Failures)." << std::endl;
    std::cout << "======================================================================" << std::endl;
    return 0;
}
