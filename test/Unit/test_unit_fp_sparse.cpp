/**
 * @file test_unit_fp_sparse.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Balanced Ternary Floating-Point (BT-FP) & T-CSR Sparse Algebra
 * @author Antigravity Prime Research Engine
 * @version 1.0.0
 */

#include "../../ternary_core.hpp"
#include "../../ternary_dense_algebra.hpp"
#include "../../ternary_fp.hpp"
#include "../../ternary_sparse.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace bt;

// ==============================================================================
// 1. BT-FP Floating-Point Tests
// ==============================================================================

void test_btfp_foundations() {
    std::cout << "  [BT-FP] Testing Unique Zero & Inherent Sign...\n";

    // Unique Zero: M = 0, E = 0
    BTFP16 z1;
    BTFP16 z2(0, 5); // Should normalize to M=0, E=0
    assert(z1.is_zero());
    assert(z2.is_zero());
    assert(z1 == z2);
    assert(z1.mantissa() == 0 && z1.exponent() == 0);
    assert(z2.mantissa() == 0 && z2.exponent() == 0);
    assert(z1.leading_trit() == Trit::Zero);

    // Inherent Sign: No sign bit
    BTFP16 pos_val(10000, 0);
    BTFP16 neg_val(-10000, 0);
    assert(pos_val.is_positive() && !pos_val.is_negative());
    assert(neg_val.is_negative() && !neg_val.is_positive());
    assert(pos_val.leading_trit() == Trit::Pos);
    assert(neg_val.leading_trit() == Trit::Neg);

    // 0-Cycle Negation
    BTFP16 neg_pos = -pos_val;
    assert(neg_pos == neg_val);
    assert((-neg_val) == pos_val);
}

void test_btfp_round_trip() {
    std::cout << "  [BT-FP] Testing IEEE-754 Round-Trip Conversions...\n";

    // Exact powers of 3
    std::vector<double> powers = {1.0, 3.0, 9.0, 27.0, 81.0, 1.0 / 3.0, 1.0 / 9.0, 1.0 / 27.0};
    for (double p : powers) {
        BTFP16 fp = BTFP16::from_double(p);
        double back = fp.to_double();
        double rel_err = std::abs((back - p) / p);
        assert(rel_err < 1e-6);

        BTFP16 fp_neg = BTFP16::from_double(-p);
        double back_neg = fp_neg.to_double();
        double rel_err_neg = std::abs((back_neg - (-p)) / p);
        assert(rel_err_neg < 1e-6);
    }

    // Typical integers and reals
    std::vector<double> test_vals = {0.0, 2.0, 5.0, 7.5, 42.0, -17.25, 100.0, 0.125, -0.0625};
    for (double val : test_vals) {
        BTFP16 fp16 = BTFP16::from_double(val);
        BTFP32 fp32 = BTFP32::from_double(val);

        if (val == 0.0) {
            assert(fp16.is_zero() && fp32.is_zero());
            assert(fp16.to_double() == 0.0 && fp32.to_double() == 0.0);
        } else {
            double back16 = fp16.to_double();
            double back32 = fp32.to_double();
            double err16 = std::abs((back16 - val) / val);
            double err32 = std::abs((back32 - val) / val);

            // BTFP16 has ~15.8 bits (rel error < 1e-4)
            assert(err16 < 1e-4);
            // BTFP32 has ~31.7 bits (rel error < 1e-9)
            assert(err32 < 1e-9);
            // Higher precision format is at least as accurate
            assert(err32 <= err16 + 1e-12);
        }
    }
}

void test_btfp_arithmetic() {
    std::cout << "  [BT-FP] Testing Arithmetic Operations (Add, Sub, Mul, Div)...\n";

    BTFP16 a = BTFP16::from_double(12.0);
    BTFP16 b = BTFP16::from_double(4.0);

    // Addition: 12 + 4 = 16
    BTFP16 sum = a + b;
    double sum_val = sum.to_double();
    assert(std::abs(sum_val - 16.0) < 1e-4);

    // Subtraction: 12 - 4 = 8
    BTFP16 diff = a - b;
    double diff_val = diff.to_double();
    assert(std::abs(diff_val - 8.0) < 1e-4);

    // Exact Zero Cancellation: a - a = 0
    BTFP16 cancel = a - a;
    assert(cancel.is_zero());
    assert(cancel.to_double() == 0.0);

    // Multiplication: 12 * 4 = 48
    BTFP16 prod = a * b;
    double prod_val = prod.to_double();
    assert(std::abs(prod_val - 48.0) < 1e-4);

    // Division: 12 / 4 = 3
    BTFP16 quot = a / b;
    double quot_val = quot.to_double();
    assert(std::abs(quot_val - 3.0) < 1e-4);

    // Multiplication by 1 and 0
    BTFP16 one = BTFP16::from_double(1.0);
    BTFP16 zero = BTFP16::from_double(0.0);
    assert((a * one) == a);
    assert((a * zero).is_zero());

    // Sign rules
    BTFP16 neg_b = -b;
    assert((a * neg_b).is_negative());
    assert(((-a) * neg_b).is_positive());

    // Division by zero throws exception
    bool caught = false;
    try {
        [[maybe_unused]] auto bad = a / zero;
    } catch (const std::domain_error&) {
        caught = true;
    }
    assert(caught && "Division by zero must throw std::domain_error");
}

void test_btfp_zero_bias_rounding() {
    std::cout << "  [BT-FP] Verifying Zero-Bias Truncation Rounding [-0.5, +0.5]...\n";

    // Verify analytical mean truncation error across various trit discard widths
    for (int k = 1; k <= 7; ++k) {
        double mean_err = BTFP16::verify_zero_bias_mean(k);
        assert(std::abs(mean_err) < 1e-15 && "Balanced truncation mean error must be bit-exact 0.0");
    }

    // Compare with binary truncation bias
    // In binary fixed-point, dropping k bits introduces systematic downward bias of (2^k - 1) / (2 * 2^k) ~ -0.5 LSB
    // In balanced ternary, mean error is mathematically 0.0!
    double ternary_bias = BTFP16::verify_zero_bias_mean(4);
    assert(ternary_bias == 0.0);
}

// ==============================================================================
// 2. T-CSR Sparse Linear Algebra Tests
// ==============================================================================

void test_tcsr_creation() {
    std::cout << "  [T-CSR] Testing Matrix Creation & 1-Bit Sign Bitplane Encoding...\n";

    // 4x5 test sparse matrix
    // Row 0: [+1,  0, -1,  0,  0]
    // Row 1: [ 0,  0,  0,  0,  0]
    // Row 2: [ 0, +1,  0,  0, +1]
    // Row 3: [-1,  0,  0, -1,  0]
    std::vector<std::tuple<size_t, size_t, Trit>> triplets = {
        {0, 0, Trit::Pos},
        {0, 2, Trit::Neg},
        {2, 1, Trit::Pos},
        {2, 4, Trit::Pos},
        {3, 0, Trit::Neg},
        {3, 3, Trit::Neg}
    };

    TCSRMatrix mat = TCSRMatrix::from_triplets(4, 5, triplets);
    assert(mat.rows == 4);
    assert(mat.cols == 5);
    assert(mat.nnz == 6);

    // Verify row pointers
    assert(mat.row_ptr[0] == 0);
    assert(mat.row_ptr[1] == 2);
    assert(mat.row_ptr[2] == 2); // Empty row 1
    assert(mat.row_ptr[3] == 4);
    assert(mat.row_ptr[4] == 6);

    // Verify element access via binary search
    assert(mat.get(0, 0) == Trit::Pos);
    assert(mat.get(0, 1) == Trit::Zero);
    assert(mat.get(0, 2) == Trit::Neg);
    assert(mat.get(1, 0) == Trit::Zero);
    assert(mat.get(2, 1) == Trit::Pos);
    assert(mat.get(2, 4) == Trit::Pos);
    assert(mat.get(3, 0) == Trit::Neg);
    assert(mat.get(3, 3) == Trit::Neg);
    assert(mat.get(3, 4) == Trit::Zero);

    // Verify 1-bit sign packing
    assert(mat.is_pos(0) == true);  // +1
    assert(mat.is_pos(1) == false); // -1
    assert(mat.is_pos(2) == true);  // +1
    assert(mat.is_pos(3) == true);  // +1
    assert(mat.is_pos(4) == false); // -1
    assert(mat.is_pos(5) == false); // -1
}

void test_tcsr_density_and_memory() {
    std::cout << "  [T-CSR] Testing Density & Memory Compression Metrics...\n";

    size_t M = 1000, N = 1000;
    size_t target_nnz = 50000; // 5% density
    std::vector<std::tuple<size_t, size_t, Trit>> triplets;
    triplets.reserve(target_nnz);

    for (size_t i = 0; i < target_nnz; ++i) {
        // A permutation of flattened positions: no repeated coordinates.
        // The old two modulo-1000 sequences repeated after only 1000 entries.
        const size_t position = (i * 37) % (M * N);
        size_t r = position / N;
        size_t c = position % N;
        Trit v = (i % 2 == 0) ? Trit::Pos : Trit::Neg;
        triplets.emplace_back(r, c, v);
    }

    TCSRMatrix mat = TCSRMatrix::from_triplets(M, N, triplets);
    double dens = mat.density();
    assert(dens > 0.04 && dens < 0.06);

    // Non-zero value storage reduction: exactly 32x vs FP32
    assert(mat.value_memory_reduction_vs_float() == 32.0);
    assert(mat.value_memory_reduction_vs_double() == 64.0);

    // Overall memory savings vs Dense Float
    double mem_savings_dense = mat.memory_savings_vs_dense_float();
    std::cout << "    -> Density: " << std::fixed << std::setprecision(2) << (dens * 100.0) << "%\n";
    std::cout << "    -> T-CSR Memory: " << mat.memory_bytes() << " bytes\n";
    std::cout << "    -> Dense FP32 Memory: " << mat.dense_float_memory_bytes() << " bytes\n";
    std::cout << "    -> Memory Savings vs Dense Float: " << mem_savings_dense << "x\n";
    std::cout << "    -> Non-zero Value Storage Reduction: " << mat.value_memory_reduction_vs_float() << "x\n";

    // Memory savings should be significant (> 10x for 5% density)
    assert(mem_savings_dense > 10.0);
}

void test_tcsr_spmv() {
    std::cout << "  [T-CSR] Testing Multiplier-Free SpMV (Sparse Matrix-Vector Multiply)...\n";

    // 3x4 matrix:
    // [ +1,  0, -1,  0 ]
    // [  0, +1, +1, -1 ]
    // [ -1,  0,  0, +1 ]
    std::vector<std::tuple<size_t, size_t, Trit>> triplets = {
        {0, 0, Trit::Pos}, {0, 2, Trit::Neg},
        {1, 1, Trit::Pos}, {1, 2, Trit::Pos}, {1, 3, Trit::Neg},
        {2, 0, Trit::Neg}, {2, 3, Trit::Pos}
    };
    TCSRMatrix A = TCSRMatrix::from_triplets(3, 4, triplets);

    // Integer vector x = [10, 20, 30, 40]
    // y[0] = 1*10 + (-1)*30 = -20
    // y[1] = 1*20 + 1*30 + (-1)*40 = 10
    // y[2] = (-1)*10 + 1*40 = 30
    std::vector<int64_t> x_int = {10, 20, 30, 40};
    auto y_int = A.spmv(x_int);
    assert(y_int.size() == 3);
    assert(y_int[0] == -20);
    assert(y_int[1] == 10);
    assert(y_int[2] == 30);

    // Double vector x = [1.5, -2.5, 3.5, 0.5]
    // y[0] = 1.5 - 3.5 = -2.0
    // y[1] = -2.5 + 3.5 - 0.5 = 0.5
    // y[2] = -1.5 + 0.5 = -1.0
    std::vector<double> x_dbl = {1.5, -2.5, 3.5, 0.5};
    auto y_dbl = A.spmv(x_dbl);
    assert(std::abs(y_dbl[0] - (-2.0)) < 1e-9);
    assert(std::abs(y_dbl[1] - 0.5) < 1e-9);
    assert(std::abs(y_dbl[2] - (-1.0)) < 1e-9);

    // Unified BT-FP16 vector
    std::vector<BTFP16> x_fp = {
        BTFP16::from_double(10.0),
        BTFP16::from_double(20.0),
        BTFP16::from_double(30.0),
        BTFP16::from_double(40.0)
    };
    auto y_fp = A.spmv(x_fp);
    assert(std::abs(y_fp[0].to_double() - (-20.0)) < 1e-4);
    assert(std::abs(y_fp[1].to_double() - 10.0) < 1e-4);
    assert(std::abs(y_fp[2].to_double() - 30.0) < 1e-4);
}

void test_tcsr_spmm() {
    std::cout << "  [T-CSR] Testing Multiplier-Free SpMM (Sparse Matrix-Matrix Multiply)...\n";

    // A is 3x3:
    // [ +1,  0, -1 ]
    // [  0, +1,  0 ]
    // [ -1,  0, +1 ]
    std::vector<std::tuple<size_t, size_t, Trit>> triplets_A = {
        {0, 0, Trit::Pos}, {0, 2, Trit::Neg},
        {1, 1, Trit::Pos},
        {2, 0, Trit::Neg}, {2, 2, Trit::Pos}
    };
    TCSRMatrix A = TCSRMatrix::from_triplets(3, 3, triplets_A);

    // Dense B (3x2):
    // [ 1, 2 ]
    // [ 3, 4 ]
    // [ 5, 6 ]
    std::vector<int64_t> B = {1, 2, 3, 4, 5, 6};

    // Expected C = A * B (3x2):
    // Row 0: [1 - 5, 2 - 6] = [-4, -4]
    // Row 1: [3, 4]         = [ 3,  4]
    // Row 2: [-1 + 5, -2 + 6] = [ 4,  4]
    auto C = A.spmm(B, 2);
    assert(C.size() == 6);
    assert(C[0] == -4 && C[1] == -4);
    assert(C[2] ==  3 && C[3] ==  4);
    assert(C[4] ==  4 && C[5] ==  4);

    // Sparse-Sparse SpMM: C_sparse = A * A
    // Expected A * A:
    // Row 0: [+2, 0, -2]
    // Row 1: [ 0, 1,  0]
    // Row 2: [-2, 0, +2]
    DenseMatrixInt64 C_sp = A.spmm(A);
    assert(C_sp.rows == 3 && C_sp.cols == 3);
    assert(C_sp(0, 0) ==  2 && C_sp(0, 1) == 0 && C_sp(0, 2) == -2);
    assert(C_sp(1, 0) ==  0 && C_sp(1, 1) == 1 && C_sp(1, 2) ==  0);
    assert(C_sp(2, 0) == -2 && C_sp(2, 1) == 0 && C_sp(2, 2) ==  2);
}

void test_fp_and_sparse_guards() {
    std::cout << "  [BT-FP & T-CSR] Testing FP/Sparse Edge Cases & Invariant Guards...\n";

    // 1. BTFP::from_double domain errors
    bool caught_nan = false;
    try {
        BTFP16::from_double(std::numeric_limits<double>::quiet_NaN());
    } catch (const std::domain_error&) {
        caught_nan = true;
    }
    assert(caught_nan && "from_double on NaN must throw std::domain_error");

    bool caught_inf = false;
    try {
        BTFP16::from_double(std::numeric_limits<double>::infinity());
    } catch (const std::domain_error&) {
        caught_inf = true;
    }
    assert(caught_inf && "from_double on Inf must throw std::domain_error");

    bool caught_neginf = false;
    try {
        BTFP16::from_double(-std::numeric_limits<double>::infinity());
    } catch (const std::domain_error&) {
        caught_neginf = true;
    }
    assert(caught_neginf && "from_double on -Inf must throw std::domain_error");

    // 2. Finite subnormal inputs remain representable in the ternary exponent domain
    BTFP16 subnormal = BTFP16::from_double(1e-310);
    assert(subnormal.is_positive() && "Finite subnormal input must not silently flush to zero");
    assert(std::abs(subnormal.to_double() / 1e-310 - 1.0) < 1e-4);

    // 3. Configured exponent domains reject overflow AND underflow. Never
    // silently turn a very large positive value into a small value or zero.
    for (int32_t exponent : {100, -100}) {
        bool rejected = false;
        try { BTFP<10,3> outside(1, exponent); }
        catch (const std::overflow_error&) { rejected = true; }
        assert(rejected);
    }

    // 4. Strict weak ordering for operator<
    BTFP16 v_neg10(-10);
    BTFP16 v_neg5(-5);
    BTFP16 v_zero(0);
    BTFP16 v_pos5(5);
    BTFP16 v_pos10(10);

    assert(v_neg10 < v_neg5);
    assert(v_neg5 < v_zero);
    assert(v_zero < v_pos5);
    assert(v_pos5 < v_pos10);

    // Irreflexivity: !(a < a)
    assert(!(v_zero < v_zero));
    assert(!(v_pos5 < v_pos5));
    assert(!(v_neg10 < v_neg10));

    // Asymmetry: a < b => !(b < a)
    assert(v_neg10 < v_pos10 && !(v_pos10 < v_neg10));

    // Distinct numbers with same sign but differing exponents
    BTFP16 big_exp(100, 20);
    BTFP16 small_exp(100, 5);
    assert(small_exp < big_exp);
    assert(!(big_exp < small_exp));

    // Negative differing exponents: larger exponent is more negative (smaller)
    BTFP16 neg_big_exp(-100, 20);
    BTFP16 neg_small_exp(-100, 5);
    assert(neg_big_exp < neg_small_exp);
    assert(!(neg_small_exp < neg_big_exp));

    // std::sort verification
    std::vector<BTFP16> to_sort = {v_pos10, v_neg5, v_zero, v_pos5, v_neg10};
    std::sort(to_sort.begin(), to_sort.end());
    assert(to_sort[0] == v_neg10);
    assert(to_sort[1] == v_neg5);
    assert(to_sort[2] == v_zero);
    assert(to_sort[3] == v_pos5);
    assert(to_sort[4] == v_pos10);

    // 5. TCSRMatrix push_element monotonic row check & int8_t overload
    TCSRMatrix mat(4, 4);
    mat.push_element(1, 0, static_cast<int8_t>(1)); // int8_t overload
    mat.push_element(1, 2, static_cast<int8_t>(-1));
    bool caught_row_order = false;
    try {
        mat.push_element(0, 1, static_cast<int8_t>(1)); // row 0 < current row 1!
    } catch (const std::invalid_argument&) {
        caught_row_order = true;
    }
    assert(caught_row_order && "Out-of-order row push must throw std::invalid_argument");

    // Out of bounds row/col
    bool caught_oob = false;
    try {
        mat.push_element(5, 0, Trit::Pos);
    } catch (const std::out_of_range&) {
        caught_oob = true;
    }
    assert(caught_oob && "Out of bounds push_element must throw std::out_of_range");

    // 6. TCSRMatrix spmm column bounds check
    // Create matrix A (2x2) with an invalid column index = 5 (exceeding B's rows = 2)
    TCSRMatrix A_bad(2, 2);
    A_bad.push_element(0, 0, Trit::Pos);
    A_bad.finalize();
    // Tamper col_indices to simulate corrupted or mismatched dimensions
    A_bad.col_indices[0] = 10; // Col index 10 >= B.rows (2)

    TCSRMatrix B_mat(2, 2);
    B_mat.push_element(0, 0, Trit::Pos);
    B_mat.finalize();

    bool caught_spmm_oob = false;
    try {
        A_bad.spmm(B_mat);
    } catch (const std::out_of_range&) {
        caught_spmm_oob = true;
    }
    assert(caught_spmm_oob && "spmm with column index exceeding B.rows must throw std::out_of_range");
}

void test_btfp_integer_rounding_and_mul() {
    std::cout << "  [BT-FP] Testing Integer Rounded Normalization & Multiplication...\n";

    // 1. Verify round_div_pow3_fast64 matches k sequential calls to round_div3
    auto seq_round_div_pow3 = [](int64_t val, int k) -> int64_t {
        for (int i = 0; i < k && val != 0; ++i) {
            val = fp_detail::round_div3(val);
        }
        return val;
    };

    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    auto splitmix64 = [&seed]() -> uint64_t {
        uint64_t z = (seed += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };

    for (int iter = 0; iter < 5000; ++iter) {
        int64_t val = static_cast<int64_t>(splitmix64() >> 10) - (1LL << 53);
        int k = static_cast<int>(splitmix64() % 38);
        assert(fp_detail::round_div_pow3_fast64(val, k) == seq_round_div_pow3(val, k));
        assert(fp_detail::round_div_pow3_fast128(val, k) == seq_round_div_pow3(val, k));
    }

    // 2. Independent exact-distance search across all relevant binades for an
    // int64 input and the 20-trit BTFP32 mantissa. A common 3^24 denominator
    // permits exact comparisons; even the largest candidate fits signed i128.
    auto candidate_normalize_btfp32 = [](int64_t m, int32_t e) -> std::pair<int64_t, int32_t> {
        if (!m) return {0,0};
        __int128 magnitude=m; if(magnitude<0) magnitude=-magnitude;
        __int128 target=magnitude*fp_detail::pow3_128(24);
        __int128 best_error=static_cast<__int128>(1)<<126,best_value=0;
        int64_t best_m=0; int32_t best_e=0;
        for(int k=-24;k<=24;++k) {
            __int128 scale=fp_detail::pow3_128(k+24);
            __int128 candidate=(target+scale/2)/scale;
            candidate=std::clamp(candidate,static_cast<__int128>(BTFP32::min_norm_mantissa()),
                                static_cast<__int128>(BTFP32::max_mantissa()));
            __int128 value=candidate*scale,error=value-target;
            if(error<0) error=-error;
            if(error<best_error || (error==best_error && value>best_value)) {
                best_error=error;best_value=value;
                best_m=static_cast<int64_t>(candidate);best_e=e+k;
            }
        }
        return {m<0?-best_m:best_m,best_e};
    };

    for (int iter = 0; iter < 5000; ++iter) {
        // Test both very small (|m| < min_m) and very large (|m| > max_m) mantissas
        int64_t raw_m = (iter % 2 == 0)
            ? (static_cast<int64_t>(splitmix64() >> 6) - (1LL << 57))
            : (static_cast<int64_t>(splitmix64() % 2000000LL) - 1000000LL);
        int32_t raw_e = static_cast<int32_t>(splitmix64() % 200) - 100;

        BTFP32 fast_fp(raw_m, raw_e);
        auto [ref_m, ref_e] = candidate_normalize_btfp32(raw_m, raw_e);
        assert(fast_fp.mantissa() == ref_m && fast_fp.exponent() == ref_e);
    }

    // 3. Verify BTFP32::mul across boundary cases (min_m, max_m) and 10,000 random pairs
    const int64_t min_m32 = BTFP32::min_norm_mantissa();
    const int64_t max_m32 = BTFP32::max_mantissa();
    std::vector<int64_t> boundary_m = {
        min_m32, min_m32 + 1, (min_m32 + max_m32) / 2, max_m32 - 1, max_m32,
        -min_m32, -(min_m32 + 1), -((min_m32 + max_m32) / 2), -(max_m32 - 1), -max_m32
    };

    for (int64_t ma : boundary_m) {
        for (int64_t mb : boundary_m) {
            BTFP32 fa(ma, 3);
            BTFP32 fb(mb, -2);
            BTFP32 prod = fa * fb;
            auto [ref_m, ref_e] = candidate_normalize_btfp32(ma * mb, 1);
            assert(prod.mantissa() == ref_m && prod.exponent() == ref_e);
        }
    }

    for (int iter = 0; iter < 10000; ++iter) {
        int64_t ma = min_m32 + static_cast<int64_t>(splitmix64() % static_cast<uint64_t>(max_m32 - min_m32 + 1));
        int64_t mb = min_m32 + static_cast<int64_t>(splitmix64() % static_cast<uint64_t>(max_m32 - min_m32 + 1));
        if (splitmix64() & 1ULL) ma = -ma;
        if (splitmix64() & 1ULL) mb = -mb;
        int32_t ea = static_cast<int32_t>(splitmix64() % 100) - 50;
        int32_t eb = static_cast<int32_t>(splitmix64() % 100) - 50;

        BTFP32 fa(ma, ea);
        BTFP32 fb(mb, eb);
        BTFP32 prod = fa * fb;
        auto [ref_m, ref_e] = candidate_normalize_btfp32(ma * mb, ea + eb);
        assert(prod.mantissa() == ref_m && prod.exponent() == ref_e);
    }

    std::cout << "    -> Exact-distance BTFP32 normalization/mul regressions passed (20,100 samples).\n";
}

void test_btfp_fma_sqrt_pow_and_clamping() {
    std::cout << "  [BT-FP] Testing fma, sqrt, abs, neg, pow_int & mul() Exponent-Clamping Fix...\n";

    // 1. O(1) abs(), neg(), and explicit operator double()
    BTFP32 pos = BTFP32::from_double(42.5);
    BTFP32 neg = pos.neg();
    assert(neg.is_negative());
    assert(neg == -pos);
    assert(neg.abs() == pos);
    assert(pos.abs() == pos);
    assert(BTFP32(0).abs().is_zero());
    assert(BTFP32(0).neg().is_zero());
    assert(std::abs(static_cast<double>(pos) - 42.5) / 42.5 < 1e-9);
    assert(std::abs(static_cast<double>(neg) - (-42.5)) / 42.5 < 1e-9);

    // 2. Single-rounding fma(a, b, c) vs double-rounded (a * b + c)
    // Catastrophic cancellation test: a * b is an exact 38-trit product that would lose
    // low trits if rounded to 20 trits before adding c = -(high part of a * b).
    int64_t m_a = BTFP32::min_norm_mantissa() + 12345;
    int64_t m_b = BTFP32::min_norm_mantissa() + 67891;
    __int128_t exact_prod = static_cast<__int128_t>(m_a) * static_cast<__int128_t>(m_b);
    BTFP32 fa(m_a, 0);
    BTFP32 fb(m_b, 0);
    BTFP32 ab_rounded = fa * fb; // Rounded to 20 trits
    BTFP32 fc = -ab_rounded;     // Exact negation of the rounded product

    // Separate mul + add cancels to 0, losing the residual!
    BTFP32 double_rounded = (fa * fb) + fc;
    assert(double_rounded.is_zero());

    // Single-rounding FMA retains the exact 128-bit residual: exact_prod - ab_rounded!
    BTFP32 single_rounded = BTFP32::fma(fa, fb, fc);
    __int128_t ab_val = static_cast<__int128_t>(ab_rounded.mantissa()) *
                        fp_detail::pow3_128(ab_rounded.exponent());
    __int128_t expected_residual = exact_prod - ab_val;
    assert(expected_residual != 0);
    BTFP32 expected_fma = BTFP32::normalize_128(expected_residual, 0);
    assert(single_rounded == expected_fma);

    // General FMA accuracy across various scales
    BTFP32 x = BTFP32::from_double(3.5);
    BTFP32 y = BTFP32::from_double(-4.0);
    BTFP32 z = BTFP32::from_double(15.25);
    BTFP32 fma_res = BTFP32::fma(x, y, z);
    double exact_xyz = x.to_double() * y.to_double() + z.to_double();
    assert(std::abs(fma_res.to_double() - exact_xyz) < 1e-9);
    assert(std::abs(fma_res.to_double() - 1.25) < 1e-7);

    // 3. sqrt() exact squares, parity alignment, midpoint rounding & negative guard
    assert(BTFP32(0).sqrt().is_zero());
    for (double sq_val : {1.0, 4.0, 9.0, 16.0, 25.0, 81.0, 729.0, 2.0, 3.0, 5.0, 0.25, 1.0 / 9.0}) {
        BTFP32 s = BTFP32::from_double(sq_val).sqrt();
        double expected = std::sqrt(sq_val);
        assert(std::abs(s.to_double() - expected) / expected < 1e-8);
    }

    bool caught_neg_sqrt = false;
    try {
        [[maybe_unused]] auto bad = BTFP32(-9).sqrt();
    } catch (const std::domain_error&) {
        caught_neg_sqrt = true;
    }
    assert(caught_neg_sqrt && "BTFP32::sqrt on negative value must throw std::domain_error");

    // 4. pow_int(exp) binary exponentiation
    BTFP32 three(3);
    assert(three.pow_int(0) == BTFP32(1));
    assert(BTFP32(0).pow_int(0) == BTFP32(1));
    assert(BTFP32(0).pow_int(5).is_zero());
    assert(three.pow_int(4) == BTFP32(81));
    assert(std::abs(three.pow_int(-2).to_double() - (1.0 / 9.0)) < 1e-12);
    assert(BTFP32(-2).pow_int(5) == BTFP32(-32));
    assert(BTFP32(-2).pow_int(6) == BTFP32(64));

    // 5. Arithmetic uses the SAME configured exponent policy as constructors.
    using Bounded = BTFP<10,3>;
    Bounded d1(Bounded::min_norm_mantissa(), -12);
    Bounded d2(Bounded::min_norm_mantissa(), -12);
    bool rejected = false;
    try { auto product = d1 * d2; (void)product; }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected && "out-of-domain product must not silently clamp");
    Bounded high(Bounded::max_mantissa(), 13);
    rejected = false;
    try { auto product = high * high; (void)product; }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);

}

// ==============================================================================
// Main Test Runner
// ==============================================================================

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  [LLVM Unit Test Layer 1] BT-FP & T-CSR Sparse Algebra Verification\n";
    std::cout << "======================================================================\n";

    // Phase 3.1: BT-FP Tests
    test_btfp_foundations();
    test_btfp_round_trip();
    test_btfp_arithmetic();
    test_btfp_zero_bias_rounding();
    test_btfp_integer_rounding_and_mul();
    test_btfp_fma_sqrt_pow_and_clamping();

    // Phase 3.2: T-CSR Sparse Linear Algebra Tests
    test_tcsr_creation();
    test_tcsr_density_and_memory();
    test_tcsr_spmv();
    test_tcsr_spmm();

    // Phase 3.3: Invariant & Exception Guards
    test_fp_and_sparse_guards();

    std::cout << "======================================================================\n";
    std::cout << "  -> PASSED: All BT-FP & T-CSR Unit Tests 100% Verified Successfully!\n";
    std::cout << "======================================================================\n";
    return 0;
}
