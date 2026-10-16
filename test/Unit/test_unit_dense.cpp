/**
 * @file test_unit_dense.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Dense Algebra & Tiled GEMM
 */

#include "../../ternary_dense_algebra.hpp"
#include <iostream>
#include <cassert>

using namespace bt;

void test_kleene_logic() {
    TritWord64 a(0b1010, 0b0101);
    TritWord64 b(0b1100, 0b0011);
    TritWord64 wand = kleene_and(a, b);
    TritWord64 wor = kleene_or(a, b);
    TritWord64 wnot = kleene_not(a);
    assert(wand.isValid());
    assert(wor.isValid());
    assert(wnot.isValid());
}

void test_tiled_gemm() {
    size_t M = 16, K = 64, N = 16;
    TernaryMatrix A(M, K);
    TernaryMatrix B(K, N);

    // Fill with known values
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            A.set(i, k, (i % 2 == 0) ? Trit::Pos : Trit::Neg);
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            B.set(k, j, (j % 2 == 0) ? Trit::Pos : Trit::Zero);
        }
    }

    DenseMatrixInt64 C_ref(M, N);
    gemm(A, B, C_ref);

    DenseMatrixInt64 C_tiled(M, N);
    gemm_tiled_l1_l2(A, B, C_tiled, 8, 8, 2);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            assert(C_ref(i, j) == C_tiled(i, j));
        }
    }
}

void test_codec_and_gemm_guards() {
    std::cout << "  [Dense] Testing Codec Guards & MultiTrit GEMM Edge Cases...\n";

    // 1. to_int64 with num_trits > 39
    TritWord64 w(0x55, 0xAA);
    bool caught_overflow = false;
    try {
        Codec::to_int64(w, 40);
    } catch (const std::invalid_argument&) {
        caught_overflow = true;
    }
    assert(caught_overflow && "to_int64 with num_trits > 39 must throw std::invalid_argument");

    // 2. float_to_trits with NaN and Inf
    std::vector<Trit> trits;
    bool caught_nan = false;
    try {
        Codec::float_to_trits(std::numeric_limits<double>::quiet_NaN(), trits);
    } catch (const std::domain_error&) {
        caught_nan = true;
    }
    assert(caught_nan && "float_to_trits with NaN must throw std::domain_error");

    bool caught_inf = false;
    try {
        Codec::float_to_trits(std::numeric_limits<double>::infinity(), trits);
    } catch (const std::domain_error&) {
        caught_inf = true;
    }
    assert(caught_inf && "float_to_trits with Inf must throw std::domain_error");

    // 3. float_to_trits with fractional_trits clamped
    Codec::float_to_trits(1.2345, trits, 45); // Should clamp to 38 without buffer overflow
    assert(!trits.empty());

    // 4. gemm_multitrit with empty planes
    MultiTritMatrix A_empty(4, 4, 0);
    MultiTritMatrix B_empty(4, 4, 0);
    DenseMatrixInt64 C(4, 4);
    C.data[0] = 42;
    gemm_multitrit(A_empty, B_empty, C);
    assert(C.data[0] == 0 && "gemm_multitrit on empty planes must zero C and safely return");
}

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Dense Algebra & GEMM Tests...\n";
    test_kleene_logic();
    test_tiled_gemm();
    test_codec_and_gemm_guards();
    std::cout << "  -> PASSED: All Dense Algebra Unit Tests Verified.\n";
    return 0;
}
