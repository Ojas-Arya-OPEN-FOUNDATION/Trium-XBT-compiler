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

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Dense Algebra & GEMM Tests...\n";
    test_kleene_logic();
    test_tiled_gemm();
    std::cout << "  -> PASSED: All Dense Algebra Unit Tests Verified.\n";
    return 0;
}
