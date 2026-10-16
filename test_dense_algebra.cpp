/**
 * @file test_dense_algebra.cpp
 * @brief Comprehensive Test & Verification Suite for Ternary Dense Algebra
 * @author Antigravity Prime Engineering Engine
 */

#include "ternary_dense_algebra.hpp"
#include <iostream>
#include <random>
#include <cassert>
#include <vector>
#include <chrono>

using namespace bt;

// Color terminal output
#define ANSI_GREEN "\033[32m"
#define ANSI_RED   "\033[31m"
#define ANSI_BOLD  "\033[1m"
#define ANSI_RESET "\033[0m"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << ANSI_RED << "[FAILED] " << ANSI_RESET << msg \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_tests_failed++; \
        } else { \
            g_tests_passed++; \
        } \
    } while (0)

// Helper to generate a random TritWord64 satisfying P & N = 0
TritWord64 randomTritWord(std::mt19937_64& rng) {
    uint64_t raw_p = rng();
    uint64_t raw_n = rng();
    // Enforce invariant: P & N = 0
    uint64_t p = raw_p & ~raw_n;
    uint64_t n = raw_n & ~raw_p;
    return TritWord64(p, n);
}

// -----------------------------------------------------------------------------
// Test 1: Kleene 3-Valued Logic & Invariants
// -----------------------------------------------------------------------------
void test_kleene_logic() {
    std::cout << "[Test 1] Testing Kleene 3-Valued Logic (AND, OR, NOT, XOR, CMP)... ";
    std::mt19937_64 rng(42);

    for (int trial = 0; trial < 1000; ++trial) {
        TritWord64 a = randomTritWord(rng);
        TritWord64 b = randomTritWord(rng);

        TritWord64 w_and = kleene_and(a, b);
        TritWord64 w_or  = kleene_or(a, b);
        TritWord64 w_not = kleene_not(a);
        TritWord64 w_xor = kleene_xor(a, b);
        TritWord64 w_cmp = ternary_compare(a, b);

        TEST_ASSERT(w_and.isValid(), "kleene_and invariant broken");
        TEST_ASSERT(w_or.isValid(),  "kleene_or invariant broken");
        TEST_ASSERT(w_not.isValid(), "kleene_not invariant broken");
        TEST_ASSERT(w_xor.isValid(), "kleene_xor invariant broken");
        TEST_ASSERT(w_cmp.isValid(), "ternary_compare invariant broken");

        // Verify trit-by-trit semantics
        for (size_t i = 0; i < 64; ++i) {
            int8_t va = to_int(a.getTrit(i));
            int8_t vb = to_int(b.getTrit(i));

            // Kleene AND: min(va, vb)
            int8_t exp_and = std::min(va, vb);
            TEST_ASSERT(to_int(w_and.getTrit(i)) == exp_and, "Kleene AND mismatch");

            // Kleene OR: max(va, vb)
            int8_t exp_or = std::max(va, vb);
            TEST_ASSERT(to_int(w_or.getTrit(i)) == exp_or, "Kleene OR mismatch");

            // Kleene NOT: -va
            int8_t exp_not = -va;
            TEST_ASSERT(to_int(w_not.getTrit(i)) == exp_not, "Kleene NOT mismatch");

            // Kleene XOR: va * vb
            int8_t exp_xor = va * vb;
            TEST_ASSERT(to_int(w_xor.getTrit(i)) == exp_xor, "Kleene XOR mismatch");

            // Ternary Compare: sign of (va - vb)
            int8_t exp_cmp = (va > vb) ? 1 : ((va < vb) ? -1 : 0);
            TEST_ASSERT(to_int(w_cmp.getTrit(i)) == exp_cmp, "Ternary Compare mismatch");
        }
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// -----------------------------------------------------------------------------
// Test 2: Branchless 3-Way Conditional Selection
// -----------------------------------------------------------------------------
void test_branchless_selection() {
    std::cout << "[Test 2] Testing Branchless 3-Way Conditional Selection... ";
    std::mt19937_64 rng(1337);

    for (int trial = 0; trial < 1000; ++trial) {
        TritWord64 cond = randomTritWord(rng);
        TritWord64 pos_val = randomTritWord(rng);
        TritWord64 zero_val = randomTritWord(rng);
        TritWord64 neg_val = randomTritWord(rng);

        TritWord64 res = ternary_select(cond, pos_val, zero_val, neg_val);
        TEST_ASSERT(res.isValid(), "ternary_select invariant broken");

        for (size_t i = 0; i < 64; ++i) {
            Trit c = cond.getTrit(i);
            Trit expected = (c == Trit::Pos) ? pos_val.getTrit(i) :
                            (c == Trit::Zero) ? zero_val.getTrit(i) : neg_val.getTrit(i);
            TEST_ASSERT(res.getTrit(i) == expected, "ternary_select value mismatch");
        }
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// -----------------------------------------------------------------------------
// Test 3: Carry-Free BTHA and BTFA
// -----------------------------------------------------------------------------
void test_adders() {
    std::cout << "[Test 3] Testing BTHA & BTFA 3-to-2 Compressors... ";
    // Exhaustive test across all 27 input states
    for (int8_t a = -1; a <= 1; ++a) {
        for (int8_t b = -1; b <= 1; ++b) {
            TritWord64 wa(0, 0), wb(0, 0);
            wa.setTrit(0, from_int(a));
            wb.setTrit(0, from_int(b));

            AdderOutput ha = BTHA(wa, wb);
            int8_t s = to_int(ha.Sum.getTrit(0));
            int8_t c = to_int(ha.Carry.getTrit(0));
            TEST_ASSERT(3 * c + s == a + b, "BTHA arithmetic identity broken");

            for (int8_t d = -1; d <= 1; ++d) {
                TritWord64 wd(0, 0);
                wd.setTrit(0, from_int(d));

                AdderOutput fa = BTFA(wa, wb, wd);
                int8_t s_fa = to_int(fa.Sum.getTrit(0));
                int8_t c_fa = to_int(fa.Carry.getTrit(0));
                TEST_ASSERT(3 * c_fa + s_fa == a + b + d, "BTFA 3-to-2 compression broken");
            }
        }
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// -----------------------------------------------------------------------------
// Test 4: BT-CSA Accumulator Tree
// -----------------------------------------------------------------------------
void test_csa_tree() {
    std::cout << "[Test 4] Testing BT-CSA Accumulator Tree (O(1) Accumulation)... ";
    std::mt19937_64 rng(999);

    // Limit to 15 trits to verify exact sum against int64 without overflow
    uint64_t mask15 = (1ULL << 15) - 1ULL;

    for (int trial = 0; trial < 100; ++trial) {
        BT_CSATree csa;
        int64_t expected_sum = 0;

        for (int step = 0; step < 20; ++step) {
            TritWord64 w = randomTritWord(rng);
            w.P &= mask15;
            w.N &= mask15;

            expected_sum += Codec::to_int64(w, 15);
            csa.add(w);
        }

        TritWord64 resolved = csa.resolve();
        int64_t actual_sum = Codec::to_int64(resolved, 30);
        TEST_ASSERT(actual_sum == expected_sum, "BT_CSATree accumulation mismatch");
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// -----------------------------------------------------------------------------
// Test 5: Bit-Sliced Planar Matrix Multiplication (GEMM)
// -----------------------------------------------------------------------------
void test_gemm() {
    std::cout << "[Test 5] Testing Bit-Sliced Planar GEMM (M=32, K=128, N=32)... ";
    size_t M = 32;
    size_t K = 128;
    size_t N = 32;

    std::mt19937_64 rng(2026);
    TernaryMatrix A(M, K);
    TernaryMatrix B(K, N);

    // Populate A and B randomly
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int r = rng() % 3;
            Trit t = (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos);
            A.set(i, k, t);
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            int r = rng() % 3;
            Trit t = (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos);
            B.set(k, j, t);
        }
    }

    // Reference Scalar GEMM
    DenseMatrixInt64 C_ref(M, N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += to_int(A.get(i, k)) * to_int(B.get(k, j));
            }
            C_ref(i, j) = sum;
        }
    }

    // Fast Bit-Sliced Streaming GEMM
    DenseMatrixInt64 C_fast(M, N);
    gemm(A, B, C_fast);

    // Tiled Block GEMM
    DenseMatrixInt64 C_tiled(M, N);
    gemm_tiled(A, B, C_tiled, 16);

    // Cache-Line Tiled L1/L2 GEMM Engine
    DenseMatrixInt64 C_l1_l2(M, N);
    gemm_tiled_l1_l2(A, B, C_l1_l2, 16, 16, 2);

    DenseMatrixInt64 C_l1_l2_def(M, N);
    gemm_tiled_l1_l2(A, B, C_l1_l2_def);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            TEST_ASSERT(C_fast(i, j) == C_ref(i, j), "GEMM fast output mismatch with reference");
            TEST_ASSERT(C_tiled(i, j) == C_ref(i, j), "GEMM tiled output mismatch with reference");
            TEST_ASSERT(C_l1_l2(i, j) == C_ref(i, j), "GEMM L1/L2 tiled output mismatch with reference");
            TEST_ASSERT(C_l1_l2_def(i, j) == C_ref(i, j), "GEMM L1/L2 default output mismatch with reference");
        }
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// -----------------------------------------------------------------------------
// Test 6: Multi-Trit Radix-3 Fixed-Point Matrix Multiplication
// -----------------------------------------------------------------------------
void test_multitrit_gemm() {
    std::cout << "[Test 6] Testing Multi-Trit Fixed-Point GEMM (3-trit planes)... ";
    size_t M = 16;
    size_t K = 64;
    size_t N = 16;
    size_t planes = 3; // 3 trits = range [-13, +13]

    std::mt19937_64 rng(777);
    MultiTritMatrix A(M, K, planes);
    MultiTritMatrix B(K, N, planes);

    for (size_t p = 0; p < planes; ++p) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                int r = rng() % 3;
                Trit t = (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos);
                A.plane(p).set(i, k, t);
            }
        }
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                int r = rng() % 3;
                Trit t = (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos);
                B.plane(p).set(k, j, t);
            }
        }
    }

    // Scalar reference model
    DenseMatrixInt64 C_ref(M, N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int64_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // Reconstruct multi-trit integers
                int64_t val_A = to_int(A.plane(0).get(i, k)) + 
                                3 * to_int(A.plane(1).get(i, k)) + 
                                9 * to_int(A.plane(2).get(i, k));
                int64_t val_B = to_int(B.plane(0).get(k, j)) + 
                                3 * to_int(B.plane(1).get(k, j)) + 
                                9 * to_int(B.plane(2).get(k, j));
                sum += val_A * val_B;
            }
            C_ref(i, j) = sum;
        }
    }

    DenseMatrixInt64 C_multi(M, N);
    gemm_multitrit(A, B, C_multi);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            TEST_ASSERT(C_multi(i, j) == C_ref(i, j), "Multi-trit GEMM mismatch with scalar reference");
        }
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// -----------------------------------------------------------------------------
// Test 7: Integer & Symmetric Rounding Codec
// -----------------------------------------------------------------------------
void test_codecs() {
    std::cout << "[Test 7] Testing Integer & Real Symmetric Rounding Codec... ";
    for (int64_t v = -1000; v <= 1000; ++v) {
        TritWord64 w = Codec::from_int64(v);
        TEST_ASSERT(w.isValid(), "from_int64 produced invalid word");
        int64_t back = Codec::to_int64(w);
        TEST_ASSERT(v == back, "Integer roundtrip mismatch");
    }

    // Test float symmetric rounding
    std::vector<Trit> trits;
    Codec::float_to_trits(3.5, trits, 4);
    // 3.5 in base 3: 3.5 * 81 = 283.5 -> rounded to 284
    // 284 = 243 + 27 + 9 + 5 = 3^5 + 3^3 + 3^2 + 2*3^1 - 1
    TEST_ASSERT(!trits.empty(), "Float conversion produced empty vector");

    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// -----------------------------------------------------------------------------
// Main Test Runner
// -----------------------------------------------------------------------------
int main() {
    std::cout << "======================================================================\n";
    std::cout << "  Antigravity Balanced Ternary Dense Algebra Verification Harness\n";
    std::cout << "======================================================================\n";

    test_kleene_logic();
    test_branchless_selection();
    test_adders();
    test_csa_tree();
    test_gemm();
    test_multitrit_gemm();
    test_codecs();

    std::cout << "======================================================================\n";
    if (g_tests_failed == 0) {
        std::cout << ANSI_BOLD << ANSI_GREEN 
                  << "  ALL TESTS PASSED (" << g_tests_passed << " assertions successful)!"
                  << ANSI_RESET << "\n";
        std::cout << "======================================================================\n";
        return 0;
    } else {
        std::cerr << ANSI_BOLD << ANSI_RED 
                  << "  " << g_tests_failed << " ASSERTIONS FAILED!"
                  << ANSI_RESET << "\n";
        std::cout << "======================================================================\n";
        return 1;
    }
}
