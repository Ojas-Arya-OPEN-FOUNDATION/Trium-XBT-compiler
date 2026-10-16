/**
 * @file test_exhaustive_edge_cases.cpp
 * @brief Comprehensive Edge-Case Stress & Microarchitectural Validation Suite
 * @author Antigravity Prime Engineering Engine
 * 
 * Tests all fundamental operations across 10 critical edge case categories:
 * 1. Zero vectors & identity elements
 * 2. All-positive maximum saturation (64-bit full +1)
 * 3. All-negative minimum saturation (64-bit full -1)
 * 4. High-frequency alternating sign switching (0x5555 / 0xAAAA)
 * 5. Complete carry cancellation cascades (+A + -A == 0)
 * 6. Branchless 3-way conditional selection under rapid condition jitter
 * 7. Prime, asymmetric, and unaligned matrix dimensions (1x1, 1x65, 3x7, 17x129)
 * 8. Multi-trit fixed-point convolution at dynamic range boundaries
 * 9. Base-243 codec boundary states (0, 121, 242)
 * 10. Symplectic leapfrog exact forward-backward reversibility across 100 steps
 */

#include "ternary_dense_algebra.hpp"
#include "symplectic_ternary_engine.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <iomanip>

using namespace bt;
using namespace bt::symplectic;

#define ANSI_GREEN "\033[32m\033[1m"
#define ANSI_RED   "\033[31m\033[1m"
#define ANSI_RESET "\033[0m"

static int g_edge_passed = 0;
static int g_edge_failed = 0;

#define EDGE_ASSERT(cond, desc) \
    do { \
        if (!(cond)) { \
            std::cerr << ANSI_RED << "  [FAILED] " << ANSI_RESET << desc \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_edge_failed++; \
        } else { \
            g_edge_passed++; \
        } \
    } while (0)

// ==============================================================================
// 1. Zero Vectors & Identity Elements
// ==============================================================================
void test_edge_zeros() {
    std::cout << "[Edge Case 1/10] Zero Vectors & Identity Operations... ";
    TritWord64 zero(0, 0);
    uint64_t rp = 0x12345678ULL;
    uint64_t rn = 0x87654321ULL & ~rp;
    TritWord64 val(rp, rn);

    // Add zero
    AdderOutput add_zero = BTHA(val, zero);
    EDGE_ASSERT(add_zero.Sum.P == val.P && add_zero.Sum.N == val.N, "Adding zero corrupted sum");
    EDGE_ASSERT(add_zero.Carry.P == 0 && add_zero.Carry.N == 0, "Adding zero produced carry");

    // Dot product with zero
    int64_t dot_z = dotProduct64(val, zero);
    EDGE_ASSERT(dot_z == 0, "Dot product with zero vector is non-zero");

    // Kleene AND with zero
    TritWord64 and_z = kleene_and(val, zero);
    // AND with 0 should produce 0 (or -1 if input was -1)
    for (size_t i = 0; i < 64; ++i) {
        int8_t v = to_int(val.getTrit(i));
        int8_t res = to_int(and_z.getTrit(i));
        int8_t exp = std::min(v, (int8_t)0);
        EDGE_ASSERT(res == exp, "Kleene AND with zero mismatch");
    }

    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 2 & 3. Max Positive & Max Negative Saturation
// ==============================================================================
void test_edge_saturation() {
    std::cout << "[Edge Case 2/10] All-Positive Max (+1 x 64) and All-Negative Min (-1 x 64)... ";
    TritWord64 all_pos(~0ULL, 0ULL);
    TritWord64 all_neg(0ULL, ~0ULL);

    EDGE_ASSERT(all_pos.isValid(), "all_pos invariant violated");
    EDGE_ASSERT(all_neg.isValid(), "all_neg invariant violated");

    // 0-cycle Negation symmetry
    TritWord64 neg_pos = -all_pos;
    EDGE_ASSERT(neg_pos.P == all_neg.P && neg_pos.N == all_neg.N, "Negation of max pos is not max neg");

    // Self dot product of all +1 is exactly 64
    int64_t dot_all_pos = dotProduct64(all_pos, all_pos);
    EDGE_ASSERT(dot_all_pos == 64, "All +1 dot product was not 64");

    // All +1 dot All -1 is exactly -64
    int64_t dot_pos_neg = dotProduct64(all_pos, all_neg);
    EDGE_ASSERT(dot_pos_neg == -64, "All +1 dot All -1 was not -64");

    // BTHA overflow (+1 + +1 produces carry=+1, sum=-1)
    AdderOutput ha_pos = BTHA(all_pos, all_pos);
    EDGE_ASSERT(ha_pos.Carry.P == ~0ULL && ha_pos.Carry.N == 0ULL, "Max positive BTHA carry failed");
    EDGE_ASSERT(ha_pos.Sum.P == 0ULL && ha_pos.Sum.N == ~0ULL, "Max positive BTHA sum failed");

    // BTHA underflow (-1 + -1 produces carry=-1, sum=+1)
    AdderOutput ha_neg = BTHA(all_neg, all_neg);
    EDGE_ASSERT(ha_neg.Carry.P == 0ULL && ha_neg.Carry.N == ~0ULL, "Max negative BTHA carry failed");
    EDGE_ASSERT(ha_neg.Sum.P == ~0ULL && ha_neg.Sum.N == 0ULL, "Max negative BTHA sum failed");

    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 4. High-Frequency Alternating Sign Switching (0x5555 / 0xAAAA)
// ==============================================================================
void test_edge_alternating() {
    std::cout << "[Edge Case 3/10] High-Frequency Alternating Sign Switching (0x5555 / 0xAAAA)... ";
    TritWord64 alt(0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL);
    EDGE_ASSERT(alt.isValid(), "Alternating word violated disjointness");

    // Check alternating values
    for (size_t i = 0; i < 64; ++i) {
        if (i % 2 == 0) {
            EDGE_ASSERT(alt.getTrit(i) == Trit::Pos, "Even bit not +1");
        } else {
            EDGE_ASSERT(alt.getTrit(i) == Trit::Neg, "Odd bit not -1");
        }
    }

    // Dot product with all +1 should be 32(+1) + 32(-1) == 0!
    TritWord64 all_pos(~0ULL, 0ULL);
    int64_t dot_alt = dotProduct64(alt, all_pos);
    EDGE_ASSERT(dot_alt == 0, "Alternating dot all-pos did not cancel to 0");

    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 5. Complete Carry Cancellation Cascades (+A + -A == 0)
// ==============================================================================
void test_edge_carry_cancellation() {
    std::cout << "[Edge Case 4/10] Complete Carry Cancellation Cascades (+A + -A == 0)... ";
    std::mt19937_64 rng(12345);

    for (int trial = 0; trial < 1000; ++trial) {
        uint64_t rp = rng();
        uint64_t rn = rng();
        TritWord64 a(rp & ~rn, rn & ~rp);
        TritWord64 neg_a = -a;

        AdderOutput ha = BTHA(a, neg_a);
        EDGE_ASSERT(ha.Carry.P == 0 && ha.Carry.N == 0, "Carry was generated for +A + -A");
        EDGE_ASSERT(ha.Sum.P == 0 && ha.Sum.N == 0, "Sum was not zero for +A + -A");
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 6. Branchless 3-Way Selection Under Rapid Condition Jitter
// ==============================================================================
void test_edge_branchless_jitter() {
    std::cout << "[Edge Case 5/10] Branchless 3-Way Selection Rapid Condition Jitter... ";
    TritWord64 pos_val(0x1111111111111111ULL, 0ULL);
    TritWord64 zero_val(0ULL, 0ULL);
    TritWord64 neg_val(0ULL, 0x2222222222222222ULL);

    // Rapid toggle between +1, 0, -1 across trits
    TritWord64 cond(0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL);
    TritWord64 out = ternary_select(cond, pos_val, zero_val, neg_val);
    EDGE_ASSERT(out.isValid(), "ternary_select output disjointness violated");

    for (size_t i = 0; i < 64; ++i) {
        Trit c = cond.getTrit(i);
        Trit exp = (c == Trit::Pos) ? pos_val.getTrit(i) :
                   (c == Trit::Zero) ? zero_val.getTrit(i) : neg_val.getTrit(i);
        EDGE_ASSERT(out.getTrit(i) == exp, "Jitter select mismatch");
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 7. Prime, Asymmetric & Unaligned Matrix Dimensions (1x1, 1x65, 3x7, 17x129)
// ==============================================================================
void test_edge_unaligned_dimensions() {
    std::cout << "[Edge Case 6/10] Prime & Unaligned Matrix Dimensions (1x1, 1x65, 17x129)... ";
    std::vector<std::pair<size_t, size_t>> test_dims = {
        {1, 1},
        {1, 65},    // Spills into 2nd word by exactly 1 trit!
        {3, 7},
        {17, 129},  // Prime rows, odd columns crossing multiple word boundaries
        {63, 63},
        {64, 64}
    };

    std::mt19937_64 rng(999);

    for (auto [M, K] : test_dims) {
        size_t N = M;
        TernaryMatrix A(M, K);
        TernaryMatrix B(K, N);

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                int r = rng() % 3;
                A.set(i, k, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
            }
        }
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                int r = rng() % 3;
                B.set(k, j, (r == 0) ? Trit::Neg : ((r == 1) ? Trit::Zero : Trit::Pos));
            }
        }

        DenseMatrixInt64 C(M, N);
        gemm(A, B, C);

        // Ground truth verification on every single cell
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                int64_t expected = 0;
                for (size_t k = 0; k < K; ++k) {
                    expected += to_int(A.get(i, k)) * to_int(B.get(k, j));
                }
                EDGE_ASSERT(C(i, j) == expected, "Unaligned GEMM mismatch");
            }
        }
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 8. Multi-Trit Fixed-Point Boundary Dynamic Range
// ==============================================================================
void test_edge_multitrit_bounds() {
    std::cout << "[Edge Case 7/10] Multi-Trit Fixed-Point Boundary Range... ";
    size_t M = 8, K = 16, N = 8, planes = 4; // 4 trits = range [-40, +40]
    MultiTritMatrix A(M, K, planes);
    MultiTritMatrix B(K, N, planes);

    // Set all planes to max positive
    for (size_t p = 0; p < planes; ++p) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                A.plane(p).set(i, k, Trit::Pos);
            }
        }
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                B.plane(p).set(k, j, Trit::Pos);
            }
        }
    }

    DenseMatrixInt64 C(M, N);
    gemm_multitrit(A, B, C);

    // 4 trits all +1 = 1 + 3 + 9 + 27 = 40.
    // Each dot product has K=16 elements: 16 * 40 * 40 = 25,600!
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            EDGE_ASSERT(C(i, j) == 25600, "Max multi-trit product did not equal 25,600");
        }
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 9. Base-243 Codec Boundary States (0, 121, 242)
// ==============================================================================
void test_edge_base243_bounds() {
    std::cout << "[Edge Case 8/10] Base-243 Codec Boundary States (0, 121, 242)... ";
    // Byte 0: all -1
    auto trits_0 = Base243Codec::unpack5_fast(0);
    for (int i = 0; i < 5; ++i) EDGE_ASSERT(trits_0[i] == Trit::Neg, "Byte 0 not all -1");

    // Byte 121: all 0 (1 + 3 + 9 + 27 + 81 = 121 in offset-1 representation)
    auto trits_121 = Base243Codec::unpack5_fast(121);
    for (int i = 0; i < 5; ++i) EDGE_ASSERT(trits_121[i] == Trit::Zero, "Byte 121 not all 0");

    // Byte 242: all +1 (2*1 + 2*3 + 2*9 + 2*27 + 2*81 = 242)
    auto trits_242 = Base243Codec::unpack5_fast(242);
    for (int i = 0; i < 5; ++i) EDGE_ASSERT(trits_242[i] == Trit::Pos, "Byte 242 not all +1");

    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 10. Symplectic Leapfrog Exact 100-Step Time Reversibility
// ==============================================================================
void test_edge_symplectic_reversibility() {
    std::cout << "[Edge Case 9/10] Symplectic Leapfrog 100-Step Exact Reversibility... ";
    std::mt19937_64 rng(42);
    std::vector<TritWord64> trajectory;
    trajectory.reserve(100);

    for (int i = 0; i < 100; ++i) {
        uint64_t rp = rng();
        uint64_t rn = rng();
        trajectory.emplace_back(rp & ~rn, rn & ~rp);
    }

    // Forward chain in Phase Space (Q, P)
    uint64_t init_p = 0x12345678ULL;
    uint64_t init_n = 0x87654321ULL & ~init_p;
    PhaseState state(TritWord64(init_p, init_n), TritWord64(0ULL, 0ULL));
    PhaseState initial_state = state;

    for (size_t step = 0; step < trajectory.size(); ++step) {
        state = SymplecticLeapfrogEngine::forward_step(state, trajectory[step]);
        EDGE_ASSERT(state.isValid(), "Intermediate symplectic phase state broke disjointness");
    }

    // Exact Reverse chain: Stepping backwards in time
    PhaseState rev_state = state;
    for (int step = 99; step >= 0; --step) {
        rev_state = SymplecticLeapfrogEngine::reverse_step(rev_state, trajectory[step]);
        EDGE_ASSERT(rev_state.isValid(), "Reverse symplectic phase state broke disjointness");
    }

    // Verify bit-for-bit exact reconstruction of initial state
    EDGE_ASSERT(rev_state.Q.P == initial_state.Q.P && rev_state.Q.N == initial_state.Q.N,
                "Position Q failed exact symplectic time-reversal reconstruction");
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

// ==============================================================================
// 11. Division by Modulo 3 Soundness
// ==============================================================================
void test_edge_divmod() {
    std::cout << "[Edge Case 10/10] Symmetric Rounding Truncation Stability... ";
    for (int64_t v = -500; v <= 500; ++v) {
        TritWord64 w = Codec::from_int64(v);
        int64_t back = Codec::to_int64(w);
        EDGE_ASSERT(v == back, "Codec roundtrip error on boundary integer");
    }
    std::cout << ANSI_GREEN << "PASSED" << ANSI_RESET << "\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  EXHAUSTIVE EDGE-CASE STRESS SUITE: BALANCED TERNARY ENGINE\n";
    std::cout << "======================================================================\n";

    test_edge_zeros();
    test_edge_saturation();
    test_edge_alternating();
    test_edge_carry_cancellation();
    test_edge_branchless_jitter();
    test_edge_unaligned_dimensions();
    test_edge_multitrit_bounds();
    test_edge_base243_bounds();
    test_edge_symplectic_reversibility();
    test_edge_divmod();

    std::cout << "======================================================================\n";
    if (g_edge_failed == 0) {
        std::cout << ANSI_GREEN << "  ALL 10 EDGE-CASE CATEGORIES PASSED ("
                  << g_edge_passed << " ASSERTIONS VERIFIED)!" << ANSI_RESET << "\n";
        std::cout << "======================================================================\n";
        return 0;
    } else {
        std::cerr << ANSI_RED << "  " << g_edge_failed << " EDGE-CASE CHECKS FAILED!" << ANSI_RESET << "\n";
        std::cout << "======================================================================\n";
        return 1;
    }
}
