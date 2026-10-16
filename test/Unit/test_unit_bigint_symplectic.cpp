/**
 * @file test_unit_bigint_symplectic.cpp
 * @brief Comprehensive Unit Tests for Arbitrary-Precision TritBigInt & Multi-Word Symplectic Integrator
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "../../ternary_bigint.hpp"
#include "../../symplectic_ternary_engine.hpp"

#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <compare>

using namespace bt;
using namespace bt::symplectic;

// ==============================================================================
// 1. Invariant Verification & Dual-Rail Disjointness (256-trit and 512-trit)
// ==============================================================================

void test_invariants_and_primitives() {
    std::cout << "[Test 1/6] Invariant Verification & Dual-Rail Primitives..." << std::endl;

    // 256-trit TritBigInt<4>
    TritBigInt<4> a256;
    assert(a256.is_zero());
    assert(a256.verify_invariants());
    assert(a256.isValid());
    assert(TritBigInt<4>::num_words() == 4);
    assert(TritBigInt<4>::total_trits() == 256);
    assert(TritBigInt<4>::precision_bits() >= 405);

    // 512-trit TritBigInt<8>
    TritBigInt<8> a512;
    assert(a512.is_zero());
    assert(a512.verify_invariants());
    assert(a512.isValid());
    assert(TritBigInt<8>::num_words() == 8);
    assert(TritBigInt<8>::total_trits() == 512);
    assert(TritBigInt<8>::precision_bits() >= 811);

    // Test getTrit & setTrit across all boundary bits
    const size_t test_indices[] = {0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255};
    for (size_t idx : test_indices) {
        a256.setTrit(idx, Trit::Pos);
        assert(a256.getTrit(idx) == Trit::Pos);
        assert(a256.verify_invariants());

        a256.setTrit(idx, Trit::Neg);
        assert(a256.getTrit(idx) == Trit::Neg);
        assert(a256.verify_invariants());

        a256.setTrit(idx, Trit::Zero);
        assert(a256.getTrit(idx) == Trit::Zero);
        assert(a256.verify_invariants());
    }

    // Negation: zero-cycle rail swap
    a256.setTrit(0, Trit::Pos);
    a256.setTrit(63, Trit::Neg);
    a256.setTrit(64, Trit::Pos);
    a256.setTrit(192, Trit::Neg);

    TritBigInt<4> neg_a = -a256;
    assert(neg_a.verify_invariants());
    assert(neg_a.getTrit(0) == Trit::Neg);
    assert(neg_a.getTrit(63) == Trit::Pos);
    assert(neg_a.getTrit(64) == Trit::Neg);
    assert(neg_a.getTrit(192) == Trit::Pos);
    assert(-neg_a == a256);

    // Integer conversion roundtrips
    int64_t sample_vals[] = {0, 1, -1, 42, -42, 1000000, -1000000, 123456789012345LL, -123456789012345LL};
    for (int64_t v : sample_vals) {
        TritBigInt<4> b = TritBigInt<4>::from_int64(v);
        assert(b.verify_invariants());
        assert(b.to_int64(39) == v);
    }

    // String conversion
    TritBigInt<4> str_val;
    str_val.setTrit(0, Trit::Pos);
    str_val.setTrit(1, Trit::Neg);
    str_val.setTrit(2, Trit::Zero);
    std::string s = str_val.to_trit_string(3);
    assert(s == "+-0");

    std::cout << "  -> PASSED" << std::endl;
}

// ==============================================================================
// 2. Multi-Word Shift Operations (Exact Multiplication/Division by 3^k)
// ==============================================================================

void test_shift_operations() {
    std::cout << "[Test 2/6] Multi-Word Shift Operations..." << std::endl;

    TritBigInt<4> a;
    a.setTrit(0, Trit::Pos);
    a.setTrit(63, Trit::Neg);
    a.setTrit(64, Trit::Pos);
    a.setTrit(127, Trit::Neg);
    a.setTrit(195, Trit::Pos);

    for (size_t k = 0; k <= 256; ++k) {
        TritBigInt<4> left = a.shift_trits_left(k);
        assert(left.verify_invariants());
        for (size_t i = 0; i < 256; ++i) {
            Trit expected = (i >= k) ? a.getTrit(i - k) : Trit::Zero;
            assert(left.getTrit(i) == expected);
        }

        TritBigInt<4> right = a.shift_trits_right(k);
        assert(right.verify_invariants());
        for (size_t i = 0; i < 256; ++i) {
            Trit expected = (i + k < 256) ? a.getTrit(i + k) : Trit::Zero;
            assert(right.getTrit(i) == expected);
        }
    }

    // Operator overloads << and >>
    TritBigInt<4> shifted = (a << 64) >> 64;
    // Lower 64 trits of shifted should be a's lower 64 trits, trits >= 192 dropped
    for (size_t i = 0; i < 192; ++i) {
        assert(shifted.getTrit(i) == a.getTrit(i));
    }
    for (size_t i = 192; i < 256; ++i) {
        assert(shifted.getTrit(i) == Trit::Zero);
    }

    std::cout << "  -> PASSED" << std::endl;
}

// ==============================================================================
// 3. Cross-Word Carry-Lookahead / Ripple-Carry Addition & Subtraction
// ==============================================================================

void test_cross_word_carry_addition() {
    std::cout << "[Test 3/6] Cross-Word Carry Ripple Addition & Subtraction..." << std::endl;

    // Small integer addition & subtraction verified against int64
    for (int64_t x = -500; x <= 500; x += 37) {
        for (int64_t y = -500; y <= 500; y += 43) {
            TritBigInt<4> ax = TritBigInt<4>::from_int64(x);
            TritBigInt<4> ay = TritBigInt<4>::from_int64(y);
            TritBigInt<4> sum = ax + ay;
            TritBigInt<4> diff = ax - ay;
            assert(sum.verify_invariants());
            assert(diff.verify_invariants());
            assert(sum.to_int64() == x + y);
            assert(diff.to_int64() == x - y);
        }
    }

    // --- Critical Carry Ripple Test Across Bit Boundaries ---

    // Boundary 1: Bit 63 -> Bit 64
    // Word 0 all 1s (64 trits of +1) + 1:
    // sum = 3^64 - (3^64 - 1)/2 = (3^64 + 1)/2
    // Word 0 becomes all -1s, Word 1 bit 0 becomes +1
    {
        TritBigInt<4> a, b;
        a.words[0] = TritWord64(~0ULL, 0ULL); // 64 trits of +1
        b.words[0] = TritWord64(1ULL, 0ULL);  // +1 at bit 0

        TritBigInt<4> c = a + b;
        assert(c.verify_invariants());
        assert(c.words[0].P == 0ULL && c.words[0].N == ~0ULL); // all -1s in word 0
        assert(c.words[1].P == 1ULL && c.words[1].N == 0ULL);  // +1 at bit 0 of word 1 (bit 64)
        assert(c.words[2].P == 0ULL && c.words[2].N == 0ULL);
        assert(c.words[3].P == 0ULL && c.words[3].N == 0ULL);

        // Subtraction: c - b == a
        TritBigInt<4> recovered_a = c - b;
        assert(recovered_a.verify_invariants());
        assert(recovered_a == a);
    }

    // Boundary 2: Bit 63 -> Bit 64 -> Bit 128
    // Word 0 and Word 1 all 1s (128 trits of +1) + 1:
    // Word 0 and Word 1 become all -1s, Word 2 bit 0 becomes +1 (bit 128)
    {
        TritBigInt<4> a, b;
        a.words[0] = TritWord64(~0ULL, 0ULL);
        a.words[1] = TritWord64(~0ULL, 0ULL);
        b.words[0] = TritWord64(1ULL, 0ULL);

        TritBigInt<4> c = a + b;
        assert(c.verify_invariants());
        assert(c.words[0].P == 0ULL && c.words[0].N == ~0ULL);
        assert(c.words[1].P == 0ULL && c.words[1].N == ~0ULL);
        assert(c.words[2].P == 1ULL && c.words[2].N == 0ULL);
        assert(c.words[3].P == 0ULL && c.words[3].N == 0ULL);

        TritBigInt<4> recovered_a = c - b;
        assert(recovered_a == a);
    }

    // Boundary 3: Bit 63 -> Bit 64 -> Bit 128 -> Bit 192 (Full 192-Trit Carry Ripple Chain)
    // Word 0, 1, 2 all 1s (192 trits of +1) + 1:
    // Word 0, 1, 2 become all -1s, Word 3 bit 0 becomes +1 (bit 192)
    {
        TritBigInt<4> a, b;
        a.words[0] = TritWord64(~0ULL, 0ULL);
        a.words[1] = TritWord64(~0ULL, 0ULL);
        a.words[2] = TritWord64(~0ULL, 0ULL);
        b.words[0] = TritWord64(1ULL, 0ULL);

        TritBigInt<4> c = a + b;
        assert(c.verify_invariants());
        assert(c.words[0].P == 0ULL && c.words[0].N == ~0ULL);
        assert(c.words[1].P == 0ULL && c.words[1].N == ~0ULL);
        assert(c.words[2].P == 0ULL && c.words[2].N == ~0ULL);
        assert(c.words[3].P == 1ULL && c.words[3].N == 0ULL);

        TritBigInt<4> recovered_a = c - b;
        assert(recovered_a == a);
    }

    // Negative Boundary: Word 0, 1, 2 all -1s + (-1):
    // Word 0, 1, 2 become all +1s, Word 3 bit 0 becomes -1
    {
        TritBigInt<4> a, b;
        a.words[0] = TritWord64(0ULL, ~0ULL);
        a.words[1] = TritWord64(0ULL, ~0ULL);
        a.words[2] = TritWord64(0ULL, ~0ULL);
        b.words[0] = TritWord64(0ULL, 1ULL); // -1 at bit 0

        TritBigInt<4> c = a + b;
        assert(c.verify_invariants());
        assert(c.words[0].P == ~0ULL && c.words[0].N == 0ULL);
        assert(c.words[1].P == ~0ULL && c.words[1].N == 0ULL);
        assert(c.words[2].P == ~0ULL && c.words[2].N == 0ULL);
        assert(c.words[3].P == 0ULL && c.words[3].N == 1ULL);

        TritBigInt<4> recovered_a = c - b;
        assert(recovered_a == a);
    }

    // 512-trit (8-word) carry ripple across 7 words (448 trits) into Word 7
    {
        TritBigInt<8> a8, b8;
        for (size_t w = 0; w < 7; ++w) {
            a8.words[w] = TritWord64(~0ULL, 0ULL);
        }
        b8.words[0] = TritWord64(1ULL, 0ULL);

        TritBigInt<8> c8 = a8 + b8;
        assert(c8.verify_invariants());
        for (size_t w = 0; w < 7; ++w) {
            assert(c8.words[w].P == 0ULL && c8.words[w].N == ~0ULL);
        }
        assert(c8.words[7].P == 1ULL && c8.words[7].N == 0ULL);

        TritBigInt<8> rec8 = c8 - b8;
        assert(rec8 == a8);
    }

    std::cout << "  -> PASSED" << std::endl;
}

// ==============================================================================
// 4. Trit & Multi-Word Multiplication
// ==============================================================================

void test_multiplication() {
    std::cout << "[Test 4/6] Trit & Multi-Word Multiplication..." << std::endl;

    TritBigInt<4> a = TritBigInt<4>::from_int64(12345);

    // mul_trit
    assert(a.mul_trit(Trit::Zero).is_zero());
    assert(a.mul_trit(Trit::Pos) == a);
    assert(a.mul_trit(Trit::Neg) == -a);

    // Multi-word multiplication verified against 64-bit int
    int64_t factors[][2] = {
        {12345, -6789},
        {-999, -999},
        {1000000, 42},
        {-3, 81},
        {0, 999999},
        {1, -123456789}
    };

    for (auto& pair : factors) {
        int64_t x = pair[0];
        int64_t y = pair[1];
        TritBigInt<4> ax = TritBigInt<4>::from_int64(x);
        TritBigInt<4> ay = TritBigInt<4>::from_int64(y);
        TritBigInt<4> prod = ax * ay;
        assert(prod.verify_invariants());
        assert(prod.to_int64() == x * y);
    }

    // Multi-word multiplication distributive property: a * (b + c) == (a * b) + (a * c)
    TritBigInt<4> m_a, m_b, m_c;
    m_a.words[0] = TritWord64(0x12345678ULL, ~0x12345678ULL & 0x87654321ULL);
    m_b.words[0] = TritWord64(0x55555555ULL, 0x0ULL);
    m_b.words[1] = TritWord64(0x0ULL, 0x33333333ULL);
    m_c.words[0] = TritWord64(0x0ULL, 0x11111111ULL);
    m_c.words[1] = TritWord64(0x44444444ULL, 0x0ULL);

    TritBigInt<4> lhs = m_a * (m_b + m_c);
    TritBigInt<4> rhs = (m_a * m_b) + (m_a * m_c);
    assert(lhs.verify_invariants());
    assert(rhs.verify_invariants());
    assert(lhs == rhs);

    // Commutativity: a * b == b * a
    assert(m_a * m_b == m_b * m_a);

    // Power-of-3 scaling: a * 3^k == a << k
    for (size_t k = 1; k <= 5; ++k) {
        int64_t p3 = 1;
        for (size_t i = 0; i < k; ++i) p3 *= 3;
        TritBigInt<4> scale_mul = m_a * TritBigInt<4>::from_int64(p3);
        TritBigInt<4> scale_shift = m_a << k;
        assert(scale_mul == scale_shift);
    }

    std::cout << "  -> PASSED" << std::endl;
}

// ==============================================================================
// 5. Three-Way Spaceship Comparison (<=>) & Relational Operators
// ==============================================================================

void test_spaceship_comparisons() {
    std::cout << "[Test 5/6] Spaceship (<=>) & Relational Comparisons..." << std::endl;

    // Small integer comparisons
    int64_t test_vals[] = {-1000, -50, -1, 0, 1, 2, 50, 1000};
    for (int64_t v1 : test_vals) {
        for (int64_t v2 : test_vals) {
            TritBigInt<4> b1 = TritBigInt<4>::from_int64(v1);
            TritBigInt<4> b2 = TritBigInt<4>::from_int64(v2);

            assert((b1 == b2) == (v1 == v2));
            assert((b1 != b2) == (v1 != v2));
            assert((b1 < b2) == (v1 < v2));
            assert((b1 > b2) == (v1 > v2));
            assert((b1 <= b2) == (v1 <= v2));
            assert((b1 >= b2) == (v1 >= v2));

            if (v1 < v2) {
                assert((b1 <=> b2) < 0);
            } else if (v1 > v2) {
                assert((b1 <=> b2) > 0);
            } else {
                assert((b1 <=> b2) == 0);
            }
        }
    }

    // Cross-word boundary comparisons (differing only at bit 63, 64, 128, 192, 255)
    TritBigInt<4> zero;
    const size_t diff_bits[] = {0, 63, 64, 127, 128, 191, 192, 255};
    for (size_t bit : diff_bits) {
        TritBigInt<4> pos, neg;
        pos.setTrit(bit, Trit::Pos);
        neg.setTrit(bit, Trit::Neg);

        assert(pos > zero);
        assert(zero > neg);
        assert(pos > neg);
        assert((pos <=> zero) > 0);
        assert((zero <=> pos) < 0);
        assert((neg <=> zero) < 0);
        assert((pos <=> pos) == 0);
    }

    // Dominance of highest differing trit:
    // a has +1 at bit 128, and -1 at ALL lower bits 0..127.
    // b has 0 at bit 128, and +1 at ALL lower bits 0..127.
    // Since 3^128 > 2 * (3^128 - 1)/2 = 3^128 - 1, a MUST be > b!
    TritBigInt<4> big_a, big_b;
    big_a.setTrit(128, Trit::Pos);
    for (size_t i = 0; i < 128; ++i) {
        big_a.setTrit(i, Trit::Neg);
        big_b.setTrit(i, Trit::Pos);
    }
    assert(big_a > big_b);
    assert((big_a <=> big_b) > 0);
    assert((big_b <=> big_a) < 0);

    std::cout << "  -> PASSED" << std::endl;
}

// ==============================================================================
// 6. MultiWordSymplecticEngine Leapfrog Trajectory Integration (10,000 Steps)
// ==============================================================================

void test_symplectic_engine_reversibility() {
    std::cout << "[Test 6/6] Multi-Word Symplectic Engine 10,000-Step Time-Reversibility..." << std::endl;

    using Engine256 = MultiWordSymplecticEngine<4>;
    using BigInt256 = TritBigInt<4>;
    using State256 = MultiWordSymplecticState<4>;

    // Initial non-trivial state with high entropy across all 4 words (256 trits)
    BigInt256 q0, p0;
    q0.words[0] = TritWord64(0x13579BDF02468ACEULL, 0x0ULL);
    q0.words[1] = TritWord64(0x0ULL, 0xECA86420FDB97531ULL);
    q0.words[2] = TritWord64(0x5A5A5A5A5A5A5A5AULL, 0x0ULL);
    q0.words[3] = TritWord64(0x0ULL, 0xA5A5A5A5A5A5A5A5ULL);

    p0.words[0] = TritWord64(0x0ULL, 0x1111222233334444ULL);
    p0.words[1] = TritWord64(0x5555666677778888ULL, 0x0ULL);
    p0.words[2] = TritWord64(0x0ULL, 0x9999AAAABBBBCCCCULL);
    p0.words[3] = TritWord64(0xDDDEEEFFF0001112ULL, 0x0ULL);

    State256 state_initial(q0, p0);
    assert(state_initial.isValid());

    // 1. Harmonic Potential Test: F(Q) = -Q
    // Verify first step non-trivial rotation in phase space: (Q, P) -> (P, -Q)
    State256 step1 = Engine256::forward_step(state_initial);
    assert(step1.isValid());
    assert(step1.Q == p0);
    assert(step1.P == -q0);
    assert(!(step1 == state_initial));

    constexpr size_t STEPS = 10000;
    State256 curr = state_initial;

    for (size_t i = 0; i < STEPS; ++i) {
        curr = Engine256::forward_step(curr);
        if (i % 2500 == 0) {
            assert(curr.isValid() && "Dual-rail disjointness invariant violated during forward integration!");
        }
    }
    assert(curr.isValid());

    // Backward Leapfrog Integration (Exact Inverse)
    for (size_t i = 0; i < STEPS; ++i) {
        curr = Engine256::reverse_step(curr);
        if (i % 2500 == 0) {
            assert(curr.isValid() && "Dual-rail disjointness invariant violated during reverse integration!");
        }
    }
    assert(curr.isValid());

    // 100% Bit-Exact Recovery of Initial State
    assert(curr == state_initial);
    assert(curr.Q == state_initial.Q);
    assert(curr.P == state_initial.P);

    // 2. Nonlinear Bit-Sliced Potential Test: F(Q) = -(Q >> 1) over 10,000 Steps
    // Explores quasi-periodic phase space with high entropy
    auto nonlinear_force = [](const BigInt256& q) {
        return -(q >> 1);
    };

    curr = state_initial;
    for (size_t i = 0; i < STEPS; ++i) {
        curr = Engine256::forward_step(curr, nonlinear_force);
        if (i % 2500 == 0) {
            assert(curr.isValid() && "Disjointness violated in nonlinear potential!");
        }
    }
    assert(curr.isValid());
    // Phase space has dynamically evolved away from initial state
    assert(!(curr == state_initial));

    for (size_t i = 0; i < STEPS; ++i) {
        curr = Engine256::reverse_step(curr, nonlinear_force);
        if (i % 2500 == 0) {
            assert(curr.isValid() && "Disjointness violated in nonlinear reverse potential!");
        }
    }
    assert(curr.isValid());
    // Exact 100% Bit-Exact Recovery under Nonlinear Force
    assert(curr == state_initial);

    // 3. 512-Trit (8-Word) Symplectic Leapfrog Integration over 1,000 Steps
    using Engine512 = MultiWordSymplecticEngine<8>;
    using BigInt512 = TritBigInt<8>;
    using State512 = MultiWordSymplecticState<8>;

    BigInt512 q0_512, p0_512;
    for (size_t w = 0; w < 8; ++w) {
        if (w % 2 == 0) {
            q0_512.words[w] = TritWord64(0x0123456789ABCDEFULL, 0x0ULL);
            p0_512.words[w] = TritWord64(0x0ULL, 0xFEDCBA9876543210ULL);
        } else {
            q0_512.words[w] = TritWord64(0x0ULL, 0x5555555555555555ULL);
            p0_512.words[w] = TritWord64(0xAAAAAAAAAAAAAAAAULL, 0x0ULL);
        }
    }

    State512 state512(q0_512, p0_512);
    assert(state512.isValid());

    State512 step1_512 = Engine512::forward_step(state512);
    assert(step1_512.isValid());
    assert(step1_512.Q == p0_512);
    assert(step1_512.P == -q0_512);
    assert(!(step1_512 == state512));

    State512 curr512 = state512;
    for (size_t i = 0; i < 1000; ++i) {
        curr512 = Engine512::forward_step(curr512);
        assert(curr512.isValid());
    }

    for (size_t i = 0; i < 1000; ++i) {
        curr512 = Engine512::reverse_step(curr512);
        assert(curr512.isValid());
    }
    assert(curr512 == state512);

    std::cout << "  -> PASSED (10,000 steps forward + 10,000 steps reverse: 100% Bit-Exact Recovery)" << std::endl;
}

// ==============================================================================
// Main Runner
// ==============================================================================

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Balanced Ternary Arbitrary-Precision BigInt & Symplectic Unit Tests" << std::endl;
    std::cout << "======================================================================" << std::endl;

    test_invariants_and_primitives();
    test_shift_operations();
    test_cross_word_carry_addition();
    test_multiplication();
    test_spaceship_comparisons();
    test_symplectic_engine_reversibility();

    std::cout << "======================================================================" << std::endl;
    std::cout << "  ALL 6 UNIT TEST SUITES PASSED WITH 0 FAILURES." << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
