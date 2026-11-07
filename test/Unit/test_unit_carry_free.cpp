/**
 * @file test_unit_carry_free.cpp
 * @brief Conformance tests for the symplectic-ternary carry-free accumulator.
 *
 * Every assertion here has a machine-checked counterpart in
 * verification/verify_redundant.py (obligations V1-V7).  This file pins the
 * C++ implementation to the same reference model.
 */

#include "../../ternary_carry_free.hpp"
#include "../../ternary_dense_algebra.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace bt;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

static int g_checks = 0;
static int g_fails = 0;

static void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  [FAIL] %s\n", what);
    }
}

// ---------------------------------------------------------------------------
// Reference model: exact integer value of a word (the oracle)
// ---------------------------------------------------------------------------

using i128 = __int128;

/// val(W) = sum_i (P_i - N_i) * 3^i, exact in __int128 for w <= 80.
static i128 word_value_ref(const TritWord64& w, unsigned width = 64) {
    i128 acc = 0, p3 = 1;
    for (unsigned i = 0; i < width; ++i) {
        int t = (int)((w.P >> i) & 1ULL) - (int)((w.N >> i) & 1ULL);
        acc += (i128)t * p3;
        p3 *= 3;
    }
    return acc;
}

/// The conserved quantity H(S,C) = val(S) + 3 val(C).
static i128 energy_ref(const TritWord64& S, const TritWord64& C,
                       unsigned width = 64) {
    return word_value_ref(S, width) + 3 * word_value_ref(C, width);
}

static TritWord64 random_word(std::mt19937_64& rng, unsigned width) {
    uint64_t P = 0, N = 0;
    for (unsigned i = 0; i < width; ++i) {
        unsigned r = rng() % 3;
        if (r == 1) P |= (1ULL << i);
        else if (r == 2) N |= (1ULL << i);
    }
    return TritWord64(P, N);
}

// ---------------------------------------------------------------------------
// T1 -- the carry transfer semigroup Sigma_7
// ---------------------------------------------------------------------------

static void test_semigroup() {
    // The 5 source functions are those induced by a raw digit sum in [-2,+2],
    // and index == raw + 3.
    const int expect[5][3] = {
        {-1, -1, 0},   // raw = -2
        {-1,  0, 0},   // raw = -1
        { 0,  0, 0},   // raw =  0
        { 0,  0, 1},   // raw = +1
        { 0,  1, 1},   // raw = +2
    };
    for (int raw = -2; raw <= 2; ++raw) {
        int idx = carry_transfer_index(raw);
        check(idx == raw + 3, "transfer index == raw + 3");
        for (int c = 0; c < 3; ++c)
            check(kCarryTransfer[idx][c] == expect[raw + 2][c],
                  "source transfer matches the spec");
    }

    // Monotone, and oscillation <= 1 (carries cannot amplify).
    for (int i = 0; i < 7; ++i) {
        check(kCarryTransfer[i][0] <= kCarryTransfer[i][1] &&
                  kCarryTransfer[i][1] <= kCarryTransfer[i][2],
              "semigroup element is monotone");
        check(kCarryTransfer[i][2] - kCarryTransfer[i][0] <= 1,
              "semigroup element has oscillation <= 1");
    }

    // Composition is closed, associative, and matches the semantic definition.
    for (int f = 0; f < 7; ++f) {
        for (int g = 0; g < 7; ++g) {
            int h = kCompose[f][g];
            check(h >= 0 && h < 7, "composition stays inside Sigma_7");
            for (int c = 0; c < 3; ++c) {
                int got = kCarryTransfer[h][c];
                int want = kCarryTransfer[g][kCarryTransfer[f][c] + 1];
                check(got == want, "composition matches g(f(c))");
            }
            for (int k = 0; k < 7; ++k) {
                check(kCompose[kCompose[f][g]][k] ==
                          kCompose[f][kCompose[g][k]],
                      "composition is associative");
            }
        }
    }

    // Genuine semigroup: not commutative, not idempotent.
    check(kCompose[5][5] == 6 && kCompose[5][5] != 5, "not idempotent");
    check(kCompose[1][4] != kCompose[4][1], "not commutative");
}

// ---------------------------------------------------------------------------
// T2 -- sizing rule
// ---------------------------------------------------------------------------

static void test_sizing() {
    // N accumulations of the largest n-trit word must fit.
    for (unsigned n = 2; n <= 12; ++n) {
        i128 cap = 1;                       // (3^n - 1)/2
        for (unsigned i = 1; i < n; ++i) cap = 3 * cap + 1;
        unsigned long long M = (unsigned long long)cap;
        for (unsigned long long N : {1ULL, 3ULL, 16ULL, 256ULL}) {
            unsigned w = required_width(n, M, N);
            check(w > n, "sizing gives headroom");
            // Check the sized width really can hold N*M.
            i128 cw = 1;
            for (unsigned i = 1; i < w; ++i) cw = 3 * cw + 1;
            check(cw >= (i128)M * (i128)N, "sized width holds N*M");
        }
    }
}

// ---------------------------------------------------------------------------
// T3 -- energy conservation, per step (the core invariant, V2)
// ---------------------------------------------------------------------------

static void test_energy_conservation() {
    std::mt19937_64 rng(0xC0FFEEu);
    for (int iter = 0; iter < 20000; ++iter) {
        // width 12 leaves bit 63 clear, so no top-trit drop is possible and
        // the invariant must hold exactly.
        constexpr unsigned W = 12;
        TritWord64 S = random_word(rng, W);
        TritWord64 C = random_word(rng, W - 1);   // top trit of C stays 0
        TritWord64 X = random_word(rng, W);

        i128 before = energy_ref(S, C, W);
        CarryFreeStep st = carry_free_add(S, C, X);
        i128 after = energy_ref(st.S, st.C, W);

        if (after != before + word_value_ref(X, W)) {
            check(false, "H(S',C') == H(S,C) + val(X)");
            return;
        }
    }
    check(true, "H(S',C') == H(S,C) + val(X)  [20000 random cases]");
}

// ---------------------------------------------------------------------------
// T4 -- resolve_parallel agrees with the ripple BT_CSATree::resolve (D2)
// ---------------------------------------------------------------------------

static void test_resolve_matches_ripple() {
    std::mt19937_64 rng(0xBEEFu);
    int mismatch = 0, ripple_mismatch = 0;
    for (int iter = 0; iter < 50000; ++iter) {
        constexpr unsigned W = 20;               // well below 63: no drop
        TritWord64 S = random_word(rng, W);
        TritWord64 C = random_word(rng, W - 1);

        TritWord64 fast = resolve_parallel(S, C);

        BT_CSATree tree(S, C);
        bool of = false;
        TritWord64 slow = tree.resolve(&of);

        if (word_value_ref(fast, W) != word_value_ref(slow, W)) ++mismatch;
        TritWord64 bounded = resolve_bounded_ripple(S, C);
        if (word_value_ref(bounded, W) != word_value_ref(slow, W))
            ++ripple_mismatch;
    }
    check(mismatch == 0, "resolve_parallel == BT_CSATree::resolve [50000 cases]");
    check(ripple_mismatch == 0,
          "resolve_bounded_ripple == BT_CSATree::resolve [50000 cases]");
}

// ---------------------------------------------------------------------------
// T5 -- restore returns the exact canonical value (V4)
// ---------------------------------------------------------------------------

static void test_restore_exact() {
    std::mt19937_64 rng(0x5EEDu);
    int bad = 0, tested = 0, skipped = 0;
    constexpr unsigned W = 18;
    i128 cap = 1;                                  // (3^W - 1)/2
    for (unsigned i = 1; i < W; ++i) cap = 3 * cap + 1;

    for (int iter = 0; iter < 200000 && tested < 50000; ++iter) {
        TritWord64 S = random_word(rng, W);
        TritWord64 C = random_word(rng, W - 1);
        i128 want = energy_ref(S, C, W);
        // Only meaningful when the value is actually representable in W trits;
        // otherwise the modular wrap is expected (and is a sizing issue, not
        // a correctness issue -- see required_width()).
        if (want > cap || want < -cap) { ++skipped; continue; }
        ++tested;
        if (word_value_ref(resolve_parallel(S, C), W) != want) ++bad;
        if (word_value_ref(resolve_bounded_ripple(S, C), W) != want) ++bad;
    }
    check(bad == 0 && tested == 50000,
          "restore returns exactly val(S) + 3 val(C) [50000 in-range cases]");
}

// ---------------------------------------------------------------------------
// T6 -- end to end: N accumulations, one restore, exact sum (V3/V6)
// ---------------------------------------------------------------------------

static void test_end_to_end() {
    std::mt19937_64 rng(0xFACEu);
    int bad = 0, oflow = 0;

    for (int iter = 0; iter < 20000; ++iter) {
        constexpr unsigned N = 32;
        constexpr unsigned W = 24;      // sized: 32 * (3^8) fits comfortably
        std::vector<TritWord64> xs;
        i128 expect = 0;
        for (unsigned i = 0; i < N; ++i) {
            TritWord64 x = random_word(rng, 8);   // |val| <= (3^8-1)/2
            xs.push_back(x);
            expect += word_value_ref(x, W);
        }

        CarryFreeAccumulator acc;
        for (const auto& x : xs) acc.add(x);
        if (acc.overflowed()) ++oflow;
        i128 got = word_value_ref(acc.resolve(), W);
        if (got != expect) ++bad;
    }
    check(bad == 0, "32 carry-free accumulations restore to the exact sum");
    check(oflow == 0, "correctly sized accumulator never overflows");
}

// ---------------------------------------------------------------------------
// T7 -- edge cases (V6)
// ---------------------------------------------------------------------------

static void test_edge_cases() {
    struct Case {
        const char* name;
        uint64_t P, N;
    };
    const Case cases[] = {
        {"all +1", ~0ULL, 0ULL},
        {"all -1", 0ULL, ~0ULL},
        {"zero", 0ULL, 0ULL},
        {"lsb +1", 1ULL, 0ULL},
        {"lsb -1", 0ULL, 1ULL},
        {"alternating", 0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL},
    };

    for (const auto& c : cases) {
        TritWord64 X(c.P, c.N);
        // Sized accumulator: 4 accumulations of a 16-trit operand.
        constexpr unsigned W = 24;
        TritWord64 Xw(c.P & ((1ULL << 16) - 1), c.N & ((1ULL << 16) - 1));
        i128 expect = 4 * word_value_ref(Xw, W);

        CarryFreeAccumulator acc;
        for (int i = 0; i < 4; ++i) {
            bool ok = acc.add(Xw);
            if (!ok) check(false, "edge case overflowed");
        }
        i128 got = word_value_ref(acc.resolve(), W);
        if (got != expect) {
            std::printf("  [%s] expect %lld got %lld\n", c.name,
                        (long long)expect, (long long)got);
            check(false, "edge case restores exactly");
        } else {
            check(true, "edge case restores exactly");
        }
    }
}

// ---------------------------------------------------------------------------
// T8 -- defect D1 is now detected rather than silent
// ---------------------------------------------------------------------------

static void test_overflow_detection() {
    TritWord64 S(0, 0);
    TritWord64 C(1ULL << 63, 0);   // a carry parked at the very top trit
    CarryFreeStep st = carry_free_add(S, C, TritWord64(0, 0));
    check(st.overflow, "top-trit carry in C is reported, not silently dropped");

    bool of = false;
    (void)resolve_parallel(S, C, &of);
    check(of, "resolve reports the top-trit carry too");

    // And a clear top trit reports no overflow.
    TritWord64 C_ok(1ULL << 62, 0);
    check(!carry_free_add(S, C_ok, TritWord64(0, 0)).overflow,
          "no false overflow report");
}

// ---------------------------------------------------------------------------

int main() {
    std::printf("carry-free accumulator conformance\n");
    test_semigroup();
    test_sizing();
    test_energy_conservation();
    test_resolve_matches_ripple();
    test_restore_exact();
    test_end_to_end();
    test_edge_cases();
    test_overflow_detection();

    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
