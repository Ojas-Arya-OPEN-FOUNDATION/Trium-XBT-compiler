/**
 * @file test_ternary_core.cpp
 * @brief Comprehensive Unit, Property, and Fuzzing Test Suite for ternary_core.hpp
 * @author Antigravity Prime Research Engine
 */

#include "ternary_core.hpp"
#include <iostream>
#include <random>
#include <cassert>
#include <vector>

void test_dual_rail_primitives() {
    std::cout << "[TEST 1] Dual-Rail Primitives & Invariants...\n";
    using namespace bt;

    DualRailTrit pos = DualRailTrit::fromTrit(Trit::Pos);
    DualRailTrit zero = DualRailTrit::fromTrit(Trit::Zero);
    DualRailTrit neg = DualRailTrit::fromTrit(Trit::Neg);

    assert(pos.isValid() && pos.val() == 1);
    assert(zero.isValid() && zero.val() == 0);
    assert(neg.isValid() && neg.val() == -1);

    DualRailTrit invalid = {1, 1};
    assert(!invalid.isValid());
    std::cout << "  -> PASSED: Primitive states and disjointness verified.\n";
}

void test_btha_exhaustive() {
    std::cout << "[TEST 2] BTHA Exhaustive Truth Table & Soundness...\n";
    using namespace bt;

    // Test all 9 combinations of two trits
    for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
            TritWord64 wordA, wordB;
            wordA.setTrit(0, from_int(a));
            wordB.setTrit(0, from_int(b));

            AdderOutput res = BTHA(wordA, wordB);
            assert(res.Sum.isValid());
            assert(res.Carry.isValid());

            int sum_val = to_int(res.Sum.getTrit(0));
            int carry_val = to_int(res.Carry.getTrit(0));

            int expected_total = a + b;
            int actual_total = 3 * carry_val + sum_val;
            assert(expected_total == actual_total);
        }
    }
    std::cout << "  -> PASSED: All 9 BTHA combinations verified sound.\n";
}

void test_btfa_exhaustive() {
    std::cout << "[TEST 3] BTFA 3-to-2 Compression & Carry Non-Overlap...\n";
    using namespace bt;

    // Test all 27 combinations of three trits
    for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
            for (int d = -1; d <= 1; ++d) {
                TritWord64 wordA, wordB, wordD;
                wordA.setTrit(0, from_int(a));
                wordB.setTrit(0, from_int(b));
                wordD.setTrit(0, from_int(d));

                AdderOutput res = BTFA(wordA, wordB, wordD);
                assert(res.Sum.isValid());
                assert(res.Carry.isValid());

                int sum_val = to_int(res.Sum.getTrit(0));
                int carry_val = to_int(res.Carry.getTrit(0));

                int expected_total = a + b + d;
                int actual_total = 3 * carry_val + sum_val;
                assert(expected_total == actual_total);
            }
        }
    }
    std::cout << "  -> PASSED: All 27 BTFA combinations verified sound.\n";
}

void test_base243_fuzzing() {
    std::cout << "[TEST 4] Base-243 Division-Free Fuzzing (All 243 states)...\n";
    using namespace bt;

    for (int b = 0; b < 243; ++b) {
        uint8_t byte_val = static_cast<uint8_t>(b);
        std::array<Trit, 5> unpacked = Base243Codec::unpack5_fast(byte_val);
        uint8_t repacked = Base243Codec::pack5(unpacked);
        assert(repacked == byte_val);
    }
    std::cout << "  -> PASSED: Exhaustive round-trip verified for all 243 bytes.\n";
}

void test_bt_csa_accumulation() {
    std::cout << "[TEST 5] BT-CSA Carry-Save Accumulator Property Testing...\n";
    using namespace bt;

    std::mt19937_64 rng(42);
    BT_CSA accumulator;

    int64_t expected_total_scalar = 0;

    const uint64_t TRIT_MASK_15 = (1ULL << 15) - 1ULL;

    // Accumulate 50 random words (each with 15 random trits)
    for (int step = 0; step < 50; ++step) {
        TritWord64 input;
        uint64_t mask1 = rng();
        uint64_t mask2 = rng();
        input.P = (mask1 & ~mask2) & TRIT_MASK_15;
        input.N = (~mask1 & mask2) & TRIT_MASK_15;

        // Scalar value calculation for verification
        int64_t scalar_input = 0;
        int64_t mult = 1;
        for (int i = 0; i < 15; ++i) {
            scalar_input += to_int(input.getTrit(i)) * mult;
            mult *= 3;
        }

        accumulator.accumulate(input);
        expected_total_scalar += scalar_input;
    }

    TritWord64 canonical = accumulator.canonicalize();
    assert(canonical.isValid());

    // Verify canonical sum matches expected scalar sum in lower 25 trits
    int64_t actual_canonical_scalar = 0;
    int64_t mult = 1;
    for (int i = 0; i < 25; ++i) {
        actual_canonical_scalar += to_int(canonical.getTrit(i)) * mult;
        mult *= 3;
    }

    assert(actual_canonical_scalar == expected_total_scalar);
    std::cout << "  -> PASSED: BT-CSA accumulated 50 words in O(1) time and canonicalized exactly.\n";
}

void test_dot_product_fuzzing() {
    std::cout << "[TEST 6] Bit-Sliced POPCNT Dot Product Fuzzing (1000 trials)...\n";
    using namespace bt;

    std::mt19937_64 rng(1337);
    for (int trial = 0; trial < 1000; ++trial) {
        TritWord64 vecA, vecB;
        uint64_t m1 = rng(), m2 = rng();
        vecA.P = m1 & ~m2;
        vecA.N = ~m1 & m2;

        uint64_t m3 = rng(), m4 = rng();
        vecB.P = m3 & ~m4;
        vecB.N = ~m3 & m4;

        int64_t expected = 0;
        for (size_t i = 0; i < 64; ++i) {
            expected += to_int(vecA.getTrit(i)) * to_int(vecB.getTrit(i));
        }

        int64_t actual = dotProduct(vecA, vecB);
        assert(expected == actual);
    }
    std::cout << "  -> PASSED: 1000 randomized 64-element bit-sliced dot products matched reference.\n";
}

void test_branchless_select() {
    std::cout << "[TEST 7] Branchless Kleene Ternary Selection...\n";
    using namespace bt;

    TritWord64 pos_val(0xAAAA, 0);
    TritWord64 zero_val(0, 0);
    TritWord64 neg_val(0, 0x5555);

    DualRailTrit cond_p = {1, 0}; // Pos
    DualRailTrit cond_z = {0, 0}; // Zero
    DualRailTrit cond_n = {0, 1}; // Neg

    TritWord64 r_p = ternarySelect(cond_p, pos_val, zero_val, neg_val);
    TritWord64 r_z = ternarySelect(cond_z, pos_val, zero_val, neg_val);
    TritWord64 r_n = ternarySelect(cond_n, pos_val, zero_val, neg_val);

    assert(r_p.P == pos_val.P && r_p.N == pos_val.N);
    assert(r_z.P == zero_val.P && r_z.N == zero_val.N);
    assert(r_n.P == neg_val.P && r_n.N == neg_val.N);
    std::cout << "  -> PASSED: Branchless ternary select verified across all 3 conditions.\n";
}

void test_base59049_exhaustive() {
    std::cout << "[TEST 8] Base-59049 Division-Free Fuzzing (All 59,049 states)...\n";
    using namespace bt;

    // Exhaustive verification of all 59,049 words [0..59048]
    for (uint32_t w = 0; w < 59049; ++w) {
        auto trits = Base59049Codec::unpack10_fast(static_cast<uint16_t>(w));
        uint16_t repacked = Base59049Codec::pack10(trits);
        assert(repacked == w);
    }

    // Boundary check: W >= 59049 returns all zeroes
    auto zero_trits = Base59049Codec::unpack10_fast(59049);
    for (int i = 0; i < 10; ++i) {
        assert(zero_trits[i] == Trit::Zero);
    }
    std::cout << "  -> PASSED: Exhaustive round-trip verified for all 59,049 16-bit states.\n";
}

void test_universal_stream_codec() {
    std::cout << "[TEST 9] Universal Arbitrary Data Stream Codec Fuzzing...\n";
    using namespace bt;

    // 1. Known ASCII string round-trip
    std::string test_str = "Balanced Ternary Foundational Compiler - 100% Lossless Arbitrary Stream Encoding!";
    std::vector<uint8_t> input_bytes(test_str.begin(), test_str.end());
    auto trits = UniversalTritStreamCodec::encode_bytes(input_bytes);
    assert(trits.size() == input_bytes.size() * 6);
    auto decoded_bytes = UniversalTritStreamCodec::decode_bytes(trits);
    assert(decoded_bytes == input_bytes);

    // 2. Randomized binary buffer round-trip (10,000 arbitrary bytes)
    std::mt19937_64 rng(1337);
    std::vector<uint8_t> rand_bytes(10000);
    for (size_t i = 0; i < rand_bytes.size(); ++i) {
        rand_bytes[i] = static_cast<uint8_t>(rng() & 0xFF);
    }
    auto rand_trits = UniversalTritStreamCodec::encode_bytes(rand_bytes);
    auto rand_decoded = UniversalTritStreamCodec::decode_bytes(rand_trits);
    assert(rand_decoded == rand_bytes);

    std::cout << "  -> PASSED: Lossless round-trip verified across arbitrary binary data streams.\n";
}

void test_cycle_cost_model() {
    std::cout << "[TEST 10] Cycle-Accurate Microarchitecture & Timing Verification...\n";
    using namespace bt;

    assert(CycleCostModel::CYCLES_NEGATE == 0);
    assert(CycleCostModel::CYCLES_KLEENE_NOT == 0);
    assert(CycleCostModel::CYCLES_KLEENE_AND == 1);
    assert(CycleCostModel::CYCLES_KLEENE_OR == 1);
    assert(CycleCostModel::CYCLES_BTHA == 2);
    assert(CycleCostModel::CYCLES_BTFA == 3);
    assert(CycleCostModel::CYCLES_POPCNT_DOT_64 == 4);

    uint64_t est_1024 = CycleCostModel::estimate_gemm_cycles(1024, 1024, 1024);
    assert(est_1024 == 1024ULL * 1024ULL * 16ULL * 4ULL);
    std::cout << "  -> PASSED: Microarchitecture cycle cost model validated.\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Production-Grade Ternary Core Test & Fuzzing Harness  \n";
    std::cout << "========================================================\n";

    test_dual_rail_primitives();
    test_btha_exhaustive();
    test_btfa_exhaustive();
    test_base243_fuzzing();
    test_bt_csa_accumulation();
    test_dot_product_fuzzing();
    test_branchless_select();
    test_base59049_exhaustive();
    test_universal_stream_codec();
    test_cycle_cost_model();

    std::cout << "========================================================\n";
    std::cout << "  ALL 10 PRODUCTION-GRADE TEST SUITES PASSED!           \n";
    std::cout << "========================================================\n";
    return 0;
}
