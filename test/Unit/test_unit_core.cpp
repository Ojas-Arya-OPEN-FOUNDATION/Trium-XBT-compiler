/**
 * @file test_unit_core.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Ternary Core Primitives & Codecs
 */

#include "../../ternary_core.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace bt;

void test_primitives() {
    DualRailTrit pos = {1, 0};
    DualRailTrit neg = {0, 1};
    DualRailTrit zero = {0, 0};
    assert(pos.val() == 1);
    assert(neg.val() == -1);
    assert(zero.val() == 0);

    assert(to_int(from_int(1)) == 1);
    assert(to_int(from_int(-1)) == -1);
    assert(to_int(from_int(0)) == 0);

    uint64_t raw_p = 0x12345678ULL;
    uint64_t raw_n = ~raw_p & 0x87654321ULL;
    TritWord64 w(raw_p, raw_n);
    assert(w.isValid());
}

void test_btha() {
    AdderOutput ha = BTHA(TritWord64(1, 0), TritWord64(1, 0));
    assert(ha.Sum.N == 1 && ha.Sum.P == 0);
    assert(ha.Carry.P == 1 && ha.Carry.N == 0);
}

void test_base243() {
    for (uint32_t b = 0; b < 243; ++b) {
        auto trits = Base243Codec::unpack5_fast(static_cast<uint8_t>(b));
        uint8_t repacked = Base243Codec::pack5(trits);
        assert(repacked == b);
    }
}

void test_base59049() {
    for (uint32_t w = 0; w < 59049; w += 137) { // sampled across entire 16-bit range
        auto trits = Base59049Codec::unpack10_fast(static_cast<uint16_t>(w));
        uint16_t repacked = Base59049Codec::pack10(trits);
        assert(repacked == w);
    }
}

void test_stream_codec() {
    std::string text = "LLVM Unit Test Layer 1 - Balanced Ternary Foundation";
    std::vector<uint8_t> in_bytes(text.begin(), text.end());
    auto trits = UniversalTritStreamCodec::encode_bytes(in_bytes);
    auto out_bytes = UniversalTritStreamCodec::decode_bytes(trits);
    assert(out_bytes == in_bytes);
}

void test_bounds_and_overflow() {
    std::cout << "  [Core] Testing TritWord64 Bounds Guards & CSA Overflow Tracking...\n";

    // 1. TritWord64 bounds checks (prevent UB in release builds)
    TritWord64 w(0x0F, 0xF0);
    assert(w.getTrit(64) == Trit::Zero);
    assert(w.getTrit(65) == Trit::Zero);
    assert(w.getTrit(1000) == Trit::Zero);

    // setTrit out of bounds should safely no-op
    uint64_t orig_P = w.P;
    uint64_t orig_N = w.N;
    w.setTrit(64, Trit::Pos);
    w.setTrit(100, Trit::Neg);
    assert(w.P == orig_P);
    assert(w.N == orig_N);

    // 2. BT_CSA Canonicalize with carry-out overflow
    BT_CSA acc;
    // Set MSB carries that ripple out past bit 63
    TritWord64 max_word(1ULL << 63, 0);
    acc.accumulate(max_word);
    acc.accumulate(max_word); // 1 + 1 = carry +1 at bit 64
    auto canon_res = acc.canonicalize_with_overflow();
    assert(canon_res.overflow && "Carry out of MSB must be tracked");
    assert(canon_res.overflow_trit == Trit::Pos);

    // Standard canonicalize should run without crash or infinite loop
    TritWord64 resolved = acc.canonicalize();
    (void)resolved;
}

void test_swar_tree_reduction() {
    std::cout << "  [Core] Testing Douglas W. Jones SWAR BCT-to-Binary Tree Reduction...\n";

    // 1. Reference sequential Horner BCT-to-binary
    auto bct_to_bin_sequential = [](uint32_t bct) -> uint32_t {
        uint32_t total = 0;
        uint32_t mult = 1;
        for (int i = 0; i < 16; ++i) {
            uint32_t t = (bct >> (2 * i)) & 0x3U;
            assert(t < 3);
            total += t * mult;
            mult *= 3;
        }
        return total;
    };

    // 2. Verify bct_to_bin_swar32 on all 0s, 1s, and 2s
    assert(bct_to_bin_swar32(0) == 0);
    uint32_t all_ones = 0x55555555U; // 16 01_2 trits
    uint32_t all_twos = 0xAAAAAAAAU; // 16 10_2 trits
    assert(bct_to_bin_swar32(all_ones) == bct_to_bin_sequential(all_ones));
    assert(bct_to_bin_swar32(all_twos) == bct_to_bin_sequential(all_twos));
    assert(bct_to_bin_swar32(all_twos) == 43046720U); // 3^16 - 1

    // 3. Verify bct_to_bin_swar32 across 2,000 pseudo-random 16-trit words
    uint32_t seed = 0x9E3779B9U;
    auto lcg = [&seed]() {
        seed = seed * 1664525U + 1013904223U;
        return seed;
    };

    for (int iter = 0; iter < 2000; ++iter) {
        uint32_t bct = 0;
        for (int i = 0; i < 16; ++i) {
            uint32_t t = lcg() % 3;
            bct |= (t << (2 * i));
        }
        uint32_t ref = bct_to_bin_sequential(bct);
        uint32_t swar = bct_to_bin_swar32(bct);
        assert(ref == swar && "bct_to_bin_swar32 must match sequential Horner 100%");
    }

    // 4. Verify spread_16_to_32
    for (uint32_t x : {0x0000U, 0xFFFFU, 0xAAAAU, 0x5555U, 0x1234U, 0x8001U}) {
        uint32_t s = spread_16_to_32(static_cast<uint16_t>(x));
        for (int i = 0; i < 16; ++i) {
            uint32_t bit = (x >> i) & 1U;
            uint32_t s_bit = (s >> (2 * i)) & 1U;
            assert(bit == s_bit);
            assert(((s >> (2 * i + 1)) & 1U) == 0);
        }
    }

    // 5. Reference sequential Horner to_int64
    auto seq_to_int64 = [](const TritWord64& word, size_t num_trits) -> int64_t {
        int64_t total = 0;
        int64_t p3 = 1;
        for (size_t i = 0; i < num_trits; ++i) {
            total += static_cast<int64_t>(to_int(word.getTrit(i))) * p3;
            if (i + 1 < num_trits) p3 *= 3;
        }
        return total;
    };

    // 6. Test swar_to_int64 against sequential for various widths and boundary cases
    for (size_t width = 1; width <= 39; ++width) {
        // Single positive trit at MSB
        TritWord64 w_pos;
        w_pos.setTrit(width - 1, Trit::Pos);
        assert(swar_to_int64(w_pos, width) == seq_to_int64(w_pos, width));

        // Single negative trit at MSB
        TritWord64 w_neg;
        w_neg.setTrit(width - 1, Trit::Neg);
        assert(swar_to_int64(w_neg, width) == seq_to_int64(w_neg, width));

        // Alternating trits
        TritWord64 w_alt;
        for (size_t i = 0; i < width; ++i) {
            w_alt.setTrit(i, (i % 2 == 0) ? Trit::Pos : Trit::Neg);
        }
        assert(swar_to_int64(w_alt, width) == seq_to_int64(w_alt, width));
    }

    // 7. Verify swar_to_int64 across 5,000 pseudo-random 39-trit words
    for (int iter = 0; iter < 5000; ++iter) {
        TritWord64 w;
        for (size_t i = 0; i < 39; ++i) {
            int r = static_cast<int>(lcg() % 3) - 1; // {-1, 0, 1}
            w.setTrit(i, from_int(static_cast<int8_t>(r)));
        }
        int64_t ref = seq_to_int64(w, 39);
        int64_t swar = swar_to_int64(w, 39);
        assert(ref == swar && "swar_to_int64 must match sequential conversion 100%");
    }

    // 8. num_trits > 39 must throw std::invalid_argument
    bool threw = false;
    try {
        TritWord64 w;
        swar_to_int64(w, 40);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw && "num_trits > 39 must throw std::invalid_argument");

    std::cout << "    -> Douglas W. Jones SWAR reduction verified bit-exact 100% across all cases.\n";
}

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Core Primitives Tests...\n";
    test_primitives();
    test_btha();
    test_base243();
    test_base59049();
    test_stream_codec();
    test_bounds_and_overflow();
    test_swar_tree_reduction();
    std::cout << "  -> PASSED: All Core Unit Tests Verified.\n";
    return 0;
}
