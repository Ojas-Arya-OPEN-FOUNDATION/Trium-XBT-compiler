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

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Core Primitives Tests...\n";
    test_primitives();
    test_btha();
    test_base243();
    test_base59049();
    test_stream_codec();
    std::cout << "  -> PASSED: All Core Unit Tests Verified.\n";
    return 0;
}
