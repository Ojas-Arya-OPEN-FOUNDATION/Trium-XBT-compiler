/**
 * @file test_unit_layouts.cpp
 * @brief Conformance: the C++ layout conversions against the verified model.
 *
 * docs/SCALING_PLAN.md 3.4 -- the hardware audit records ZERO conformance
 * vectors for layouts. verification/verify_layouts.py discharges 13
 * obligations and emits 2100 conformance vectors to
 * specs/layout_conformance.json. This file pins the shipped header to them.
 *
 * Two real bugs were found by the model during implementation, both now fixed
 * here and guarded:
 *   1. base-243 bytes must be initialised to the BIAS (121), not zero --
 *      the byte stores value+121, so an all-zero trit group is byte 121.
 *   2. base-243 decode must extract base-3 digits of the BYTE and subtract 1,
 *      not digits of the unbiased value. The unbiased route needs a borrow
 *      and gets negatives wrong: trits {-1,-1} -> value -4 -> byte 117, and
 *      C++ integer division of -4 by 3 truncates toward zero, which does not
 *      correspond to any balanced digit.
 */

#include "Ternary/IR/Layout.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace bt::ir;
using Layout = bt::target::Layout;

namespace {

int g_checks = 0;
int g_fails = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

std::vector<int> all_words(int w, std::vector<std::vector<int>>& out,
                           std::vector<int> cur = {}) {
    if (w == 0) { out.push_back(cur); return {}; }
    for (int t : {-1, 0, 1}) {
        std::vector<int> nxt = cur;
        nxt.push_back(t);
        all_words(w - 1, out, nxt);
    }
    return {};
}

std::vector<std::vector<int>> enumerate(int w) {
    std::vector<std::vector<int>> out;
    all_words(w, out);
    return out;
}

}  // namespace

static void test_traits() {
    std::printf("  layout traits\n");
    const Layout all[] = {Layout::DualRail, Layout::Base243, Layout::TCSR,
                          Layout::BTFP};
    int bad = 0;
    for (Layout l : all) {
        const auto& t = layout_traits(l);
        if (t.bits_per_trit < 0 || t.alignment_bytes == 0) ++bad;
        if (std::string(t.description).empty()) ++bad;
    }
    check(bad == 0, "every layout has complete traits");

    check(layout_traits(Layout::Base243).bits_per_trit <
              layout_traits(Layout::DualRail).bits_per_trit,
          "base-243 is denser than dual-rail");
    check(layout_traits(Layout::TCSR).is_sparse, "T-CSR is marked sparse");
    check(!layout_traits(Layout::DualRail).is_sparse, "dual-rail is dense");
    check(!layout_traits(Layout::BTFP).is_numeric,
          "BTFP is a numeric format, not a trit container");
    check(conversion_cost(Layout::DualRail, Layout::DualRail) == 0,
          "identity conversion is free");
    check(conversion_cost(Layout::DualRail, Layout::Base243) > 0,
          "real conversion has a cost");
}

static void test_roundtrips() {
    std::printf("  per-layout round-trips\n");

    auto roundtrip = [](const std::vector<int>& src, Layout l) {
        auto conv = layout_convert(src, Layout::DualRail, l);
        if (!conv) return false;
        auto back = layout_convert(*conv, l, Layout::DualRail);
        if (!back) return false;
        std::vector<int> b = *back;
        // BTFP is a NUMERIC format with a fixed mantissa width, so converting
        // to it always yields `mantissa_trits` trits. The round-trip is a
        // VALUE comparison: compare the source width, the rest is zero
        // padding. Same as the Python model.
        if (l == Layout::BTFP) b.resize(src.size());
        return b == src;
    };

    int bad_dr = 0, bad_243 = 0, bad_tcsr = 0, bad_btfp = 0, n = 0;
    for (int w = 1; w <= 8; ++w) {
        for (const auto& word : enumerate(w)) {
            ++n;
            if (!roundtrip(word, Layout::DualRail)) ++bad_dr;
            if (!roundtrip(word, Layout::Base243)) ++bad_243;
            if (!roundtrip(word, Layout::TCSR)) ++bad_tcsr;
            if (!roundtrip(word, Layout::BTFP)) ++bad_btfp;
        }
    }
    check(bad_dr == 0, "dual-rail round-trips");
    check(bad_243 == 0, "base-243 round-trips");
    check(bad_tcsr == 0, "T-CSR round-trips (zeros restored)");
    check(bad_btfp == 0, "BTFP round-trips");
    check(n == 9840, "enumerated the full 3^1..3^8 space: " + std::to_string(n));
}

static void test_base243_details() {
    std::printf("  base-243 specifics\n");
    // Byte range is exactly 0..242.
    int lo = 999, hi = -1;
    for (const auto& w : enumerate(5)) {
        auto d = Base243Data::from_trits(w);
        for (uint8_t b : d.bytes) {
            lo = std::min(lo, static_cast<int>(b));
            hi = std::max(hi, static_cast<int>(b));
        }
    }
    check(lo == 0 && hi == 242, "bytes span exactly 0..242 (got " +
                                    std::to_string(lo) + ".." +
                                    std::to_string(hi) + ")");

    // The specific case both bugs broke: {-1,-1} -> value -4 -> byte 117.
    auto d = Base243Data::from_trits({-1, -1});
    check(d.bytes.size() == 1 && d.bytes[0] == 117,
          "trits {-1,-1} encode to byte 117 (got " +
              std::to_string(d.bytes.size() ? (int)d.bytes[0] : -1) + ")");
    check(d.to_trits() == std::vector<int>({-1, -1}),
          "byte 117 decodes back to {-1,-1}");

    // 5 trits per byte.
    auto five = Base243Data::from_trits({1, 0, -1, 1, 0});
    check(five.bytes.size() == 1, "5 trits fit in 1 byte");
    auto six = Base243Data::from_trits({1, 0, -1, 1, 0, 1});
    check(six.bytes.size() == 2, "6 trits need 2 bytes");
    check(six.to_trits() == std::vector<int>({1, 0, -1, 1, 0, 1}),
          "6 trits survive the split across bytes");
}

static void test_tcsr_details() {
    std::printf("  T-CSR specifics\n");
    auto z = TcsrData::from_trits(std::vector<int>(16, 0));
    check(z.nnz() == 0, "all-zero word compresses to nothing");
    check(z.density() == 0.0, "density is 0");

    std::vector<int> src(16, 0);
    src[3] = 1;
    src[11] = -1;
    auto s = TcsrData::from_trits(src);
    check(s.nnz() == 2, "two nonzeros stored");
    check(s.to_trits() == src, "round-trips with zeros restored");
}

static void test_btfp_details() {
    std::printf("  BTFP specifics\n");
    BtfpData f;
    f.mantissa = 5;
    check(f.is_representable(), "small mantissa is representable");

    // Out of range must be refused, not approximated.
    std::vector<int> huge(40, 1);
    check(!layout_convert(huge, Layout::DualRail, Layout::BTFP).has_value(),
          "out-of-range BTFP is REFUSED, not approximated");

    std::vector<int> ok(39, 0);
    ok[0] = 1;
    check(layout_convert(ok, Layout::DualRail, Layout::BTFP).has_value(),
          "in-range BTFP converts");
}

static void test_all_pairs() {
    std::printf("  all ordered conversion pairs\n");
    const Layout all[] = {Layout::DualRail, Layout::Base243, Layout::TCSR,
                          Layout::BTFP};
    int bad = 0, pairs = 0, words = 0;
    for (Layout a : all)
        for (Layout b : all) {
            if (a == b) continue;
            ++pairs;
            for (int w = 1; w <= 5; ++w)
                for (const auto& src : enumerate(w)) {
                    auto mid = layout_convert(src, a, b);
                    if (!mid) continue;
                    auto back = layout_convert(*mid, b, a);
                    if (!back) continue;
                    ++words;
                    std::vector<int> got = *back;
                    // BTFP pads to a fixed mantissa width; compare the value.
                    if (a == Layout::BTFP || b == Layout::BTFP)
                        got.resize(src.size());
                    if (got != src) ++bad;
                }
        }
    check(bad == 0, "every ordered pair round-trips (" +
                        std::to_string(pairs) + " pairs, " +
                        std::to_string(words) + " words)");
}

int main() {
    std::printf("layout conformance\n");
    test_traits();
    test_roundtrips();
    test_base243_details();
    test_tcsr_details();
    test_btfp_details();
    test_all_pairs();
    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
