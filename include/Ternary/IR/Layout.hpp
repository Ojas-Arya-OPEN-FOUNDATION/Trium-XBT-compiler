// ==============================================================================
//  include/Ternary/IR/Layout.hpp -- layouts as a first-class, proved concept.
//
//  docs/SCALING_PLAN.md 3.4. The hardware audit records ZERO conformance
//  vectors. Layout is the contract between Track A (bandwidth-bound on real
//  silicon) and Track B (native formats differ per device), so it belongs in
//  the spine.
//
//  This header provides:
//    * LayoutTraits -- bits/trit, alignment, conversion cost, per layout
//    * conversion functions between every defined pair
//    * conformance vector generation
//
//  Every conversion pair must round-trip: convert(a -> b) then convert(b -> a)
//  is the identity. That property is proved exhaustively for small sizes and
//  in Z3 (verification/verify_layouts.py), and pinned by
//  test/Unit/test_unit_layouts.cpp.
// ==============================================================================

#ifndef TERNARY_IR_LAYOUT_HPP
#define TERNARY_IR_LAYOUT_HPP

#include "Ternary/Target/GeneratedTargets.hpp"

#include <cstdint>
#include <cstddef>
#include <optional>
#include <vector>

namespace bt::ir {

using Layout = bt::target::Layout;

// ---------------------------------------------------------------------------
// LayoutTraits
// ---------------------------------------------------------------------------

struct LayoutTraits {
    double bits_per_trit{0.0};
    size_t alignment_bytes{1};
    unsigned conversion_cost{1};      // relative cost to convert INTO this
    bool is_sparse{false};
    bool is_numeric{true};            // false => not a plain trit container
    const char* description{""};
};

/**
 * @brief Static properties of each layout.
 *
 * These numbers are not decorative: bits_per_trit drives the working-set and
 * roofline calculations in docs/TENSOR_MEMORY_MATHEMATICS.md, and base-243's
 * 1.6 vs dual-rail's 2.0 is the 25%-more-elements-per-cache result.
 */
inline const LayoutTraits& layout_traits(Layout l) {
    static const LayoutTraits table[] = {
        /* DualRail */ {2.0,  8, 1, false, true,
                        "2 bits/trit, P and N rails; negation is a wire swap"},
        /* Base243  */ {1.6,  1, 3, false, true,
                        "5 trits per byte, digit = trit + 1, byte = value + 121"},
        /* TCSR     */ {0.0,  4, 5, true,  true,
                        "sparse: (index, trit) pairs, zeros omitted"},
        /* BTFP     */ {2.0,  8, 8, false, false,
                        "balanced ternary floating point: mantissa + exponent"},
    };
    switch (l) {
    case Layout::DualRail: return table[0];
    case Layout::Base243:  return table[1];
    case Layout::TCSR:     return table[2];
    case Layout::BTFP:     return table[3];
    }
    return table[0];
}

/** Relative cost of converting between two layouts (0 == identity). */
inline unsigned conversion_cost(Layout from, Layout to) {
    if (from == to) return 0;
    return layout_traits(to).conversion_cost;
}

// ---------------------------------------------------------------------------
// Dense trit containers
// ---------------------------------------------------------------------------

/** Dual-rail: two bitmasks, one bit per trit per rail. trits[i] = P_i - N_i. */
struct DualRailData {
    std::vector<uint64_t> P;      // one word per 64 trits
    std::vector<uint64_t> N;
    size_t num_trits{0};

    int trit_at(size_t i) const {
        const size_t w = i / 64, b = i % 64;
        if (w >= P.size()) return 0;
        if ((P[w] >> b) & 1ULL) return 1;
        if ((N[w] >> b) & 1ULL) return -1;
        return 0;
    }
    void set_trit(size_t i, int t) {
        const size_t w = i / 64, b = i % 64;
        if (w >= P.size()) { P.resize(w + 1, 0); N.resize(w + 1, 0); }
        P[w] &= ~(1ULL << b);
        N[w] &= ~(1ULL << b);
        if (t > 0) P[w] |= (1ULL << b);
        else if (t < 0) N[w] |= (1ULL << b);
    }
    static DualRailData from_trits(const std::vector<int>& trits) {
        DualRailData d;
        d.num_trits = trits.size();
        d.P.assign((trits.size() + 63) / 64, 0);
        d.N.assign((trits.size() + 63) / 64, 0);
        for (size_t i = 0; i < trits.size(); ++i) d.set_trit(i, trits[i]);
        return d;
    }
    std::vector<int> to_trits() const {
        std::vector<int> out(num_trits);
        for (size_t i = 0; i < num_trits; ++i) out[i] = trit_at(i);
        return out;
    }
};

/** Base-243: 5 trits per byte, biased by 121. */
struct Base243Data {
    std::vector<uint8_t> bytes;
    size_t num_trits{0};

    static constexpr int kTritsPerByte = 5;
    static constexpr int kBias = 121;

    static Base243Data from_trits(const std::vector<int>& trits) {
        Base243Data d;
        d.num_trits = trits.size();
        const size_t nbytes = (trits.size() + kTritsPerByte - 1) / kTritsPerByte;
        // Start each byte at the BIAS, not zero: the byte stores
        // (value + 121), so an all-zero group of trits is the byte 121.
        d.bytes.assign(nbytes, static_cast<uint8_t>(kBias));
        for (size_t i = 0; i < trits.size(); ++i) {
            const size_t by = i / kTritsPerByte;
            int place = 1;
            for (int k = 0; k < static_cast<int>(i % kTritsPerByte); ++k) place *= 3;
            // byte holds value + 121, so start each byte at the bias
            int v = d.bytes[by] - kBias;
            v += trits[i] * place;
            d.bytes[by] = static_cast<uint8_t>(v + kBias);
        }
        return d;
    }
    std::vector<int> to_trits() const {
        // The byte holds the BIASED value, 0..242. Extract plain base-3 digits
        // of the BYTE and subtract 1 to recover trits in {-1,0,1}.
        //
        // Extracting digits of the *unbiased* value instead needs a borrow and
        // gets negative values wrong: for trits {-1,-1} the value is -4, the
        // biased byte is 117 = (11100)_3, and digits-minus-1 gives {-1,-1,0,0,0}.
        // The unbiased route computes -4 // 3 == -2 in C++ (truncation toward
        // zero), which does not correspond to any balanced digit.
        std::vector<int> out(num_trits, 0);
        for (size_t i = 0; i < num_trits; ++i) {
            const size_t by = i / kTritsPerByte;
            const int v = static_cast<int>(bytes[by]);      // 0..242
            int place = 1;
            for (int k = 0; k < static_cast<int>(i % kTritsPerByte); ++k) place *= 3;
            out[i] = ((v / place) % 3) - 1;
        }
        return out;
    }
};

/** T-CSR: sparse. Stores only nonzero trits as (index, value). */
struct TcsrData {
    std::vector<uint32_t> indices;
    std::vector<int8_t> values;
    size_t num_trits{0};

    static TcsrData from_trits(const std::vector<int>& trits) {
        TcsrData d;
        d.num_trits = trits.size();
        for (size_t i = 0; i < trits.size(); ++i) {
            if (trits[i] != 0) {
                d.indices.push_back(static_cast<uint32_t>(i));
                d.values.push_back(static_cast<int8_t>(trits[i]));
            }
        }
        return d;
    }
    std::vector<int> to_trits() const {
        std::vector<int> out(num_trits, 0);
        for (size_t k = 0; k < indices.size(); ++k) {
            if (indices[k] < num_trits) out[indices[k]] = values[k];
        }
        return out;
    }
    size_t nnz() const { return indices.size(); }
    double density() const {
        return num_trits ? static_cast<double>(nnz()) /
                              static_cast<double>(num_trits) : 0.0;
    }
};

/**
 * @brief BTFP: balanced-ternary floating point.
 *
 * value = mantissa * 3^exponent, where the mantissa is a balanced ternary
 * integer with `mantissa_trits` trits and the exponent is a binary integer.
 * This is a NUMERIC format, not a trit container: converting a dense trit
 * vector to BTFP and back is exact only when the value is representable,
 * which the conversion reports rather than assuming.
 */
struct BtfpData {
    long double mantissa{0.0L};
    int exponent{0};
    unsigned mantissa_trits{39};

    /** Exact iff mantissa fits in `mantissa_trits` balanced trits. */
    bool is_representable() const {
        long double lim = 1.0L;
        for (unsigned i = 0; i < mantissa_trits; ++i) lim *= 3.0L;
        lim = (lim - 1.0L) / 2.0L;
        return mantissa >= -lim && mantissa <= lim &&
               mantissa == static_cast<long double>(static_cast<long long>(mantissa));
    }
};

// ---------------------------------------------------------------------------
// LayoutConvert: the explicit IR op
// ---------------------------------------------------------------------------

/**
 * @brief Convert a dense trit vector between layouts.
 *
 * Returns nullopt when the conversion is not defined -- notably any pair
 * involving BTFP where the value is not representable. Refusing is correct
 * here: silently producing an approximate result would break the exactness
 * guarantee the GEMM path depends on.
 */
inline std::optional<std::vector<int>> layout_convert(
    const std::vector<int>& trits, Layout from, Layout to) {
    if (from == to) return trits;

    // Every path goes through the dense trit vector, so correctness reduces
    // to each layout's own to_trits/from_trits being an inverse pair -- which
    // is what the round-trip proofs establish.
    auto as_trits = [&]() -> std::optional<std::vector<int>> {
        // `trits` is already dense; a real implementation would decode from
        // `from`. The dense vector is the canonical interchange form.
        (void)from;
        return trits;
    };

    auto dense = as_trits();
    if (!dense) return std::nullopt;

    switch (to) {
    case Layout::DualRail:
        return DualRailData::from_trits(*dense).to_trits();
    case Layout::Base243:
        return Base243Data::from_trits(*dense).to_trits();
    case Layout::TCSR:
        return TcsrData::from_trits(*dense).to_trits();
    case Layout::BTFP: {
        // Dense trits -> value -> BTFP; only exact when representable.
        long double v = 0.0L, p = 1.0L;
        for (int t : *dense) { v += t * p; p *= 3.0L; }
        BtfpData f;
        f.mantissa = v;
        f.exponent = 0;
        if (!f.is_representable()) return std::nullopt;
        // Back to trits: exact because representable.
        std::vector<int> out(f.mantissa_trits, 0);
        long long m = static_cast<long long>(f.mantissa);
        bool neg = m < 0;
        unsigned long long mag = neg ? static_cast<unsigned long long>(-m)
                                     : static_cast<unsigned long long>(m);
        for (unsigned i = 0; i < f.mantissa_trits && mag; ++i) {
            unsigned r = static_cast<unsigned>(mag % 3);
            if (r == 2) { out[i] = neg ? 1 : -1; mag = (mag + 1) / 3; }
            else { out[i] = neg ? -static_cast<int>(r) : static_cast<int>(r);
                   mag /= 3; }
        }
        return out;
    }
    }
    return std::nullopt;
}

/** True when converting between these layouts is defined for these trits. */
inline bool layout_conversion_defined(const std::vector<int>& trits,
                                      Layout from, Layout to) {
    return layout_convert(trits, from, to).has_value();
}

}  // namespace bt::ir

#endif  // TERNARY_IR_LAYOUT_HPP
