/**
 * @file ternary_bigint.hpp
 * @brief Arbitrary-Precision Balanced Ternary Integer Math Library (TritBigInt)
 * @author Antigravity Prime Research & Engineering Engine
 * @version 1.0.0
 * 
 * Supports multi-word bit-sliced balanced ternary arithmetic:
 * - Register-width bitplane storage (P, N rails per 64-trit word)
 * - Zero-cycle negation via rail swap
 * - Cross-word carry-lookahead / ripple-carry addition & subtraction
 * - Multi-word shift-and-add multiplication & power-of-3 trit shifts
 * - SWAR-accelerated binary conversions, spaceship comparisons, and invariant validation
 */

#ifndef TERNARY_BIGINT_HPP
#define TERNARY_BIGINT_HPP

#include "ternary_core.hpp"
#include "ternary_dense_algebra.hpp"

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <array>
#include <string>
#include <compare>
#include <bit>
#include <algorithm>
#include <ostream>
#include <stdexcept>

namespace bt {

template <size_t NumWords = 4>
class TritBigInt {
    static_assert(NumWords > 0, "TritBigInt requires at least 1 word");

public:
    std::array<TritWord64, NumWords> words{};

    constexpr TritBigInt() noexcept = default;

    constexpr explicit TritBigInt(const std::array<TritWord64, NumWords>& w) noexcept : words(w) {
        assert(verify_invariants() && "Invariant violation: P and N must be disjoint in all words");
    }

    constexpr explicit TritBigInt(int64_t val) noexcept {
        *this = from_int64(val);
    }

    // Number of 64-trit words
    static constexpr size_t num_words() noexcept { return NumWords; }

    // Total trit capacity
    static constexpr size_t total_trits() noexcept { return 64 * NumWords; }

    // Approximate equivalent binary bits (log2(3) approx 1.5849625)
    static constexpr size_t precision_bits() noexcept {
        return static_cast<size_t>(64 * NumWords * 1.5849625);
    }

    // Verify dual-rail disjointness invariant (P & N == 0) across all words
    constexpr bool verify_invariants() const noexcept {
        for (size_t w = 0; w < NumWords; ++w) {
            if ((words[w].P & words[w].N) != 0) return false;
        }
        return true;
    }

    constexpr bool isValid() const noexcept {
        return verify_invariants();
    }

    // Check if the integer is exactly zero
    constexpr bool is_zero() const noexcept {
        for (size_t w = 0; w < NumWords; ++w) {
            if (words[w].P != 0 || words[w].N != 0) return false;
        }
        return true;
    }

    // Get trit at index [0 .. 64 * NumWords - 1]
    constexpr Trit getTrit(size_t idx) const noexcept {
        if (idx >= 64 * NumWords) return Trit::Zero;
        size_t w = idx / 64;
        size_t bit = idx % 64;
        return words[w].getTrit(bit);
    }

    // Set trit at index [0 .. 64 * NumWords - 1]
    constexpr void setTrit(size_t idx, Trit v) noexcept {
        if (idx >= 64 * NumWords) return;
        size_t w = idx / 64;
        size_t bit = idx % 64;
        words[w].setTrit(bit, v);
    }

    // 0-Cycle Negation: swaps P and N in each word
    constexpr TritBigInt negate() const noexcept {
        TritBigInt res;
        for (size_t w = 0; w < NumWords; ++w) {
            res.words[w] = -words[w];
        }
        return res;
    }

    constexpr TritBigInt operator-() const noexcept {
        return negate();
    }

    // Exact Shift Left by k trits (multiplication by 3^k)
    constexpr TritBigInt shift_trits_left(size_t k) const noexcept {
        if (k == 0) return *this;
        if (k >= 64 * NumWords) return TritBigInt{};

        TritBigInt res{};
        size_t word_shift = k / 64;
        size_t bit_shift = k % 64;

        for (size_t w = word_shift; w < NumWords; ++w) {
            if (bit_shift == 0) {
                res.words[w] = words[w - word_shift];
            } else {
                uint64_t p_low = words[w - word_shift].P << bit_shift;
                uint64_t n_low = words[w - word_shift].N << bit_shift;
                uint64_t p_high = (w > word_shift) ? (words[w - word_shift - 1].P >> (64 - bit_shift)) : 0ULL;
                uint64_t n_high = (w > word_shift) ? (words[w - word_shift - 1].N >> (64 - bit_shift)) : 0ULL;
                res.words[w] = TritWord64(p_low | p_high, n_low | n_high);
            }
        }
        return res;
    }

    // Exact Shift Right by k trits (division by 3^k with truncation towards 0)
    constexpr TritBigInt shift_trits_right(size_t k) const noexcept {
        if (k == 0) return *this;
        if (k >= 64 * NumWords) return TritBigInt{};

        TritBigInt res{};
        size_t word_shift = k / 64;
        size_t bit_shift = k % 64;

        for (size_t w = 0; w + word_shift < NumWords; ++w) {
            if (bit_shift == 0) {
                res.words[w] = words[w + word_shift];
            } else {
                uint64_t p_low = words[w + word_shift].P >> bit_shift;
                uint64_t n_low = words[w + word_shift].N >> bit_shift;
                uint64_t p_high = (w + word_shift + 1 < NumWords) ? (words[w + word_shift + 1].P << (64 - bit_shift)) : 0ULL;
                uint64_t n_high = (w + word_shift + 1 < NumWords) ? (words[w + word_shift + 1].N << (64 - bit_shift)) : 0ULL;
                res.words[w] = TritWord64(p_low | p_high, n_low | n_high);
            }
        }
        return res;
    }

    // Multi-word Balanced Ternary Addition with cross-word carry propagation
    static TritBigInt add(const TritBigInt& a, const TritBigInt& b) noexcept {
        std::array<TritWord64, NumWords> Sum{};
        std::array<TritWord64, NumWords> Carry{};

        for (size_t w = 0; w < NumWords; ++w) {
            AdderOutput ha = BTHA(a.words[w], b.words[w]);
            Sum[w] = ha.Sum;
            Carry[w] = ha.Carry;
        }

        while (true) {
            bool has_carry = false;
            for (size_t w = 0; w < NumWords; ++w) {
                if (Carry[w].P != 0 || Carry[w].N != 0) {
                    has_carry = true;
                    break;
                }
            }
            if (!has_carry) break;

            std::array<TritWord64, NumWords> ShiftedCarry{};
            for (size_t w = 0; w < NumWords; ++w) {
                uint64_t sh_P = Carry[w].P << 1;
                uint64_t sh_N = Carry[w].N << 1;
                if (w > 0) {
                    sh_P |= (Carry[w - 1].P >> 63) & 1ULL;
                    sh_N |= (Carry[w - 1].N >> 63) & 1ULL;
                }
                ShiftedCarry[w] = TritWord64(sh_P, sh_N);
            }

            for (size_t w = 0; w < NumWords; ++w) {
                AdderOutput ha = BTHA(Sum[w], ShiftedCarry[w]);
                Sum[w] = ha.Sum;
                Carry[w] = ha.Carry;
            }
        }

        TritBigInt res;
        res.words = Sum;
        return res;
    }

    // Multi-word Balanced Ternary Subtraction: a - b = a + (-b)
    static TritBigInt sub(const TritBigInt& a, const TritBigInt& b) noexcept {
        return add(a, b.negate());
    }

    // Multiplication by a single Trit
    constexpr TritBigInt mul_trit(Trit s) const noexcept {
        if (s == Trit::Pos) return *this;
        if (s == Trit::Neg) return negate();
        return TritBigInt{};
    }

    static constexpr TritBigInt mul_trit(const TritBigInt& a, Trit s) noexcept {
        return a.mul_trit(s);
    }

    // Multi-word Balanced Ternary Multiplication via shift-and-add
    static TritBigInt mul(const TritBigInt& a, const TritBigInt& b) noexcept {
        TritBigInt res{};
        for (size_t w = 0; w < NumWords; ++w) {
            uint64_t P = b.words[w].P;
            uint64_t N = b.words[w].N;
            uint64_t active = P | N;
            while (active != 0) {
                int bit = std::countr_zero(active);
                size_t k = w * 64 + bit;
                if (k < 64 * NumWords) {
                    if ((P >> bit) & 1ULL) {
                        res = add(res, a.shift_trits_left(k));
                    } else {
                        res = sub(res, a.shift_trits_left(k));
                    }
                }
                active &= active - 1;
            }
        }
        return res;
    }

    // Conversion from int64_t
    static TritBigInt from_int64(int64_t val) noexcept {
        TritBigInt res{};
        res.words[0] = Codec::from_int64(val);
        return res;
    }

    // Conversion to int64_t using SWAR on word 0
    int64_t to_int64(size_t num_trits = 39) const {
        return swar_to_int64(words[0], num_trits);
    }

    // Trit string representation (least-significant trit first, matching TritWord64::toString)
    std::string to_trit_string(size_t width = 64 * NumWords) const {
        size_t len = std::min(width, 64 * NumWords);
        std::string s;
        s.reserve(len);
        for (size_t i = 0; i < len; ++i) {
            s.push_back(trit_to_char(getTrit(i)));
        }
        return s;
    }

    std::string toString(size_t width = 64 * NumWords) const {
        return to_trit_string(width);
    }

    // Three-way Spaceship Comparison (<=>)
    constexpr std::strong_ordering operator<=>(const TritBigInt& other) const noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            size_t w = NumWords - 1 - i;
            uint64_t diff = (words[w].P ^ other.words[w].P) | (words[w].N ^ other.words[w].N);
            if (diff != 0) {
                int bit = 63 - std::countl_zero(diff);
                int ta = ((words[w].P >> bit) & 1ULL) ? 1 : (((words[w].N >> bit) & 1ULL) ? -1 : 0);
                int tb = ((other.words[w].P >> bit) & 1ULL) ? 1 : (((other.words[w].N >> bit) & 1ULL) ? -1 : 0);
                if (ta > tb) return std::strong_ordering::greater;
                if (ta < tb) return std::strong_ordering::less;
            }
        }
        return std::strong_ordering::equal;
    }

    constexpr bool operator==(const TritBigInt& other) const noexcept {
        for (size_t w = 0; w < NumWords; ++w) {
            if (words[w].P != other.words[w].P || words[w].N != other.words[w].N) return false;
        }
        return true;
    }

    constexpr bool operator!=(const TritBigInt& other) const noexcept { return !(*this == other); }
    constexpr bool operator<(const TritBigInt& other) const noexcept { return (*this <=> other) < 0; }
    constexpr bool operator>(const TritBigInt& other) const noexcept { return (*this <=> other) > 0; }
    constexpr bool operator<=(const TritBigInt& other) const noexcept { return (*this <=> other) <= 0; }
    constexpr bool operator>=(const TritBigInt& other) const noexcept { return (*this <=> other) >= 0; }

    // Arithmetic operators
    TritBigInt operator+(const TritBigInt& o) const noexcept { return add(*this, o); }
    TritBigInt operator-(const TritBigInt& o) const noexcept { return sub(*this, o); }
    TritBigInt operator*(const TritBigInt& o) const noexcept { return mul(*this, o); }

    TritBigInt& operator+=(const TritBigInt& o) noexcept { *this = add(*this, o); return *this; }
    TritBigInt& operator-=(const TritBigInt& o) noexcept { *this = sub(*this, o); return *this; }
    TritBigInt& operator*=(const TritBigInt& o) noexcept { *this = mul(*this, o); return *this; }

    TritBigInt operator<<(size_t k) const noexcept { return shift_trits_left(k); }
    TritBigInt operator>>(size_t k) const noexcept { return shift_trits_right(k); }
    TritBigInt& operator<<=(size_t k) noexcept { *this = shift_trits_left(k); return *this; }
    TritBigInt& operator>>=(size_t k) noexcept { *this = shift_trits_right(k); return *this; }
};

// Aliases for 256-trit and 512-trit BigInts
using TritBigInt256 = TritBigInt<4>; // 256 trits ~= 405.7 bits
using TritBigInt512 = TritBigInt<8>; // 512 trits ~= 811.5 bits

template <size_t NumWords>
inline std::ostream& operator<<(std::ostream& os, const TritBigInt<NumWords>& val) {
    os << val.to_trit_string();
    return os;
}

} // namespace bt

#endif // TERNARY_BIGINT_HPP
