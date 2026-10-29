# Advanced Algebra, Arbitrary-Precision BigInt & Dense Transformations
## Multi-Word Karatsuba/Toom-3 Arithmetic, NTT, and Householder Linear Algebra

> **AI Context Brief:**
> This document details the high-precision arbitrary-width arithmetic and dense linear algebra systems of Trium-XBT:
> 1. **TritBigInt**: Register-width arbitrary-precision integer arithmetic supporting hundreds or thousands of trits (equivalent to thousands of binary bits).
> 2. **Toom-Cook 3-Way & NTT Multiplication**: Sub-quadratic $\mathcal{O}(n^{\log_3 5}) \approx \mathcal{O}(n^{1.465})$ multi-word multiplication.
> 3. **Householder QR Decomposition**: Orthogonal matrix factorizations on ternary-quantized and balanced coordinates.

---

## 1. Multi-Word Planar BigInt Representation

For an arbitrary word width $W$, a BigInt is an array of $W$ 64-trit dual-rail words:
$$\text{TritBigInt}\langle W \rangle = \Big[ (P_0, N_0), (P_1, N_1), \dots, (P_{W-1}, N_{W-1}) \Big]$$

- **Bitplane Invariant**: $(P_i \ \& \ N_i) == 0$ for all $0 \le i < W$.
- **Precision**: Provides $64 \cdot W$ balanced trits, which corresponds to:
  $$\text{Equivalent Binary Precision} = 64 \cdot W \cdot \log_2(3) \approx 101.4 \cdot W \text{ bits}$$
  *(A 4-word `TritBigInt<4>` holds 256 trits $\approx 405$ binary bits of precision).*

### Three-Way Comparison (`<=>`)
In balanced ternary, comparing two numbers $A$ and $B$ is extraordinarily elegant:
1. Scan words from most significant word $W-1$ down to 0.
2. The comparison $(A \Leftrightarrow B)$ is determined entirely by the **highest-order trit where $A$ and $B$ differ**:
   $$\text{MSB}(A \oplus B) \implies \text{if } A_i > B_i \text{ return } +1; \quad \text{if } A_i < B_i \text{ return } -1$$
3. No sign-bit checks or special cases for negative numbers!

---

## 2. Toom-Cook 3-Way Multiplication

Standard BigInt multiplication runs in $\mathcal{O}(n^2)$ time.
Because balanced ternary has radix 3, **Toom-Cook 3-Way (Toom-3)** is its most natural polynomial evaluation algorithm:
- Both numbers $A$ and $B$ are split into 3 equal slices:
  $$A(x) = A_2 x^2 + A_1 x + A_0, \quad B(x) = B_2 x^2 + B_1 x + B_0 \quad (\text{where } x = 3^m)$$
- The product $C(x) = A(x) \cdot B(x)$ has degree 4 and requires 5 evaluation points:
  $$x \in \{0, 1, -1, -2, \infty\}$$
- In balanced ternary, the evaluation points $\{0, 1, -1\}$ require **zero scalar multiplications**—only additions and subtractions.
- Time Complexity:
  $$\mathcal{O}(n^{\log_3 5}) \approx \mathcal{O}(n^{1.46497})$$
  *(Significantly faster than Karatsuba's $\mathcal{O}(n^{\log_2 3}) \approx \mathcal{O}(n^{1.585})$).*

---

## 3. C++20 Implementation

```cpp
#pragma once
#include "ternary_core.hpp"
#include <array>
#include <vector>
#include <compare>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace bt {

// -----------------------------------------------------------------------------
// Arbitrary-Precision Multi-Word Balanced Ternary Integer (TritBigInt)
// -----------------------------------------------------------------------------
template <size_t NumWords = 4>
class TritBigInt {
public:
    std::array<TritWord64, NumWords> words{};

    constexpr TritBigInt() noexcept = default;
    constexpr explicit TritBigInt(const std::array<TritWord64, NumWords>& w) noexcept : words(w) {}

    static constexpr size_t total_trits() noexcept { return 64 * NumWords; }

    constexpr bool isValid() const noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            if (!words[i].isValid()) return false;
        }
        return true;
    }

    constexpr bool is_zero() const noexcept {
        for (size_t i = 0; i < NumWords; ++i) {
            if (words[i].P != 0 || words[i].N != 0) return false;
        }
        return true;
    }

    // 0-Cycle Negation: Swaps P and N rails across all words
    constexpr TritBigInt operator-() const noexcept {
        TritBigInt res;
        for (size_t i = 0; i < NumWords; ++i) {
            res.words[i] = -words[i];
        }
        return res;
    }

    // Multi-Word Carry-Lookahead Addition
    TritBigInt add(const TritBigInt& other, int8_t* carry_out = nullptr) const noexcept {
        TritBigInt result;
        int8_t carry = 0;

        for (size_t w = 0; w < NumWords; ++w) {
            int8_t next_carry = 0;
            result.words[w] = kogge_stone_add64(words[w], other.words[w], carry, &next_carry);
            carry = next_carry;
        }

        if (carry_out != nullptr) *carry_out = carry;
        return result;
    }

    TritBigInt sub(const TritBigInt& other, int8_t* borrow_out = nullptr) const noexcept {
        return this->add(-other, borrow_out);
    }

    // Spaceship operator: Natural ternary lexicographical order from MSB to LSB
    std::strong_ordering operator<=>(const TritBigInt& other) const noexcept {
        for (size_t w = NumWords; w > 0; --w) {
            size_t idx = w - 1;
            if (words[idx] == other.words[idx]) continue;

            // Find most significant differing trit
            for (int bit = 63; bit >= 0; --bit) {
                Trit a_trit = words[idx].getTrit(bit);
                Trit b_trit = other.words[idx].getTrit(bit);
                if (a_trit != b_trit) {
                    return to_int(a_trit) <=> to_int(b_trit);
                }
            }
        }
        return std::strong_ordering::equal;
    }

    bool operator==(const TritBigInt& other) const noexcept = default;
};

// -----------------------------------------------------------------------------
// Householder QR Decomposition in Ternary Dense Coordinates
// -----------------------------------------------------------------------------
class DenseTernaryMatrix {
public:
    size_t rows;
    size_t cols;
    std::vector<double> data; // Continuous column-major or row-major elements

    DenseTernaryMatrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& operator()(size_t r, size_t c) { return data[r * cols + c]; }
    double operator()(size_t r, size_t c) const { return data[r * cols + c]; }

    // Computes Q and R such that A = Q * R, where Q is orthogonal and R is upper-triangular
    void householder_qr(DenseTernaryMatrix& Q, DenseTernaryMatrix& R) const {
        size_t m = rows;
        size_t n = cols;
        R = *this;
        Q = DenseTernaryMatrix(m, m);
        
        // Initialize Q to Identity
        for (size_t i = 0; i < m; ++i) Q(i, i) = 1.0;

        for (size_t k = 0; k < std::min(m, n); ++k) {
            // Compute Householder vector for column k below diagonal
            double norm_x = 0.0;
            for (size_t i = k; i < m; ++i) norm_x += R(i, k) * R(i, k);
            norm_x = std::sqrt(norm_x);
            if (norm_x < 1e-12) continue;

            double alpha = (R(k, k) > 0) ? -norm_x : norm_x;
            double u1 = R(k, k) - alpha;
            double norm_u = std::sqrt(u1 * u1 + (norm_x * norm_x - R(k, k) * R(k, k)));

            std::vector<double> v(m - k, 0.0);
            v[0] = u1 / norm_u;
            for (size_t i = k + 1; i < m; ++i) v[i - k] = R(i, k) / norm_u;

            // Apply reflector to R: R = R - 2 * v * (v^T * R)
            for (size_t j = k; j < n; ++j) {
                double dot = 0.0;
                for (size_t i = k; i < m; ++i) dot += v[i - k] * R(i, j);
                for (size_t i = k; i < m; ++i) R(i, j) -= 2.0 * v[i - k] * dot;
            }

            // Accumulate into Q
            for (size_t i = 0; i < m; ++i) {
                double dot = 0.0;
                for (size_t j = k; j < m; ++j) dot += Q(i, j) * v[j - k];
                for (size_t j = k; j < m; ++j) Q(i, j) -= 2.0 * dot * v[j - k];
            }
        }
    }
};

} // namespace bt
```
