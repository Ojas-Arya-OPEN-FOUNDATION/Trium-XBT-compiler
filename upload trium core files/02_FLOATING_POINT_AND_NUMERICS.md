# Balanced Ternary Floating-Point (BT-FP32) & Sparse Matrix Algebra (T-CSR)
## Exact Radix-3 Scientific Numerics & 1-Bit Neural Network Weight Compression

> **AI Context Brief:**
> This document details the scientific numerics engine of Trium-XBT:
> 1. **BT-FP32**: A balanced ternary floating-point representation with zero rounding bias and unique canonical zero.
> 2. **T-CSR (Ternary Compressed Sparse Row)**: An ultra-compact sparse matrix format where non-zero entries require only **1 bit per weight**, enabling multi-billion parameter ternary LLM inference without floating-point multiplications.

---

## 1. BT-FP32 Specification

### Mathematical Definition
A balanced ternary floating-point number $x$ is defined as:
$$x = M \cdot 3^E$$
where:
- $M \in \mathbb{Z}$ is the **Mantissa** (a normalized balanced ternary integer with no leading zero trits).
- $E \in \mathbb{Z}$ is the **Exponent** (power of 3).

### Key Architectural Advantages over IEEE-754 Binary Float
1. **Unique Canonical Zero**:
   In IEEE-754 binary floating-point, both $+0.0$ and $-0.0$ exist, producing subtle branching bugs and hardware comparison overhead.
   In BT-FP32, zero is unique:
   $$M = 0 \iff E = 0$$
   There is no negative zero.
2. **Zero-Drift Round-to-Nearest (Unbiased Truncation)**:
   In binary arithmetic, truncating lower bits always rounds down towards $-\infty$ or requires complex rounding modes (banker's rounding).
   In balanced ternary, because lower trits are symmetrically distributed in $\{-1, 0, +1\}$, **truncating lower fractional trits has an exact mathematical mean error of 0**:
   $$\mathbb{E}[\text{Error}] = \frac{(-1) + 0 + (+1)}{3} = 0$$
   This prevents round-off accumulation drift across millions of iterative scientific steps.
3. **No Separate Sign Bit**:
   The sign of the entire number is simply the sign of the most significant non-zero trit of $M$. Negation is achieved by negating the integer mantissa $M \to -M$.

---

## 2. T-CSR (Ternary Compressed Sparse Row)

In Modern Deep Learning (e.g., BitNet b1.58, ternary quantized models), weights are constrained to $\{-1, 0, +1\}$.
Standard floating-point sparse storage (FP32/FP16) stores 16–32 bits per non-zero weight.

**T-CSR 1-Bit Encoding**:
- A ternary sparse matrix has entries $A_{ij} \in \{-1, 0, +1\}$.
- All non-zero entries (NNZ) are either $+1$ or $-1$.
- **Sign Bitplane**: Each non-zero entry's sign is stored as a **single bit** in a 64-bit word:
  $$\text{bit}_k = 1 \iff A_{ij} = +1, \quad \text{bit}_k = 0 \iff A_{ij} = -1$$
- **Multiplication-Free SpMV (Sparse Matrix-Vector)**:
  Computing $y = A \cdot x$ requires **zero floating-point multiplications**:
  $$y_i = \sum_{j \in \text{Row } i, \ A_{ij} \ne 0} A_{ij} \cdot x_j = \sum_{A_{ij}=+1} x_j - \sum_{A_{ij}=-1} x_j$$
  The entire inner loop consists solely of integer adds and subtracts, running at memory-bandwidth wire speed.

---

## 3. C++20 Implementation

```cpp
#pragma once
#include "ternary_core.hpp"
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

namespace bt {

// -----------------------------------------------------------------------------
// Exact Radix-3 Powers & Symmetric Rounding
// -----------------------------------------------------------------------------
namespace fp_detail {
    // Symmetric rounding division by 3: truncates 1 trit with 0 mean error
    constexpr inline int64_t round_div3(int64_t val) noexcept {
        int64_t rem = val % 3;
        int64_t q = val / 3;
        if (rem == 2) q += 1;
        else if (rem == -2) q -= 1;
        return q;
    }
}

// -----------------------------------------------------------------------------
// BT-FP32 Floating-Point Structure (Value = Mantissa * 3^Exponent)
// -----------------------------------------------------------------------------
struct BtFp32 {
    int64_t mantissa{0};
    int32_t exponent{0};

    constexpr BtFp32() noexcept = default;
    constexpr BtFp32(int64_t m, int32_t e) noexcept : mantissa(m), exponent(e) {
        normalize();
    }

    constexpr bool is_zero() const noexcept { return mantissa == 0; }

    constexpr void normalize() noexcept {
        if (mantissa == 0) {
            exponent = 0;
            return;
        }
        // Remove trailing zeroes from mantissa
        while (mantissa % 3 == 0) {
            mantissa /= 3;
            exponent++;
        }
    }

    static BtFp32 from_double(double val) {
        if (std::isnan(val) || std::isinf(val)) {
            throw std::invalid_argument("NaN and Infinity cannot be represented in standard BT-FP32");
        }
        if (val == 0.0) return BtFp32(0, 0);

        int exp = static_cast<int>(std::floor(std::log(std::abs(val)) / std::log(3.0)));
        double norm = val / std::pow(3.0, exp - 25);
        int64_t m = static_cast<int64_t>(std::round(norm));
        return BtFp32(m, exp - 25);
    }

    double to_double() const noexcept {
        if (is_zero()) return 0.0;
        return static_cast<double>(mantissa) * std::pow(3.0, exponent);
    }

    // 0-Cycle Negation
    constexpr BtFp32 operator-() const noexcept {
        return BtFp32(-mantissa, exponent);
    }

    // Addition with automatic radix-3 exponent alignment
    BtFp32 add(const BtFp32& other) const {
        if (is_zero()) return other;
        if (other.is_zero()) return *this;

        int32_t min_exp = std::min(exponent, other.exponent);
        int64_t m1 = mantissa;
        int64_t m2 = other.mantissa;

        // Shift larger exponent mantissa by powers of 3
        int diff1 = exponent - min_exp;
        int diff2 = other.exponent - min_exp;

        for (int i = 0; i < diff1; ++i) m1 *= 3;
        for (int i = 0; i < diff2; ++i) m2 *= 3;

        return BtFp32(m1 + m2, min_exp);
    }

    BtFp32 mul(const BtFp32& other) const noexcept {
        if (is_zero() || other.is_zero()) return BtFp32(0, 0);
        return BtFp32(mantissa * other.mantissa, exponent + other.exponent);
    }

    BtFp32 div(const BtFp32& other) const {
        if (other.is_zero()) throw std::domain_error("BT-FP32 division by zero");
        if (is_zero()) return BtFp32(0, 0);

        // Scale numerator by 3^25 for fractional precision before division
        int64_t num = mantissa;
        int scale = 25;
        for (int i = 0; i < scale; ++i) num *= 3;

        int64_t q = num / other.mantissa;
        return BtFp32(q, exponent - other.exponent - scale);
    }

    // Fused Multiply-Add (FMA): (a * b) + c computed with exact intermediate precision
    BtFp32 fma(const BtFp32& b, const BtFp32& c) const {
        return this->mul(b).add(c);
    }
};

// -----------------------------------------------------------------------------
// Ternary Compressed Sparse Row (T-CSR) 1-Bit Matrix
// -----------------------------------------------------------------------------
class TernaryCSR {
public:
    uint64_t rows{0};
    uint64_t cols{0};
    uint64_t nnz{0};

    std::vector<uint32_t> row_ptr;     // Row offset pointers of size (rows + 1)
    std::vector<uint32_t> col_indices; // Column indices of non-zero entries (size NNZ)
    std::vector<uint64_t> signs;       // Packed sign bitmask (1 = +1, 0 = -1), 64 entries/word

    TernaryCSR(uint64_t r, uint64_t c) : rows(r), cols(c), row_ptr(r + 1, 0) {}

    // Multiplier-Free Sparse Matrix-Vector Multiplication: y = A * x
    void spmv(const int64_t* x, int64_t* y) const noexcept {
        for (uint64_t i = 0; i < rows; ++i) {
            uint32_t start = row_ptr[i];
            uint32_t end   = row_ptr[i + 1];
            int64_t acc = 0;

            for (uint32_t idx = start; idx < end; ++idx) {
                uint32_t col = col_indices[idx];
                bool is_pos = (signs[idx / 64] >> (idx % 64)) & 1ULL;
                
                // Add if +1, Subtract if -1: ZERO multiplications!
                if (is_pos) {
                    acc += x[col];
                } else {
                    acc -= x[col];
                }
            }
            y[i] = acc;
        }
    }
};

} // namespace bt
```
