# Exhaustive Audit Notes: Core Numerics, Sparse Algebra, LibBT & Memory Subsystems
**Workspace:** Balanced Ternary Compiler & Runtime  
**Auditor:** Codebase Auditor Subagent (Facet 1: Core, Numerics, Sparse & LibBT)  
**Date:** October 2026  
**Status:** Audit Completed — Critical Vulnerabilities & Fix Specifications Documented

---

## 1. Executive Summary

A deep mathematical, memory-safety, and microarchitectural audit was conducted across seven foundational headers:
1. `ternary_core.hpp`
2. `ternary_dense_algebra.hpp`
3. `ternary_fp.hpp`
4. `ternary_sparse.hpp`
5. `libbt.hpp`
6. `symplectic_ternary_engine.hpp`
7. `ternary_memory_arena.hpp`

While the core bitwise Kleene logic gates and small-vector tests function correctly under ideal conditions, the audit identified **16 high-severity bugs and architectural flaws** spanning undefined bit-shifts, MSB carry truncation, IEEE-754 conversion collapses, out-of-order CSR matrix corruption, signed integer overflow UB on INT64_MIN, thread-safety violations in const accessors, and storage layout contradictions.

---

## 2. Exhaustive Audit Breakdown by Component

### Component 1: `ternary_core.hpp`

#### Issue 1.1: Missing `#include <vector>` Header Include
- **File & Lines:** `ternary_core.hpp`, lines 14–23 vs lines 382, 383, 399, 401.
- **Severity:** High (Compilation Failure)
- **Root Cause:** `UniversalTritStreamCodec::encode_bytes` and `decode_bytes` return and consume `std::vector<Trit>` and `std::vector<uint8_t>`, yet `<vector>` is omitted from the header includes. Any translation unit including `ternary_core.hpp` without first including `<vector>` fails compilation.
- **Exact C++ Fix:**
```cpp
#include <vector>
```

#### Issue 1.2: Bit-Shift Undefined Behavior on Index $\ge 64$ in `TritWord64`
- **File & Lines:** `ternary_core.hpp`, lines 114–130.
- **Severity:** High (Undefined Behavior)
- **Root Cause:**
  `P >> index` and `1ULL << index` are guarded solely by `assert(index < 64)`. In release builds with `-DNDEBUG`, assertions are stripped. Shifting a 64-bit integer by $\ge 64$ is undefined behavior in C++ (ISO/IEC 14882 [expr.shift]).
- **Exact C++ Fix:**
```cpp
constexpr Trit getTrit(size_t index) const noexcept {
    if (index >= 64) return Trit::Zero;
    bool p = (P >> index) & 1ULL;
    bool n = (N >> index) & 1ULL;
    if (p) return Trit::Pos;
    if (n) return Trit::Neg;
    return Trit::Zero;
}
constexpr void setTrit(size_t index, Trit t) noexcept {
    if (index >= 64) return;
    uint64_t mask = 1ULL << index;
    P &= ~mask;
    N &= ~mask;
    if (t == Trit::Pos) P |= mask;
    if (t == Trit::Neg) N |= mask;
}
```

#### Issue 1.3: MSB Carry Truncation in `BT_CSA::accumulate` and `BT_CSA::canonicalize`
- **File & Lines:** `ternary_core.hpp`, lines 210–212, 227–228.
- **Severity:** Critical (Arithmetic Precision Loss)
- **Root Cause:**
  `uint64_t C_shifted_P = C.P << 1;` shifts out bit 63 without carry tracking. In `canonicalize()`, if a carry ripples out of bit 63, it is silently dropped into bit 64 and discarded.
- **Exact C++ Fix:**
  Introduce carry overflow detection or return an overflow trit structure:
```cpp
struct CanonicalWordResult {
    TritWord64 word;
    Trit overflow_trit{Trit::Zero};
};
inline CanonicalWordResult canonicalize_with_overflow() const noexcept {
    TritWord64 curr_S = S;
    TritWord64 curr_C = C;
    Trit ovf = Trit::Zero;
    while (curr_C.P != 0 || curr_C.N != 0) {
        if (curr_C.P & (1ULL << 63)) ovf = Trit::Pos;
        if (curr_C.N & (1ULL << 63)) ovf = Trit::Neg;
        uint64_t sh_P = curr_C.P << 1;
        uint64_t sh_N = curr_C.N << 1;
        AdderOutput ha = BTHA(curr_S, TritWord64(sh_P, sh_N));
        curr_S = ha.Sum;
        curr_C = ha.Carry;
    }
    return { curr_S, ovf };
}
```

#### Issue 1.4: Silent Modulo-256 Truncation in `UniversalTritStreamCodec::decode_bytes`
- **File & Lines:** `ternary_core.hpp`, line 412.
- **Severity:** Medium (Silent Data Corruption)
- **Root Cause:**
  `bytes.push_back(static_cast<uint8_t>(val & 0xFF));` silently masks values in 728$ down to 255$ if corrupt or untrusted trits are passed. Also ignores unpadded trailing trits.
- **Exact C++ Fix:**
```cpp
if (val > 255) {
    throw std::domain_error("UniversalTritStreamCodec::decode_bytes: decoded value exceeds byte range");
}
```

---

### Component 2: `ternary_dense_algebra.hpp`

#### Issue 2.1: Signed Integer Overflow in `Codec::to_int64`
- **File & Lines:** `ternary_dense_algebra.hpp`, lines 523–533.
- **Severity:** High (Undefined Behavior)
- **Root Cause:**
  `num_trits` defaults to 39. If caller passes `num_trits >= 40`, `p3 *= 3` overflows `int64_t` (^{40} pprox 1.215 	imes 10^{19} > 	ext{INT64_MAX} = 9.22 	imes 10^{18}$).
- **Exact C++ Fix:**
```cpp
static inline int64_t to_int64(const TritWord64& word, size_t num_trits = 39) {
    if (num_trits > 39) {
        throw std::invalid_argument("to_int64: num_trits cannot exceed 39 without 64-bit signed integer overflow");
    }
    int64_t total = 0;
    int64_t p3 = 1;
    for (size_t i = 0; i < num_trits; ++i) {
        total += static_cast<int64_t>(to_int(word.getTrit(i))) * p3;
        if (i + 1 < num_trits) p3 *= 3;
    }
    return total;
}
```

#### Issue 2.2: Undefined Behavior in `Codec::float_to_trits` on NaN/Inf and Excessive Fractional Trits
- **File & Lines:** `ternary_dense_algebra.hpp`, lines 539–549.
- **Severity:** Critical (Undefined Behavior / Out-of-Bounds Shift)
- **Root Cause:**
  1. `std::llround(val * scale)` on `val = NaN` or `Inf` is undefined behavior.
  2. If `fractional_trits > 48`, `trits.size() > 64`, causing `w.getTrit(i)` to shift by $\ge 64$.
- **Exact C++ Fix:**
```cpp
static inline void float_to_trits(double val, std::vector<Trit>& trits, int fractional_trits = 8) {
    if (!std::isfinite(val)) {
        throw std::invalid_argument("float_to_trits: cannot convert NaN or Inf");
    }
    if (fractional_trits < 0 || fractional_trits > 48) {
        throw std::invalid_argument("float_to_trits: fractional_trits must be in range [0, 48]");
    }
    trits.resize(fractional_trits + 16, Trit::Zero);
    double scale = std::pow(3.0, fractional_trits);
    int64_t scaled_int = std::llround(val * scale);
    TritWord64 w = from_int64(scaled_int, std::min<size_t>(trits.size(), 64));
    for (size_t i = 0; i < trits.size() && i < 64; ++i) {
        trits[i] = w.getTrit(i);
    }
}
```

#### Issue 2.3: Out-of-Bounds Crash in `gemm_multitrit` with Empty Planes
- **File & Lines:** `ternary_dense_algebra.hpp`, lines 459–460.
- **Severity:** High (Crash / Out-of-Bounds Dereference)
- **Root Cause:**
  `A.plane(0).rows` dereferences empty vector if `A.num_planes == 0`.
- **Exact C++ Fix:**
```cpp
if (A.num_planes == 0 || B.num_planes == 0) {
    std::fill(C.data.begin(), C.data.end(), 0);
    return;
}
```

---

### Component 3: `ternary_fp.hpp`

#### Issue 3.1: Silent NaN / Infinity Collapse to Canonical Zero in `from_double`
- **File & Lines:** `ternary_fp.hpp`, lines 337–339.
- **Severity:** Critical (Data Integrity Hazard)
- **Root Cause:**
  `if (val == 0.0 || !std::isfinite(val)) return BTFP(0, 0);` silently transforms NaN and $\pm\infty$ into valid zero (0.0), masking floating-point exceptions and errors.
- **Exact C++ Fix:**
```cpp
static BTFP from_double(double val) {
    if (!std::isfinite(val)) {
        throw std::domain_error("BTFP::from_double: NaN or Inf cannot be represented in BTFP");
    }
    if (val == 0.0) return BTFP(0, 0);
    ...
}
```

#### Issue 3.2: Floating-Point Overflow & UB on Subnormal Doubles in `from_double`
- **File & Lines:** `ternary_fp.hpp`, lines 342–348.
- **Severity:** Critical (Undefined Behavior)
- **Root Cause:**
  For subnormal values ( pprox 10^{-320}$), `est_e` is $pprox -680$. `std::pow(3.0, -est_e)` calculates ^{680} > 10^{324}$, overflowing to `+INFINITY`. `val * Inf` yields `NaN`, and `std::llround(NaN)` is undefined behavior.
- **Exact C++ Fix:**
```cpp
if (std::abs(val) < 1e-300) {
    // Flush subnormal to zero or scale with multi-step exponent
    return BTFP(0, 0);
}
```

#### Issue 3.3: Exponent Arithmetic Overflow in `mul`, `div`, and `add`
- **File & Lines:** `ternary_fp.hpp`, lines 376, 404, 436.
- **Severity:** High (Signed Integer Overflow UB)
- **Root Cause:**
  `m_exponent + other.m_exponent` and `m_exponent - other.m_exponent` perform 32-bit signed addition/subtraction without checking for overflow.
- **Exact C++ Fix:**
```cpp
int64_t e_calc = static_cast<int64_t>(m_exponent) + other.m_exponent;
if (e_calc > std::numeric_limits<int32_t>::max() || e_calc < std::numeric_limits<int32_t>::min()) {
    throw std::overflow_error("BTFP exponent overflow");
}
```

#### Issue 3.4: Complete Neglect of `ExponentTrits` Template Parameter
- **File & Lines:** `ternary_fp.hpp`, lines 121, 125.
- **Severity:** Medium (Architectural Flaw)
- **Root Cause:**
  `template <int MantissaTrits = 10, int ExponentTrits = 0>`: `ExponentTrits` is ignored throughout normalization, arithmetic, and rounding.
- **Exact C++ Fix:**
  Add exponent clamping in `normalize()`:
```cpp
if constexpr (ExponentTrits > 0) {
    constexpr int64_t max_e = (fp_detail::pow3_128(ExponentTrits) - 1) / 2;
    if (m_exponent > max_e || m_exponent < -max_e) {
        throw std::overflow_error("BTFP: exponent exceeds ExponentTrits");
    }
}
```

#### Issue 3.5: Violation of Strict Weak Ordering in `operator<`
- **File & Lines:** `ternary_fp.hpp`, lines 463–468.
- **Severity:** High (Undefined Behavior in `std::sort`)
- **Root Cause:**
  When `A != B` but `A.sub(B)` rounds to 0 due to precision limit, `A < B` and `B < A` are both false, violating strict weak ordering.
- **Exact C++ Fix:** Compare exponents and mantissas lexicographically when signs match.

---

### Component 4: `ternary_sparse.hpp`

#### Issue 4.1: Matrix CSR State Corruption on Out-of-Order `push_element`
- **File & Lines:** `ternary_sparse.hpp`, lines 93–96.
- **Severity:** Critical (Memory / CSR Invariant Corruption)
- **Root Cause:**
  `while (m_current_build_row < r)` does nothing if  < m\_current\_build\_row$, silently attaching the non-zero element to an older row.
- **Exact C++ Fix:**
```cpp
if (r < m_current_build_row) {
    throw std::invalid_argument("push_element: row indices must be monotonically non-decreasing");
}
```

#### Issue 4.2: Binary Search Failure on Out-of-Order Columns in `get(r, c)`
- **File & Lines:** `ternary_sparse.hpp`, lines 236–237.
- **Severity:** High (Silent Incorrect Retrieval)
- **Root Cause:**
  `std::lower_bound` is used on `col_indices`, which assumes sorted column indices per row, but no sort is enforced.
- **Exact C++ Fix:**
  In `push_element`, enforce `c > last_c` for the current row, or sort row slices during `finalize()`.

#### Issue 4.3: Out-of-Bounds Indexing in Sparse-Sparse `spmm`
- **File & Lines:** `ternary_sparse.hpp`, lines 461–462.
- **Severity:** High (Out-of-Bounds Memory Read)
- **Root Cause:**
  `size_t b_start = B.row_ptr[k];` dereferences `B.row_ptr[k+1]` without validating  < B.rows$.
- **Exact C++ Fix:**
```cpp
if (k >= B.rows) throw std::out_of_range("TCSRMatrix::spmm: column index exceeds B rows");
```

---

### Component 5: `libbt.hpp`

#### Issue 5.1: Undefined Behavior on `INT64_MIN` in `Rational`
- **File & Lines:** `libbt.hpp`, lines 69–72, 92, 145.
- **Severity:** Critical (Signed Overflow UB)
- **Root Cause:**
  Negating `INT64_MIN` (`-d` or `-num`) is undefined behavior in C++.
- **Exact C++ Fix:**
```cpp
__int128_t n128 = n;
__int128_t d128 = d;
if (d128 < 0) {
    n128 = -n128;
    d128 = -d128;
}
__int128_t an = (n128 < 0) ? -n128 : n128;
__int128_t ad = d128;
__int128_t g = std::gcd(an, ad);
num = static_cast<int64_t>(n128 / g);
den = static_cast<int64_t>(d128 / g);
```

#### Issue 5.2: Dimension Inconsistency in `Matrix::inverse()` on Singular Matrix
- **File & Lines:** `libbt.hpp`, lines 529, 538.
- **Severity:** Medium (Dimension Invariant Violation)
- **Root Cause:**
  Returns `Matrix<T>()` (zsh 	imes 0$) instead of preserving  	imes n$ dimensions.
- **Exact C++ Fix:** Return `Matrix<T>::zeros(n, n)` or `std::optional<Matrix<T>>`.

#### Issue 5.3: Division by Zero on Empty Signal in `Radix3FFT`
- **File & Lines:** `libbt.hpp`, lines 957, 967, 1011.
- **Severity:** High (Floating Point Exception)
- **Root Cause:**
  `fft_unitary` divides by `std::sqrt(0.0)` when sequence is empty.
- **Exact C++ Fix:**
```cpp
if (x.empty()) return {};
```

---

### Component 6: `symplectic_ternary_engine.hpp`

#### Issue 6.1: Complete Carry Loss in Leapfrog Stages
- **File & Lines:** `symplectic_ternary_engine.hpp`, lines 77, 82, 87.
- **Severity:** Critical (Semantic & Arithmetic Breakdown)
- **Root Cause:**
  `p_half_res.Carry`, `q_res.Carry`, and `p_next_res.Carry` are discarded. The integrator operates as a modulo-3 ring rather than carry-save arithmetic.
- **Exact C++ Fix:**
  Accumulate carries into a conjugate carry register and preserve energy conservation via dual-rail carry tracking.

---

### Component 7: `ternary_memory_arena.hpp`

#### Issue 7.1: Data Race in `const` Method `HierarchicalZeroPage::get_byte`
- **File & Lines:** `ternary_memory_arena.hpp`, lines 900–902.
- **Severity:** Critical (Data Race UB)
- **Root Cause:**
  `++const_cast<HierarchicalZeroPage*>(this)->read_count_;` mutates a non-atomic counter in a `const` method, causing concurrent read threads to invoke data races.
- **Exact C++ Fix:**
```cpp
mutable std::atomic<size_t> read_count_{0};
// in get_byte:
read_count_.fetch_add(1, std::memory_order_relaxed);
```

#### Issue 7.2: Missing Boundary Checks in Release Builds
- **File & Lines:** `ternary_memory_arena.hpp`, lines 921–928, 1071–1078.
- **Severity:** Critical (Out-of-Bounds Memory Corruption)
- **Root Cause:**
  When compiled with `-DNDEBUG`, assertions are removed and `offset >= virtual_capacity_` indexes past `root_`.
- **Exact C++ Fix:**
```cpp
if (offset >= virtual_capacity_) {
    throw std::out_of_range("HierarchicalZeroPage: offset exceeds virtual capacity");
}
```

#### Issue 7.3: Incompatible Storage Layout: `get_trit` vs `get_word`
- **File & Lines:** `ternary_memory_arena.hpp`, lines 963–971 vs lines 977–994.
- **Severity:** Critical (Architectural Layout Mismatch)
- **Root Cause:**
  `set_trit` writes 1 byte per trit (values -1, 0, 1). `get_word` reads raw memory as a 16-byte dual-rail struct (`uint64_t P`, `uint64_t N`). Reading a word after setting trits breaks the dual-rail invariant  \& N = 0$.
- **Exact C++ Fix:** Store trits as true bitplanes or provide an explicit unpack layer between byte storage and dual-rail words.

