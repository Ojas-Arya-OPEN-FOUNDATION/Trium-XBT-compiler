# Exhaustive Mathematical Audit & Redesign: Floating-Point, Exact Rational, Dense/Sparse Linear Algebra & Radix-3 FFT

**Target Headers**:
- `ternary_fp.hpp` (`bt::BTFP<MantissaTrits, ExponentTrits>`)
- `libbt.hpp` (`bt::Rational`, `bt::Matrix<T>`, `bt::Radix3FFT`)
- `ternary_dense_algebra.hpp` (`bt::MultiTritMatrix`, `bt::gemm_multitrit`, `bt::TernaryMatrix`, `bt::DenseDualRailPlanes`)
- `ternary_sparse.hpp` (`bt::TCSRMatrix`)

---

## 1. `ternary_fp.hpp` — Balanced Ternary Floating-Point (`BTFP<MantissaTrits, ExponentTrits>`)

### 1.1 Current State & Architectural Gaps
`BTFP<W, E>` represents a floating-point value as $V = M \cdot 3^e$, where $M \in \mathbb{Z}$ is normalized such that either $M = 0, e = 0$ (unique canonical zero) or:
$$\frac{3^{W-1} + 1}{2} \le |M| \le \frac{3^W - 1}{2}$$
Currently, `BTFP` provides `+`, `-`, `*`, `/`, unary `operator-()`, `truncate_fractional()`, `to_double()`, and `from_double()`. However, several foundational mathematical primitives and exact interop bridges are missing, and a critical rounding bug exists in `mul()` when `ExponentTrits > 0`.

### 1.2 Critical Bug Found in `BTFP::mul` Exponent Clamping
In `BTFP::mul()` (lines 634–641 of `ternary_fp.hpp`), when `ExponentTrits > 0`:
```cpp
if constexpr (ExponentTrits > 0) {
    constexpr int64_t max_e = (fp_detail::pow3_128(ExponentTrits) - 1) / 2;
    if (e_prod > max_e) {
        e_prod = static_cast<int32_t>(max_e);
    } else if (e_prod < -max_e) {
        e_prod = static_cast<int32_t>(-max_e);
    }
}
```
Unlike `normalize()`, which pushes excess exponent delta into the mantissa or symmetric-rounds via `round_div_pow3_fast64`, `mul()` **silently clamps `e_prod` without scaling `q`**, corrupting the value by factors of $3^{\Delta e}$ on underflow/overflow when `ExponentTrits > 0`.
- **Fix**: Delegate the final construction in `mul()` to `BTFP((prod < 0) ? -q : q, e_prod)` or invoke `normalize()` when `ExponentTrits > 0`.

### 1.3 Mathematical Specifications for Missing `BTFP` Functions

#### A. O(1) Sign & Magnitude Primitives: `neg()` and `abs()`
Because balanced ternary has no separate sign bit and its mantissa range $[-\frac{3^W-1}{2}, +\frac{3^W-1}{2}]$ is completely symmetric (no two's complement `INT_MIN` asymmetry!), both negation and absolute value are exact $O(1)$ operations that preserve normalization without calling `normalize()`:
```cpp
constexpr BTFP neg() const noexcept {
    return -(*this);
}

constexpr BTFP abs() const noexcept {
    BTFP res;
    res.m_mantissa = (m_mantissa < 0) ? -m_mantissa : m_mantissa;
    res.m_exponent = m_exponent;
    return res;
}
```

#### B. Exact Single-Rounding Fused Multiply-Add: `fma(a, b, c)`
Standard evaluation of `a * b + c` suffers from double rounding: `a * b` rounds a $2W$-trit intermediate product to $W$ trits, and `+ c` rounds again. Furthermore, if $a \cdot b \approx -c$, catastrophic cancellation destroys up to $W$ trits of accuracy.
- **Mathematical Design**:
  Given $a = M_a 3^{e_a}$, $b = M_b 3^{e_b}$, $c = M_c 3^{e_c}$:
  1. If $a = 0$ or $b = 0$, return $c$.
  2. Compute the exact product in 128-bit arithmetic:
     $$P = \text{static\_cast<\text{\_\_int128\_t}>}(M_a) \cdot M_b, \quad e_p = e_a + e_b$$
     Since $|M_a|, |M_b| \le \frac{3^{40}-1}{2} < 2^{63}$, $|P| < \frac{3^{80}}{4} \approx 3.69 \times 10^{37} < 2^{125}$, which fits in signed `__int128_t` with zero overflow for all valid `MantissaTrits <= 40`!
  3. If $c = 0$, return `normalize_128(P, e_p)` (a single rounding).
  4. Let $\Delta = e_p - e_c$.
     - **Case 1 ($\Delta == 0$)**: Exact sum $S = P + M_c$ at exponent $e_p$, followed by a single `normalize_128(S, e_p)`.
     - **Case 2 ($\Delta > 0$)**: $P$ has a larger exponent than $c$.
       - The product $P$ already has $\sim 2W$ trits of precision. To avoid overflowing `__int128_t` when shifting $P$ left by $3^\Delta$, keep up to $2W + 4$ trits in the accumulator:
       - If $\Delta > 2W + 4$ (or when $P \cdot 3^\Delta$ would exceed $2^{126}$), instead of discarding $c$ prematurely when $\Delta \le W + 4$, shift $P$ left by $k_1 = \min(\Delta, 59 - \text{tritlen}(P))$ and symmetric-round $M_c$ right by $\Delta - k_1$ using `round_div_pow3_128`, or for $W \le 20$: since $2W + 4 \le 44$, if $\Delta \le 38$, $P \cdot 3^\Delta$ fits directly in `__int128_t`! Specifically, for $W \le 20$, $|P| < 3^{40}/4$, so for $\Delta \le 38$, $|P| \cdot 3^\Delta < 3^{78}/4 < 2^{122}$, allowing **100% exact alignment** $S = P \cdot 3^\Delta + M_c$ at exponent $e_c$, followed by a single `normalize_128(S, e_c)`!
       - If $\Delta > 38$ (for $W \le 20$), since $P$ has at least $2W - 1$ trits and $M_c$ has $W$ trits, $M_c$ is shifted more than $38 - W \ge 18$ trits below the least significant trit of $P$, so $c$ cannot affect the rounded $W$-trit result of $P$ (except at an exact midpoint, which in balanced ternary only occurs if the dropped trits are $\pm \frac{3^k-1}{2}$, impossible if a zero trit intervenes). Thus `normalize_128(P, e_p)` is exact.
     - **Case 3 ($\Delta < 0$)**: $e_c > e_p$. Let $d = -\Delta = e_c - e_p > 0$.
       - If $d > W + 4$: $P$ is at least $4$ trits below the ULP of $M_c$, so return $c$.
       - If $d \le W + 4$: since $|M_c| < 3^W/2$ and $d \le 44$, for $W \le 20$ we have $W + d \le 64$ ($|M_c| \cdot 3^d < 3^{64}/2 < 2^{101}$), which fits comfortably in `__int128_t`! Thus $S = M_c \cdot 3^d + P$ at exponent $e_p$ is computed **100% exactly** in `__int128_t` and rounded once via `normalize_128(S, e_p)`.

#### C. Exact Balanced Ternary Square Root: `sqrt()`
- **Mathematical Design**:
  Given $x = M \cdot 3^e$:
  1. If $M < 0$, throw `std::domain_error("BTFP::sqrt: negative argument")`.
  2. If $M == 0$, return `BTFP(0, 0)`.
  3. **Exponent Parity Alignment**: We want $\sqrt{M \cdot 3^e} = \sqrt{M' \cdot 3^{2k}} = \sqrt{M'} \cdot 3^k$.
     Scale $M$ by $3^{S}$ into `__int128_t` such that the total exponent $e - S$ is **even**, and $M' = M \cdot 3^S$ has at least $2W + 4$ trits:
     - Choose base shift $S_0 = W + 4$.
     - If $(e - S_0)$ is odd, set $S = S_0 + 1$; else $S = S_0$.
     - Then $e_{\text{out}} = (e - S) / 2$, and $M_{\text{scaled}} = \text{static\_cast<\text{\_\_int128\_t}>}(M) \cdot 3^S$.
     - For $W \le 20$, $S \le 25$, so $M_{\text{scaled}} < 3^{45} < 2^{72}$, fitting inside `__int128_t` (and for $W \le 38$, use $S = W + 2$ so $M_{\text{scaled}} < 3^{79} < 2^{126}$).
  4. **Exact Nearest-Integer Square Root (`isqrt_nearest_128`)**:
     We seek the integer $R = \lfloor \sqrt{M_{\text{scaled}}} + \frac{1}{2} \rfloor$.
     Notice the algebraic identity:
     $$r = \lfloor \sqrt{N} \rfloor, \quad \text{round}(\sqrt{N}) = \begin{cases} r + 1 & \text{if } N - r^2 > r \\ r & \text{otherwise} \end{cases}$$
     because $(r + \frac{1}{2})^2 = r^2 + r + \frac{1}{4}$, and since $N, r \in \mathbb{Z}$, $N > r^2 + r + \frac{1}{4} \iff N - r^2 > r$.
     Compute $r = \lfloor \sqrt{M_{\text{scaled}}} \rfloor$ via 128-bit Newton-Raphson / Heron iteration seeded from `std::sqrt(static_cast<double>(M_scaled))`, refine until $r^2 \le M_{\text{scaled}} < (r+1)^2$, apply the exact midpoint test `if (M_scaled - r * r > r) ++r;`, and return `normalize_128(r, e_out)`.

#### D. Integer Exponentiation: `pow_int(int64_t n)`
- Binary exponentiation by squaring in $O(\log |n|)$ multiplications:
  - `n == 0`: return `BTFP(1)` (even for 0, matching `std::pow(0, 0) == 1`).
  - `n < 0`: compute `pow_int(-n)` and invert via `BTFP(1).div(...)` (or invert first/last). To minimize rounding drift, accumulate intermediate products in 128-bit normalized form `(m128, exp)` with 38-trit working precision before rounding once to `MantissaTrits` at the end!

#### E. Exact Conversion Between `BTFP` and `bt::Rational`
Currently `ternary_fp.hpp` does not know about `bt::Rational` (which is defined in `libbt.hpp`).
- **`to_rational()` / `bt:: to_rational(const BTFP&)`**:
  - If `is_zero()`, return `Rational(0, 1)`.
  - If $e \ge 0$: return `Rational::from_reduced(static_cast<__int128_t>(m_mantissa) * fp_detail::pow3_128(m_exponent), 1)`.
  - If $e < 0$: let $k = -e$. Since `m_mantissa` is normalized, its lowest trit might still be 0 if `abs(m_mantissa) == min_norm_mantissa`? Wait: $\frac{3^{W-1}+1}{2}$ is never divisible by 3, and in general `Rational(n, d)` automatically reduces via `reduce_128(n, d)`! So constructing from `(m_mantissa, pow3_128(-m_exponent))` after `reduce_128` produces the **exact reduced rational** $M / 3^{-e}$ whenever it fits in `int64_t`.
- **`from_rational(const Rational& r)`**:
  - If `r.is_zero()`, return `BTFP(0, 0)`.
  - Exact 128-bit conversion without `double` intermediate precision loss!
  - Extract powers of 3 from `r.num` and `r.den` first, or directly compute:
    `BTFP(r.num, 0) / BTFP(r.den, 0)`!
    Wait: if `MantissaTrits < 40`, `BTFP(r.num, 0)` would round `r.num` to `MantissaTrits` before dividing!
    To convert `Rational(num, den)` to `BTFP<W, E>` with a **single unbiased rounding**:
    Scale `num` by $3^S$ in `__int128_t` such that $|num| \cdot 3^S / den \ge 3^{W+2}$, perform exact symmetric-rounded 128-bit integer division $\text{round}((num \cdot 3^S) / den)$, and call `normalize_128(q, -S)`. This guarantees 0 intermediate rounding error!

---

## 2. `libbt.hpp` — Generic `Matrix<ScalarT>` & Radix-3 FFT (`Radix3FFT`)

### 2.1 `Matrix<T>` Template Audit: Seamless Support for `Rational`, `double`, and `BTFP<W, E>`
Wait: in `libbt.hpp` line 269, `Matrix` is already templated as `template <typename T = Rational> class Matrix`, with aliases `using RationalMatrix = Matrix<Rational>;` and `using RealMatrix = Matrix<double>;`!
**HOWEVER**, `Matrix<BTFP32>` and `Matrix<BTFP16>` currently **fail to compile or behave suboptimally** in several methods of `Matrix<T>`:

1. **Missing `solve(const std::vector<T>& b)` and `solve(const Matrix<T>& B)`**:
   - Currently `Matrix<T>` only has `solve_upper(U, b)` and `solve_lower(L, b)`! It is **completely missing general linear system solving `solve(b)`** ($A x = b$) via Gaussian elimination / Gauss-Jordan or LU decomposition!
   - Over `Matrix<Rational>`, `A.solve(b)` computes the **100% exact rational solution vector $x = A^{-1} b$ with zero rounding error**, in $\frac{1}{3} n^3$ operations (3x faster than computing `A.inverse() * b`).

2. **Broken `BTFP` Support in `determinant()`, `inverse()`, `norm_1()`, `norm_inf()`, `norm_frobenius()`, and Vector Norms**:
   - In `determinant()` (lines 490, 502) and `inverse()` (lines 559, 571):
     ```cpp
     if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
     ```
     For `T = BTFP32`, this `if constexpr` takes the `else` branch (which was designed for `Rational`), doing **zero partial pivoting** (`while (pivot < n && A(pivot, i) == T{0}) ++pivot;`)! For floating-point types like `BTFP32`, Gaussian elimination without partial pivoting is numerically unstable. Even for `Rational`, partial pivoting or first-nonzero pivoting works, whereas `BTFP` should use partial pivoting by magnitude (`A(r, i).abs() > max_val`).
   - In `norm_1()`, `norm_inf()`, `norm_frobenius()`, `norm_l1()`, `norm_l2()`, `norm_linf()`:
     ```cpp
     if constexpr (std::is_same_v<T, Rational>) {
         col_sum += std::abs((*this)(r, c).to_double());
     } else {
         col_sum += std::abs(static_cast<double>((*this)(r, c)));
     }
     ```
     `BTFP` has `.to_double()`, **not** `explicit operator double()`! Thus `static_cast<double>(elem)` fails to compile when `T = BTFP32` unless either:
     - `BTFP` defines `explicit constexpr operator double() const noexcept { return to_double(); }`, AND/OR
     - `libbt.hpp` provides a unified `detail::to_double_scalar(const T& x)` trait that dispatches to `.to_double()` if available (via C++20 `requires { x.to_double(); }`) or `static_cast<double>(x)`.
   - Also in `determinant()` and `inverse()`:
     In `inverse()` line 615: `A(r, c) = A(r, c) - factor * A(i, c);` works, and at the end of the row reduction, `A(r, i)` is updated in the loop `for (size_t c = 0; c < n; ++c)`.

3. **Exact `Matrix<Rational>` Capabilities**:
   - Over `Matrix<Rational>`, `determinant()`, `inverse()`, and the new `solve(b)` / `rref()` / `rank()` are **100% mathematically exact** with zero rounding drift.
   - Adding `solve(const std::vector<T>& b)` and `solve(const Matrix<T>& B)` to `Matrix<T>` makes `Matrix<Rational>`, `Matrix<double>`, and `Matrix<BTFP32>` a unified, complete linear algebra container.

### 2.2 `Radix3FFT` Audit: `ifft_unitary`, $O(N \log_3 N)$ Cooley-Tukey, and `convolve_radix3`

#### A. Critical Bug Found in `Radix3FFT::ifft_unitary`!
Look at lines 1009–1026 of `libbt.hpp`:
```cpp
static std::vector<cdouble> fft_unitary(const std::vector<cdouble>& x) {
    if (x.empty()) return {};
    std::vector<cdouble> res = fft(x);
    double scale = 1.0 / std::sqrt(static_cast<double>(x.size()));
    for (auto& v : res) v *= scale;
    return res;
}

static std::vector<cdouble> ifft_unitary(const std::vector<cdouble>& X) {
    if (X.empty()) return {};
    std::vector<cdouble> res = ifft(X);
    double scale = std::sqrt(static_cast<double>(X.size()));
    for (auto& v : res) v *= scale;
    return res;
}
```
Let's check the math of `ifft_unitary`:
- The unnormalized forward DFT is $F(x)_k = \sum_{n=0}^{N-1} x_n \omega_N^{-kn}$.
- `fft(x)` computes $F(x)$.
- `fft_unitary(x)` computes $\hat{X} = \frac{1}{\sqrt{N}} F(x)$.
- `ifft(X)` computes $\frac{1}{N} F^{-1}(X) = \frac{1}{N} \sum_{k=0}^{N-1} X_k \omega_N^{+kn}$.
- Therefore, `ifft_unitary(X)` computes:
  $$\sqrt{N} \cdot \text{ifft}(X) = \frac{\sqrt{N}}{N} F^{-1}(X) = \frac{1}{\sqrt{N}} F^{-1}(X)$$
- Wait: is that right?
  $$\text{ifft\_unitary}(\text{fft\_unitary}(x)) = \frac{1}{\sqrt{N}} F^{-1}\left(\frac{1}{\sqrt{N}} F(x)\right) = \frac{1}{N} F^{-1}(F(x)) = x$$
  YES! Because `ifft(X)` already divides by $N$ (line 980: `inv_N = 1.0 / N`), multiplying `ifft(X)` by $\sqrt{N}$ yields net factor $\frac{\sqrt{N}}{N} = \frac{1}{\sqrt{N}}$, which is the exact unitary inverse transform $\mathcal{F}_u^{-1}(X) = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X_k e^{+2\pi i k n / N}$!
  - **Verification**: Both `fft_unitary` and `ifft_unitary` satisfy $\|\mathcal{F}_u(x)\|_2 = \|x\|_2$, $\|\mathcal{F}_u^{-1}(X)\|_2 = \|X\|_2$, and $\mathcal{F}_u^{-1}(\mathcal{F}_u(x)) = x$.

#### B. $O(N \log_3 N)$ Cooley-Tukey Verification for Arbitrary Powers of 3 ($N = 3^k$)
Let's trace `Radix3FFT::transform_inplace` in `libbt.hpp` (lines 933–985):
1. **Trit-reversal permutation**: For each $i \in [0, 3^k - 1]$, `trit_reverse(i, k)` reverses the $k$ base-3 digits of $i$. Since digit-reversal is an involution ($\text{rev}(\text{rev}(i)) = i$), swapping when `i < rev` permutes the array in-place in $O(N k) = O(N \log_3 N)$ time (or $O(N)$ if incremental trit-reversal is used).
2. **Radix-3 Butterfly Stages**:
   - Outer loop: `for (size_t length = 3; length <= N; length *= 3)` runs $k = \log_3 N$ stages.
   - In stage `length` ($m = \text{length}/3$), the primitive 3rd roots of unity are:
     $$w_{3,1} = e^{\pm 2\pi i / 3} = -\frac{1}{2} \pm i \frac{\sqrt{3}}{2}, \quad w_{3,2} = e^{\pm 4\pi i / 3} = -\frac{1}{2} \mp i \frac{\sqrt{3}}{2} = \overline{w_{3,1}}$$
   - For each block of size `length` and offset $j \in [0, m-1]$:
     $$u_0 = a[i+j], \quad u_1 = a[i+j+m] \cdot \omega_{\text{length}}^{\pm j}, \quad u_2 = a[i+j+2m] \cdot \omega_{\text{length}}^{\pm 2j}$$
     $$a[i+j] = u_0 + u_1 + u_2$$
     $$a[i+j+m] = u_0 + u_1 w_{3,1} + u_2 w_{3,2}$$
     $$a[i+j+2m] = u_0 + u_1 w_{3,2} + u_2 w_{3,1}$$
   - Why does $a[i+j+2m]$ have $w_{3,2}$ on $u_1$ and $w_{3,1}$ on $u_2$?
     Because for frequency index $j + 2m$:
     $$\omega_{\text{length}}^{(j+2m) \cdot r} = \omega_{\text{length}}^{j r} \cdot (\omega_{\text{length}}^m)^{2r} = \omega_{\text{length}}^{j r} \cdot w_{3,1}^{2r}$$
     For $r = 1$: $w_{3,1}^2 = w_{3,2}$.
     For $r = 2$: $w_{3,1}^4 = w_{3,1}^3 \cdot w_{3,1} = w_{3,1}$.
   - Thus the butterfly equations are **100% mathematically exact** for arbitrary $N = 3^k$!
   - **Guard condition**: In `for (size_t length = 3; length <= N; length *= 3)`, since $N = 3^k$, `length` takes values $3^1, 3^2, \dots, 3^k = N$ and terminates (as long as $3 N$ does not overflow `size_t`, which holds for all $k \le 39$ on 64-bit platforms).

#### C. Clean 1D Discrete Convolution via Radix-3 FFT: `convolve_radix3(a, b)`
- **Mathematical Specification**:
  Given two sequences $a$ of length $N_a$ and $b$ of length $N_b$:
  1. If $a$ or $b$ is empty, return `{}`.
  2. The linear discrete convolution $c = a * b$ has exact length $L = N_a + N_b - 1$, where:
     $$c[n] = \sum_{k=\max(0, n - N_b + 1)}^{\min(n, N_a - 1)} a[k] \, b[n - k], \quad n \in [0, L - 1]$$
  3. Find the smallest power of 3, $N = 3^k$, such that $N \ge L$:
     ```cpp
     size_t N = 1;
     while (N < L) N *= 3;
     ```
  4. Zero-pad $a$ and $b$ to length $N$: $A_{\text{pad}}[0..N-1]$, $B_{\text{pad}}[0..N-1]$.
  5. Transform in-place via `Radix3FFT::transform_inplace(A_pad, false)` and `Radix3FFT::transform_inplace(B_pad, false)`.
  6. Pointwise multiply in the frequency domain:
     $$\forall m \in [0, N-1]: \quad C_{\text{pad}}[m] = A_{\text{pad}}[m] \cdot B_{\text{pad}}[m]$$
  7. Inverse transform in-place: `Radix3FFT::transform_inplace(C_pad, true)`.
  8. Truncate $C_{\text{pad}}$ to length $L = N_a + N_b - 1$.
  9. **Overloads to provide**:
     - `convolve_radix3(const std::vector<cdouble>& a, const std::vector<cdouble>& b)` -> `std::vector<cdouble>` (complex convolution).
     - `convolve_radix3(const std::vector<double>& a, const std::vector<double>& b)` -> `std::vector<double>` (real convolution, extracting `.real()`).
     - `convolve_radix3_int64(const std::vector<int64_t>& a, const std::vector<int64_t>& b)` -> `std::vector<int64_t>` (exact integer / balanced ternary polynomial convolution via `std::llround(c[i].real())`!). Because balanced ternary trits are in $\{-1, 0, +1\}$, the maximum convolution magnitude for length $N \le 3^{15}$ is $\le N$, well within IEEE-754 `double`'s 53-bit exact integer mantissa ($2^{53} \approx 9 \times 10^{15}$), so `llround` recovers the **100% exact integer convolution** in $O(N \log_3 N)$ time!
     - Also provide `circular_convolve_radix3(a, b)` for exact $N = 3^k$ cyclic convolution without zero-padding.

---

## 3. `ternary_dense_algebra.hpp` & `ternary_sparse.hpp` — `MultiTritMatrix` & `gemm_multitrit`

### 3.1 Audit of `MultiTritMatrix` and `gemm_multitrit`
Let's inspect `MultiTritMatrix` and `gemm_multitrit` in `ternary_dense_algebra.hpp` (lines 803–875):
- `MultiTritMatrix(size_t r, size_t c, size_t p)` stores `num_planes = p` instances of `TernaryMatrix(r, c)`.
- Each plane $p \in \{0, 1, \dots, P-1\}$ holds a balanced ternary matrix $A^{(p)} \in \{-1, 0, +1\}^{M \times K}$ representing positional weight $3^p$:
  $$A = \sum_{p=0}^{P-1} A^{(p)} \cdot 3^p, \quad B = \sum_{q=0}^{Q-1} B^{(q)} \cdot 3^q$$
- `gemm_multitrit(A, B, C)` computes:
  $$C = A \cdot B = \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} \left(A^{(p)} B^{(q)}\right) \cdot 3^{p+q}$$
  using `gemm(A.plane(p), B.plane(q), plane_prod)` and accumulating into a 128-bit buffer `accum[idx] += plane_prod[idx] * pow3[p + q]`, then checking for `int64_t` overflow and writing to `C`.

### 3.2 Critical Usability & Completeness Gaps in `MultiTritMatrix`
While the positional weighting $3^{p+q}$ in `gemm_multitrit` is mathematically exact, **`MultiTritMatrix` is missing all encoding, decoding, and element access methods!**
Specifically:
1. **No `encode_int_matrix` / `decode_int_matrix`**:
   - Users currently have to manually decompose every integer entry $A(r, c)$ into balanced trits and loop over `plane(p).set(r, c, t)`!
   - There is no `MultiTritMatrix::encode_int_matrix(const DenseMatrixInt64& M, size_t num_planes)` or `MultiTritMatrix::to_int_matrix() const`.
2. **No Element Accessors `get_int64(r, c)` / `set_int64(r, c, val)`**:
   - Setting or reading a single multi-trit integer at $(r, c)$ across all $P$ planes requires manual loops.
3. **Suboptimal Plane-by-Plane Transpose in `gemm_multitrit`**:
   - In `gemm_multitrit` (line 860):
     ```cpp
     for (size_t p = 0; p < A.num_planes; ++p) {
         for (size_t q = 0; q < B.num_planes; ++q) {
             gemm(A.plane(p), B.plane(q), plane_prod);
     ```
     Inside `gemm(A.plane(p), B.plane(q), plane_prod)`, **`B.plane(q).transpose()` is recomputed $P$ times for every plane $q$ of $B$** ($P \times Q$ transpositions instead of $Q$ transpositions!), and it calls the un-tiled `gemm` instead of `gemm_autotuned`!
   - Pre-transposing all $Q$ planes of $B$ once before the loop (or using `DenseDualRailPlanes`) reduces transpose overhead by $P\times$ and enables L1d cache-autotuned 4x2 register microkernels for every $(p, q)$ plane pair!

### 3.3 Mathematical Specification for Complete Multi-Trit Encoding & Reconstruction
In balanced ternary, any signed integer $x \in \left[-\frac{3^P - 1}{2}, +\frac{3^P - 1}{2}\right]$ has a **unique** $P$-trit decomposition:
$$x = \sum_{p=0}^{P-1} t_p \cdot 3^p, \quad t_p \in \{-1, 0, +1\}$$
computed via the balanced division recurrence:
$$r = x \bmod 3, \quad q = x / 3$$
$$\text{if } r == 2: (t_p = -1, q \leftarrow q + 1); \quad \text{else if } r == -2: (t_p = +1, q \leftarrow q - 1); \quad \text{else}: t_p = r$$
- **`MultiTritMatrix::max_representable(size_t num_planes)`**:
  Returns $\frac{3^P - 1}{2}$.
- **`MultiTritMatrix::min_planes_for_range(int64_t max_abs_val)`**:
  Computes the minimum $P$ such that $\frac{3^P - 1}{2} \ge |x_{\max}|$.
- **`MultiTritMatrix::encode_int_matrix(const DenseMatrixInt64& src, size_t num_planes = 0)`**:
  - If `num_planes == 0`, automatically determines the minimal $P \ge 1$ sufficient to represent $\max_{r,c} |src(r, c)|$!
  - If `num_planes > 0`, validates that every entry $src(r, c)$ satisfies $|src(r, c)| \le \frac{3^P - 1}{2}$ (throwing `std::overflow_error` if any entry exceeds the $P$-trit representable range, preventing silent high-trit truncation).
  - Decomposes each $src(r, c)$ into $P$ balanced trits and populates `plane(0) ... plane(P-1)`.
- **`MultiTritMatrix::decode_int_matrix() const` (and `to_int_matrix() const`)**:
  - Reconstructs `DenseMatrixInt64` by evaluating $\sum_{p=0}^{P-1} A^{(p)}(r, c) \cdot 3^p$.
- **Exact Identity Guarantee**:
  For any integer matrices $A \in \mathbb{Z}^{M \times K}$ and $B \in \mathbb{Z}^{K \times N}$ with entries bounded by $\frac{3^P-1}{2}$ and $\frac{3^Q-1}{2}$ ($P + Q \le 38$):
  $$\text{gemm\_multitrit}(\text{encode}(A, P), \text{encode}(B, Q)) \equiv A \cdot B \quad (\text{100\% bit-exact integer product!})$$

### 3.4 Sparse-Dense & Multi-Trit Interop (`ternary_sparse.hpp`)
In `ternary_sparse.hpp`, `TCSRMatrix` already provides multiplier-free `spmv(const T* x, T* y)` and `spmm(const T* B, size_t B_cols, T* C)` templated on `T`.
- Because `spmv` and `spmm` only use `T(0)`, `+=`, and `-=`, they **already work out-of-the-box** with `T = double`, `T = int64_t`, `T = bt::Rational`, and `T = bt::BTFP32`!
- Adding a convenience overload `TCSRMatrix::spmm(const Matrix<T>& B) -> Matrix<T>` in `libbt.hpp` (or when `Matrix<T>` is available) allows multiplying a 1-bit-per-nonzero sparse ternary matrix directly against `Matrix<Rational>`, `Matrix<double>`, or `Matrix<BTFP32>` with **zero multiplications**!

---

## 4. Summary of Concrete Mathematical Enhancements
| File | Component | Enhancement / Fix | Mathematical Property |
| :--- | :--- | :--- | :--- |
| `ternary_fp.hpp` | `BTFP::fma(a, b, c)` | 128-bit exact product $M_a M_b 3^{e_a+e_b} + M_c 3^{e_c}$ with single `normalize_128` rounding | Eliminates double-rounding and catastrophic cancellation |
| `ternary_fp.hpp` | `BTFP::sqrt()` | Parity-aligned $3^S$ scaling + 128-bit Newton-Raphson `isqrt` with exact $N - r^2 > r$ midpoint rounding | Correctly rounded radix-3 square root |
| `ternary_fp.hpp` | `BTFP::abs()`, `neg()`, `pow_int(n)` | $O(1)$ mantissa sign ops; 128-bit binary exponentiation by squaring | Zero-bias, $O(1)$ sign control without re-normalization |
| `ternary_fp.hpp` | `BTFP::mul()` | Fix `ExponentTrits > 0` clamping to renormalize mantissa via `normalize()` | Eliminates silent $3^{\Delta e}$ value corruption on exponent clamp |
| `libbt.hpp` | `BTFP` $\leftrightarrow$ `Rational` | `to_rational(BTFP)` and `from_rational<BTFP>(Rational)` via 128-bit scaled division | 100% exact rational extraction & single-rounding rational import |
| `libbt.hpp` | `Matrix<ScalarT>` | Partial pivoting & norm traits for `BTFP`, plus general linear solver `solve(b)` / `solve(B)` | 100% exact $Ax=b$ over `Rational`; stable pivoting over `BTFP32` & `double` |
| `libbt.hpp` | `Radix3FFT` | Add `convolve_radix3` (complex, real, and exact `int64_t`) & `circular_convolve_radix3` | $O(N \log_3 N)$ polynomial/signal convolution via base-3 Cooley-Tukey |
| `ternary_dense_algebra.hpp` | `MultiTritMatrix` | Add `encode_int_matrix`, `decode_int_matrix`, `get_int64`, `set_int64`, and pre-transposed `gemm_multitrit` | Complete signed multi-trit radix-3 decomposition & $P\times$ fewer transposes |
