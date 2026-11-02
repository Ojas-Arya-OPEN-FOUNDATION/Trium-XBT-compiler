# More capable floating-point, exact numerical routines and SDK 1.2

Date: **2026-10-30**. This is the next implementation slice after [SDK 1.1](SDK_1_1.md), following [the numerical expansion plan](NUMERICAL_EXPANSION_PLAN.md). README and license grants remain unchanged. ABI major **1**, 24-byte finite-FP records and existing backend IDs are preserved; the additive API/library/package version is now **1.2 / 1.2.0**.

## 1. What is newly usable

All of these operate on **finite balanced-ternary floating-point values, widths 1..40**, not binary64 relabeled as ternary. Values remain normalized `M * 3^e`, with canonical zero, nearest rounding and halfway ties away from zero. Domain/range failures remain explicit.

| New operation | Numerical contract |
|---|---|
| `reciprocal` | Round the exact ratio `1/x` once; reject zero |
| `cbrt` | Signed real cube root, including negative inputs; integer-powered midpoint comparisons select the nearest representable result |
| `hypot` | Round `sqrt(x*x + y*y)` once, without rounding or range-checking the intermediate squares |
| `pow_int` | Compute the exact integer power, then round once; exponent -16..16; negative powers invert the exact power, not an already-rounded result; `x^0 = 1`, including `0^0` |
| `sum_exact` | Sum the exact values of the represented inputs, then round once |
| `dot_exact` | Accumulate exact products and sum, then round once; no intermediate product rounding |
| `norm2` | Round the square root of the exact sum of squares once |
| `polynomial_exact` | Ascending coefficients `[c0,c1,...]`; evaluate the entire exact polynomial, then round once—not multiply-rounded Horner |

The original add/subtract/multiply/divide/sqrt/FMA and binary64 conversion contracts are unchanged. The new exact reciprocal is separate from the older **approximate** Householder reciprocal.

### Why this matters

Repeatedly rounded accumulation can lose small residuals. At width 3, take the represented values `1053`, `1`, `-1053`: the sequential rounded sum is zero, while the new exact sum returns **1**. Dot products can cancel products whose individual exponents would be outside the output format; the exact accumulator does not reject those products prematurely. Hypot and norm similarly avoid false intermediate square overflow/underflow.

These are useful building blocks for residuals, low-bit/ternary-weight experiments, distances, normalization, polynomial approximations and scoped simulation calculations. They are not a full AI framework, floating GPU GEMM, general linear solver or proof of an application's numerical stability. "Exact" refers to the **represented inputs** and accumulation before final rounding; it does not undo input quantization from binary64/decimal data.

## 2. Algorithms and predictable limits

The implementation uses checked, fixed **2048-bit unsigned scratch integers**, signed-magnitude accumulation and exact ratios. No external arbitrary-precision library and no binary floating intermediate is used in these new qualified operations.

Roots compare the exact radicand with powered rational midpoints, including the two-unit gap at ternary binade boundaries. Hypot has a proven conservative far-tail rule when the input exponent separation exceeds the format width, so it does not allocate or iterate proportional to `INT32_MAX`. Range checks apply to the final rounded exponent.

The `exact-reductions-v1` profile has explicit bounds:

- At most **4096** array terms.
- At most **256 trits** between active term exponents, **after stripping all exact powers of three from each coefficient**, and **before cancellation**.
- Polynomial degree at most **16** (at most 17 supplied coefficients, including trailing zeros).
- Integer power exponent **-16..16**.
- Existing normalized input records and int32 final exponent bounds.

For sum the active terms are values; for dot they are exact products; for norm they are exact squares; for polynomial they are `ci*x^i`. Zero terms do not widen the exponent window. Removing coefficient powers of three prevents artificial failures for values such as `x=1` at width 40. The window is not simply the difference of the original record exponents, and equal large terms are not canceled before admission. A widely separated sum may therefore be refused even if a more elaborate algorithm could compute it.

Beyond a count/window/power/degree bound the SDK returns **RESOURCE (6)**, never a shortened array, dropped term, approximate fallback or partly written result. Invalid records/domain inputs return **INVALID (1)**; final exponent range failures return **OVERFLOW (2)**. Empty sums/dots/norms and empty polynomials yield canonical zero; reductions have an explicit width. These math functions have their own fixed limits, not a source context's interpreter work budget. They are synchronous CPU routines, not backend-selectable GPU or source-language BTFP lowering.

Outputs commit after validation/calculation and may alias an input record. Raw C storage must still be valid, aligned and large enough. The caller must not mutate it concurrently. Scratch/count bounds are not a global application memory quota or hard-real-time guarantee.

## 3. SDK and application surface

### C and discovery

- Extended `bt_sdk_float_apply` IDs: reciprocal **7**, cbrt **8**, hypot **9**; original IDs **1..6** unchanged.
- New `bt_sdk_float_powi`.
- New `bt_sdk_float_reduce`: sum **1**, dot **2**, norm2 **3** in its separate reduction-op domain.
- New `bt_sdk_float_polynomial`.
- `bt_sdk_info` advertises API minor 2, extended operation names, numerical profile, reduction names and limits.

All functions are exported through the existing shared SDK. A common internal record decoder now enforces the same canonical/reserved-field checks for old and new operations. The compiled CPU module cache, Vulkan resource cache, signatures and ownership APIs from SDK 1.1 remain intact.

### Python

```python
import trium

with trium.Context(library="/absolute/path/libtrium_sdk.so") as ctx:
    three = ctx.float_from_ratio(3, width=40)
    four = ctx.float_from_ratio(4, width=40)
    five = ctx.float_apply("hypot", three, four)
    assert ctx.float_to_double(five) == 5.0

    norm = ctx.float_reduce("norm2", [three, four])
    dot = ctx.float_reduce("dot", [three, four], [three, four])
    assert ctx.float_to_double(norm) == 5.0
    assert ctx.float_to_double(dot) == 25.0
    reciprocal_cube = ctx.float_powi(three, -3)
    # Inputs are represented ternary values; this result has one final rounding.

    x = ctx.float_from_ratio(2, width=40)
    coeff = [ctx.float_from_ratio(n, width=40) for n in (1, 2, 3)]
    assert ctx.float_to_double(ctx.float_polynomial(x, coeff)) == 17.0
```

Reductions infer width from the first input unless explicitly supplied; an empty Python reduction defaults to width 20 unless `width=` is supplied. Mixed-width records are rejected. Iterables are bounded while materializing; infinite iterables do not grow without limit. New calls are feature-gated against older libraries and report API 1.2 as required. These context-free C math operations are exposed as Python `Context` conveniences; they do not execute source or depend on an open source-module context.

### C++

`Ternary/Math/Numerical.hpp` provides typed span-based `sum_exact`, `dot_exact`, `norm2`, and `polynomial_exact`. For a vector and an explicit width/profile:

```cpp
#include <Ternary/Math/Numerical.hpp>
using F = bt::BTFP<40>;
std::vector<F> inputs{F(3), F(4)};
auto norm = bt::math::norm2<40, 0>(inputs); // 5
```

The owning SDK `bt::sdk::TernaryFloat` exposes `.pow_int`, extended `.apply`, static `.reduce`, and `.polynomial`. Installed consumers request `find_package(TriumXBT 1.2 CONFIG REQUIRED)`. Relocated C and C++ examples exercise new routines as well as existing source execution.

The Python wheel still contains wrappers, not a bundled native library. Rust's earlier compiler bindings remain unchanged; no new Rust finite-FP wrapper or local Rust qualification is implied. Public distribution still needs the [existing component-license clarification](LICENSE_BOUNDARIES.md); no license grant was changed.

## 4. Older low-level helpers: real corrections

The new interval checker `tools/verify_cordic_constants.py` computes **84 binary64 angle/scale constants** from exact rational intervals and exact square-root midpoint comparisons. It found and corrected **36 stored constants**. Most differences were a few ulps; one circular micro-angle had approximately **6.37327e-11** absolute error. The checker verifies that an interval selects a unique binary64 value and runs in CTest. `--rewrite` is an explicit maintainer action, not runtime regeneration.

On the deterministic survey of 2001 circular inputs from -100 to 100, the maximum sin/cos absolute error changed from approximately **6.3735e-11 to 3.8581e-15**. The corresponding sampled positive-octant atan2 survey improved from approximately **6.3733e-11 to 4.4409e-16**. These are sampled observations, **not universal error bounds**.

Other robustness changes:

- Circular sin/cos preserves negative-zero sine and returns NaNs for nonfinite raw helper input.
- Atan2 handles IEEE axes, signed zeros and nonfinite inputs explicitly, and scales finite vectors before iteration to avoid overflow near binary64 maxima. If scaling underflows one component, the resulting signed zero is preserved.
- Reduced hyperbolic helpers explicitly reject inputs outside their declared domain with NaN; negative zero is preserved.
- Legacy BTFP exp uses fused residual calculation and a half-bin correction to keep the reduced helper input in range.
- New checked scratch `divmod(u64)` is verified by exact reconstruction, and old power-of-three rounding helpers are regression-tested at signed 64/128-bit minima and maxima.

The full older CORDIC/Householder algorithms remain **approximate**. Certifying stored constants does not certify huge-angle range reduction, every transcendental result, all exponent profiles or arbitrary host floating environments. They are not silently promoted into the exact SDK operation set.

## 5. Verification and evidence

**Final local qualification:** Release **49 passed / 4 skipped of 53** (25.97 s); Debug ASan/UBSan **49 passed / 4 skipped of 53** (63.14 s). The new routines passed **13,326** exact result/error/ABI checks. Existing **54,022** finite-FP checks, **15,372** older rational/BTFP cases and **1,744** native differential cases remain passing. All **14 root Z3 scripts** and the **84-constant interval check** passed. Relocated C/C++ clients, an isolated installed Python wheel, and installed numerical headers compiled **without linking the SDK** passed.

The four explicit skips were physical CUDA qualification, Rocq, Vulkan offline tools and Vulkan runtime execution. This round used a Vulkan-disabled SDK because the earlier GPU tool/runtime build artifacts were absent; no GPU success is inferred from these CPU results.

The independent oracle uses Python `Fraction`, integer roots and powered rational midpoint comparisons—not tolerance-based checks—to compare result mantissa/exponent. It exercises all widths 1..40, small exhaustive formats, deterministic random data, negative roots/powers, canonical zero, binade boundaries, cancellation/permutations, full count/degree bounds, extreme int32 exponents, range/resource errors and raw ABI output transactions/aliasing. Existing exact foundation cases remain separate nonregressions.

The new Z3 script checks **51** scoped root/tail/capacity/address model obligations. As with the preceding milestones, these are not a whole-C++/allocator/driver proof. Legacy-helper numerical surveys use explicit tolerances and are not mixed with exact result certification.

Actual final test counts, toolchain/platform, source/library hashes, installed-consumer results and proof output are recorded in [the numerical evidence directory](evidence/2026-10-30-numerical-expansion/). Physical GPU testing remains isolated and user-run; this milestone does not add GPU BTFP execution. Previous SDK 1.1 software Vulkan evidence is historical, not a new physical qualification.

## 6. Next numerical gates

Further work can build on these primitives: certified bounded exp/log with explicit approximation/error contracts, wider sparse-exponent accumulators, larger powers, conditioned linear algebra/solvers, transcendental range reduction and framework-owned tensors. Each needs its own algorithms, limits and independent references. This delivery expands useful mathematical building blocks without pretending all advanced numerics or future ternary hardware are complete.
