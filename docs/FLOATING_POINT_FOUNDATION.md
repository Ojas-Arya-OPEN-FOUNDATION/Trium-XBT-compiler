# Finite ternary floating-point foundation — profile 1

Date: 2026-10-29. This is the focused numerical milestone following
[R1/R2](R1_R2_DELIVERY.md). README and license declarations remain unchanged.
Machine-readable contract: [`specs/ternary-float-v1.json`](../specs/ternary-float-v1.json).

## In plain terms

We retained the mathematical format and removed known arithmetic restrictions.
**All widths 1..40 now have addition, subtraction, multiplication, division,
square root and single-rounding FMA, without the previous i128 alignment or
W=40 square-root rejection.** A genuine out-of-range final result still raises
an error; that is the format's deliberate finite-range policy.

Applications can call these operations through C, C++ or Python. This is actual
CPU execution on binary hardware. It is not GPU floating-point execution,
generated FPGA RTL, or a physical ternary chip.

## 1. The numerical contract

A nonzero value is `M * 3^e`. For width W:

- `1 <= W <= 40`.
- `L=(3^(W-1)+1)/2 <= abs(M) <= H=(3^W-1)/2`.
- Mantissa storage is int64; exponent storage is int32.
- Zero is exactly `(0,0)`; no signed BTFP zero, NaN, infinity or subnormal.
- Round to the nearest normalized value, halfway ties **away from zero**.
- First round on the normalized lattice with unbounded mathematical exponents,
  then check the stored/configured exponent range. Nonzero underflow/overflow
  rejects; it does not round to zero or clamp to the nearest endpoint.

Adjacent mantissas inside a binade differ by one. Across a binade boundary,
`3L=H+2`: the gap is two units, not one. This matters for normalization, ratios,
FMA ties and square root. All must select the nearest actual representable value.

`BTFP<W,E>` retains its additional exponent-trit bound when E>0 (capped by int32
storage). The SDK's profile-1 values expose the full int32 exponent domain.
Alternative rounding/special-value formats require a new explicit profile; this
milestone does not silently redefine all future kinds of floating point.

## 2. Arithmetic improvements

### Checked wider scratch arithmetic, with a fast path

`ternary_wide.hpp` supplies internal fixed-capacity unsigned scratch integers.
They are checked, allocation-free and use no external library. They do not
change the public mantissa width or create an arbitrary-precision public type.
The existing exact i128 path remains for alignments that fit; 512-bit scratch
handles the close cases that previously failed.

Addition/subtraction and FMA keep exact aligned coefficients through one final
rounding. FMA does not round or range-check the product before adding c. A final
answer can therefore succeed even when a separately rounded multiply would
raise an exponent-range error.

For exponent gaps greater than 160, materializing a power of three is unnecessary:
coefficient bounds prove the remote term cannot cross a nonzero rounding distance.
Its **sign is retained as a sticky perturbation** to resolve exact midpoint ties.
For example at W=3, `6*7=42` is halfway between 39 and 45. A tiny negative c must
choose 39, not 45, even when c's exponent is INT32_MIN. Tests cover both signs and
construct analogous ties at every odd width from 3 through 39.

### Square root, including W=40

The old overscaled radicand is replaced with adaptive exponent-parity scaling.
After scaling, `N < 9L^2`; both `4N` and the squared candidate midpoint fit i128
for every supported width. Binary-search neighbors and exact squared midpoint
comparisons avoid a binary-floating estimate and avoid double rounding.

### Exact binary64 conversions

`from_double` decodes the input's actual IEEE significand/exponent and rounds
that exact rational to ternary once. There is no logarithmic estimate, `pow`,
`llround` or approximate binary scaling in the conversion.

`to_double` rounds the exact ternary value to IEEE binary64 with **ties-to-even**,
including subnormals, normal/subnormal transitions, overflow and signed underflow
zero. This is distinct from ternary ties-away rounding and does not depend on the
host floating rounding mode. NaN/infinity import rejects; finite export may produce
IEEE infinity when the ternary magnitude exceeds binary64 range. Both binary
input zeros become the one canonical BTFP zero. Floating exception flags/payloads
are not promised.

1280-bit scratch is enough for every finite binary64 input and all nontrivial
binary64 exports; conservative magnitude exits avoid exponent-sized work at
int32 extremes. Supported native toolchains still require unsigned/signed i128
(GCC/Clang); a no-i128 MSVC implementation remains a separate portability task.

## 3. Application APIs

### Python

```python
# PYTHONPATH=$PWD/ffi/python
# BT_SDK_LIBRARY=$PWD/build/sdk/libtrium_sdk.so
import trium
with trium.Context() as ctx:
    nine = ctx.float_from_ratio(9, width=40)
    root = ctx.float_apply('sqrt', nine)
    assert ctx.float_to_double(root) == 3.0
    half = ctx.float_from_ratio(1, 2, width=40)
    imported = ctx.float_from_double(0.5, width=40)
    assert imported == half
    print(half.mantissa, half.exponent)  # inspect the actual ternary result
```

`TernaryFloat` is an immutable value record, not an ordinary Python float.
`float_apply` accepts `add`, `sub`, `mul`, `div`, `sqrt`, `fma`; all operands must
have the same width. Pass `(operation,a,b,c)` for FMA. These mathematical methods
use the context's loaded library, not GPU dispatch or compiler module execution.

### C and C++

The C header exposes `bt_sdk_float_value`, `bt_sdk_float_from_ratio`,
`bt_sdk_float_from_double`, `bt_sdk_float_to_double` and `bt_sdk_float_apply`.
Value records must be canonical, widths equal and reserved fields zero. Failure
leaves result storage unchanged. Domain/invalid inputs use SDK status 1;
nonzero exponent overflow/underflow use status 2. Records are native ABI structs,
**not a serialized wire format or a proposed hardware encoding**.

`bt::sdk::TernaryFloat` wraps those C functions, while header-only consumers can
continue using `bt::BTFP<W,E>` directly. Relocated C/C++ example clients now use
W40 square root through the installed SDK. Rust floating-point wrappers have not
been added in this milestone; the existing Rust source remains separately pending.

The language's `.bt float` is still binary64. This API does not silently turn
source floating-point programs into ternary arithmetic.

## 4. Independent reference and verification

`test/fp_reference.py` is an independent Python arbitrary-precision Fraction
oracle: it compares exact distances among neighboring normalized candidates.
Its square-root oracle compares exact squared rational midpoints, not an
approximate root. It does not call or reproduce the C++ scratch implementation.

`test/floating_foundation.py` checks exact mantissa/exponent pairs and binary64
bits, rather than accepting an epsilon. It includes:

- All 40 widths, rational imports and all six operations.
- Exhaustive pairs over small-format W1/W2/W3 neighborhoods, and W1 FMA triples.
- Binary64 normals/subnormals, extrema, signed zero, bitwise random inputs,
  export boundaries, ties-to-even and 1,000 random W40 binary roundtrips.
- INT32 exponent endpoints, parity-sensitive roots, remote FMA tails, alignment
  thresholds, malformed records, mixed widths and output transactions.
- Genuine range/domain failures, separately from successful numerical results.

The existing Fraction regression also now includes W40 FMA/sqrt rather than
skipping them. `test_unit_fp_foundation.cpp` covers constexpr sticky ties,
configured exponent bounds, fused cancellation and scratch carries/borrows.

`verify_z3_floating_foundation.py` adds limb/carry/borrow model lemmas, square-root
capacity and midpoint lemmas, and explicit remote-tail/binary64 capacity bounds.
These support the design but are **not a whole-C++ or RTL refinement proof**.
No physical GPU/FPGA/ternary device is used by these floating-point tests.

Executed counts, tools and logs are retained in
[evidence/2026-10-29-floating-foundation](evidence/2026-10-29-floating-foundation/).
Historical R1/R2 evidence remains separate and is not overwritten.

Final local results:

| Configuration | Passed | Skipped | Failed | Time |
|---|---:|---:|---:|---:|
| Release (45 registered tests) | 43 | 2 | 0 | 15.97 s |
| Debug ASan/UBSan (45 registered tests) | 43 | 2 | 0 | 39.87 s |

The independent foundation runner passed **54,022** exact checks. The earlier
Fraction regression passed **15,372** cases including the newly enabled W40
sqrt/FMA cases. All **12** root Z3 scripts passed. These counts describe scoped
checks, not a proof of every possible program or every bit of C++.

The two CTest skips remain physical CUDA qualification and native Rocq. Existing
Vulkan integer-map execution passed through SwiftShader CPU software Vulkan; it
is not physical-GPU or BTFP-GPU execution. The CPU-only SDK configuration separately
passed integration and all 54,022 foundation checks. Installed/relocated C/C++
consumers and a rebuilt/installed Python wheel exercised the new W40 API.
Python-hosted sanitizer tests disable CPython host leak reporting; instrumented
C++ tests retain leak detection.


## 5. What remains

This completes the scoped primitive implementation and independent-reference
milestone, not every numerical function or platform. Continue with measured
performance tuning, deeper implementation-level proofs, correctly bounded or
correctly rounded transcendental algorithms, alternative numerical profiles and
future hardware operators. `pow_int` and transcendental/Householder routines are
not newly certified as single-rounding primitives by this work.

Herbie and FloPoCo are not installed/integrated here. Herbie-style suggestions
must be validated against the ternary contract and must not silently replace
strict expressions with different floating-point behavior. Hardware generators
must implement this contract explicitly; choosing binary exponent/mantissa widths
alone is not a radix-3 implementation. Physical synthesis remains Step 6.
