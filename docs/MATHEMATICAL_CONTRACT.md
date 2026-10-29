# Mathematical contract — Phase 1

Status: executable specification for the remediated standalone compiler and runtime.
Last validation: 2026-10-29. See [validation and remaining gates](PHASE1_VALIDATION.md).
This document separates intended mathematics, implementation restrictions, and proof evidence.
The versioned stable subset and reusable library APIs are specified in
[CORE_SEMANTICS_V0_1.md](CORE_SEMANTICS_V0_1.md).
It is not a claim that the entire compiler is formally verified or production-certified.

## 1. Domains must not be conflated

A balanced trit is an element of `T = {-1, 0, +1}`. A positional word of width `w`
denotes `V(t) = sum(t[i] * 3^i, i=0..w-1)`. Its symmetric integer range is
`[-H_w, H_w]`, where `H_w = (3^w - 1)/2`.

A packed bit-sliced word instead stores **64 independent lanes** in two unsigned
64-bit rails `(P,N)`, with invariant `P & N == 0`. Lane `i` is
`((P >> i) & 1) - ((N >> i) & 1)`. Negation swaps the rails. This representation
is not a signed 64-bit binary integer and a full 64-trit positional value does
not fit in `int64_t`. Low-level codecs with caller-selected widths are not
implicitly checked scalar conversions.

The `.bt` scalar integer domain is **39 positional trits**:

```
I = [-2026277576509488133, +2026277576509488133]
```

Scalar `+`, `-`, and `*` mean mathematical integer operations followed by a range
check in `I`. They do not wrap, saturate, or silently drop high trits. Operands
in `I` have exact signed-128-bit intermediates, including multiplication.
The interpreter raises an error; generated native code sets its internal error
state, propagates it across calls, and exits with status 1. Native positional
vector literals are currently limited to 39 trits. Wider interpreter literals
infer a separate `packedword64` type, which cannot implicitly enter scalar
arithmetic or conditions; literals wider than 64 trits are rejected. Wider packed-library operations
must not be advertised as part of the scalar backend equivalence claim.

## 2. Logic, comparisons, and control flow

Digit-wise strong-Kleene AND/OR are respectively `min(a,b)` and `max(a,b)` on
`-1 < 0 < +1`. The operation historically called ternary XOR is digit-wise trit
multiplication, **not** binary XOR or a promise of an arbitrary field operation.
Missing high positional digits are zero. Negation changes every trit's sign.

Ordinary comparisons return `+1` for true and `-1` for false. Spaceship comparison
returns the sign of the numeric ordering. `if` and `while` take the true path
only for a positive condition; this is not C-style nonzero truthiness.

`select(c, positive, zero, negative)` evaluates the condition once, then evaluates
**only** the selected branch. Types are checked in all branches before execution.
Integral conditions partition into positive, zero, and negative. For binary64
conditions, signed zero takes the zero branch. The current policy for a computed
NaN condition is the zero branch in `select` and false in `if`/`while`, because
ordered comparisons with NaN are false. This is a documented policy, not a new
ternary numeric value. Infinity is classified by sign.

Bindings are lexically scoped. Functions have checked arity, parameter types,
return types, and forward signatures. Non-void functions require statically
recognized returning paths. Only `int -> float` is an implicit floating conversion;
float-to-int and implicit trit/packed-word-to-float conversions are rejected.
Native aggregate function ABIs and native global captures are rejected, rather
than lowered to guessed LLVM signatures. Unsupported native matrix operations
and aggregate reassignment are diagnosed.

Top-level return values must be integral (or void). The native entry ABI returns
`i32`; operating systems further restrict process status, commonly to 8 bits on
POSIX. Full scalar-range return values are therefore **not** a portable process
communication channel. Use output or an explicit ABI instead.

Analysis precedes program effects. Failed interpreter execution restores root
bindings. Already printed output and other external effects are not rolled back;
this is not a general transactional-I/O guarantee.

## 3. `.bt float` is binary64, not BTFP

Language `float` means host IEEE-754 binary64 on the validated platform, lowered
to LLVM `double`. Basic operations use ordinary binary64 rounding; no `fast-math`
reassociation is requested. `sqrt`, `abs`, and `fma` use the standard host/LLVM
operations; `fma` denotes fused binary64 multiply-add. Finite literals are parsed
with range checks. Out-of-range or nonfinite literals are diagnosed. Computations
may produce IEEE infinities or NaNs. No portable NaN payload or floating exception
flag preservation is promised.

Interpreter formatting uses 17 significant digits and a `[float64]` label.
Native printing currently uses six fractional digits. Printed text is not a
bit-exact cross-backend oracle; tests use explicit tolerances and independent
expected values. Target libm implementations and rounding-environment changes
remain outside the current cross-machine equivalence claim.

## 4. BTFP library: a separate ternary floating model

`BTFP<W,E>` denotes `M * 3^e`, with integer mantissa `M`, stored signed-int32
exponent `e`, and canonical zero `(0,0)`. Supported template widths are `1..40`.
For a valid normalized nonzero value:

```
(3^(W-1)+1)/2 <= |M| <= (3^W-1)/2
```

`E=0` means the full **finite int32 exponent domain**, not unlimited precision.
For `E>0`, the requested exponent range is also bounded by `(3^E-1)/2` and by
int32 storage. Thus `E>=21` imposes no additional bound beyond int32.
Exponent sums, differences, alignment, and normalization use wide intermediates;
nonrepresentable stored exponents raise `overflow_error`, not signed C++ overflow.
Operations that can fail are not `noexcept`.

Configured exponent domains now use **checked rejection for both overflow and
underflow**, consistently in constructors and arithmetic. The earlier legacy
clamping policy could turn a huge positive input into a small value or zero and
has been removed. A nonzero normalized result outside the configured range raises
`overflow_error`; it does not silently saturate or flush to zero. BTFP has no
gradual-underflow/subnormal encoding. Canonical zero remains representable.

Normalization and multiplication now choose the nearest member of the **whole
normalized representable set**, with halfway ties away from zero. Mantissa-binade
boundaries have a two-unit gap: for `W=1`, input 5 rounds to 3, while the midpoint
6 rounds to 9. Rounding a quotient and immediately switching binades was not
sufficient. The implementation retains the original widened magnitude and clamps
the final rounded candidate within the selected binade.

`from_ratio(n,d,e)` rounds the exact rational `(n/d)*3^e` directly; `div` uses
this operation without intermediate guard-digit rounding. All widths **1..40**
are now supported for division. Full-int64 ratio operands, including INT64_MIN,
are handled by unsigned magnitude arithmetic. Midpoint ties are explicit.
Square root compares the original integer radicand with the final mantissa
midpoint, rather than first rounding an integer root and then rounding again.
For nonzero exact-normalization inputs, `3^(1-W)` is a conservative relative
rounding-error envelope, checked separately in the independent reference suite.
This is not a universal formal proof of the entire C++ numeric implementation.

Addition/FMA have finite, checked signed-i128 alignment capacity; overly wide
alignments raise an error instead of wrapping. **FMA no longer pre-rounds a small
addend:** accepted nonzero alignment paths retain the exact product plus addend
before final rounding, and excessive alignments are explicitly rejected. For
product exponent difference `d>=0`, capacity is `d<=max(0,78-2W)`; for `d<0`,
`-d<=78-W`. Zero-product/addend cases bypass unnecessary alignment. Square-root
scaling still requires `W+S<=81`; W=40 sqrt is explicitly rejected. Capacity errors
can occur even when arbitrary-precision evaluation could represent the answer.

`test/numeric_reference.py` compares exact Fraction distances across neighboring
binades, independently of the C++ algorithm; it covers 15,072 rational/BTFP cases.
The C++ BTFP32 regression also searches candidate binades by exact distance instead
of treating the old sequential normalizer as a mathematical oracle.

Odd-power integer division has no exact half ties for integer numerators. That
fact is not a distribution-independent statistical zero-bias theorem.
BTFP-to/from-binary64 conversion remains approximate; finite subnormal imports are
supported. Nonfinite imports and invalid sqrt/division domains raise errors.
Transcendental/Householder routines remain approximations. `.bt float` remains
ordinary binary64 and does **not** call these ternary library operations.

### Exact rational library

`include/Ternary/Math/Rational.hpp` supplies `Rational<W>` for widths 1..40.
Canonical values have a positive denominator, coprime components, and zero `0/1`.
Construction and arithmetic reduce by the exact gcd **before** final component
range checks. Add/subtract/multiply/divide/compare/nearest-round use widened
integer arithmetic, not floating-point surrogates. A zero denominator/divisor is
a domain error; an unrepresentable reduced component is an overflow error.
Two width-40 cross-products and their sum fit signed i128; the corresponding
capacity obligation is checked in Rocq and Z3. This is a reusable C++ library API,
not a new `.bt` source type or implicit change to binary64 arithmetic.

## 5. Sparse integer algebra and ABI

A CSR matrix has `rows+1` monotone row offsets, first offset zero, last offset
`nnz`, valid columns `< cols`, and enough sign bits. Sign bit 1 means coefficient
`+1`, bit 0 means `-1`. Offsets/columns use uint32 storage and `nnz<=UINT32_MAX`.
For each row, `y[r] = sum(sign[k] * x[column[k]])` over that row's entries.

Checked C++ integral SpMV/SpMM and the checked C ABI accumulate in signed i128,
then range-check the destination integer type. At most `UINT32_MAX` 64-bit terms
fit comfortably in i128. Checked C++/C outputs are copied only after all result
checks succeed, including when buffers alias. Generated LLVM SpMV also uses i128
and checks int64 results, but may write earlier internal result rows before a
later error; its caller stops on the error state. Matrix result components have
an int64 domain, distinct from scalar `I`.

`bt_csr_spmv_checked` accepts explicit buffer lengths. Raw-pointer APIs still
require actual allocated, live buffers of the claimed lengths; no C function can
validate arbitrary forged pointers. The old CSR entry point remains a
compatibility API without independently knowable allocation lengths. Rust marks
raw-pointer CSR access unsafe.

C ABI exceptions are caught and mapped to thread-local status/message queries.
Neutral return values alone do not indicate success: callers must inspect status.
The Python wrapper's self-test is not a guarantee that every convenience method
automatically checks every ABI status. Allocation failure is a reported runtime
failure, not proof that arbitrary-sized workloads can be executed.

## 6. Mixed radix and quantized tensors

A tri-radix scalar denotes `M * 2^e2 * 3^e3 * 5^e5`. The finite-float importer
stores a binary32 significand exactly as an integer and a power of two, optionally
factoring threes. Signed zero distinctions and nonfinite IEEE values are not
encoded. Decimal imports require coefficients in the 39-trit scalar domain;
larger coefficients are rejected rather than truncated. Conversion back to double
is not a universally exact mixed-radix arithmetic engine.

A block tensor quantizes finite binary32 input with a shared positive scale `s`
and `P` balanced trit planes (`1<=P<=16`). A decoded component is approximately
`s*q`, with integer `q` in `[-H_P,H_P]`. In the ideal real-arithmetic, unclipped
nearest-integer model, `|x-s*q|<=s/2`. Actual scale storage, subnormal floors,
clipping, and binary32 reconstruction introduce additional rounding behavior.
Decode clamps to finite binary32 range. No unconditional lossless or zero-bias
claim follows. Storage shape, rail disjointness, padding, scales, and active masks
are validated by checked decoding and encoded GEMM, because storage is public.

Encoded GEMM evaluates products of the **quantized** inputs; it is not exactly
SGEMM of the original input floats. Plane cutoffs introduce additional,
input-dependent error. Error bounds and useful speedups must be measured for
specific data, precision, hardware, and workloads.

## 7. Backend refinement obligations

A backend should preserve accepted-program results, sign-based control flow,
branch laziness, and observable failures under the domains above. Representation,
vector lane count, byte order, alignment, integer overflow policy, and floating
policy must be explicit per backend. Invalid rail states must not be interpreted
as additional trits. Native ternary or mixed-radix hardware requires a separately
specified ABI and instruction semantics, not just a target name.

Currently exercised native execution is Linux x86-64. Assembly emission for
AArch64 and standard RISC-V is not execution on those machines. Generic shader
text or placeholder XBT instruction strings are not program compilation or GPU
execution. The compiler now implements a restricted program-dependent CUDA
int39 map lowering and a real driver runtime; NVPTX emission and device-free
validation have passed, while physical GPU qualification is still pending.
Unsupported general GPU/custom-XBT programs are rejected, not substituted with
templates. Current scope and evidence are recorded in
[STEPS_1_TO_5_STATUS.md](STEPS_1_TO_5_STATUS.md). A portable mathematical
specification is the foundation for more backends, not evidence of untested
hardware support.

## 8. Evidence boundaries

`verify_z3_phase1.py` proves scalar intermediate bounds/range predicates, a
reduced-width accepted-truncation model, integer sign-selection partitioning,
the ideal quantization inequality, and exponent/accumulator capacity models.
`test/phase1_differential.py` independently verifies and executes actual emitted
LLVM through llvmlite, compares it with the interpreter and Python expected
values, and exercises errors and driver safety. C++ tests cover additional
runtime and boundary behavior; sanitizer builds detect executed memory/UB faults.

Neither the solver models nor finite differential tests prove that every compiler
path implements these mathematics. Both Rocq source files have now been checked
through jsCoq 0.17.1's Coq kernel
using an import-prefix compatibility adapter, including theorem assumption checks.
Native Rocq 9 and Lean execution remain unvalidated locally. Full
frontend-to-backend refinement, exhaustive
floating-point equivalence, concurrency proofs, arbitrary malformed public-object
safety, and universal CPU/GPU support remain open validation gates.
