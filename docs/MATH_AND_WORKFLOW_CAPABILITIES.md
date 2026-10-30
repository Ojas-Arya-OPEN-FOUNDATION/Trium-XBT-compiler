# Mathematics, binary workflows, AI/simulation and dual-rail accuracy

Date: 2026-10-30. Current executable SDK scope, not an all-workload claim.
The [focused floating-point milestone](FLOATING_POINT_FOUNDATION.md) adds all-width
CPU BTFP primitives, exact-reference validation and explicit SDK access; it does
not add GPU floating-point/tensor execution.
The [SDK 1.1 execution milestone](SDK_1_1.md) adds reusable native CPU execution
for pure loop-free scalar functions, mixed typed invocation, discoverable
signatures and reusable Vulkan resources. This makes repeated application calls
practical without turning the foundation into an AI or simulation framework.

## Can this mathematical core serve modern binary workflows?

**Yes, for declared numerical profiles.** The mathematics runs on conventional
binary CPUs; ternary hardware is unnecessary. A ternary representation is a
choice of number system/storage, not a reason to reject ordinary binary inputs.
However, representation correctness, numerical approximation, supported algorithms
and performance are four separate questions.

| Workflow | Available now | Not implied |
|---|---|---|
| Exact bounded integers | Checked balanced integers, int39 source/kernel profile, widths 1..40 in library | Arbitrary precision or unlimited intermediate capacity |
| Exact fractions | Reduced rational library, exact decimal parsing within documented capacity, canonical wire format | Every rational fits the bounded numerator/denominator |
| Ordinary floating operations | Source binary64 arithmetic/functions; separate BTFP library with declared rounding/capacity rules | Certified correctly rounded transcendental functions or identical IEEE/BTFP representations |
| Integer/ternary-weight matrix work | New CPU row-major int39 matrix multiply with checked products, increasing-k checked sums and transactional output; existing library packed contractions | GPU GEMM, production BLAS throughput, arbitrary dtype/tensor layouts |
| GPU elementwise work | Pure int39 map source compiled to validated SPIR-V and executed through Vulkan; CPU/optional CUDA references | Arbitrary graph compilation, GPU rational/BTFP/tensor execution |
| Simulation | Scalar loops/functions, binary64 operations and library numeric primitives can implement scoped simulations | A ready-made PDE/physics/solver framework, stability guarantees or arbitrary-precision integration |
| AI applications | Useful foundations for exact low-bit/ternary-weight experiments and host integration | A complete training/inference engine, FP16/BF16 GPU GEMM, autograd or automatic PyTorch/JAX compilation |

For AI expansion, the next contracts are shape/layout/strides, quantization and
scaling, accumulator dtype/range, reduction order, batching, and framework/device
ownership. For simulation, explicit precision/error goals, algorithms, time-step
stability and reproducibility matter. These are not automatically solved by
converting binary data to ternary.

## What was strengthened in this implementation

- `Math/Interchange.hpp`: strict bounded decimal/integer/rational parsing, exact
  reduction, nonthrowing result adapters, checked dual-rail conversion, canonical
  dual-rail/rational little-endian wire formats and checked dense matrix multiply.
- `Math/Float.hpp`: exact decimal → bounded rational → existing BTFP ratio rounding,
  with no binary64 intermediate. Decimal input currently allows 18 fractional
  places; accumulator/input-component capacities are explicit. Result rounding
  is still BTFP rounding, not an assertion that all decimal inputs are exact BTFP.
- SDK matrix multiply checks every int39 product and each increasing-k sum.
  A later cancellation does not erase an earlier overflow. Host output is committed
  only after success, including when input/output storage overlaps.
- Tests cover all 40 widths, full-int64 extrema, more than 200,000 scalar/rail
  roundtrips, tractable small-width exhaustive encodings, malformed inputs,
  canonical serialization and 1,000 matrix reference cases.

The exact-decimal parser intentionally has a bounded accumulator and rejects
unsupported inputs rather than silently approximating. It is not a general
unlimited decimal parser. Existing source `float` semantics are unchanged.

## Dual rail: what it encodes and what needed checking

For each trit, use `(positive, negative)` bits:

| Trit | Positive | Negative |
|---|---:|---:|
| +1 | 1 | 0 |
| 0 | 0 | 0 |
| -1 | 0 | 1 |
| Invalid | 1 | 1 |

**The representation itself does not need a fourth state or a redesign.** What
needs strengthening is its checked boundary and the meaning of data encoded in it.
New `encode_integer`/`decode_integer`, text and wire APIs explicitly check:

1. Disjoint rails; overlapping bits never become another number.
2. Declared width 1..64 and no silently ignored high bits.
3. Signed conversion, including INT64_MIN and INT64_MAX without signed overflow.
4. Positional-value overflow: a 64-trit word can represent far more than int64,
   so not every rail word can be converted to int64.
5. Canonical bytes/version/reserved fields and exact roundtrips.

A packed 64-trit vector, a positional int39 and a binary uint64 bitmask are distinct
objects. Dual rail consumes **two bits per trit**; it is not automatically smaller
or faster for every datatype or application. Base-243 packing and vectorization
have different tradeoffs and need separate measurements.

Legacy low-level helpers still have valid-input preconditions. New checked
interchange/SDK APIs do not silently bless invalid raw structures. The portable
Vulkan int39 representation is a **pair of uint32 words holding one signed
integer**, not the positive/negative dual-rail representation; this distinction
is intentional.

## Exact versus approximate conversion

- A representable integer or a trit value -1/0/+1 converts exactly.
- Rational `1/2` is exact in the rational library, but cannot be represented by a
  finite base-3 fractional expansion. Finite BTFP must round it.
- General FP32/FP64 activations converted to a fixed ternary format therefore need
  an explicit rounding/quantization/scaling profile and accuracy measurements.
- Keeping source binary64 data as binary64 is legitimate; there is no requirement
  to quantize every binary workflow into ternary to use the compiler/library.

Thus the right promise is **exact encoding where representable, documented
rounding otherwise, explicit rejection outside capacity**—not automatic lossless
conversion of every real number into a finite ternary word.

## Example: currently executable host integration

```python
# PYTHONPATH=$PWD/ffi/python
# BT_SDK_LIBRARY=$PWD/build/sdk/libtrium_sdk.so
import trium
with trium.Context() as context:
    assert context.decode(*context.encode(-(2**63))) == -(2**63)
    result = context.matmul([1, 2, 3, 4], [5, 6, 7, 8], 2, 2, 2)
    assert result == [19, 22, 43, 50]
    with context.compile('func transform(x:int)->int{return x*3;}') as module:
        values, metrics = module.map('transform', [-2, 0, 2], backend='cpu')
        assert values == [-6, 0, 6]
```

This is real CPU integration. `backend='vulkan'` explicitly requests the Vulkan
path and fails clearly if unavailable; it never disguises CPU fallback as a GPU.
Physical-device instructions are in [HARDWARE_GPU_QUALIFICATION.md](HARDWARE_GPU_QUALIFICATION.md).
