# Stable core semantics 0.1 and mathematical library boundary

Date: 2026-10-29. This is the initial **versioned contract**, not a declaration
that the entire repository, every matrix extension, or every backend is stable.
The full mathematical context is in [MATHEMATICAL_CONTRACT.md](MATHEMATICAL_CONTRACT.md).
Current implementation, reproducible evidence and qualification boundaries:
[STEPS_1_TO_5_STATUS.md](STEPS_1_TO_5_STATUS.md). The later SDK/Vulkan work is
recorded in [R1_R2_DELIVERY.md](R1_R2_DELIVERY.md). The machine-readable operation
matrix is [core-profile-v0.1.json](../specs/core-profile-v0.1.json).

## 1. Supported core and error policy

The audited common interpreter/native profile consists of scalar declarations,
lexical scopes, assignment, scalar functions/forward calls, checked integer
`+ - *`, scalar negation, digit-wise logic, comparisons, binary64 math intrinsics,
sign-based `if`/`while`, lazy three-way selection, scalar printing, and integral
program exits. Both paths must reject invalid programs before their execution.
A runtime error is observable failure, never an arbitrary replacement numeric value.

- `int`: 39-trit symmetric integer range `[-2026277576509488133,2026277576509488133]`.
- `trit`: only `-1,0,1`. An explicit trit binding/parameter/return may receive
  another trit or a statically known trit-valued literal (including unary sign).
  Arbitrary integer narrowing is rejected; a future checked conversion must be
  explicit. Inferred bindings of single trit literals widen to `int`, so ordinary
  counters initialized to zero do not acquire an accidental constrained type.
- Decimal zero is treated as an integer by semantic analysis even though the
  lexer retains its legacy trit token. Other decimal integer tokens are integers.
- `tritword`: a positional value limited to 39 trits in the common scalar profile.
  Its arithmetic is checked scalar arithmetic, not unchecked packed arithmetic.
- `packedword64`: up to 64 independent trit lanes, **not** a scalar integer.
  Long literals infer this type. Interpreter printing, rail negation and digit
  logic are permitted; arithmetic, ordering conditions, and implicit scalar
  conversions are rejected. Native support for this type is not implemented.
  Literals wider than 64 trits are rejected on both paths.
- `float`: binary64, distinct from library BTFP. Only `int -> float` is an implicit
  floating conversion. A binding's type is preserved through assignment, and a
  mixed `select` result is promoted even when its selected branch is integral.
  Builtin `sqrt` and `fma` explicitly define scalar-to-binary64 input conversion.
- Predicate results are `+1/-1`; spaceship results are `-1/0/+1`. Float equality
  with NaN is false, inequality is true, ordered comparisons are false, and float
  spaceship with NaN is zero. `select(NaN,...)` chooses its zero branch; NaN is
  false in `if`/`while`. Signed zeros are equal and choose the zero branch.
- `if`/`while` mean **positive**, not merely nonzero. Only one `select` branch is
  evaluated. Native and interpreter external output formatting still differs.

No optimizer may reassociate checked arithmetic without proving that observable
intermediate failures are preserved. For example, `(H+1)-1` fails whereas
`H+(1-1)` succeeds. Likewise, strict binary64 arithmetic must not be silently
changed to fast-math or a ternary approximation.

The existing source notation allows compact positional trit literals. Use spaces
around arithmetic operators to avoid their legacy lexical interaction with runs
of `+`, `-`, and `0`. Free-form numeric syntax and a redesigned unambiguous packed
literal spelling are not part of this stabilization claim.

### Hardware intrinsics

`planar_dot64`, `btha_sum_p/n`, and `btha_carry_p/n` now have both interpreter and
native implementations. Their arguments are language integers interpreted as
64-bit two's-complement bit patterns. The bit formulas are defined for all such
patterns; interpreting them as ternary hardware operations additionally requires
disjoint input rails. Results have language `int` type. Mask results outside the
39-trit scalar range raise errors rather than being silently recoded/truncated.
A true unsigned bitmask/rail ABI belongs in a later dedicated IR/type extension.

### Outside the stable common profile

Dynamic matrix typing/shape inference, general mutable aggregates, aggregate
function ABIs, global captures in native functions, unrestricted packed-word
native execution, parallel language effects, imports under hostile resource
loads, BTFP transcendental correctness, and GPU execution are not certified here.
Unsupported operations must be rejected, not substituted with an approximate
implementation. Existing interpreter matrix extensions remain experimental.

Matrix constructors now reject negative/overflowing dimensions, excess values,
and nontrit values before narrowing. Ternary matrix addition/subtraction returns
integer components, not sign-clipped trits. Canonical C++ CSR construction rejects
duplicate coordinates and requires strictly increasing columns per row, so `get`
and multiplication cannot disagree about duplicate entries. The C ABI's checked
CSR sum interface still permits duplicates as additive entries; it has a separate
contract and does not expose the canonical C++ lookup operation.

## 2. Reusable integer mathematics library

`include/Ternary/Math/Integer.hpp` is hardware-independent at the value-contract
level. It implements:

| API | Contract |
|---|---|
| `BalancedInteger<W>`, `1<=W<=40` | Strong value type, bounds `±(3^W-1)/2`, canonical zero |
| Unary negation, `+`, `-`, `*` | Exact mathematical result or overflow exception |
| `/`, `%` | Euclidean quotient/remainder, not C truncating-remainder semantics |
| `divmod_euclidean(a,b)` | `a=b*q+r`, `0<=r<abs(b)` for either divisor sign |
| `round_quotient(a,b)` | Nearest integer, exact half-way ties away from zero |

Division by zero raises `domain_error`. Quotients outside signed int64 raise
`overflow_error`, including `INT64_MIN/-1`. The full-int64 helpers widen before
negating, dividing, or correcting a remainder, including `INT64_MIN` divisors.
The strong balanced type applies its own narrower bounds to final values.
The compiler's `ternary_numeric_contract.hpp` delegates its bounds/checking to this
library, so this is not an unrelated demonstration implementation.

For nonzero `b`, Euclidean quotient/remainder are uniquely specified by the two
conditions above. Nearest rounding is specified independently of host floating
point: let `n=abs(a)`, `d=abs(b)`, and
`u=floor((2*n+d)/(2*d))`; return `u` or `-u` according to the quotient sign.
The C++ implementation uses an equivalent widened integer expression.

These library `/` and `%` APIs do **not** add new source-language tokens/operators;
that language extension requires a separate grammar, type, and lowering change.
No claim is made that a finite-width library can represent every mathematical
integer or that these APIs constitute every kind of mathematics.

## 3. Rocq proof foundations and actual evidence

`proofs/rocq/CoreMath.v` defines unbounded mathematical integer models for:

- Trit valuation, positional decoding, width capacity, and rail-independent
  negation; proofs of decoding bounds and value preservation.
- Checked integer acceptance/rejection, and exactness of accepted add/sub/mul.
- Euclidean division for both divisor signs; reconstruction, remainder bounds,
  zero-divisor rejection, and uniqueness.
- Thunk-based integer sign selection and the three sign cases.
- A counterexample disproving unconditional reassociation of checked arithmetic.

The existing `BalancedTernaryFoundations.v` covers rail encoding, BTHA/BTFA,
packing bounds, and integer reversible updates. These are model-level theorems,
not proofs that the C++ compiler or machine hardware implements those models.

`ImplementationBridge.v` adds mathematical counterparts of the actual truncating
quotient/remainder correction, the C++ nearest-quotient formula, rational i128
capacity/reduction, and ternary-binade midpoint comparisons.

`CoreLowering.v` adds multiplication-precheck, interval, intermediate-width and
observable-error preservation obligations used by the core CPU/CUDA lowering.

**Executed locally:** all four files' proof bodies and theorem assumption checks
passed using jsCoq 0.17.1's Coq kernel: **28 + 18 + 11 + 9 = 66 results**.
Only the standard-library import prefix was adapted from `Stdlib` to `Coq` for
that compatibility check; proof bodies were unchanged. This is genuine Coq-kernel
checking, but it is **not native Rocq 9 execution**. The wrapper checks explicit
completion markers and error diagnostics because jsCoq's CLI can exit zero after
an error. It also rejects source-level unchecked declarations and inspects
transitive theorem assumptions.

`round_quotient` is tested against independent Python arithmetic; the bridge now
proves the widened formula equivalent to the mathematical rounding model and its
half-interval/tie properties. Euclidean correction likewise has an algorithm-level
proof. Full C++ refinement and executable proof extraction remain separate gates.
Binary64 semantics must eventually use an IEEE model, not an invented rational
substitute. BTFP/quantizer error bounds need their own separate developments.

Reproduce the optional local compatibility check:

```sh
npm install --prefix build/proof-tools --no-audit --no-fund jscoq@0.17.1
python3 test/check_rocq.py \
  --jscoq build/proof-tools/node_modules/jscoq/dist-cli/cli.cjs
```

Without native Rocq or an explicitly selected jsCoq CLI, CTest reports `Core_Rocq`
as a **skip**. `BT_JSCOQ_CLI` selects the same compatibility checker in CTest.
`.github/workflows/core-foundation.yml` adds a native Rocq 9.1.1/stdlib 9.1.0 job,
separate from C++ Debug/Release/sanitizer jobs. That workflow has not been run on
GitHub in this session. Local compatibility proof checks do not replace its gate.

## 4. Whole-program typed scalar IR

`include/Ternary/IR/Core.hpp` and `Program.hpp` define backend-independent typed
expressions **and whole programs**: stable symbol IDs, typed function signatures,
forward calls, lexical blocks, bindings/assignments, loops, returns, conditionals,
and lazy three-way selection. Explicit integer casts and integer-to-float nodes
make promotion visible rather than dependent on a runtime representation.
Verification checks symbols, signatures, arity, scope, types, return paths,
statement shapes and expression/statement nesting bounds.

Effects distinguish `Pure`, `MayFail`, and `Observable`. Print and potentially
observable user calls cannot be treated as speculatable arithmetic; effect union
never downgrades observable operands. `SignSelect` alternatives remain lazy.

`ternary_ir_lowering.hpp` performs checked AST-to-core lowering.
`include/Ternary/IR/Execute.hpp` executes the verified program without evaluating
AST nodes. Run it with:

```sh
build/phase1-release/bt-compiler --core-run program.bt
```

The executor supports checked int39 arithmetic/comparison, constrained trits and
positional scalars, binary64 promotion/math, digitwise ternary logic, planar/BTHA
intrinsics, functions, recursion, scopes, loops and lazy selection. Its CLI profile
is scalar-only (the native frontend support rules apply). A 10-million-operation
execution budget and 512-call depth limit bound runaway execution; the C++ API
accepts a different operation budget, with zero meaning unlimited.

All **525** baseline differential values now also check this IR route. A separate
seeded generated-program test checks **211** whole-program outputs against native
LLVM, the AST interpreter and an independent Python calculation, including loop
bodies, shadowing, recursion, forward calls, promotions and block selection.

The default supported scalar LLVM route now consumes this verified core through
`include/Ternary/IR/LLVM.hpp`, including functions, checked failures and lazy
control flow. `--legacy-llvm` explicitly selects the older AST-native compatibility
route for aggregate extensions; there is no automatic aggregate fallback.
`--core-opt` runs conservative core transformations and interval-based narrowing
before execution/emission. It also selects core execution for `run`.

A pure loop-free `(int)->int` CUDA map subset now emits program-dependent LLVM/PTX
and has an actual CUDA-driver runtime. Offline compilation/emulation is tested;
physical GPU qualification remains pending. Upstream MLIR integration is not
completed by this standalone implementation. See the current status report for
limits, commands, support and measured performance.

## 5. Stabilization status

The concrete silent semantic mismatches found in this audit have fixes and
regression coverage. There is no known outstanding silent mismatch in the audited
scalar cases described here. This is an issue-status statement, **not a proof
that the supported language has no bugs**. Continuous grammar/IR fuzzing, broader generated programs, formal refinement,
backend-wide resource policies, and real device qualification remain mandatory before a production-readiness claim.
