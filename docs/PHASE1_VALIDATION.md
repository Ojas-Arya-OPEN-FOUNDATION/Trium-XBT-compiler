# Phase 1 remediation and validation record

> Subsequent focused numerical delivery: [FLOATING_POINT_FOUNDATION.md](FLOATING_POINT_FOUNDATION.md).
> Historical limits/counts below describe the earlier snapshot; W40 sqrt and
> wider FMA alignment are now implemented and separately validated.

> Historical foundation-pass record. For the later shared LLVM/CUDA/optimization
> implementation and current validation, see [STEPS_1_TO_5_STATUS.md](STEPS_1_TO_5_STATUS.md).
> Counts and measurements below describe the earlier pass, not the current build.

Date: **2026-10-29**. Branch: `arena/f3c30e8e-trium-xbt-compiler`.
Scope: standalone compiler/runtime safety and common mathematical semantics.
README deliberately unchanged. Subsequent core stabilization details are in
[CORE_SEMANTICS_V0_1.md](CORE_SEMANTICS_V0_1.md). Work is saved on the session branch; this record
is not a release certification or an end-to-end formal-equivalence proof.

## Changes implemented

- Mandatory semantic analysis before interpretation and LLVM generation: checked
  names, arity, signatures, conversions, scopes, return paths, and backend support.
  Native code rejects unsupported aggregates/global captures instead of guessing.
- Lexical interpreter frames, forward function registration, root-binding rollback
  on runtime failure, and propagation of program exit status.
- A symmetric 39-trit scalar integer domain, checked i128 arithmetic, and generated
  error propagation across native functions. Lazy typed sign-selection lowering.
- Common binary64 `.bt float` semantics, standard math intrinsics, checked literal
  parsing, and honest interpreter formatting labels.
- Argv-only process execution and private temporary directories; unsupported GPU,
  custom-XBT and invalid command-line modes no longer report fabricated success.
- Escaped LLVM source/debug filenames; native local stack allocations are hoisted
  out of loops. Sparse descriptors use distinct internal identities for scopes.
- BTFP exponent arithmetic widened and checked; throwing arithmetic is no longer
  hidden behind `noexcept`. Wide accumulator limits, huge rounding shifts,
  conversion subnormals, exponential range, and W=1 multiplication are guarded.
  A premature FMA dominance shortcut was corrected and given a regression case.
  Follow-up work replaced binade-boundary rounding, direct rational division and
  final sqrt midpoint comparison; FMA now rejects inexact alignment shortcuts.
- Checked sparse integral SpMV/SpMM use i128, validated public CSR storage, explicit
  destination range checks, and transactional C++ output. Native LLVM SpMV now
  checks i128-to-int64 results and propagates failure.
- Thread-local C ABI error reporting and centralized exception boundaries; an
  explicit-length checked CSR API. Rust raw-pointer CSR access is unsafe.
- Reciprocal-square-root extreme-range scaling, exponential conversion guards,
  subnormal-safe quantizer scales, tensor storage/padding/mask checks, and GEMM
  dimension/output guards. Nonfinite tri-radix imports and oversized decimal
  coefficients are rejected rather than silently reinterpreted.
- Out-of-tree test tool paths and failure propagation, explicit dependency skips,
  Release-active assertions, and a clean-build `check-all` target covering every
  registered executable. Added independent differential, safety, FFI, and Z3 tests.

The follow-up adds a reduced exact-rational library, whole-program scalar core IR
and executor, generated-program testing, algorithm-level proofs, and private
mangled native function symbols that cannot interpose host-library names.
Current measured performance is in [FOUNDATION_PROGRESS_2026_10_29.md](FOUNDATION_PROGRESS_2026_10_29.md).

Mathematical definitions and deliberate limitations are in
[MATHEMATICAL_CONTRACT.md](MATHEMATICAL_CONTRACT.md).

## Observed results on the final tested source

| Validation | Result | What it establishes |
|---|---|---|
| GNU 12.2 Debug, full CTest | **28 passed / 29 registered**, native-Rocq dependency skip | Registered standalone tests and dependencies exercised |
| GNU 12.2 Release, `check-all` from a newly configured build directory, then rerun after final edits | **28 passed / 29 registered**, native-Rocq dependency skip | Build dependencies and tests work with optimized code and active assertions |
| GNU 12.2 Debug with ASan + UBSan | **28 passed / 29 registered**, native-Rocq dependency skip | No sanitizer findings on exercised instrumented paths |
| Independent LLVM/interpreter/Python differential | **525 expected values passed across native/AST/core/Python**, plus diagnostic/overflow/driver cases | Actual emitted IR verifies and executes through a separate LLVM toolchain |
| Phase 1 Z3 obligations | Passed | Stated integer, selection, ideal quantization, exponent, accumulator models |
| Eight additional Z3 scripts | All passed on rerun (9 total) | Their existing finite/SMT obligations remain green |
| Native frontend suite | Passed using bundled Clang 22.1.8 | Actual generated LLVM compilation, loading, and execution on this x86-64 host |
| Codegen suite | All 7 files passed | Includes real x86-64, AArch64, and standard RISC-V assembly emission |
| jsCoq 0.17.1 proof checking | **57 results checked**, no theorem axioms reported | 28 foundations + 18 core + 11 algorithm-bridge results; import-prefix adapter, not native Rocq 9 |
| Whole-program generation | **211 outputs passed** across native/AST/core/Python | Seeded loops, scopes, calls, recursion, selection and native symbol-isolation cases |
| Exact rational/BTFP oracle | **15072 cases passed** | Independent exact Fraction distance oracle, widths 1..40, binade boundaries and full-int64 ratios |
| Integer-math rational oracle | **3289 cases passed** | Full-int64 sign/extreme cases against Python bigint/Fraction |
| `git diff --check` | Clean | No patch whitespace errors |
| README diff | Empty | User's README restriction preserved |

### Important qualifications

The Clang tool was obtained from the PyPI `ziglang` distribution, not an official
standalone Clang install. A **local, untracked** compatibility adapter invokes
`zig cc`, removes unsupported `-x ir` for `.ll` inputs, and translates
`riscv64-unknown-linux-gnu` to Zig's `riscv64-linux-gnu`. Compiler failures are not
suppressed. Native and cross-assembly results must be understood in that toolchain
context; no other machine was used for execution.

The independent oracle uses llvmlite 0.50.0 to parse, verify, and emit objects from
actual generated LLVM. It is independent of the Clang/dlopen adapter, but its
finite test set is not exhaustive. Float comparisons respect the current native
six-decimal print format; they are not bitwise comparisons of all binary64 values.

ASan/UBSan instrument the C++ compiler, tests, and runtime. Generated LLVM machine
code is not automatically sanitizer-instrumented. The Python FFI test preloads
GNU ASan because Python itself is an unsanitized host. Leak detection is disabled
**only for that Python-host test** to avoid attributing Python-host allocations
to the runtime; native C++ test runs used `detect_leaks=1`. UBSan used
`halt_on_error=1` throughout the sanitizer CTest run.

Optional upstream LLVM/MLIR integration was not built; the standalone MLIR-related
tests do not substitute for that integration. Rust was not compiled in this
sandbox. Native Lean/Lake and Rocq were unavailable. All three Rocq source files were instead
checked by jsCoq 0.17.1's Coq kernel with an import-prefix adapter; native Rocq 9
execution remains an open local gate. GPU runtimes,
actual AArch64/RISC-V hardware, and native ternary hardware were not available.

## Reproduction

Use a Python virtual environment; the system Python may enforce PEP 668.
Optional model/differential dependencies are pinned in
`requirements/verification.txt`. CMake >=3.20, Ninja, a C++20 compiler, and POSIX
process/dynamic-loading facilities are needed by the validated native path.
Make a compatible `clang` discoverable on PATH for the native/codegen suites.
Missing optional tools must be reported as skips, not counted as proof success.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install cmake ninja -r requirements/verification.txt

cmake -S . -B build/debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBT_ENABLE_MLIR=OFF
cmake --build build/debug --target check-all -j 2

cmake -S . -B build/release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBT_ENABLE_MLIR=OFF
cmake --build build/release --target check-all -j 2

cmake -S . -B build/sanitized -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBT_ENABLE_MLIR=OFF -DBT_ENABLE_ASAN=ON
cmake --build build/sanitized -j 2
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/sanitized --output-on-failure

python test/phase1_differential.py build/debug/bt-compiler
for script in verify_z3*.py; do python "$script" || exit 1; done
```

Local validation logs are under `build/review/`; build outputs/logs are deliberately
ignored rather than added to Git. The virtual environment and local Zig adapter
are environment dependencies, not repository deliverables. `BT_RUNTIME_LIBRARY`
selects the Python wrapper's library; CTest sets it to the just-built shared runtime.

## Remaining gates before calling Phase 1 fully production-ready

1. **BTFP numeric specification closure.** The legacy bounded-exponent clamping
   policy has been replaced with checked rejection. Establish exact rounding/error contracts for every
   operation/precision, and use a high-precision oracle for cancellation, exponent
   boundaries, transcendental functions, and wide FMA. Currently unsupported exact
   accumulator ranges intentionally fail. They do not imply arbitrary precision.
2. **Full language/backend refinement.** Model the typed AST, effects, exceptions,
   packed words versus scalars, and aggregate dimensions. Prove or systematically
   check lowering against that model. Existing solver scripts prove models, not
   extraction of every C++ implementation path. Execute Lean and native Rocq proofs with
   pinned official toolchains and CI before citing those native-toolchain results. Coq-kernel compatibility checks
   now exist, but do not prove compiler refinement.
3. **Fuzzing and resource limits.** Add continuous parser/AST/IR/malformed-storage
   fuzzing, bounded recursion/compile-time work, import-cycle tests, allocation
   budgets, and long-running session stress. Sanitizer success is finite evidence,
   not proof that arbitrary public mutable objects are safe in every library API.
4. **ABI/client hardening.** Compile/test Rust, make Python status handling uniform,
   audit every legacy pointer-only interface and optional API, specify ABI/version
   compatibility, and test allocation failures and races under supported platforms.
5. **Portability qualification.** Execute on actual supported CPU architectures,
   build upstream LLVM/MLIR integration, document toolchain/OS support, and add
   continuous integration. Current host process utilities are POSIX-oriented;
   portable Windows support cannot be inferred from generic C++ kernels.
6. **GPU and ternary backend execution.** Implement program-dependent GPU lowering,
   memory/layout and capability contracts, runtime launching, and differential
   tests on real devices. Define future ternary/mixed-radix instruction semantics
   before implementing a custom backend. Shader templates and cross-assembly are
   not evidence of universal execution or performance.

This closes many critical defects and establishes a substantially stronger,
repeatably tested Phase 1 baseline. It does **not** close all the gates above, and
scaling/performance claims should wait for their corresponding evidence.
