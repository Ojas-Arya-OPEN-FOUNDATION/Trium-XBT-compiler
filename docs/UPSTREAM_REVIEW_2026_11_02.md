# Latest upstream review — 2026-11-02

> **Update after the deeper audit.** The latest upstream commit is now
> [`5a33d9c`](https://github.com/Ojas-Arya-OPEN-FOUNDATION/Trium-XBT-compiler/commit/5a33d9ce7e75e531d8be27574b3d7713a45a2eda),
> which changes only the CI workflow; source findings remain those of `82ae251`.
> A deeper, reproduced audit is in
> [DEEP_AUDIT_2026_11_02.md](DEEP_AUDIT_2026_11_02.md), with
> [30 machine-readable findings](evidence/2026-11-02-deep-audit/findings.json)
> and logs for memory corruption, thread-pool crashes, FFI loading, numerical
> edges, proof gaps and CI failures. Repairs for those findings are recorded in
> [AUDIT_FIXES_2026_11_02.md](AUDIT_FIXES_2026_11_02.md).

## What was checked

Fetched `origin` and deepened the shallow history, without merging, resetting,
checking out another branch or editing application sources. Reviewed upstream
`main` at [`82ae25146453e57d49869933a1e2dc0ca9e18bb7`](https://github.com/Ojas-Arya-OPEN-FOUNDATION/Trium-XBT-compiler/commit/82ae25146453e57d49869933a1e2dc0ca9e18bb7).
Latest commit date: November 2, 2026, 14:16:25 IST.

The session branch remains `arena/f3c30e8e-trium-xbt-compiler`, at base
`a7e456e7ae46878437ef2ccc4bc9ba5532840bbe`, with its existing uncommitted work
preserved. Upstream has **55 commits after that base**. This is not 55 commits of
entirely absent functionality: comparing upstream blobs to actual local files
(including untracked files) found **260 identical, 19 different and 18 absent**.
The comparison preceded creation of these review documents.

Much of upstream now commits the foundation/SDK/numerical work already present
in this session. The most recent additional differences include performance
changes, safety checks, portability adjustments and expanded numerical SMT work.
The session's chip-side research plan and source inventory are not in this
upstream snapshot. `hw/` has no changes between the session base and latest main.

## Improvements present upstream

| Area | Changes and practical value |
|---|---|
| Compiler foundation | Typed whole-program Core IR, scoped symbols, conservative optimization, independent execution and LLVM/native lowering. These make backend semantics and tests more systematic. |
| Reusable SDK | ABI 1/API 1.2, shared SDK, C/C++ APIs, native runtime, Python integration and Rust SDK bindings. CMake exports installable SDK/math targets and tests relocated consumers. |
| Numerics | Exact finite reciprocal/cube root/hypot/integer powers; single-round reductions and polynomial evaluation; audited CORDIC tables and bounded scratch/resource contracts. These mostly match the already delivered local SDK 1.2 work. |
| Wide arithmetic | `times_pow3()` multiplies in chunks of up to `3^40`; shifts move whole limbs and remaining bits instead of multiplying by two repeatedly. `quotient()` checks whether trial multiplication fits scratch storage. These changes target real arithmetic overhead. |
| Compiler optimization | Attempts to use i64 add/sub instead of i128 and adds a single-trit logic fast path. The add/sub implementation introduces the semantic regression described below; do not qualify its speedup as a correct compiler improvement yet. |
| Boundary hardening | Adds C-ABI null-buffer checks and zero-length handling for SGEMM/RMSNorm/softmax, thread-safe process error formatting and conditional close-on-exec spawn attributes where the platform exposes them. These do not constitute complete buffer-bounds or platform-security qualification. |
| Portability/build | Additional macOS SDK-path handling, quoted CMake include paths, Rust runtime linking/crate-type corrections, and CI changes. macOS-specific behavior was inspected but not executed here. |
| Verification | More reference/integration tests, CI jobs, benchmark targets, and 312 numerical-model SMT obligations in the current numerical script. These are scoped obligations, not 312 complete implementation proofs. |
| GPU infrastructure | Portable GPU build/qualification/benchmark tools and runtime paths. Physical-GPU qualification remains separate; no new device run was performed. |
| Documentation/tool configuration | Specifications, qualification archives, roadmap updates, curated component dossiers and agent/IDE instruction files. Instruction files are configuration/documentation, not evidence of a newly running evolutionary optimizer. |

Notable recent commits: `0bb8d0c` (wide arithmetic), `32b09ba` (compiler
add/sub optimization), `9ee4bb3` (boundary hardening), `3ace7f0` (SMT work),
`82ae251` (SDK evidence and documentation).

## Focused checks performed on isolated upstream code

Built both the unchanged workspace compiler and an exported upstream snapshot
with g++ 12.2.0. Used the existing independent LLVM test adapter with llvmlite
0.50.0 to parse/verify emitted LLVM, produce actual native objects and run them.
Temporary Python dependencies were installed only under ignored build storage.

- **Ordinary arithmetic:** both compilers and interpreters return 12 for `5+7`.
- **Wide arithmetic benchmark:** all 100,000 power-scaling comparisons and
  100,000 shift comparisons agree with the serial reference for its sampled
  valid ranges. Across three trials, ratios of mean baseline/optimized times
  were **12.704x** and **17.059x**, respectively. These are microkernel results
  in this sandbox, not whole-program/SDK speedups or universal hardware claims.
- **Numerical Z3 script:** upstream `verify_z3_numerical.py` passes **312**
  scoped obligations with Z3 5.3.3. Some added checks establish constant chunk
  decompositions and algebraic identities; this does not prove every line of
  the limb implementation, all overflow handling or RTL.
- **Existing backend regression suite:** upstream
  `test/core_backend_regression.py` fails at case 1, the first overflow case.
  The focused execution below explains that failure.

Full new Release/ASan regression suites, macOS, Rust and physical GPU/FPGA tests
were not run locally in this review. Previously archived SDK 1.2 results should
not be described as a fresh qualification of this changed upstream compiler.

## Two correctness issues requiring attention

### 1. Native add/sub silently saturate instead of reporting overflow

Files: `include/Ternary/IR/LLVM.hpp` and `ternary_llvm_codegen.hpp`.

The contract and interpreter still reject results outside the balanced 39-trit
integer domain. New native lowering clamps to the endpoints. With
`H = (3^39 - 1)/2 = 2026277576509488133`:

| Program expression | Local native/reference | Upstream native | Upstream interpreter |
|---|---|---|---|
| `H + 1` | Error, exit 1 | Returns H, exit 0 | Error, exit 1 |
| `-H - 1` | Error, exit 1 | Returns -H, exit 0 | Error, exit 1 |
| `(H + 1) - 1` | Error, exit 1 | Returns H-1, exit 0 | Error, exit 1 |

This changes program meaning and causes backend disagreement. The new arithmetic
benchmark compares two saturating implementations, so agreement in that
benchmark would not validate the existing checked-overflow contract.

**Recommended repair:** retain i64 add/sub where input bounds make the host
operation safe, but branch to the existing error/status path when the result
exceeds the language domain. Do not clamp unless a separately specified
saturating operation is requested. Re-run optimized/unoptimized differential and
error-propagation tests before accepting a compiler performance claim.

### 2. Failed RISC-V compilation can report false success

File: `bt_compiler.cpp`, assembly-emission path.

If the invoked backend returns failure for a RISC-V target, the new fallback
writes a fixed `bt_main` assembly body returning zero, sets success and reports
assembly generation as successful. It does not compile the input program.

Reproduced by substituting a test `clang` that exits 1 to represent a missing
backend, with source `print(42); return 7;`:

- Workspace compiler: correctly reports failure, exit 1, no assembly output.
- Upstream compiler: reports success, exit 0, emits a stub containing `li a0, 0`.

This is a failure-path test, not a physical RISC-V execution claim. Nevertheless,
it directly demonstrates that the produced assembly does not preserve source
semantics. The fallback also assumes RV64 assembly whenever the triple contains
`riscv`, rather than implementing a generally valid target backend.

**Recommended repair:** fail explicitly when no real target compiler exists,
or implement source-dependent lowering and qualify it. A placeholder may be
emitted only as an explicitly labeled non-executable demonstration, not as
successful compilation.

## GitHub CI state

The [latest run](https://github.com/Ojas-Arya-OPEN-FOUNDATION/Trium-XBT-compiler/actions/runs/41415612321)
for this exact upstream commit is **failed**:

- `release`, `debug`, `sanitizers`: failed at `check-all`.
- `rust-client`, `native-rocq`: successful.

Full job-log download failed at the log storage host. The job metadata and
annotations were available through GitHub API. The locally reproduced compiler
regression must not be presented as the proven sole cause of all CI failures.
Likewise, the two successful jobs are useful upstream evidence, but do not make
the entire changed tree qualified.

## Recommendation

Retain the numerical/SDK additions and independently useful wide-arithmetic
optimization. Fix checked-overflow behavior and remove the false-success
RISC-V fallback before treating latest main as a clean foundation for the next
hardware implementation. Then re-run full differential, Release and sanitizer
qualification and investigate any remaining CI failures.

Do not blindly pull over this working tree: upstream already contains much of
its work, while local research documents and some files differ. Synchronization
should be a separate deliberate operation preserving those changes. This review
made **no merge, reset, application-source fix, commit or push**. README and
license files were not changed.

## Evidence

Compact [results/provenance JSON](evidence/2026-11-02-upstream-review/results.json),
[wide benchmark output](evidence/2026-11-02-upstream-review/wide-benchmark.txt),
and [backend regression failure](evidence/2026-11-02-upstream-review/backend-regression.txt)
are archived separately from generated build artifacts. The JSON includes the
executed overflow/fallback cases and exact source/commit identities.
