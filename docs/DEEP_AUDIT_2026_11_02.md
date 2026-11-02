# Deep audit of the Trium-XBT compiler, 2026-11-02

> **Repairs:** the findings in this audit have been addressed on this branch; see
> `docs/AUDIT_FIXES_2026_11_02.md` for the per-finding change, regression test and
> residual limitation. This document remains the description of the audited
> upstream snapshot and its evidence.

**Scope:** upstream `main` at `82ae25146453e57d49869933a1e2dc0ca9e18bb7`, isolated from
this workspace (base commit `a7e456e`). The only later upstream commit,
`5a33d9ce7e75e531d8be27574b3d7713a45a2eda`, changes CI workflow files only, so every
source-level finding below still applies. No merge, rebase or upstream edit was made.

**Method:** source reading plus independent execution of the code being audited.
A `git archive` snapshot of upstream was configured twice from scratch
(`Release` and `Debug + AddressSanitizer`), built with GCC 12.2 and run under CTest
(53 tests). On top of that I wrote separate probes for allocators, the thread pool,
process spawning, the numerical kernels, the legacy Python FFI, and the compiler
driver, plus a 2,067-case native-versus-reference differential fuzz, a 300,000-case
rounding fuzz, a `llvmlite` harness that reproduces the emitted-code semantics, a
`pyslang` parse of the SystemVerilog, and an inspection of the two latest CI runs.
Every finding is labelled **reproduced** (executed here) or **static** (established by
reading the code, without a failing run).

**Not tested, and therefore not claimed:** no real Clang backend (the container has no
Clang, so cross-compilation was exercised with a fake failing tool and with
`llvmlite`), no macOS host, no Rocq kernel, no Vulkan device or driver, no CUDA GPU,
no Verilator/Icarus/Yosys/SBY/EQY, no physical ternary hardware, no fuzzing of the
frontend with a coverage-guided engine. Timings are one container, not a benchmark
suite.

**Result: 30 findings — 2 critical, 10 high, 11 medium, 7 low; 21 reproduced, 9 static.**
The machine-readable list, with files, evidence and suggested repairs, is
`findings.json` next to this document; raw outputs are in the same directory.

---

## 1. Verdict

The audited snapshot is **not safe to sync into this branch as-is**. It contains three
classes of defect that change observable behaviour, not just code quality:

1. **A semantic regression in integer arithmetic.** The 39-trit `int39` type is
   documented as checked: a result outside the representable range must be an error.
   The native backend now saturates instead. The interpreter and the differential
   suites still report an error, so the same program gives different answers depending
   on the backend, and the project's own tests fail.
2. **A fabricated compiler success.** If RISC-V cross-compilation fails, the driver
   writes a fixed `li a0,0; ret` assembly file and exits 0. A build system cannot tell
   the difference between "compiled" and "silently did nothing".
3. **Memory-safety and concurrency defects.** Integer wrap-around in three allocator
   APIs produces allocations far smaller than the published size; the thread pool both
   terminates on a worker exception and uses a destroyed stack latch when the calling
   thread throws; the legacy Python FFI loads a shared library from the current working
   directory.

Outside those, the numerical kernels had reproducible edge-case failures (NaN on exact
cancellation, `std::terminate` from a `noexcept` function), two "complete" Rocq
theorems prove `True`, and the hardware "co-simulation" test only searches the RTL
source for module names.

What held up should be said as clearly: the exact-arithmetic and floating-point
references, the Z3 obligation set, the native-versus-reference differential (2,067
cases, zero mismatches other than the overflow regression), 300,000 one-shot and
50,000 sequential rounding checks, the 20,000-pair adder model comparison, the SDK
status/JSON contract, and 49 of 53 CTest tests in both Release and ASan builds.

---

## 2. Findings index

| ID | Sev | Status | Finding |
|----|-----|--------|---------|
| C-01 | high | reproduced | Native Int39 add/sub saturate instead of reporting overflow |
| C-02 | high | reproduced | Failed RISC-V compile replaced by a stub, reported as success |
| C-03 | medium | static | Arithmetic benchmark encodes clamping as expected semantics |
| C-04 | low | static | Compiled loops have no execution budget; the interpreter does |
| M-01 | critical | reproduced | Caller-thread exception leaves queued tasks with a destroyed latch |
| M-02 | high | reproduced | Worker exception calls `std::terminate` |
| M-03 | high | reproduced | Concurrent `initialize`/`reinit` can SIGSEGV |
| M-04 | critical | reproduced | `DualRailPlanes::allocate` capacity arithmetic wraps |
| M-05 | high | reproduced | `allocate_aligned` rounds oversized requests with wrap-around |
| M-06 | high | reproduced | `Base243Arena::allocate<T>(count)` multiplies without overflow check |
| M-07 | medium | reproduced | `noexcept` accessors check bounds only with `assert` |
| M-08 | low | static | `parallel_gemm` validates shapes only with `assert` |
| S-01 | high | reproduced | Legacy FFI loads `libbt_runtime` from the current directory |
| S-02 | medium | static | Legacy FFI cannot validate buffer lengths |
| S-03 | medium | reproduced | Child processes inherit unrelated descriptors on Linux |
| S-04 | low | static | macOS SDK path differs between driver and native JIT |
| N-01 | medium | reproduced | GEMM underflows tiny products to zero |
| N-02 | high | reproduced | GEMM returns NaN on exact cancellation of large values |
| N-03 | medium | reproduced | RMSNorm overflows to infinity for subnormal input, `eps=0` |
| N-04 | medium | reproduced | `TriRadixScalar::to_double` returns NaN despite an exact value |
| N-05 | medium | reproduced | SiLU/GELU return NaN for negative infinity |
| N-06 | high | reproduced | `xbt_exp_btfp32` is `noexcept` but can throw (`std::terminate`) |
| V-01 | medium | static | Two "complete" Rocq theorems are literally `True` |
| V-02 | medium | static | Hardware "co-simulation" only searches strings |
| V-03 | high | reproduced | Latest CI is failing (overflow regressions) |
| V-04 | medium | reproduced | Vulkan tests are skipped; CI installs no shader tooling |
| V-05 | low | reproduced | Speedup/numerical claims are not reproduced exactly |
| A-01 | low | static | C ABI activation entry points do not check null buffers |
| A-02 | low | reproduced | TCSR `get()` hides the builder's open row until `finalize()` |
| A-03 | low | static | Map admission uses static statement count, not loop iterations |

---

## 3. Compiler semantics

### C-01 — checked `int39` became saturating `int39` (high, reproduced)

`include/Ternary/IR/LLVM.hpp` gained a `clamp64` helper used by the new add/sub path
(lines 42–46 and 72–75), and `ternary_llvm_codegen.hpp` gained
`clamp_integer_64`, which replaced the checked i128 widening at lines 952–960. The
documented contract (`docs/CORE_SEMANTICS_V0_1.md`, the Z3 obligation set, and
`core_backend_regression.py`) says an out-of-range result is an error.

Measured with `H = 2026277576509488133`:

| program | native backend | interpreter |
|---------|----------------|-------------|
| `H + 1` | prints `H`, exit 0 | error, exit 1 |
| `-H - 1` | prints `-H`, exit 0 | error, exit 1 |
| `(H + 1) - 1` | prints `H - 1`, exit 0 | error, exit 1 |

This is not a theoretical divergence: the same source gives three different answers
depending on how it is run, and a program that relies on overflow detection silently
continues with a wrong value. It is also the direct cause of the failing CI (V-03):
`Core_Backend_Regression`, `Core_Optimized_Differential` and
`Phase1_LLVM_Differential` assert the error status.

The workspace compiler (base `a7e456e`) still has the checked version, so this branch
does **not** need to import the regression. The audit did not find a soundness hole in
the still-checked `fits()` fast path: it only elides the check when interval analysis
proves the result is inside `int39`, which is sound for the representable operands.

### C-02 — fabricated RISC-V output (high, reproduced)

`bt_compiler.cpp:968–1000` catches `ret != 0` from the backend when the triple contains
`riscv`, writes a 10-line stub (`addi sp, sp, -16` … `li a0, 0` … `ret`), sets
`ret = 0`, and prints the normal success line. Two separate problems follow:

* the artifact cannot contain the program, so an object file built from it is a no-op;
* the driver's status is the only signal a build system has, and it says success.

Reproduced with `--target=riscv64 -S` and a `clang` shim that exits 1: upstream exits
0 and writes `li a0, 0`; the workspace compiler exits 1 and writes no artifact.
The named target `riscv64-xbt` is rejected elsewhere in the driver, which makes the
silent path look like support for ordinary RISC-V while it is only a fallback for a
missing backend. A correct implementation must propagate the backend failure (or
select a real backend); it must never synthesize code.

### C-03 — the benchmark adopted the wrong semantics (medium, static)

`test/benchmark_arithmetic.cpp:105–145` computes its "expected" values by clamping and
compares the compiler against that. After C-01 this makes the benchmark pass while the
documented contract fails; before C-01 it would have caught the bug. Benchmarks must
assert the contract (error on overflow) and may time only the non-overflow fast path.

### C-04 — no execution budget in compiled code (low, static)

`include/Ternary/IR/Execute.hpp:16–21` gives the reference interpreter a tick budget
and raises `ResourceLimit`; `LLVM.hpp:178–183` emits an unguarded loop. The SDK
documents this distinction, so this is a design limitation rather than a contract
violation, but it is a real gap for anyone using AOT output as a service. Either state
it in the emitted-code documentation or add an opt-in back-edge tick check.

---

## 4. Memory safety and concurrency

### M-01 / M-02 — the thread pool loses exceptions in both directions (critical / high, reproduced)

`parallel_for` (`ternary_kokkos_bridge.hpp:373–409`) runs the last chunk on the calling
thread and otherwise enqueues lambdas that capture `&sync_latch` and `&func`. If the
callback throws in the caller's chunk, the function unwinds immediately, destroying
`sync_latch` while worker tasks still hold a pointer to it. ASan reports
`stack-use-after-scope` inside `std::latch::count_down` at line 404. If the callback
throws in a worker, nothing catches it, the exception escapes the thread function, and
the process calls `std::terminate` (M-02); `parallel_reduce` has the same two problems
at lines 436–481.

Consequences: a numeric edge case inside a parallel GEMM can kill the process, and a
correctness error in the calling thread causes a use-after-scope that may hang or
corrupt. Repair: a task guard that catches everything, records a `std::exception_ptr`
and always counts the latch down; the calling thread records its own exception, waits
for all chunks, then rethrows. Both paths are covered by the new regression test.

### M-03 — concurrent initialization races (high, reproduced)

`ThreadPool::init` calls `shutdown` but neither is serialized for `reinit` callers, and
`reinit` reads `workers`/`active_threads` without a lock. Concurrent `initialize()` from
several threads (documented as allowed by the Kokkos-compatible surface) crashes with
SIGSEGV. Separately, `enqueue` can push a task while `shutdown` is draining, after which
the task is never executed and its latch waiter blocks forever. Repair: one recursive
lifecycle mutex around `init`/`reinit`/`shutdown`, and `enqueue` runs the task inline if
the pool is stopping.

### M-04 / M-05 / M-06 — three allocator wrap-arounds (critical / high / high, reproduced)

* `DualRailPlanes::allocate(SIZE_MAX)` computes `capacity = (words + 7) & ~7` and
  `bytes = capacity * 8`, then `2 * bytes`; each step wraps. The object publishes
  `num_words = SIZE_MAX`, `capacity = 0`, `bytes = 0` and allocates one vector-sized
  block. ASan then reports a heap-buffer-overflow **write** in `set_word`.
* `mem::allocate_aligned` rounds `bytes + alignment - 1` the same way. A request for
  `SIZE_MAX - 31` returns a 64-byte block, so callers that trust their own request size
  write out of bounds. This is the more dangerous form because the API has no failure
  signal on a normal machine.
* `Base243Arena::allocate<T>(count)` evaluates `count * sizeof(T)` with no check; a
  huge count produces a small block. `allocate_bytes` also lacks a check on
  `padding + bytes` and on the slow-path chunk size.

Repair (implemented in this branch): reject non-power-of-two alignment, reject byte and
word counts whose rounding or double-rail multiplication overflows `size_t`, and check
`count * sizeof(T)` and `num_trits + 4` before allocating. Regression tests assert the
`std::length_error`/`std::invalid_argument` for each of the three reproduced requests
and that ordinary sizes still allocate correctly and stay 64-byte aligned.

### M-07 / M-08 — release builds lose the checks (medium / low)

`TernaryMatrix::word/get/set`, `DenseMatrixInt64::operator()`, `DualRailPlanes` accessors
and `TCSRMatrix::get_sign/is_pos` check bounds only with `assert`, and several are
`noexcept`. Under `NDEBUG` an out-of-range index is an out-of-bounds read; `set` is an
out-of-bounds write; `get_sign(100)` segfaults (ASan traces in
`matrix-ndebug-oob.txt` and `tcsr-ndebug-oob.txt`). `parallel_gemm` likewise validates
matrix shapes only with asserts, so a dimension mismatch in a release build indexes out
of bounds. These are caller-contract violations, but a numeric library should fail
loudly: the repair makes the checks unconditional (throwing `std::out_of_range` /
`std::invalid_argument`) and drops `noexcept` from the checked accessors.

---

## 5. Security and process handling

### S-01 — library preloading through the working directory (high, reproduced)

`ffi/python/bt_ffi.py:17–45` searched `os.getcwd()` first for `libbt_runtime.so`.
Importing the module from an attacker-controlled directory executes that library.
Reproduced with a fixture shared object whose constructor writes a marker file:
`import bt_ffi` created the marker. This is code execution triggered by an ordinary
import, and the module is the documented legacy Python interface. Repair: remove the
CWD from the search list entirely; keep the explicit `BT_RUNTIME_LIBRARY`, the
package-relative paths and the system loader. A regression test builds the same fixture
and asserts the constructor never runs.

### S-02 — unchecked lengths in the legacy FFI (medium, static)

`csr_spmv` checks `row_ptr` only with `assert` (removed by `python -O`), never checks
`signs` capacity or `col_indices` monotonicity, and `xbt_rmsnorm` accepts a `weight`
list whose length differs from `x`. The C functions take no length parameters, so a
mismatch reads past the ctypes buffer. `xbt_sgemm` had the same `assert`-only pattern.
Repair: explicit `ValueError` checks for every length, `row_ptr[0] == 0`,
`row_ptr[-1] == nnz`, monotonic offsets, the `ceil(nnz/64)` signs size, index bounds,
and `len(weight) == len(x)`.

### S-03 — descriptor inheritance (medium, reproduced)

`bt_process.hpp` requests close-on-exec only through `POSIX_SPAWN_CLOEXEC_DEFAULT`,
which glibc does not define (it is a BSD/macOS flag). On Linux a spawned compiler
therefore inherits every descriptor the parent has open, including readable files and
pipes. Reproduced: the child could see the parent's pipe descriptor. Repair: enumerate
`/proc/self/fd` and add a `posix_spawn_file_actions_addclose` for every descriptor
above 2, in addition to the platform flag where available. A regression test confirms
the child can no longer see a parent pipe.

### S-04 — two macOS SDK selection policies (low, static)

`bt_process.hpp::clang_command` prefers `SDKROOT` and then falls back to the
CommandLineTools path; `lib/SDK/NativeRuntime.cpp` prefers the hardcoded path. With
`SDKROOT` set, the driver and the native JIT can compile against different SDKs.
Repair: one `platform_toolchain_flags()` helper used by both (implemented).

---

## 6. Numerical behaviour

| ID | Input | Result | Expected |
|----|-------|--------|----------|
| N-01 | 1×1 GEMM, `A=B=1e-22` | `0` | `9.8e-45` (a float subnormal) |
| N-02 | `[1e38,1e38]·[1e38,-1e38]` | `NaN` | `0` (exact cancellation) |
| N-03 | RMSNorm of `1e-45` values, `eps=0` | `inf` | finite (`~1`) |
| N-04 | TriRadix `2^-2000 · 5^800` | `NaN` | `1.306e-43` (exactly representable) |
| N-05 | SiLU/GELU at `-inf` | `NaN` | `0` |
| N-06 | `xbt_exp_btfp32(BTFP32(1000))` | `std::terminate` | catchable `domain_error` |

Causes and repairs:

* **N-01/N-02** — the block-scale product and the row sum were computed in `float`.
  `scale_A * scale_B` underflows for tiny inputs, and the sum of large terms reaches
  `±inf`, whose sum is NaN. Multiplying the two scales and accumulating the row in
  `double`, then converting once, fixes both (verified by the same inputs).
* **N-03** — the reciprocal RMS is computed in `double` but was narrowed to `float`
  before the multiply, so a subnormal mean (`1/sqrt ≈ 1e45`) became `inf`. Keeping the
  reciprocal in `double` and handling `mean_sq == 0` explicitly fixes it.
* **N-04** — `ldexp(m, exp2)` underflowed to zero and `pow(5, exp5)` overflowed to
  infinity, giving `0 * inf`. The converter now renormalizes after each factor step and
  applies one `scalbn`, so exponents cancel before the range is exhausted.
* **N-05** — `x * sigmoid(x)` at `-inf` is the indeterminate `-inf * 0`; the limit is
  0. Non-finite inputs are special-cased.
* **N-06** — `BTFP32::from_double` throws `domain_error` for a non-finite result;
  marking the wrapper `noexcept` converted that into `std::terminate`. The wrapper is
  no longer `noexcept` and is documented as throwing.

These are not all equally severe. N-02 and N-06 change program behaviour or crash the
process; N-01/N-03/N-04/N-05 are edge cases at the boundary of the documented domain,
but the documentation does not state those boundaries, and the "drop-in replacement"
wording for `xbt_sgemm_unmodified` invites the comparison.

What was checked and held: the exact-reference tests for checked integer arithmetic,
the FP foundation oracle, 300,000 rounding checks (one-shot and sequential), the
20,000 Kogge-Stone model pairs, and the Z3 models. No defect was found in Kogge-Stone
carry semantics, the codec's packing/unpacking, the dual-rail invariant, or the
`finish()` exponent window in the new numerical path.

---

## 7. Verification and process

### V-01 — vacuous terminal proofs (medium, static)

`CoreLowering.v:65` states `Theorem core_lowering_complete : True` and
`ImplementationBridge.v:96` states the same for `implementation_bridge_complete`. Both
check with `Proof. exact I.` and `check_rocq.py` only verifies that the named theorem
exists and has no axioms, so the proof gate passes while two "complete" obligations
prove nothing. The other terminal obligations (`core_math_complete`,
`verlet_kdk_invertible`) are real. Repair: substantive statements over the existing
models (failure propagation and exact multiplication; the correction + rounding
identities) and a checker rule that rejects a theorem whose statement is `True` or
`0 = 0`.

### V-02 — "co-simulation" that does not simulate (medium, static)

`test/Unit/test_unit_xbt_hw.cpp` opens `hw/riscv_xbt_extension.sv` and asserts that six
module names appear in the text, then prints "Cycle-Exact Co-Simulation". The C++ model
is checked against reference arithmetic, which is valuable, but it is never compared
with the RTL. `pyslang` parses the file cleanly (one benign `unique case` warning); no
elaboration, simulation, formal or equivalence check was run or is installed. Repair:
rename the check to a source inventory, lint with Verilator when present, and state that
co-simulation is pending.

### V-03 — CI is red (high, reproduced)

CI runs `41415612321` (commit `82ae251`) and `41419040453` (`5a33d9c`) both fail in the
debug, release and sanitizer jobs at "Build and test"; `native-rocq` and `rust-client`
pass. Failed-job logs could not be downloaded (results-receiver EOF), so I cannot quote
the job output, but the failure reproduces locally on the audited source with exactly
the three overflow-expectation tests. This is consistent with C-01 causing the CI
failure. Fix C-01/C-02 before syncing, then rerun.

### V-04 — Vulkan is never exercised in CI (medium, reproduced)

Both `SDK_Vulkan_Offline` and `SDK_Vulkan_Execution` return 77 here because
`glslangValidator` and `spirv-val` are absent, and the CI workflow installs only
`clang`. The SPIR-V emitter, the integer-emulation rules and the Vulkan ABI therefore
have no CI coverage; the only validation they have is review. Repair: install
`glslang-tools` and `spirv-tools` in CI (device execution remains optional).

### V-05 — benchmark claims are not reproduced (low, reproduced)

The wide-arithmetic benchmark measured 12.7×/17.1× in this container against the
recorded 18.37×/40.13×. The exact-operation assertions pass, so this is a measurement
scope problem, not an arithmetic one: single-machine microbenchmarks are being quoted
as product claims. Report hardware, medians and ranges, and keep timings out of
pass/fail gates.

---

## 8. Smaller API issues

* **A-01** — `xbt_rmsnorm` and `xbt_softmax` in `lib/Ternary/ternary_c_abi.cpp` did not
  check `x`/`out` for null when `n > 0`, unlike the rest of the C ABI (fixed here).
* **A-02** — `TCSRMatrix::get()` returns `Zero` for the row the builder is currently
  filling because `row_ptr[r + 1]` is not written until the next row starts or
  `finalize()` runs. Documented as "call finalize", but silent wrong answers are a poor
  default for an element accessor (fixed here with a regression test).
* **A-03** — `bt_sdk_map` admits a batch by a static statement count; loop iterations
  are not included, and the interpreter's per-invocation budget resets for every
  element, so the aggregate is effectively unbounded. The SDK correctly documents that
  this is not a sandbox, so this is a hardening gap, not an escape. The batch path now
  also enforces a cumulative execution budget.

---

## 9. What held up

* **Exact integer core:** add/sub/mul/divmod, trit logic, Kogge-Stone and Base-243
  codec tests pass, including the exhaustive 256-byte unpack loop.
* **Differential result:** 2,067 native-versus-reference cases, 128 mismatches, all of
  them the known C-01 clamp; zero other mismatches.
* **Rounding:** 250,000 one-shot and 50,000 sequential checks against exact arithmetic.
* **Z3:** 312 obligations discharged; no counterexample.
* **SDK:** 13 API probes (status codes, JSON, ownership, close semantics) behave as
  documented; 1,744 native/reference cases pass.
* **Suite:** 49 of 53 CTest tests pass in both Release and ASan builds. Of the four
  local failures, three are C-01 and one (`Layer1_Unit_Frontend_FP_Sparse`) is an
  environment artifact: no real Clang in this container.
* **FP foundation:** the oracle tests over rounding, FMA and conversions pass.
* **RTL parse:** the SystemVerilog parses without errors.

These are the parts that make the project worth continuing; they are stated as evidence
from this container, not as a security or correctness guarantee.

---

## 10. Recommended repair order

1. **C-01, C-02, V-03** — semantics and honest compilation; this also turns CI green.
2. **M-01/M-02/M-03** — thread-pool exception and lifecycle safety (crashes reachable
   from ordinary GEMM code).
3. **M-04/M-05/M-06** — allocator overflow checks (memory corruption).
4. **S-01/S-02** — legacy FFI loading and length validation (code execution).
5. **M-07/M-08, A-01, A-02, A-03, S-03, S-04** — hardening and contract checks.
6. **N-01…N-06** — numerical edge cases with explicit domain documentation.
7. **V-01, V-02, V-04, V-05** — proof/test integrity and honest reporting.
8. Only then consider syncing upstream into this workspace, and rerun the full suite.

## 11. Reproducing

* Snapshot: `git archive 82ae251 | tar -x -C build/upstream-review/snapshot`
* Builds: `cmake -S build/upstream-review/snapshot -B build/deep-audit/release
  -DBT_ENABLE_MLIR=OFF -DCMAKE_BUILD_TYPE=Release` (and the same with
  `-DBT_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug`), then `ctest`.
* Probes: `build/deep-audit/probes*.cpp`, `build/deep-audit/check_latest.py`,
  `build/deep-audit/differential_fuzz.py`, `build/deep-audit/sdk_probe.py`,
  `build/deep-audit/act.cpp`, `build/deep-audit/fuzz_round.cpp`.
* Evidence: `docs/evidence/2026-11-02-deep-audit/` (`SHA256SUMS` covers every file).

A finite audit cannot prove the absence of defects. The limits listed at the top —
especially the missing real backend, GPU and RTL simulators — are where any remaining
high-severity issue is most likely to hide.
