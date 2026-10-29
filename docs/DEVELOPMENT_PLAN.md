# Trium-XBT: goals and ordered development plan

Date: 2026-10-29. Planning baseline: the validated foundation described in
[FOUNDATION_PROGRESS_2026_10_29.md](FOUNDATION_PROGRESS_2026_10_29.md).

**Purpose:** one place to understand the destination, the current position and the
next development order. This is a plan, not a claim that future stages are already
implemented. Numeric contracts and measured results remain in their linked
technical documents. README remains unchanged. Implementation progress through
the first five stages is now recorded in
[STEPS_1_TO_5_STATUS.md](STEPS_1_TO_5_STATUS.md); the gates below remain the
acceptance criteria, not an assertion that every hardware gate has closed.

## 1. Your goals, restated

The goal is to develop **a production-grade balanced-ternary compiler and reusable
mathematical foundation** with consistent, explicitly specified behaviour across:

1. Today's binary CPUs.
2. Real binary GPU execution.
3. Future native ternary and mixed-radix hardware.

The important requirements you have emphasized are:

- Establish the mathematics deeply before broad hardware expansion.
- Share mathematical meaning between implementations instead of inventing a
  different interpretation for each backend.
- Use Z3, Rocq/Coq and independent reference implementations, including ordinary
  cases, boundaries and failures.
- Build reusable backend-independent mathematical components, not only isolated
  compiler demonstrations.
- Improve correctness, reliability, organization and measured performance toward
  production use.
- Distinguish implemented features from tested/qualified hardware support.
- Explain progress positively and factually, without arbitrary maturity
  percentages or excessive self-criticism.
- Use parallel/multi-agent work where available; never substitute that for clear
  ownership and integration checks. No additional agents have been claimed in
  this session.
- Leave README changes until the end.

**Not the goal:** merely display ternary digits, generate target templates, or
rename ordinary binary64 arithmetic as ternary floating point. Balanced-ternary
semantics can execute through binary representations today; native ternary
hardware is a later implementation of the specified operations. Performance is
an engineering target to measure, not a guarantee that every ternary workload
will be faster than every binary implementation.

## 2. The intended architecture

```text
                     Mathematical contracts
                   + reusable numeric library
                    + independent references
                              |
Source -> parser/type checking -> verified typed core IR
                                      |
                         +------------+-------------+
                         |            |             |
                    Reference      CPU backend    GPU backend
                    execution       via LLVM      + device runtime

Upstream MLIR integration is a lowering/infrastructure layer introduced where
appropriate; it is not a stage after native machine-code execution.

Future ternary/mixed-radix lowering also consumes the same specified operations,
using target capability/layout rules and an executable instruction model.
```

The diagram is architectural intent, not a statement that all arrows exist now.
An operation's value, type, rounding, overflow and side effects come from the
contract. A backend chooses its representation/instructions, not different
mathematics. Different capabilities require explicit profiles and diagnostics.

## 3. Where we actually are

Already working:

- Checked scalar source-language execution and actual LLVM execution on this
  x86-64 host.
- Backend-independent numeric components, including reduced exact rationals and
  bounded ternary floating arithmetic.
- AST-to-whole-program scalar core lowering, verification and independent core
  execution through `--core-run`.
- Existing matrix/sparse CPU functionality in the exercised paths.
- Default scalar core-to-LLVM CPU compilation, with explicit legacy aggregate
  compatibility rather than silent fallback.
- Independent numeric/program comparisons, Debug/Release/sanitizer matrices,
  ten Z3 scripts and 66 assumption-free Coq-kernel-checked results.
- A restricted CUDA map emitter and actual-driver runtime; offline PTX, emulation
  and driver-protocol tests pass. Physical GPU qualification remains open.
- Conservative opt-in optimization with matched-boundary measurements.

The immediate qualification handoff:

> The shared scalar pipeline is implemented. Run the CUDA device gate and the
> configured Rust/native-Rocq CI jobs, extend fuzz/performance campaigns, and
> qualify each additional named toolchain/platform before a broader release.

Having multiple execution routes currently helps cross-check correctness. The
long-term goal is **one authoritative semantic pipeline**, not three separately
maintained definitions. Existing tests/proofs are meaningful evidence, not an
end-to-end proof of all C++/LLVM behaviour. Native Rocq CI and real GPU execution
are not claimed as completed.

## 4. What a gate means

A gate is an objective acceptance checklist for one declared scope. It is not a
percentage, a requirement to solve all mathematics, or a ban on research.

A milestone is accepted when:

1. Its scope and unsupported cases are explicit.
2. Its implementation is integrated, not only a standalone demonstration.
3. Tests/reference comparisons and applicable proof obligations pass.
4. Reproduction commands, limitations and results are recorded.
5. Existing supported behaviour has not regressed.

Use these statuses consistently:

- **Planned:** design/scope only.
- **Implemented:** code exists and is integrated.
- **Validated:** the declared checks have passed.
- **Qualified:** a specific target/toolchain/device configuration has passed its
  release-level checks. This does not qualify every platform.

## 5. Ordered development stages

### Stage 1 — Consolidate the mathematical profile

**Deliverable:** an unambiguous supported-operation contract, using the existing
v0.1 contract as its baseline rather than restarting the mathematics.

Work:

- Record supported types, domains, operations, conversions, rounding, overflow,
  capacity errors and observable effects in one operation/support matrix.
- Expand cancellation, exponent-boundary, wide-FMA and conversion reference
  coverage; preserve counterexamples as regression tests.
- Keep `.bt` binary64, exact-rational library arithmetic and ternary BTFP clearly
  separate. A C++ library type is not automatically a new source-language type.
- State exact guarantees versus approximations, particularly transcendentals.
- Run the native Rocq CI gate when the work is submitted through GitHub; strengthen
  implementation/proof links without describing model proofs as C++ refinement.

**Gate 1:** every operation included in the declared profile has a documented
meaning, checked domain/error rules and independent validation appropriate to its
claim. Approximate or unsupported operations are labeled, not silently treated
as exact. Existing validation remains green.

**Impact:** prevents backend disagreement and gives optimizations clear rules.
It does not require proving every possible function before useful development.

### Stage 2 — Make core IR the shared scalar compiler pipeline

**Deliverable:** source -> checked core IR -> LLVM -> native CPU execution.

Implementation order:

1. Constants, symbol bindings, loads/stores and explicit conversions.
2. Checked arithmetic, comparisons and supported scalar intrinsics.
3. Lazy selection, blocks, conditions and loops.
4. Functions, returns and error/side-effect propagation.
5. Differential validation, then migration of the supported scalar default route.

Extend core representation for aggregate shapes/storage before migrating those
operations. Until then, aggregate compatibility must be an explicit profile;
there must not be an unexplained silent fallback.

**Gate 2:** all operations in the scalar profile lower through core IR, and values,
errors and observable effects agree across reference/core/native execution.
Test emitted LLVM independently and retain meaningful mismatch reproductions.

**Impact:** fixes and semantics are maintained centrally. This is the key enabling
step for reliable optimization, upstream MLIR integration and new backends.

### Stage 3 — Qualify the CPU pipeline and runtime

**Deliverable:** a dependable, reproducible CPU release candidate for explicitly
named toolchains/platforms, plus integration of the shared lowering with upstream
LLVM/MLIR infrastructure as appropriate.

Work:

- Establish reproducible builds and a toolchain/platform support matrix.
- Harden ABI versioning, error reporting, allocation/lifetime rules and clients.
- Test Rust/Python clients in supported environments, not just headers/bindings.
- Add continuous generated-program/IR/storage fuzzing, resource policies and
  regression handling.
- Benchmark compile time, execution time and memory separately.
- Execute on other CPU architectures before calling them qualified; assembly
  emission alone does not establish execution correctness.

**Gate 3:** the selected configurations pass correctness, ABI/client, sanitizer,
stress and reproducibility checks. Performance baselines and measurement
boundaries are recorded.

**Impact:** turns a working compiler into a predictable tool users can integrate.
It provides a strong reference platform for device debugging.

### Stage 4 — Deliver one real GPU path

**Deliverable:** a small but genuine program-dependent GPU backend and runtime on
one available, explicitly selected device/runtime combination.

Work:

- Choose the first GPU/runtime based on actual hardware access.
- Specify the supported parallel subset and typed shape/storage/layout rules.
- Start with useful bounded kernels, such as elementwise ternary operations and
  a dot product or sparse matrix-vector operation; do not promise that arbitrary
  scalar control flow automatically becomes an efficient GPU kernel.
- Implement allocation, transfers, launches, synchronization and device errors.
- Preserve declared checked arithmetic and reduction ordering; any relaxed float
  or reduction profile must be separately explicit, not a hidden semantic change.
- Compare device outputs/errors with the CPU/reference implementation and measure
  transfers separately from kernel execution.

**Gate 4:** generated programs actually execute on the selected GPU, pass the
supported subset's reference comparisons and memory/error tests, and have
reproducible end-to-end measurements. Templates or compiled-but-unlaunched kernels
do not pass this gate.

**Impact:** establishes the first real accelerator implementation. Additional GPU
APIs/devices follow after this pattern works, not all at once.

### Stage 5 — Optimize under the established semantics

**Deliverable:** demonstrated improvements without changing declared behaviour.

Start with constant folding, dead-code elimination and range analysis, then
packing/vectorization and algebraic/kernel transformations where justified.
Measure representative workloads, not only arithmetic microbenchmarks.

Some safe local improvements may happen earlier. The large optimization campaign
belongs after reliable reference/native/device comparisons exist.

**Gate 5, per optimization:** value/error/effect preservation checks pass, a
reproducer exists for discovered failures, and measured benefit justifies the
complexity. A particular transformation need not improve every workload; its
applicability and tradeoffs must be explicit.

**Impact:** produces trustworthy speed and memory improvements rather than
performance gains obtained by weakening arithmetic.

Example: with checked bound H, `(H+1)-1` fails, while `H+(1-1)` succeeds. A compiler
cannot silently reassociate the first into the second merely because they are
algebraically equal over unbounded integers.

### Stage 6 — Extend to native ternary/mixed-radix targets

**Deliverable:** a precise executable target specification, simulator and backend
mapping, followed by physical-device qualification when suitable hardware exists.

Work:

- Specify instructions, encodings, data layouts, conversions, memory and failures.
- Extend existing software ALU/instruction models into the selected executable
  target profile.
- Differentially compare simulator/backend behaviour with shared reference
  semantics; add RTL/device co-simulation when hardware tooling is available.
- Keep ISA-model validation, simulation and physical execution distinct.

**Gate 6:** simulator/lowering conformance passes for the declared target; physical
hardware claims require physical execution evidence.

**Impact:** the same foundation can support future hardware without starting the
language and mathematics over. Target research may proceed earlier, but does not
replace finishing the current CPU/GPU pipeline.

## 6. Keep the work organized

- **One main integration milestone at a time.** Now: mathematical-profile
  consolidation followed by shared core-to-LLVM lowering.
- **Small work packages.** Each has a contract, implementation, acceptance tests,
  evidence and a clear next dependency.
- **Three coordinated workstreams when parallel workers are available:**
  mathematics/reference validation; IR/backend implementation; testing/runtime
  tooling. All integrate through the same operation contract and gates.
- **Two scorecards:** correctness/qualification and performance. Do not replace
  either with an invented overall completion percentage.
- **Explicit support matrix:** distinguish source/library/core/CPU/GPU support;
  an operation being present in one layer does not imply every other layer.
- **No new backend proliferation yet.** Finish one shared CPU pipeline and then
  one real GPU path before expanding many target combinations.
- **Change control:** evolve versioned contracts deliberately; preserve test
  counterexamples; update this plan when priorities change. README updates remain
  deferred as requested.

## 7. Current handoff after the Steps 1–5 implementation pass

The original mathematical-support-matrix and core-to-LLVM work package is now
implemented and locally validated. Its operation matrix, executable code,
proof/test evidence and benchmark samples are linked from the current status
report. Continue in this order:

1. Execute the configured Rust and native-Rocq CI jobs on submission.
2. Run `CUDA_Device_Qualification` on a real NVIDIA device and record its exact
   driver, architecture, output/error comparisons and transfer/kernel timings.
3. Extend long-running source/IR fuzzing and representative performance coverage;
   retain the opt-in optimizer while mixed performance results are investigated.
4. Qualify additional CPU hardware/toolchains and improve source-level debugging.
5. Integrate upstream MLIR/aggregate storage deliberately, without weakening the
   scalar contracts or treating the standalone mock dialect as upstream MLIR.

Future ternary hardware, simultaneous CUDA/Vulkan/Metal proliferation and
speculative checked-arithmetic reassociation remain outside this work package.

The simple development order is:

> **Define the mathematics -> unify the compiler -> qualify CPU execution ->
> execute on one real GPU -> optimize -> extend to future ternary hardware.**
