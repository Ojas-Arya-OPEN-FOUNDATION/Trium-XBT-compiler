# Proposal: reusable mathematics, compiler SDK, and cross-vendor GPU execution

**Date:** 2026-10-29. **Status: R1/R2 implementation approved by the user.**
First executable delivery and remaining gates are recorded in
[R1_R2_DELIVERY.md](R1_R2_DELIVERY.md). This remains the full acceptance roadmap,
not a claim that every work package is complete. The subsequent
[FLOATING_POINT_FOUNDATION.md](FLOATING_POINT_FOUNDATION.md) implements the focused
primitive/capacity milestone. The subsequent [SDK 1.1 milestone](SDK_1_1.md) adds
compiled native CPU execution, typed discovery, bounded caches and persistent
Vulkan dispatch resources. Broader tensor/tuning/hardware packages below remain
separate gates. README remains unchanged.

## 1. Objective and honest end state

Develop Trium-XBT from a tested scalar/compiler foundation into an embeddable,
versioned mathematical compiler/runtime stack:

1. Reusable mathematical components with clearer contracts, conversions,
   serialization, error handling and independent validation.
2. A genuine application SDK: compile source, inspect diagnostics/exports, create
   buffers, select a device, execute a supported function/kernel, and manage every
   resource without invoking the command-line driver.
3. A real Vulkan/SPIR-V backend for a declared cross-vendor GPU subset, with
   capability-driven representation, reference comparisons and explicit fallback.
4. Measured, correctness-preserving hardware adaptation—not an automatic-fastest
   promise, hidden numerical relaxation, or a universal GPU claim.

**Stage 6 remains a separate long-term ISA/simulation/RTL/synthesis/hardware track.**
Keep and develop it, but do not make it a prerequisite for this software SDK.
This proposal does not implement a processor, all framework compilers, arbitrary
GPU programs, simultaneous multi-GPU scheduling, or distributed execution.

Current verified capabilities are recorded in [STEPS_1_TO_5_STATUS.md](STEPS_1_TO_5_STATUS.md).
Research and primary-source references are in [SDK_GPU_RESEARCH.md](SDK_GPU_RESEARCH.md).
The work packages below mix delivered and pending acceptance items. See the
current delivery report before interpreting an individual future-tense item.

## 2. What the repository already provides—and what must change

Keep the checked int39 language semantics, library integer/rational/BTFP
contracts, verified core, CPU LLVM emitter, conservative optimizer, CUDA map
reference, ABI-major-1 runtime and existing differential/proof corpus.

Close these concrete gaps:

- CLI/compiler logic is not a stable host-application compilation API.
- Internal STL/shared-pointer IR objects are not a portable public binary ABI.
- Runtime bindings do not yet own contexts/modules/devices/buffers/executions.
- Installation, exported CMake packages, relocatable deployment and complete
  binding distribution are missing.
- Exact conversion/serialization and structured numeric failure APIs need expansion.
- Native `__int128` and POSIX/native-tool assumptions need explicit capability
  boundaries; current successful Linux execution does not establish MSVC support.
- GPU validation/lowering rules currently live partly inside the CUDA emitter;
  extract them rather than independently redefining arithmetic for Vulkan.
- A Vulkan label/template is not program-dependent compilation or execution.
- Root AGPLv3 license text and Rust MIT metadata require a provenance/scope review
  before distribution. No license will be silently changed or rights inferred.

## 3. Proposed component architecture

```text
Public clients
  C ABI             C++ RAII             Python             Rust
    └──────────────────┴────────────────────┴─────────────────┘
                    versioned SDK facade
       context / diagnostic / module / function / buffer / device
                              │
           compiler service + mathematical contracts
               checked frontend → verified core IR
                              │
                    portable kernel profile
                 ┌────────────┼──────────────┐
              CPU core     CPU LLVM     GPU legalization
                                         ├─ uint32-word int39
                                         └─ native int64 (feature gated)
                                                │
                           Vulkan/SPIR-V         CUDA compatibility
                              runtime             existing path
```

Proposed build/package boundaries:

- `TriumXBT::Math`: C++20 header/library target, no GPU/LLVM runtime dependency.
- `TriumXBT::SDK`: stable C entry points and C++ wrappers; CPU-only build supported.
- Internal compiler/runtime/backend targets: implementation details, not a frozen
  public plugin ABI. The CLI calls the same compiler service as the SDK.
- Preserve `libbt_runtime` and ABI-major-1 entry points for existing consumers.
  New SDK symbols use `bt_sdk_*`; no silent replacement of existing layouts.
- GPU components are optional. Query/load a Vulkan loader on demand through a
  controlled function table, so lack of Vulkan does not prevent CPU SDK use.
- Compile-time shader tools and deployed runtime requirements are separate.
  An offline-generated GPU artifact should not require a shader compiler on the
  execution machine. CPU-only users should not need Vulkan or glslang to build.

Proposed paths include `include/Ternary/SDK/`, `lib/SDK/`,
`include/Ternary/Kernel/`, `lib/Backends/Vulkan/`, `cmake/`, `examples/sdk/`,
`test/SDK/` and `test/GPU/`. Exact layout is reviewed in P0, not frozen here.

## 4. Work packages, dependencies and completion gates

### P0 — Freeze scope, dependency policy, and integration contracts

**Work**

- Inventory supported math operations, library-vs-language types and ABI layouts.
- Write SDK ABI, ownership/threading, artifact and portable-kernel specifications.
- Choose compatible pinned Vulkan-Headers, glslang, SPIRV-Headers and SPIRV-Tools
  revisions; verify their build/tool environment together. Record licenses and
  dependency notices. No implicit network downloads during ordinary builds.
- Resolve licensing/provenance questions with rights holders before publication.
  Clarification may gate distribution without preventing compliant development.
- Define target tiers: Linux x86-64 first; additional hosts/devices only after
  their build/runtime tests. Record unavailable toolchains/hardware explicitly.
- Add feature flags and target-scoped CMake configuration, retaining legacy builds.

**Gate:** reviewed contracts, dependency lock/provenance, migration map and test
matrix exist; CPU baseline still builds. No changes to numerical semantics just
for convenience. **Dependency:** none. **Risk/effort:** medium, architectural.

### P1 — Advance the reusable mathematical foundation

**Work**

1. Provide common profile/limits/rounding traits. Retain current checked arithmetic,
   rational canonicalization/reduction-before-final-bounds, normalized BTFP and
   nearest-ties-away behavior; do not change defaults to gain speed. Source
   binary64 semantics remain separate; ternary rounding is not imposed on float.
2. Add structured math failures and nonthrowing `try_*` entry points. Distinguish
   invalid representation, zero divisor/domain, result overflow, declared exact
   accumulator capacity and unsupported operation. Do not infer categories by
   parsing exception text. Preserve existing throwing APIs and exception-family
   compatibility through adapters.
3. Add bounded exact integer/rational parsing and formatting, including decimal
   and explicit balanced-ternary formats. Define grammar, sign, whitespace,
   maximum input length and canonical roundtrip output before implementation.
4. Add exact rational/integer-to-BTFP conversion within declared accumulator
   capacities, and bounded exact decimal input through rational scaling. Reject
   capacity excess, never silently route an exact API through binary64. Keep
   existing approximate `from_double`/transcendentals clearly named/documented.
5. Define versioned mathematical wire formats: explicit widths, exponent policy,
   canonical fields and byte order, with bounded decoders. Never serialize native
   struct padding, `__int128`, or C++ object images as a portable format.
6. Audit cross-cancellation and gcd strategies for rational intermediates. Optimize
   only where value, reduction, failure and capacity contracts remain unchanged.
7. Isolate wide-integer implementation behind a tested internal layer. Native
   int128 remains the first qualified host path. Add a word-arithmetic path with
   its own tests before claiming unsupported compilers/platforms; no immediate
   all-MSVC/all-architecture promise.
8. Produce an explicit accumulator-bit/capacity table for sqrt/FMA/conversion.
   Evaluate a fixed wider-word extension for current restrictive cases (including
   W=40 sqrt), with proof and cost review before promising support. If added, use
   an explicitly versioned extended profile; retain the existing profile's failure
   semantics. This is bounded arithmetic, not a promise of arbitrary precision
   or correctly rounded transcendental functions.

**Tests/gate:** exhaustive tractable small widths; boundary/random testing across
all widths 1..40; Python bigint/Fraction oracle; BTFP binade/tie/exponent/capacity
cases; malformed/overlong decoder inputs; exact roundtrips; old regression corpus;
Z3/Rocq obligations for changed arithmetic. Report unsupported capacities exactly.
Separate API correctness from performance. **Dependency:** P0. **Effort:** large.

### P2 — Extract a callable compiler service and CPU SDK

**Work**

- Move orchestration out of `bt_compiler.cpp` into a library. No library path
  calls `exit`, invokes a shell, or writes unsolicited progress to stdout/stderr.
- Implement context creation, bounded source compilation, structured diagnostics,
  module/export inspection, callable function selection, and synchronous CPU
  execution. Keep the reference core path independent of optional native tools.
- Start with scalar CPU calls and pure int39 maps. GPU has a narrower profile;
  do not require all CPU functions to fit GPU restrictions.
- SDK module compilation does not execute the source's main body. Execution is
  explicit. Default export execution is pure/no-I/O; preserve CLI printing and
  program-exit semantics through the CLI adapter, not hidden host side effects.
- Native CPU emission uses the existing shared LLVM path where the qualified
  toolchain is available; explicit capability/error otherwise, not invented JIT.
- Preserve source spans/import provenance in diagnostics. Default SDK imports are
  disabled; add bounded caller-supplied virtual modules before exposing arbitrary
  filesystem access. Existing CLI import behavior remains compatible.

**Gate:** an independent C client compiles source and executes a supported export
inside its process, without a CLI subprocess; a C++ client does the same through
RAII wrappers. Failure paths and success produce structured status/diagnostics.
No internal IR/STL/LLVM/Vulkan types cross the C boundary. **Dependency:** P0/P1.
**Effort:** large; this is the central application-integration milestone.

### P3 — Stable ownership, binding, and packaging contracts

**C ABI design**

- ABI major/minor plus semantic-profile and capability negotiation; keep SDK,
  artifact, kernel and math-profile versions distinct.
- Fixed-width tags/counts and descriptors with `struct_size`, version and reserved
  fields. Validate overflow and conversions before allocating or indexing.
  Native C layouts are not wire formats.
- Opaque typed handles. Prefer context-scoped IDs with generation checks where
  stale-handle diagnostics are promised; never dereference an arbitrary freed
  pointer merely to validate it. Internal registry/refcount design is reviewed
  for contention and lifetime safety before freezing the ABI.
- Per-call status and owned diagnostics; asynchronous results must not rely on
  thread-local last-error state. Preserve that state for legacy ABI consumers.
- No C++ exception/panic crosses the ABI. Handle allocation failure without
  allocating another required error object; provide a minimal fallback status.
- Document concurrent calls, context closing, release order and resource retention.
  A call retains the objects it uses. Closing/releasing resources cannot free
  an allocation still used by another accepted invocation.
- Reject wrong context, unsupported dtype/layout, out-of-range int39 values,
  inconsistent lengths, read-only output and prohibited aliasing. C callers still
  must supply valid pointer/length pairs; the SDK is not a memory sandbox.

**Clients and packaging**

- C++ RAII/value wrappers; Python package with context managers/exceptions and
  trusted library loading; Rust owned handles/`Result` and justified `Send`/`Sync`
  implementations only after concurrency tests. Raw pointer interfaces stay unsafe.
- Math APIs are available without invoking the compiler. Initial buffers are
  contiguous CPU int64/rank-1 maps; conversion/copy policies are explicit.
- Add NumPy integration as an optional dependency. Validate byte order, dtype,
  shape, strides, size and ownership; initially reject unsupported/negative strides
  rather than silently aliasing or guessing a buffer's accessible storage.
- Implement CMake install/export/version files, pkg-config metadata if useful,
  and clean consumer examples. Python/Rust release packaging must find the correct
  native library without source-tree paths or ad-hoc `RUSTFLAGS`.
- Start with source/installable native packages on the qualified platform; signed
  multi-platform binary wheels and support claims require their own build matrix.

**Gate:** install, relocate, build/run C/C++ consumers outside the checkout; Python
and Rust exercise the installed ABI; ABI-major mismatch rejected; ASan/UBSan and
threaded lifetime/error tests pass. Missing Rust tooling is an explicit pending
check, not a passing release gate. **Dependency:** P2. **Effort:** large.

### P4 — Portable GPU kernel IR and checked word arithmetic

**First kernel profile**

- Selected pure `(int39) -> int39` elementwise maps over contiguous buffers.
- Scalar locals, checked arithmetic, comparisons, digit/intrinsic operations,
  branches and lazy selection supported according to an enumerated operation
  matrix. No source loops, recursion, user calls, I/O, shared-memory reductions,
  arbitrary pointers or floating/BTFP/rational GPU execution in the first profile.
- Extract validation from CUDA-specific emission. Specify binding numbers, element
  counts, byte sizes/strides, lane status, inactive-lane behavior and output commit.
- Backend-generated fixed-bound arithmetic helper loops are implementation details
  verified separately; they do not admit unrestricted source loops.

**Representations**

- Portable baseline: explicit uint32 word pairs for the int64/int39 representation;
  controlled word carries, comparison, signed conversion, shifts and packing.
- Checked multiplication uses sufficiently wide word intermediates or a proven
  precheck; never multiply with undefined/unchecked overflow and inspect it later.
- Optional native-int64 path only after the device feature is queried/enabled.
- Both paths preserve the same accepted values and error conditions as the CPU.
  Define host byte encoding and on-device layout rather than aliasing native C
  structs. Int39 is logical metadata over storage, not a new arbitrary NumPy dtype.

**Gate:** independent word-arithmetic tests including sign extremes, H boundaries,
all relevant shift widths, zero, multiplication overflow and digit operations;
Z3 bitvector refinement and appropriate Rocq lemmas; CPU/core/CUDA differential
agreement; profile/ABI verifier rejects unsupported kernels. **Dependency:** P1/P2.
**Effort:** large/high correctness risk; complete this before runtime tuning.

### P5 — Program-dependent Vulkan/SPIR-V lowering

**Work**

- First deployment profile: Vulkan 1.1 / its compatible SPIR-V environment, with
  minimal compute/storage-buffer features and explicit runtime limits. This is a
  proposed floor; P0 may raise it only with a recorded compatibility reason.
- Generate typed GLSL for the actual selected kernel IR, using controlled numeric
  identifiers and structured branches. No shader template unrelated to the body.
- Compile using pinned glslang; validate emitted/transformed SPIR-V using the exact
  target environment. Check actual SPIR-V interface/bindings against metadata;
  glslang source-level reflection alone is not a complete binary ABI guarantee.
- Record source/profile/compiler/backend versions, required capabilities, logical
  dtype, buffer contracts and binary checksums in bounded artifact metadata.
- Deterministic compilation inputs/tool paths; private temporary directories and
  argv-only tool execution. Do not load untrusted arbitrary executables from a
  cache or imply a checksum authenticates an artifact's publisher.
- Select a minimal optimization pass set; validate and differentially test it.

**Gate:** several distinct source kernels produce valid, distinct SPIR-V;
portable-word artifacts do not declare Int64; native variants require it;
unsupported profiles reject cleanly; interface and malformed-artifact tests pass.
This closes an offline-emission gate, not a device-execution gate.
**Dependency:** P4 and P0 toolchain. **Effort:** medium/large.

### P6 — Actual Vulkan compute runtime

**Work**

- Discover loader/instance/devices/compute queues; query limits/features and enable
  only required supported features. Support portability enumeration where needed.
- Implement one compute queue first: buffers, memory types, staging, descriptor
  sets, pipeline layout, compute pipeline, command recording, dispatch and fences.
- Handle bounds/index/count overflow, storage-buffer limits, workgroup limits,
  descriptor alignment, zero-length batches and inactive lanes before dispatch.
- Correct memory visibility, barriers, noncoherent flush/invalidate alignment and
  completed readback. Begin with simple staging; defer pooling/UMA shortcuts.
- Use scratch output/status and commit to host output only after success. Detect
  numeric lane errors, allocation/submission failure and device loss. No partial
  host output is returned as a successful result.
- Failure-safe cleanup and retained ownership on every path. Respect the runtime's
  own device/queue objects; do not disturb unrelated application device state.

**Gate:** mocked Vulkan ownership/failure tests plus actual Vulkan API execution on
an available software ICD; validation layers where available; then identical
value/error comparisons on physical devices. Software Vulkan is labeled software
execution, not GPU hardware qualification. If no ICD can be provisioned, execution
remains pending; fake functions do not substitute for this gate.
**Dependency:** P3/P5. **Effort:** large/high synchronization risk.

### P7 — Explicit dispatch, diagnostics and hardware-aware tuning

**Work**

- Device/capability inspection and explainable backend/representation selection.
- Policies: `require_gpu` fails if unavailable; explicit CPU executes CPU;
  `prefer_gpu` may use CPU only with fallback permission and a reported reason.
  Auto mode starts with legal static choices; no universal-fastest promise.
- Bounded optional tuning of valid workgroup sizes and representations. Verify
  outputs/status against the reference before accepting a candidate. Enforce
  scratch-memory and tuning-time budgets and keep a no-tuning mode.
- Cache by kernel/profile/compiler/toolchain/device/driver/features and relevant
  workload size. Invalidate stale entries and use controlled cache permissions.
- Report compile, allocation, upload, dispatch/synchronization, download and
  end-to-end boundaries separately. CPU-vs-GPU routing uses measured end-to-end
  costs when qualified, not theoretical FLOPS or vendor-name guesses.
- Expose selected backend, device type, numerical profile, representation, copies,
  fallback, tuning decision and timing availability to clients.

**Gate:** reproducible static selection without hardware; fake capability tests;
real tuning/performance gates only on actual measured devices. Report regressions
and use applicability guards, not cherry-picked speedups. **Dependency:** P6.
**Effort:** medium/large; physical measurements can block performance qualification.

### P8 — Deeper integration and asynchronous execution (follow-on release)

Build after the synchronous SDK is stable:

- Owned event/future objects, completion/error records and retained buffers.
  Before-submit cancellation may be supported; submitted GPU work is not promised
  safe preemption. A wait timeout is not completion and must not free live memory.
- Borrowed host memory initially stays synchronous; async requires owned/pinned
  storage or a tested lifetime contract. Python owners must survive native calls;
  Rust safe APIs must not permit invalid lifetimes across submission.
- Add CPU DLPack exchange with version, device/dtype/stride/offset/read-only and
  exactly-once deleter tests. Unsupported device tensors are rejected or copied
  through an explicit host policy; no arbitrary CUDA/Vulkan pointer conversion.
- Add one concrete framework adapter/example. Data interchange is not automatic
  lowering of arbitrary PyTorch/NumPy/JAX graphs.
- Design ordered reductions/dot products and then shared-core tensor/CSR storage
  as separate numerical contracts. Checked reduction order affects errors; do not
  parallel-reassociate an intermediate-overflowing sum under an 'exact' claim.

**Gate:** lifetime/race/failure-injection tests, framework reference comparisons,
layout/overflow checks, explicit copy/synchronization diagnostics. GPU external
memory/stream interoperability and reduction profiles need separate approval.
**Dependency:** P3/P6, then dedicated design reviews. **Effort:** large/optional R3.

### P9 — Qualification, examples and release evidence

- Preserve the existing Debug/Release/sanitizer, independent program/numeric,
  Z3 and Coq-kernel corpus. Add per-package install/ABI/fuzz tests and execute the
  configured Rust/native-Rocq gates where tools are available.
- Source/math/kernel/artifact fuzzing with deterministic seeds and minimized saved
  regressions; long-running campaigns separate from short PR tests.
- Publish host/compiler/driver/physical-device/profile matrix, exact versions,
  commands, actual outcomes, skip reasons, samples and limits.
- CPU-only, no-loader, software-ICD, feature-limited, native-int64 and failure paths
  are distinct tests. Emission-only tests do not count as execution tests.
- Linux first. Add actual AMD/Intel/NVIDIA device entries as access permits.
  Apple evaluates the current translation-layer ecosystem separately; Android,
  Windows and physical ARM hosts have their own packaging/runtime gates.
- License/dependency notices and provenance reviewed before publishing artifacts.
  No release is labeled generally cross-vendor-qualified on software-only tests.

**Gate:** independent consumers reproduce the supported end-to-end workflow; every
advertised platform/profile has evidence; unavailable configurations remain
unqualified. Qualification runs throughout the preceding packages.

## 5. Release slices and approval boundaries

| Slice | Scope | Usable result | What it must not claim |
|---|---|---|---|
| R1: math + CPU SDK | P0–P3, relevant P9 | Installable mathematical library and in-process C/C++/Python/Rust compiler/runtime API on qualified hosts | GPU portability, all-host support, arbitrary framework compilation |
| R2: portable GPU implementation | P4–P7, relevant P9 | Program-dependent Vulkan backend/runtime and capability selection; offline/software checks where available | Physical compatibility/performance for devices not actually tested |
| R3: qualified integration expansion | Physical P9 plus approved P8 items | Named GPU qualification, measured tuning, async/CPU DLPack and selected larger workloads | Universal GPU, zero-copy, graph compilation or multi-device execution |
| Later separate work | Additional contracts | Multi-device scheduling, further GPU APIs/frameworks, larger compiler profiles | Automatic completion from R2 |

Recommended approval: the whole **R1 then R2 program**, with reviewable delivery
checkpoints; approve P8's larger integrations individually after those foundations
work. Do not wait for hardware to deliver R1 or offline R2 work, but do not mark
blocked execution/performance gates passed. No reliable calendar estimate is
asserted before P0 toolchain provisioning and integration complexity are checked.

Critical path: P0 → P1/P2 → P3 → P4 → P5 → P6 → P7.
Math validation and packaging tests can progress independently of physical GPUs.
P8 is not a hidden prerequisite that delays all useful R1/R2 delivery.

## 6. How users would use the result

The following interfaces are **illustrative proposed UX, not existing APIs**.
Exact spelling is finalized with the ABI design, before implementation.

### Library-only C++ application

```cmake
find_package(TriumXBT CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE TriumXBT::Math)
```

Use checked integers/rationals/BTFP, exact parsing/conversion where supported,
structured errors and documented serialization without installing a GPU stack.
The existing value APIs remain usable; new `try_*` helpers make failure handling
explicit for embedded applications.

### Python numerical workflow

```python
# PROPOSED; not executable against today's repository
with trium.Context(backend="cpu") as context:
    module = context.compile(source)
    transform = module.function("transform")
    output = transform.map(numpy_int64_input)
```

After R2, choose `backend="vulkan", require_gpu=True` to require a compatible
Vulkan device. An explicitly permitted fallback reports the actual executor and
reason. NumPy int64 is storage; values are still checked against int39. The SDK
does not compile arbitrary Python syntax or every NumPy operation.

### C/C++ service or Rust application

Create a context, compile/load an approved module, inspect its export/profile,
upload or bind a supported buffer, invoke it and inspect structured status.
C++ RAII/Rust ownership manage cleanup; callers do not need to construct LLVM,
SPIR-V or Vulkan objects. The stable C ABI is the common implementation boundary,
not separate numerical reimplementations per binding.

### Offline compilation and deployed execution

Generate a versioned artifact and capability manifest on a build machine. A
runtime-only deployment validates the artifact/profile, chooses a compatible
execution path and reports the device used. Compiling source is optional at
runtime; tool dependencies and trusted-artifact policies are explicit.

## 7. How this advances the compiler

- **Mathematics becomes a product boundary:** exact inputs/outputs, stable profiles,
  reusable error handling and formats, not just arithmetic buried in headers.
- **Integration becomes real:** applications call a supported SDK and retain
  control of resources/diagnostics instead of scraping CLI output.
- **Portability becomes capability-based:** one declared kernel meaning can be
  legalized to devices with different integer features without truncation.
- **Optimization becomes inspectable:** users can see and measure the chosen
  device/representation, avoid unexpected copies and disable tuning/fallback.
- **Deployment becomes reproducible:** versioned packages/artifacts, installed
  consumer tests and provenance replace source-tree/environment assumptions.
- **Future hardware benefits without blocking software:** a cleaner semantic,
  kernel and runtime boundary can later connect to the independent Stage 6 target
  model; it does not itself prove RTL/synthesis correctness.

Success is a well-tested, usable compiler SDK for declared workflows—not a claim
that every program, GPU, numerical format or hardware design is already covered.

## 8. Approved scope and implementation tracking

The user approved proceeding with R1/R2, keeping physical GPU tests isolated for
personal laptop/cloud execution. See the delivery report for implemented code,
actual software execution and remaining acceptance work. In particular, the recommended
initial boundary is **R1 + R2: advanced reusable mathematics, a real CPU SDK, and
restricted cross-vendor Vulkan map execution**, retaining explicit hardware gates.
P8 expansion, multi-device scheduling and Stage 6 synthesis are not silently
included in that first delivery commitment.
