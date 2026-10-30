# R1/R2 executable delivery: mathematics, embedding SDK and portable GPU maps

> Subsequent focused numerical delivery: [FLOATING_POINT_FOUNDATION.md](FLOATING_POINT_FOUNDATION.md).
> Historical limits/counts below describe the earlier snapshot; W40 sqrt and
> wider FMA alignment are now implemented and separately validated.

Date: **2026-10-29**. Branch: `arena/f3c30e8e-trium-xbt-compiler`.
README unchanged. The R1/R2 program is approved; this is the first scoped
executable delivery, not certification of every item in the larger roadmap.

## What is implemented now

### R1 — reusable mathematics and application integration

- `Math/Interchange.hpp`: bounded integer/rational/decimal parsing, canonical
  rational reduction, nonthrowing result adapters, checked integer↔dual-rail
  conversion, strict trit text, versioned little-endian dual-rail/rational wires.
- `Math/Float.hpp`: exact decimal→bounded rational→BTFP ratio rounding, without
  a binary64 intermediate. This does not make every finite BTFP result exact.
- Checked CPU row-major int39 matrix multiplication, with checked products and
  increasing-k sums, shape/size/resource checks and transactional output.
- New `libtrium_sdk`, SDK ABI 1, process-local generation-free nonreused handle IDs,
  bounded self-contained source compilation, function export enumeration, pure
  scalar invocation and pure int39 batch maps. No CLI subprocess is used for CPU
  compilation/execution; existing frontend/typechecker/core are reused.
- Synchronous C API with caller-owned diagnostics, guarded exception boundaries,
  context closing/handle lifetime rules and immutable-module concurrency.
- C++ RAII wrappers; Python context managers, scalar calls/maps, exact integer
  validation, optional strict contiguous native-int64 NumPy arrays, matrix and
  dual-rail APIs. Arrays are copied for ownership/transactional semantics, not
  advertised as zero-copy.
- Owned Rust context/module bindings and tests authored, retaining contexts through
  `Arc`; **Rust execution remains a separate pending CI/toolchain check**.
- Relocatable CMake packages `TriumXBT::Math` / `TriumXBT::SDK`; C and C++ clients
  build/run outside the checkout after installation and relocation. A pure-Python
  binding wheel builds and its installed CPU use passes; it requires the native SDK separately, not a bundled
  multi-platform binary runtime.

The SDK's current CPU executor is the verified core interpreter. The existing
CLI's native LLVM route remains available, but arbitrary native module invocation
is not yet exposed as an SDK execution backend. Binary64 scalar calls support
loops/functions/recursion under core limits; the GPU map subset is narrower.
SDK imports and top-level execution are rejected; execution is always explicit.

### R2 — actual Vulkan/SPIR-V map implementation

- Shared `Kernel/Map.hpp` validates pure loop-free `(int39)->int39` functions.
- Program-dependent generated GLSL is compiled through pinned glslang and checked
  by SPIRV-Tools for Vulkan 1.1, not replaced by an unrelated shader template.
- Baseline uint32-word arithmetic preserves int39 without shader Int64 support.
  Multiplication computes/checks a full magnitude before signed conversion.
  Conditions, lazy selection, locals, returns, digit logic and supported bit
  intrinsics preserve the scalar reference profile.
- Optional native64 multiplication path: feature query/enable, division-by-zero-safe
  overflow precheck and the same buffer ABI. Explicit `native64` rejects missing
  `shaderInt64`; `auto` selects a legal feature-based path. Auto means legal
  representation selection, **not measured fastest-kernel selection**.
- Actual Vulkan loader/ICD calls: instance/device/queue discovery, portability
  enumeration, allocations, staging, descriptors, pipelines, barriers, dispatch,
  fences, coherent/noncoherent host visibility, status/readback and cleanup.
- Explicit backend selection, no silent CPU fallback. Device UUID in execution
  metrics is checked against enumeration by the benchmark runner to prevent
  incorrect device attribution if enumeration changes.
- Optional queue timestamps, host staging/submit-to-fence/total timings, device
  metadata, per-run samples and independent output checks.
- `tools/qualify_portable_gpu.py` isolates user hardware qualification and writes
  outcomes/logs/benchmarks. `tools/benchmark_portable_gpu.py` sweeps devices,
  32/64/128 workgroups and supported representations. Devices execute sequentially,
  not as a partitioned multi-GPU computation.
- CPU-only SDK builds do not require Vulkan headers/loader. GPU build dependencies
  are optional; source tool revisions are pinned by commit in
  `requirements/portable_gpu_tools.json` and provisioned by an explicit script.

## What was actually executed here

Local platform: Linux x86-64, GCC 12.2.0, C++20, Python 3.11.2, CMake 4.4.4,
Ninja 1.13.2, llvmlite 0.50.0/LLVM 22.1.0 and a Zig-provided Clang 22.1.8 adapter.

**Actual Vulkan execution passed through SwiftShader's Subzero CPU backend.**
This exercises shader compilation and the Vulkan allocation/dispatch/readback
path, unlike a fake driver. It is still **software execution on a CPU**, not
NVIDIA/AMD/Intel physical GPU qualification. The ICD is loaded directly; this
also does not qualify every system Vulkan loader path.

| Final configuration | Passed | Skipped | Failed | Time |
|---|---:|---:|---:|---:|
| Release, optional Vulkan enabled | 41 | 2 | 0 | 12.41 s |
| Debug + ASan/UBSan, optional Vulkan enabled | 41 | 2 | 0 | 30.66 s |

The skips are physical CUDA qualification and native Rocq. Vulkan API/software
execution is now an executed test, not a skip. Python-hosted sanitizer tests
preload ASan/libstdc++ and disable CPython host leak reporting; ordinary C++ tests
retain leak detection. No final sanitizer failures were detected.

Additional checks:

- Separate GPU-disabled SDK build and in-process integration tests passed.
- Relocated installed C/C++ consumers and Python wheel construction and isolated installation/invocation passed.
- All **eleven** root Z3 scripts passed, including new word-addition, multiplication
  decomposition, native64 division guards, capacity and wrapped-product
  counterexample obligations. These are model obligations, not whole-shader proofs.
- New math tests include more than 200,000 scalar/rail roundtrips, full-int64
  extrema, exhaustive small widths, all 40 library widths, malformed/canonical
  formats and 1,000 matrix reference cases.
- Ten distinct portable shader bodies were compared with CPU/independent values,
  including digit logic, BTHA masks, branching and lazy failures. Overflow cases
  include `H*9`, which can look valid after an incorrect int64 wrap.
- Native64 shaders validate offline. SwiftShader advertises no shader Int64:
  explicit native64 rejection and auto→portable execution passed. **Native64
  device execution remains for compatible hardware testing.**
- Physical-only qualification rejected this software-only environment with exit
  77 and `physical_execution: not_performed`; software-allowed qualification passed.

The earlier 66 Coq-kernel results remain historical evidence for unchanged proof
sources; no new whole-compiler or whole-SDK refinement proof is claimed.

Logs/provenance: [evidence](evidence/2026-10-29-r1-r2/).
Measured local samples: [software Vulkan benchmark JSON](benchmarks/2026-10-29-software-vulkan.json).
Those samples contain three CPU workloads and nine software-Vulkan workload/group
configurations, 1,000 elements and three repetitions, with every output checked.
They are smoke measurements of this software path—not GPU performance evidence.
Device/pipeline setup and shader compilation currently recur per call; the JSON
separates timestamp bands from full end-to-end cost. No general speedup is claimed.

## Using the result

See [HARDWARE_GPU_QUALIFICATION.md](HARDWARE_GPU_QUALIFICATION.md) for dependency
provisioning, build commands and the isolated command to run on your own machines.

```python
# PYTHONPATH=$PWD/ffi/python
# BT_SDK_LIBRARY=$PWD/build/sdk/libtrium_sdk.so
import trium
with trium.Context(representation='portable32') as context:
    with context.compile('func transform(x:int)->int{return x*3;}') as module:
        output, metrics = module.map('transform', [-2, 0, 2], backend='cpu')
        assert output == [-6, 0, 6]
        # Explicitly require a Vulkan path; missing loader/device is an error.
        # output, metrics = module.map('transform', [-2, 0, 2], backend='vulkan')
```

Use `representation='native64'` for an explicit feature requirement, or `'auto'`
for capability-based selection. Offline `module.shader(...)` uses native64 only
for an explicit native64 context; auto exports the portable baseline. Runtime
metrics report the actual selected representation and device UUID.

The C/C++ examples in `examples/sdk/` consume the installed SDK, not internal IR.
C++ modules require their context to remain open. Rust modules retain their owning
context; Python explicit context-manager closure invalidates future invocations
while accepted native calls retain the resources needed to finish.

## Boundaries and continuing work

This delivery does not close every P0–P9 acceptance item from the approved plan.
Continue with native CPU SDK invocation, richer export/diagnostic metadata,
stronger math-capacity taxonomy/extensions, a portable host-wide-integer backend,
persistent device/pipeline caches, measured cost-based tuning and versioned
loadable artifacts. Async events, DLPack/device interoperability, larger tensor/
reduction profiles and framework graph adapters remain later scoped work.

No GPU dense GEMM, FP16/BF16 tensor kernels, complete AI training framework or
universal simulation solver is claimed. The matrix SDK API is CPU/int39. For
an explanation of exact versus approximate binary↔ternary conversion and dual
rail, see [MATH_AND_WORKFLOW_CAPABILITIES.md](MATH_AND_WORKFLOW_CAPABILITIES.md).

Physical device/driver qualification, native64 execution on supporting devices,
Apple/Windows packaging, Vulkan validation layers and real device-loss/noncoherent
paths remain explicit gates. User-run logs can qualify each named configuration
without blocking the implemented CPU/software-validation work.

License texts/metadata were not relabeled. Existing root AGPLv3 text and Rust MIT
metadata still need their component boundaries documented before publication;
MIT wrapper metadata is not a statement that the entire native SDK is MIT.
See [LICENSE_BOUNDARIES.md](LICENSE_BOUNDARIES.md) for the distribution review
checklist. Stage 6 remains the independent future ternary/mixed-radix hardware track.
