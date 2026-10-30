# SDK and portable GPU design research

Reviewed: **2026-10-29**. Planning evidence, not implementation or qualification.
Subsequent approved implementation provisioned pinned shader tools and a
SwiftShader CPU Vulkan runtime; see [R1_R2_DELIVERY.md](R1_R2_DELIVERY.md).
The tool-availability observations below describe the earlier research baseline.
Primary documents were retrieved through GitHub's API. References below pin the
reviewed file revision; upstream behavior and recommendations may evolve.

## Repository findings

- `include/Ternary/Math/{Integer,Rational}.hpp` already separates bounded exact
  mathematics from execution. Both use native `__int128`, so availability on all
  C++ compilers must not be assumed. `ternary_fp.hpp` has its own wide arithmetic,
  normalization and capacity rules; unifying internals must preserve those rules.
- `include/Ternary/IR/Program.hpp` is an internal C++ model using STL types and
  shared pointers. Exporting these objects would couple consumers to C++ ABI and
  implementation details. It is not a stable cross-language binary interface.
- `include/Ternary/ternary_c_abi.h` and current bindings expose runtime operations,
  not a full context/compiler/module/buffer/device SDK. Existing error-state and
  ABI-major negotiation are useful compatibility foundations.
- `CMakeLists.txt` creates libraries/executables but has no install/export package
  rules. Global include/link/options need target-scoped cleanup before packaging.
- `ffi/python/bt_ffi.py` can search the working directory for a native library;
  the new SDK should use package-relative or explicit trusted paths, rather than
  silently loading a same-named library from an arbitrary application directory.
- `LICENSE` contains AGPLv3 text; `ffi/rust/Cargo.toml` declares MIT. This is a
  provenance/licensing question, not permission to relicense either component.
  Resolve the intended scope with rights holders before release/publication.
- The CUDA emitter/runtime is real but restricted and hardware-unqualified. The
  Vulkan/Metal labels or templates elsewhere in the repository are not an
  implemented portable program compiler/runtime.
- Local reconnaissance did not find `glslangValidator`, `spirv-val`, `vulkaninfo`,
  Rust tools or an advertised Vulkan loader in `ldconfig`. This is a setup
  observation, not a claim that provisioning or software Vulkan is impossible.

## Findings that determine the architecture

### R1 — Vulkan features must be queried and explicitly enabled

The Vulkan Guide explains capability querying and device-creation feature
selection. The specification forbids 64-bit shader integer types when
`shaderInt64` is not enabled. Consequently:

- Never infer integer support from a vendor name or API version alone.
- Define a portable uint32-word representation and an optional native-int64
  lowering. Prove equivalent values/errors for the supported operations.
- Make the chosen representation and required features inspectable.

References: [feature negotiation](https://github.com/KhronosGroup/Vulkan-Guide/blob/f71c59694335577fb1ec85b17448a53120ef7478/chapters/enabling_features.adoc),
[specification feature definitions](https://github.com/KhronosGroup/Vulkan-Docs/blob/a8b088a3fae12c8575407ea90c7c09bd653db923/chapters/features.adoc).

### R2 — Reuse the shader toolchain; validation is necessary, not sufficient

Glslang supplies GLSL-to-SPIR-V translation. SPIRV-Tools supplies assembly,
parsing, validation and optimization. Its documentation explicitly describes the
validator as incomplete. Therefore use generated, typed, program-dependent GLSL
as a bounded first lowering, validate SPIR-V for the exact target environment,
and also retain independent arithmetic, runtime, ABI and differential tests.
Do not claim that passing `spirv-val` proves compiler correctness. Restrict
SPIR-V optimization passes initially and validate after transformation.

References: [glslang](https://github.com/KhronosGroup/glslang/blob/0971d79400aa5811d24bb696a7a69565f5c4cb3b/README.md),
[SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools/blob/e39e5c5838bc4b4162c349f2a2e5f163efe5432f/README.md).

The queried release endpoints reported glslang **16.6.0** and SPIRV-Tools
**v2026.4**; Vulkan-Headers listed SDK tags including **vulkan-sdk-1.4.341.0**.
These observations are not a tested dependency combination. P0 must select and
record a mutually compatible pinned tool/header set, rather than independently
fetching whatever is newest.

### R3 — Device memory and synchronization are part of correctness

Vulkan separates memory types and requires explicit synchronization. Integrated
and discrete devices can have materially different host-visible/device-local
memory arrangements. Noncoherent mapped memory needs correctly aligned
flush/invalidate handling. A completed allocation/copy API call alone is not an
execution or visibility guarantee.

First implement a single compute queue with explicit staging, barriers, fence
completion and transactional host output. Optimize allocation/suballocation and
UMA paths only after that reference path is correct.

References: [memory allocation](https://github.com/KhronosGroup/Vulkan-Guide/blob/c42022d36a692a1b558588da7c09cda521512e54/chapters/memory_allocation.adoc),
[synchronization examples](https://github.com/KhronosGroup/Vulkan-Guide/blob/f71c59694335577fb1ec85b17448a53120ef7478/chapters/synchronization_examples.adoc).

### R4 — Apple support requires an explicit deployment decision

The reviewed MoltenVK README contains a deprecation/best-effort-maintenance
notice and recommends KosmicKrisp for Apple Silicon, while retaining MoltenVK as
its recommendation for Intel Macs and iOS. It also documents portability
enumeration and feature limitations. Do not carry an older blanket MoltenVK
recommendation into the support matrix.

Evaluate the then-supported Apple translation-layer/SDK combination in a separate
qualification gate. No Apple hardware/backend claim follows from Linux Vulkan
success. A dedicated Metal backend remains a possible later decision, not a
second simultaneous implementation commitment.

References: [MoltenVK README](https://github.com/KhronosGroup/MoltenVK/blob/0aa76ad7800f3c470103e2004d9a3dc0401693ee/README.md),
[portability initiative](https://github.com/KhronosGroup/Vulkan-Guide/blob/f71c59694335577fb1ec85b17448a53120ef7478/chapters/portability_initiative.adoc).

### R5 — Buffer interoperability requires ownership, not just an address

Python's buffer protocol specifies contiguity, format, read-only state, strides
and release of the exporter's retained buffer. DLPack specifies device identity,
offset, versioning, layout and deleter ownership. The reviewed DLPack 1.3 header
also records stricter stride requirements starting in 1.2.

Begin with validated contiguous CPU int64 buffers and retained owners. Add a
version-negotiated DLPack adapter only for explicitly supported CPU devices and
dtypes, with exactly-once release tests. Do not reinterpret a Vulkan buffer handle
as a CUDA pointer, assume a framework's stream is compatible, or use standard
float metadata to describe ternary BTFP storage. CPU buffer integration does not
mean arbitrary framework operations are compiled.

References: [Python buffer protocol](https://github.com/python/cpython/blob/d7ea5ba95e2e6fb821a3dd30ff955fce21b82502/Doc/c-api/buffer.rst),
[DLPack header](https://github.com/dmlc/dlpack/blob/94485e23519bc30e22a18397a9c8ff1ea6b2b81e/include/dlpack/dlpack.h).

### R6 — A reusable package must work outside this checkout

CMake's package guide covers installation, exported/imported namespaced targets,
version files and relocatable packages. Build-tree success is insufficient.
Install to a temporary prefix, move the prefix, and build independent consumers
without repository include paths or build-directory environment assumptions.
Keep target dependencies/features explicit and make GPU dependencies optional.

Reference: [CMake importing/exporting guide](https://github.com/Kitware/CMake/blob/9c963de99854587248d340c38f15eb44c00aeacc/Help/guide/importing-exporting/index.rst).

## Design choices versus alternatives

| Question | Proposed decision | Reason / deferred alternative |
|---|---|---|
| Public binary interface | Versioned C ABI with opaque handles | Avoid STL/compiler/backend types at the boundary; C++/Python/Rust wrap one implementation |
| External backend plugins | Internal backend interface first | Do not freeze an untested plugin ABI; external plugins require a later compatibility/security design |
| Portable GPU codegen | Typed kernel IR → generated GLSL → glslang → SPIR-V | Reuse structured-control-flow/toolchain infrastructure; direct SPIR-V or upstream MLIR lowering remains possible later |
| First GPU API | Vulkan compute, retaining existing CUDA path | One additional genuine runtime before API proliferation; not a claim of every GPU/OS |
| Baseline arithmetic | Checked int39 using uint32 words, optional native int64 | Preserve the domain where optional 64-bit features are absent; performance must be measured |
| CPU fallback | Explicit policy and reported actual executor | Preserve semantics without disguising CPU execution as GPU execution |
| Tuning | Bounded, optional, reference-checked, device/version-keyed | No automatic-fastest promise; static legal defaults work without tuning |
| Framework integration | CPU buffers/NumPy first; gated CPU DLPack next | Useful host integration without falsely promising zero-copy GPU interop or whole-framework compilation |

The implementation proposal and approval boundary are in
[SDK_MATH_GPU_IMPLEMENTATION_PLAN.md](SDK_MATH_GPU_IMPLEMENTATION_PLAN.md).
