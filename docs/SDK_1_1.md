# SDK 1.1 — compiled execution and reusable application integration

**Milestone: 2026-10-30.** This develops the existing compiler/math SDK rather than starting the future hardware track. ABI major **1** is preserved; API version is **1.1** (`0x00010001`), shared-library/package version **1.1.0**, SONAME major **1**. README and license grants are unchanged.

## 1. What applications can do now

- Compile a self-contained source module in-process, discover its exported signatures, and retain it across many calls.
- Explicitly choose reference CPU execution, compiled native CPU execution, or a supported Vulkan integral map. There is **no hidden fallback**.
- Invoke mixed typed scalar functions through a uniform C interface; use the C++ or Python convenience bindings without writing a compiler driver.
- Reuse native shared objects and Vulkan pipelines, devices and dispatch storage, instead of compiling/allocating everything for every call.
- Inspect API/layout information, structural backend eligibility, cache counters and configured bounds; explicitly invalidate caches.
- Independently use the existing exact integer/rational/interchange, finite ternary floating-point, checked integer matrix and dual-rail APIs. Those math APIs do not require compiling a source program.

The underlying mathematics remains available on ordinary binary hardware. Native execution uses conventional CPU instructions and checked helpers; Vulkan uses SPIR-V and binary GPU storage. Neither implies physical ternary execution.

## 2. Explicit execution profiles

| Backend | API selection | Supported SDK execution |
|---|---|---|
| Reference | `BT_SDK_CPU = 0`, Python `cpu` | Existing scalar interpreter, including supported loops/calls and binary64 operations; work/call-depth limits apply; source IO is forbidden |
| Vulkan | `BT_SDK_VULKAN = 1`, Python `vulkan` | Pure integral map-v1, `(int39) -> int39`; portable32 by default, feature-gated native64 multiplication; actual device selection and attribution |
| Native CPU | `BT_SDK_NATIVE = 2`, Python `native` | `native-v1-loopfree`: pure scalar int39, trit, binary64 and void functions; no source loops, calls, IO or packed values; scalar invocation and integral map-v1 |

Native compilation emits verified LLVM IR with a uniform raw-word wrapper, invokes a trusted host Clang-compatible compiler, loads a private shared object, and caches the executable. This is actual compiled execution, not the interpreter relabeled. The wrapper transports binary64 bits without integer reinterpretation errors; per-thread error state is reset on entry. Checked native arithmetic failure is reported before committing output.

Source `.bt float` is **IEEE binary64**, not BTFP. The SDK's finite BTFP operations use the distinct [finite floating specification](../specs/ternary-float-v1.json) and [qualification](FLOATING_POINT_FOUNDATION.md). This milestone does not lower BTFP arithmetic to GPU/native source types or certify transcendental functions. Binary64 comparisons were exercised in the default host floating environment; arbitrary host rounding-mode/libm/platform equivalence is not claimed.

Structural metadata answers “is this function in this profile?” It does **not** establish that a compiler, shader toolchain, loader or compatible device is operational. Native compiler candidate paths are captured when a context is created. Global SDK info shows current discovery; context cache info reports its captured path. Set `BT_SDK_CLANG` to an absolute trusted executable before creating a new context when overriding discovery. An explicit invalid override does not silently select another compiler or the interpreter.

## 3. Discovery, errors and versioning

The new additive C entry points are `bt_sdk_api_version`, `bt_sdk_info`, `bt_sdk_module_info`, `bt_sdk_cache_info`, `bt_sdk_cache_clear` and `bt_sdk_invoke_ex`. Existing `bt_sdk_invoke` remains reference execution. Existing ABI-1 structures are unchanged: options 40 bytes, metrics 96, scalar value 24, finite-float record 24 on the tested ABI. The binding validates these layouts.

JSON discovery exposes parameter indices/types, result type, reference/native/Vulkan structural eligibility, native rejection reason, static native work, backend/profile IDs, ABI layouts and cache bounds. Export lookup is indexed. Transitive IO eligibility is computed by reverse-call-graph propagation, including cycles, in linear graph work rather than repeated transitive searches for every function. Metadata is initialized under the source-cache lock before publishing handles, then immutable and reused.

Queries support a length request followed by a copy. Wrappers bound allocations and retry size changes; cache counters can change between queries and are not a globally atomic performance snapshot. Source parsing/type diagnostics retain their existing source locations in the message. This is not yet a separate machine-readable source-span/diagnostic-tree API.

| Status | Meaning |
|---|---|
| 0 | Success |
| 1 | Invalid input, handle, signature, profile, math-domain precondition or insufficient query buffer |
| 2 | Checked arithmetic/range failure |
| 3 | Other internal/tool integration failure |
| 4 | Requested backend/tool/device/profile unavailable |
| 5 | Context explicitly closed |
| 6 | Declared resource/admission limit or external tool deadline |
| 7 | Vulkan device/dispatch failure |
| 8 | Native-side allocation failure |

Numeric outputs and metrics commit only on success, including batch overflow. Creation functions instead clear their new-handle output to zero before attempting creation. Valid pointer storage/alignment/extents remain C caller obligations; no C interface can validate arbitrary raw pointer provenance. Binding-side allocations retain ordinary language allocation behavior.

## 4. Ownership, caches and concurrency

| Resource | Reuse and bound |
|---|---|
| Context source cache | LRU, 8 source entries, aggregate 16 MiB source bytes; individual source at most 8 MiB; retains immutable IR and only weak module references |
| Module native cache | At most 16 compiled function images; bounded eviction, not a promised LRU policy |
| Module Vulkan cache | At most 4 function/representation/workgroup programs; each owns its session and persistent pipeline/dispatch resources |
| Vulkan storage | Grow-only capacity while the cached program lives; valid counts may shrink or be zero; configured maximum at most 1,000,000 elements |

A source-cache hit can reuse a live module and its hot resources. If its last module handle has gone, cached IR can still avoid parsing, but executable/device resources must be recreated. There is no serialized persistent executable cache and no promise of resource reuse across distinct contexts.

Cache clear advances a checked generation and empties the source cache. Module native and Vulkan entries become stale together and are dropped on the next backend acquisition; this is logical invalidation, **not synchronous reclamation of every live module's memory**. Active calls hold shared ownership, so invalidation/eviction cannot unload their native code or destroy their GPU resources underneath them. Counters are cumulative. Single-flight backend construction is serialized per module. Dispatches sharing one Vulkan program are serialized; separate programs can own separate sessions. This is a synchronous SDK, not an asynchronous scheduler.

Vulkan reuse resets command, fence, timestamp and arithmetic-status state for each nonempty dispatch, re-records commands for the current count, and grows buffers when needed. An arithmetic overflow does not poison the device: a following valid dispatch must succeed. A device/dispatch failure poisons that cached program; explicit invalidation is required before rebuilding. Driver waits are **not** a hard-cancellable hardware deadline. Physical device loss/hang and noncoherent-memory qualification remain separate gates.

C++ modules retain their shared context state. Destroying or moving the original context wrapper therefore does not accidentally invalidate retained modules; explicit `Context.close()` deliberately invalidates subsequent calls. Rust uses `Arc`-based ownership. Python context-manager exit explicitly closes the context even if a module wrapper remains. At the C layer, callers own handle release. Calls already admitted may finish while another thread closes a context; later operations reject it. Handle retirement releases the global registry lock before native/device/filesystem destructors, so slow cleanup cannot monopolize unrelated context lookup. Moving/destroying the same C++ wrapper concurrently with a method call is ordinary invalid object-lifetime use, not supported thread safety.

Limits also include existing parser/IR/call-depth bounds and operation admission. Native work admission conservatively includes both sides of selections; the reference interpreter charges executed work. A native request can therefore be rejected where a cheaper reference path fits. These counts are IR work, not instruction counts, hard real-time guarantees, global process-memory quotas, or a hostile-code sandbox.

External tools use private mode-0700 directories, mode-0600 diagnostic capture, shell-free argument vectors and a 60-second compiler deadline with process-group termination/reaping. The embedding process must not independently reap SDK-owned children or replace their normal child lifecycle. Trusted tool configuration and an operational OS/driver remain prerequisites. A malicious compiler can escape ordinary process controls; this is not a security container.

## 5. Binding and installed-consumer scope

| Binding | Current SDK 1.1 surface | Qualification boundary |
|---|---|---|
| C | Additive version/info/cache APIs, explicit native scalar/map, existing math APIs | Linux x86-64 shared ABI and relocated C consumer exercised |
| C++20 | Owning/movable context/module RAII, typed scalar variant, metadata/cache JSON, explicit native map/invoke, existing math/BTFP headers | Retained/moved/closed lifetime, mixed native binary64/int calls and relocated consumer exercised |
| Python 3.11+ | Parsed metadata/cache dictionaries, explicit native invocation/map and metrics, finite BTFP, optional NumPy integral maps | Source and separately installed wheel exercised; NumPy inputs must be aligned, native-endian, contiguous rank-1 int64; read-only input is permitted |
| Rust | Authored native backend, discovery/cache and typed scalar bindings; context/module shared ownership | **Not locally compiled or runtime-qualified: Rust toolchain unavailable.** Finite BTFP wrapper remains absent |

Python iterable maps consume at most the public batch limit plus one element, rather than materializing an unbounded iterator. NumPy maps avoid Python element-by-element marshalling but still copy through SDK-owned transactional storage. Neither interface promises device zero-copy. Callers must not concurrently mutate input storage during a call.

CMake consumers use `find_package(TriumXBT 1.1 CONFIG REQUIRED)` and `TriumXBT::SDK` / `TriumXBT::Math`. The Python wheel contains wrappers, not a bundled native library; point `BT_SDK_LIBRARY` at the installed trusted absolute library path. Packaging is not a new license grant: see [license boundaries](LICENSE_BOUNDARIES.md).

### Python example

```python
import trium

with trium.Context(library="/absolute/path/libtrium_sdk.so") as ctx:
    with ctx.compile("func transform(x:int)->int{return x*3;}") as module:
        print(module.info["functions"])
        value, metrics = module.invoke("transform", 7, backend="native",
                                       with_metrics=True)
        assert value == 21 and metrics["backend"] == 2
        values, warm = module.map("transform", [-2, 0, 2], backend="native")
        assert values == [-6, 0, 6]
        print(ctx.cache_info)
```

Use `backend="cpu"` explicitly for the reference executor. For an eligible GPU map use `backend="vulkan"`; unavailable hardware raises an error rather than running a CPU substitute.

### C++ example

```cpp
#include <Ternary/SDK/SDK.hpp>

bt::sdk::Context context;
auto module = context.compile("func f(x:int)->int{return x+1;}");
auto result = module.invoke("f", {int64_t(7)}, BT_SDK_NATIVE);
auto value = std::get<int64_t>(result); // 8
// module retains context ownership if the original wrapper leaves scope.
```

## 6. Verification and measurement

The current evidence directory is [2026-10-30-sdk-1.1](evidence/2026-10-30-sdk-1.1/). It contains regression summaries, proof output, environment/provenance and benchmark JSON. The final results and measured table are recorded there; historical floating-point and R1/R2 evidence are not overwritten.

**Final local results:** Release **47 passed / 2 skipped of 49** (19.64 s); Debug ASan/UBSan **47 passed / 2 skipped of 49** (48.08 s). The skips are physical CUDA qualification and unavailable Rocq. A separate Vulkan-disabled SDK passed **5/5** selected tests. All **13 root Z3 scripts** passed. Both full suites retained **54,022 independent floating-foundation checks** and **15,372 earlier exact rational/BTFP cases**. Native differential execution covered **1,744 cases** plus the typed, cache and concurrency checks below. Relocated C/C++ and an isolated installed Python wheel passed.

Median warm timings for three **10,000-element NumPy** workloads (five samples; milliseconds):

| Kernel | Reference SDK | Native SDK | Reference incl. Python | Native incl. Python |
|---|---:|---:|---:|---:|
| Signed map | 0.8632 | 0.0732 | 0.8961 | 0.1050 |
| Checked polynomial | 0.8907 | 0.0634 | 0.9233 | 0.0964 |
| Ternary-weight dot | 1.0019 | 0.0396 | 1.0293 | 0.0673 |

First native calls cost approximately **43–51 ms** across the two input-mode reports; reuse matters. All warm samples reported zero backend compilation. Python list marshalling instead produced about **3.9–4.7 ms** native end-to-end medians. These are useful scoped improvements, not a general numerical speed claim. **30/30 benchmark runs** passed across list/NumPy and the three explicit backends; Vulkan here was **software SwiftShader only**. See the [evidence summary](evidence/2026-10-30-sdk-1.1/SUMMARY.md) and raw JSON for source hashes and exact attribution.

Coverage includes native/reference differential cases over checked arithmetic, digit gates, ternary-weight contraction, packed-mask helper arithmetic, lazy branches, boundary inputs, mixed binary64 signatures, NaN/signed-zero, trit and void; single-flight caching, bounded image eviction, invalidation during concurrent execution, independent TLS failures and unavailable compiler behavior. API tests exercise bounded/concurrent source reuse, large call-graph metadata publication, resource errors and unchanged outputs. C++ tests cover ownership and private bounded tool execution. Vulkan execution tests cover warm pipeline reuse, growth/shrink/zero counts, overflow recovery, concurrent calls and invalidation/rebuild.

Thirteen root Z3 scripts were run, including the new SDK generation, admission, ownership, transactional-commit and guarded-address **models**. These are scoped obligations, not a proof of the whole C++ implementation, compiler, allocator, operating system or driver. Independent exact finite-BTFP regressions remain part of the suite. ASan/UBSan instrument the host library/tests, not the generated native shared object or third-party Vulkan driver. Python-host runs disable leak detection to avoid attributing CPython's allocator lifetime to the SDK; native C++ tests retain normal leak checks.

The benchmark separates source compilation, first backend setup/compilation and five warm samples. It reports both SDK-only `total_ms` and Python-inclusive `binding_total_ms`; list and NumPy marshalling must not be conflated. Software Vulkan remains labeled software. The CPU reference is an interpreter, **not optimized BLAS**. Three small 10,000-element kernels are useful evidence of reuse and execution overhead, not general AI, simulation, physical-GPU or cross-platform performance qualification.

## 7. Remaining gates and next work

- User-run physical GPU correctness/benchmarking with [the isolated runner](HARDWARE_GPU_QUALIFICATION.md); software SwiftShader does not close that gate. Native64 runtime needs a real device exposing `shaderInt64`.
- Compile/test the Rust binding, and qualify additional OS/CPU/compiler/loader combinations. No remote CI execution is implied by local results.
- Broaden native source eligibility only alongside runtime budget/call-depth correctness and differential tests; loops/calls are not silently admitted today.
- Add measured device tuning, asynchronous execution, general tensor ownership/strides, GPU GEMM and framework integration as separate contracts. Capability-based `auto` is not a speed autotuner.
- Resolve component/package license declarations with the rights holders before a public distribution. No relicensing or generated-code exception was authorized here.
- Preserve Step 6 ISA/RTL/synthesis/physical ternary hardware as its own future track.

**Outcome:** a substantially more usable compiled CPU and reusable Vulkan SDK foundation, with explicit contracts and concrete evidence—not unrestricted program execution or a complete numerical/AI framework.
