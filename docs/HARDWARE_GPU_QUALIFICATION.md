# Run GPU correctness and benchmarks on your laptops/cloud

Physical-device qualification is intentionally separate from software development.
The local implementation executes Vulkan through **SwiftShader on a CPU**;
that is an actual Vulkan API path, not evidence of NVIDIA/AMD/Intel hardware
execution. Your device runs can close those individual qualification entries.

## 1. Build on a Linux x86-64 machine

Install Python 3.11+, a C++20 GCC/Clang toolchain, Git, CMake and Ninja. Install the
GPU vendor's working Vulkan driver/loader. Hardware virtualization/cloud access
must actually expose Vulkan compute; a GPU product name alone is insufficient.
The first host qualification is Linux; Windows/macOS packaging needs separate work.

Optional shader tools are pinned and built from source:

```sh
bash tools/build_portable_gpu_tools.sh "$PWD/build/portable-tools"
export BT_GLSLANG="$PWD/build/portable-tools/glslang-build/StandAlone/glslang"
export BT_SPIRV_VAL="$PWD/build/portable-tools/spirv-build/tools/spirv-val"
cmake -S . -B build/sdk -G Ninja -DBT_ENABLE_MLIR=OFF \
  -DBT_SDK_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
  -DBT_VULKAN_INCLUDE_DIR="$PWD/build/portable-tools/Vulkan-Headers/include"
cmake --build build/sdk -j2
```

`BT_SDK_VULKAN=OFF` remains a supported CPU-only configuration. Python binding
source is in `ffi/python/trium`; no NumPy dependency is needed for basic tests.
Native SDK and shader tools are built separately from the optional Python package.

## 2. Run the isolated qualification command

```sh
python3 tools/qualify_portable_gpu.py \
  --library "$PWD/build/sdk/libtrium_sdk.so" \
  --output build/gpu-results --count 10000 --repeats 7
```

This enumerates devices, excludes software CPU devices by default, runs reference
comparisons and arithmetic-error tests for each selected device, then benchmarks
three actual source kernels at workgroup sizes 32/64/128, testing portable32
and native64 when its feature is advertised. It runs devices
sequentially; it is not multi-device partitioning of a single workload.

Outputs include:

- `qualification.json`: actual device metadata and pass/fail/unavailable outcomes.
- Per-device correctness logs.
- Per-device benchmark JSON/logs with source, source/native-library hashes,
  driver/API/device IDs/UUID, actual backend/representation, all samples and medians.

Exit **0** means the selected tests passed, **77** means no suitable device or
runtime is available, and **1** means a failed test/tool/runtime operation. Read
the JSON even on failure. These tests qualify only the documented map subset and
the exact tested configuration—not every workload or GPU.

Attach the JSON and logs when reporting your results. Keep generated artifacts
under `build/` rather than adding large machine-specific artifacts to Git.

## 3. More focused benchmarking

```sh
python3 tools/benchmark_portable_gpu.py \
  --library "$PWD/build/sdk/libtrium_sdk.so" --backend both \
  --output build/gpu-results/benchmarks.json --count 10000 --repeats 7
```

Use `--device N` to select a device or omit it to run enumerated devices.
`--representations portable32 native64 auto` controls the variant sweep; unsupported
native64 is recorded rather than silently substituted. In the SDK use
`Context(representation='native64')` to require that feature, or `'auto'` for a
capability-based choice. Auto does not promise the fastest implementation.
`--require-hardware` excludes CPU ICDs. `--backend cpu` works without a GPU.
These are checked int39 maps/popcount contractions, **not AI GEMM/training benchmarks**.
`--backend cpu` is the reference interpreter, not an optimized BLAS baseline.
SDK 1.1 adds `--backend native` for compiled CPU execution and `--backend all`
for explicit reference/native/Vulkan comparisons. `--input numpy` avoids
element-by-element Python marshalling (requires NumPy); default input is a list.

Timing fields distinguish:

- Source compilation (recorded by the runner) and GLSL/SPIR-V tool compilation.
- Host staging upload/download copies and coherence operations.
- Submit-to-fence host wall time.
- GPU timestamps for upload, compute and download bands when the queue supports
  them; these bands include synchronization overhead. `-1` means unavailable.
- First-call `cold_start`, including backend setup/compilation, followed by warm
  samples reusing module-owned native images or Vulkan device/pipeline/dispatch
  resources. Buffers grow as needed; cache invalidation can cause rebuilding.
- Total SDK map time and separate Python-inclusive `binding_total_ms`. The latter
  includes binding input/output marshalling; SDK-only timing does not.

These caches are in-process resources, not serialized disk caches. Cold means
SDK-cache-cold, not a flushed OS/compiler cache. No general CPU-to-GPU speedup is
inferred from these columns. See [SDK 1.1 contracts and evidence](SDK_1_1.md).

No timing is fabricated for unavailable devices. The runner validates every
measured result against an independent Python calculation before accepting it.

## 4. Explicit software-only validation

If you have a software Vulkan implementation, use `--allow-software` on the
qualification command. Results retain `software: true` and do not close a
physical-device gate. Our local SwiftShader build uses its Subzero CPU backend.

The runtime normally loads `libvulkan.so.1`. For a trusted custom loader or a
compatible directly loadable ICD, set `BT_SDK_VULKAN_LIBRARY` to its absolute path.
Local tests use SwiftShader's `libvk_swiftshader.so` directly; this does not
exercise all behavior of the system Vulkan loader or hardware driver stack.

Apple translation layers, discrete-device noncoherent memory paths, device-loss
behavior and additional driver validation-layer checks still need their own
qualification. No native ternary hardware is required for any of this work.
