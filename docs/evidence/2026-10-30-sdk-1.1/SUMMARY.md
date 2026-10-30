# SDK 1.1 qualification evidence — 2026-10-30 (Asia/Calcutta)

## Actual results

| Gate | Result |
|---|---|
| Vulkan-enabled Release CTest | **47 passed, 2 skipped / 49**, 19.64 seconds |
| Debug ASan + UBSan CTest | **47 passed, 2 skipped / 49**, 48.08 seconds |
| Separate Vulkan-disabled Release SDK | **5/5** selected ownership/contracts/native/integration/Z3 tests passed, 2.48 seconds |
| Native differential execution | **1,744** checked native/reference cases, plus mixed binary64/void/trit signatures, image eviction, cache/TLS/concurrency/invalidation and failure-transaction checks |
| Independent floating foundation | **54,022** exact checks in both complete suites, all widths 1..40 |
| Earlier exact rational/BTFP regression | **15,372** cases retained and passing |
| Root Z3 scripts | **13/13**, including 7 SDK transition/admission/address model obligations |
| Installed consumers | Relocated C/C++ CMake package (minimum 1.1), and separately built/installed Python wheel with isolated import, native scalar/map/cache and finite-BTFP checks passed |
| Benchmarks | **30/30 runs** passed: list and NumPy modes, reference/native/software Vulkan, 3 kernels, 10,000 values, 5 warm samples each |
| Physical GPU | **Not performed**; remains the isolated user-run gate |
| Rust | API additions authored; **no local Rust compiler/cargo**, not qualified |

The two CTest skips are `CUDA_Device_Qualification` and `Core_Rocq`. They are not passed tests. Native CPU, shader validation and software Vulkan execution actually ran. Z3 proofs are scoped mathematical models, not whole-C++/driver proofs. Host library/tests are instrumented; generated native images and the external software Vulkan driver are not. Python-host ASan tests use `detect_leaks=0`; native tests retain leak detection.

## Measurement results

Median milliseconds, five cached calls, **NumPy input**, 10,000 elements:

| Kernel | Reference SDK | Native SDK | Reference incl. Python | Native incl. Python |
|---|---:|---:|---:|---:|
| Signed map | 0.8632 | 0.0732 | 0.8961 | 0.1050 |
| Checked polynomial | 0.8907 | 0.0634 | 0.9233 | 0.0964 |
| Ternary-weight dot | 1.0019 | 0.0396 | 1.0293 | 0.0673 |

These are three small checked kernels versus an interpreter, not BLAS/AI benchmarks. First native calls cost approximately **43–51 ms** across the list/NumPy runs, excluding separately recorded source compilation. All warm samples reported zero backend compilation time. OS/compiler caches were not flushed. List marshalling changes the result substantially: its native Python-inclusive medians were about **3.9–4.7 ms**. Do not use SDK-only timing as end-to-end Python throughput.

The Vulkan device was **SwiftShader Device (Subzero), software=true, shaderInt64=false**. Each input-mode report has 3 reference, 3 native, and 9 portable Vulkan runs (3 kernels × workgroups 32/64/128). Native64 is explicitly unsupported on this device. No physical GPU speedup, device loss recovery, noncoherent hardware memory, Apple translation-layer or cross-platform qualification is inferred.

## Files and reproducibility

- `release-ctest.txt`, `asan-ctest.txt`, `cpu-only-ctest.txt`: exact test summaries.
- `release-details.txt`, `asan-details.txt`: complete per-test output, including case counts and real execution evidence.
- `z3.txt`: all thirteen root proof scripts and results.
- `installed-python.txt`: isolated installed-wheel consumer result.
- `benchmark-list.json`, `benchmark-numpy.json`: source/SDK hashes, source compile time, cold backend setup, five warm samples, cache counters, device identity and separate SDK/Python timings.
- `environment.json`: platform, compiler/tool versions, library/source hashes, pinned GPU tool commits and qualification boundaries. The local native Clang 22.1.8 driver is supplied by Zig 0.17 through a documented adapter, not a separately installed system LLVM distribution.

Representative commands (prepend a configured Python/toolchain environment):

```sh
cmake -S . -B build/sdk -G Ninja -DBT_ENABLE_MLIR=OFF \
  -DBT_SDK_VULKAN=ON -DBT_VULKAN_INCLUDE_DIR=/absolute/Vulkan-Headers/include \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/sdk -j2
# Set BT_GLSLANG / BT_SPIRV_VAL to trusted absolute tool paths.
# Set BT_SDK_VULKAN_LIBRARY for the explicitly selected software runtime if needed.
ctest --test-dir build/sdk --output-on-failure

cmake -S . -B build/sdk-asan -G Ninja -DBT_ENABLE_MLIR=OFF \
  -DBT_SDK_VULKAN=ON -DBT_VULKAN_INCLUDE_DIR=/absolute/Vulkan-Headers/include \
  -DCMAKE_BUILD_TYPE=Debug -DBT_ENABLE_ASAN=ON
cmake --build build/sdk-asan -j1
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/sdk-asan --output-on-failure

for f in verify_z3*.py; do python3 "$f" || exit; done
python3 tools/benchmark_portable_gpu.py --library "$PWD/build/sdk/libtrium_sdk.so" \
  --output build/list.json --backend all --count 10000 --repeats 5
python3 tools/benchmark_portable_gpu.py --library "$PWD/build/sdk/libtrium_sdk.so" \
  --output build/numpy.json --backend all --input numpy --count 10000 --repeats 5
```

The separate CPU build uses `BT_SDK_VULKAN=OFF`, builds `trium_sdk` and `unit_sdk_host`, then selects `^SDK_(Contracts|Native|Host_Ownership|Integration|Z3_Models)$`. Native execution requires a trusted working host Clang-compatible compiler; physical hardware must use the isolated [qualification runner](../../HARDWARE_GPU_QUALIFICATION.md), not these software results.

No remote CI, commit, push, physical-GPU run, license modification or README rewrite was performed for this milestone. This source tree contains the cumulative earlier foundation work as well; hashes identify this SDK snapshot rather than implying a new released Git commit.
