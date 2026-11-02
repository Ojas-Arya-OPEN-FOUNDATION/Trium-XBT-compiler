# Numerical expansion / SDK 1.2 — qualification evidence

Date: 2026-10-30. Linux x86-64, GCC host build, CPU-only SDK configuration. This is a local working-tree qualification, not a tagged release or remote CI run.

## Actual results

| Gate | Result |
|---|---|
| Release CTest | **49 passed, 4 skipped / 53**, 25.97 seconds |
| Debug ASan + UBSan CTest | **49 passed, 4 skipped / 53**, 63.14 seconds |
| New advanced FP/numerical oracle | **13,326** exact result/error/ABI checks, all widths 1..40 |
| Existing finite-FP foundation | **54,022** independent checks retained and passing |
| Existing exact rational/BTFP regression | **15,372** cases retained and passing |
| Existing actual native execution | **1,744** differential cases plus typed ABI/cache/concurrency/failure checks retained and passing |
| Root Z3 scripts | **14/14** passed, including **51** new numerical model obligations |
| Low-level constants | **84** interval-certified binary64 constants; **36** corrected |
| Installed C/C++ | Relocated CMake package, minimum version 1.2; new norm/hypot and header-level exact sum exercised |
| Standalone installed math headers | Exact numerical header consumer compiled and ran **without SDK linking** |
| Installed Python | Separately built wheel, isolated import; new FP/reduction/polynomial/discovery and existing native execution passed |

The four skips are `CUDA_Device_Qualification`, `Core_Rocq`, `SDK_Vulkan_Offline`, and `SDK_Vulkan_Execution`. GPU tool/runtime binaries from previous snapshots were not present; this milestone was built with `BT_SDK_VULKAN=OFF`. Neither Vulkan nor a physical GPU was requalified here. SDK 1.1's software Vulkan qualification is historical evidence only. Rust compiler/cargo were unavailable; no new Rust finite-FP binding is claimed.

Both full suites include exact binade-gap root ties (norm of four width-1 unit values rounds 2 to 3; width-2 hypot(3,4) rounds 5 to 6), cancellations, negative powers/roots, empty results, max counts/degrees, extreme exponents, invalid records, resource bounds and unchanged/aliased outputs. New math uses fixed scratch integers, not binary floating estimates. ASan/UBSan check implementation behavior, not numerical correctness proofs; exact references serve the latter. Python hosts use leak suppression as in the existing harness; native C++ tests retain leak checks. Generated native images are not sanitizer-instrumented.

## Old helper improvement: isolated sampled comparison

`cordic-before.json` and `cordic-after.json` use the same current helper implementation and survey, changing **only the four stored constant tables**. Baseline table blocks are restored from the repository base commit recorded in `environment.json`.

| Survey | Before tables | Corrected tables |
|---|---:|---:|
| Circular sin/cos, 2001 inputs from -100 to 100 | 6.373484986305078e-11 max absolute error | 3.858025010572419e-15 |
| Positive-octant atan2, 1000 input pairs | 6.373301886242189e-11 | 4.440892098500626e-16 |

These are deterministic **sampled observations**, not universal transcendental error bounds or correctly rounded CORDIC certification. The constants themselves are checked independently using exact alternating/positive-series intervals and square-root midpoint comparisons. Circular/hyperbolic scale values, micro-angles and repeated hyperbolic entries are all included in the 84 constants.

## Files

- `release-ctest.txt`, `asan-ctest.txt`: final complete test summaries.
- `release-details.txt`, `asan-details.txt`: complete per-test output and explicit skip reasons.
- `z3.txt`: all fourteen root scripts and results; new obligations are scoped models, not a whole-C++/OS proof.
- `constants.txt`: independent interval table check.
- `cordic-before.json`, `cordic-after.json`: isolated sampled table correction results.
- `header-only.txt`: installed header-only consumer result, no native SDK link dependency.
- `installed-python.txt`: installed wheel result, imported in isolated Python mode without checkout binding imports.
- `environment.json`: source/library/wheel hashes, compiler/tool versions, SDK metadata and qualification boundaries.

## Reproduction

```sh
# Provide CMake/Ninja, Python with z3-solver/llvmlite/NumPy, and a trusted host
# Clang-compatible compiler for the existing native/LLVM execution tests.
cmake -S . -B build/numerical -G Ninja -DBT_ENABLE_MLIR=OFF \
  -DBT_SDK_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/numerical -j1
ctest --test-dir build/numerical --output-on-failure

cmake -S . -B build/numerical-asan -G Ninja -DBT_ENABLE_MLIR=OFF \
  -DBT_SDK_VULKAN=OFF -DBT_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/numerical-asan -j1
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build/numerical-asan --output-on-failure

python3 tools/verify_cordic_constants.py
for f in verify_z3*.py; do python3 "$f" || exit; done
python3 test/numerical_reference.py build/numerical/libtrium_sdk.so

g++ -std=c++20 -O2 -I. test/cordic_survey.cpp -o build/cordic-after
build/cordic-after
python3 -m pip wheel --no-deps --wheel-dir build/numerical-wheel ./ffi/python
```

The local native Clang driver came from Zig 0.17 through the adapter described in `environment.json`, not an independently installed system LLVM release. Additional CPU/compiler/OS combinations need their own qualification. No README/license change, commit/push, physical-GPU run or remote CI was performed in this milestone.
