# Floating-point foundation evidence — 2026-10-29

Contract and implementation: [FLOATING_POINT_FOUNDATION.md](../../FLOATING_POINT_FOUNDATION.md).

- `release-ctest.log`: 45 registered, 43 passed, two skipped, zero failed.
- `asan-ctest.log`: same counts, Debug + ASan/UBSan.
- `reference.log`: 54,022 exact foundation checks; 15,372 preexisting/expanded
  Fraction regression cases; constexpr and scratch-primitive unit checks.
- `z3.log`: all twelve root scripts passed, including the new floating-point
  arithmetic model lemmas. Not a whole-C++/RTL refinement proof.
- `provenance.json`: compiler/tool versions, compiled library hashes, source
  snapshot hashes and explicit validation boundaries.

CUDA physical qualification and native Rocq are the skipped CTests. Existing
integer Vulkan software execution passed through SwiftShader CPU Vulkan; no
physical GPU, BTFP GPU, FPGA or physical ternary execution is claimed.

A separate GPU-disabled SDK build passed integration and all 54,022 foundation
checks. Relocated installed C/C++ consumers exercised W40 sqrt; a rebuilt and
installed Python wrapper wheel exercised W40 sqrt and exact binary64 import.
Native SDK libraries remain separately built, not bundled into that wheel.

Sanitizer Python/ctypes tests preload the runtimes and disable CPython host leak
reporting. Ordinary instrumented C++ tests retain leak checking. Tests and model
lemmas support the declared finite profile, not all algorithms or all platforms.

Historical R1/R2 evidence remains in its earlier directory. No new Rust compile,
Herbie/FloPoCo integration or native Rocq/jsCoq run is implied by this record.
