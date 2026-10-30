# R1/R2 local evidence — 2026-10-29

Scope and interpretation: [delivery report](../../R1_R2_DELIVERY.md).

- `release-ctest.log`: 43 registered, 41 passed, two skipped, zero failures.
- `asan-ctest.log`: same counts for Debug ASan/UBSan.
- `z3.log`: all eleven root verification scripts, including SDK word/native64
  arithmetic model obligations. Not a whole-implementation correctness proof.
- `software-vulkan.log`: program-dependent shader/reference execution through
  SwiftShader Subzero CPU Vulkan; native64 offline validation and feature gates.
- `physical-gate.json`: physical-only runner correctly returns 77 on this
  software-only machine; no physical execution performed.
- `software-qualification.json`: explicitly software-allowed runner succeeds.
  Its `build/` paths identify generated local logs, not permanent public URLs.
- `provenance.json`: compiler/tool/dependency revisions, native library hashes
  and source hashes for this uncommitted implementation snapshot.
- [Benchmark samples](../../benchmarks/2026-10-29-software-vulkan.json): checked
  software/CPU measurements, not physical-GPU speed evidence.

The sanitizer test configuration preloads the runtime for ctypes tests; CPython
host leak reporting is disabled there, while ordinary instrumented C++ tests keep
leak checks. Tests exercise the source snapshot, not Windows/Apple packages.

CUDA physical qualification and native Rocq are the two skipped CTests. Rust has
a separate authored workflow but was not locally compiled. Python wheel build,
isolated installation and CPU invocation passed. Historical jsCoq kernel evidence
is kept under the earlier Steps 1–5 evidence, not rerun or counted here.
