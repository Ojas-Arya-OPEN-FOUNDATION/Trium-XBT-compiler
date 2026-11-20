# Trium-XBT Dual-Track Architecture Reference

Trium-XBT organizes present-day software execution and next-generation ternary silicon exploration inside **one unified codebase** without splitting repositories or duplicating core semantics. Both tracks share the same balanced-ternary mathematical contract (`ternary_core.hpp`, `ternary_fp.hpp`, `ternary_wide.hpp`, Lean/Z3 proofs) and diverge cleanly at target lowering, runtime packaging, and hardware verification.

---

## Track Overview

| Dimension | Track 1: Present-Day Hardware | Track 2: Next-Generation Ternary & RISC-V Xbt Hardware |
|---|---|---|
| **Goal** | Fast, exact ternary compilation, SDK runtimes, and bit-sliced BLAS on existing CPUs & GPUs | Declarative hardware onboarding, RISC-V Xbt ISA extension, RTL simulation, Chipyard RoCC, and Yosys synthesis |
| **Targets** | `targets/generic-cpu.yaml`, `targets/portable-gpu.yaml` | `targets/riscv64-xbt.yaml`, `targets/ternary-torus.yaml` (and new vendor YAMLs) |
| **Primary Entry Points** | `bt-compiler`, `bt-opt`, `libtrium_sdk.so`, `ffi/python/trium`, `ffi/rust` | `tools/gen_targets.py`, `tools/gen_backend.py`, `hw/*.sv`, `verification/chipyard/*`, `tools/run_hardware_benchmarks.py` |
| **Unified CLI** | `python3 tools/trium_workflow.py track1-check` / `track1-bench` | `python3 tools/trium_workflow.py track2-targets` / `track2-scaffold` / `track2-verify` |

---

## Track 1: Present-Day Hardware Compiler & SDK

Track 1 targets commodity x86-64 / ARM64 CPUs (via AVX-512 / AVX2 / NEON bit-sliced dual-rail kernels) and portable GPUs (Vulkan / CUDA compute).

### Core Components & Entry Points

- **Compiler & Dialect Drivers**
  - `bt_compiler.cpp` -> `bt-compiler`: standalone C++20 end-to-end balanced-ternary compiler (`--test` runs built-in regression checks).
  - `tools/bt-opt/bt-opt.cpp` -> `bt-opt`: modular optimizer driver for the `bt` MLIR dialect (works with upstream LLVM/MLIR or standalone C++20 lowering).
- **In-Process SDK (ABI v1, API v1.3)**
  - C ABI: `include/Ternary/SDK/sdk.h` + `lib/SDK/{sdk.cpp, NativeRuntime.cpp, NumericalRuntime.cpp, VulkanRuntime.cpp}` -> `libtrium_sdk.so` (`TriumXBT::SDK`).
  - C++20 Header: `include/Ternary/SDK/SDK.hpp` (`bt::sdk::Context`, `Module`, `TernaryFloat`, `matmul`, `gemm_ternary`, plus ergonomic helpers `version_string()`, `info_json()`, `quick_map(...)`, `ternary_matmul(...)`).
  - Python FFI: `ffi/python/trium/__init__.py` (`Context`, `Module`, `TernaryFloat`, `Metrics`, plus `trium.open(...)`, `trium.quick_map(...)`, `trium.ternary_matmul(...)`, and `Context.summary()`).
  - Rust FFI: `ffi/rust/src/` (checked for surface parity via `tools/check_sdk_parity.py`).
- **Bit-Sliced Ternary BLAS & Benchmarks**
  - `ternary_xbt_blas.hpp` & `include/Ternary/Kernel/`: register-blocked dual-rail micro-kernels (`bt::blas::ternary_gemm`).
  - `tools/ternary_bench.cpp`: differential correctness (`--check`) and throughput benchmark harness.

### Track 1 CMake Flags

```bash
cmake -S . -B build \
  -DBT_ENABLE_MLIR=ON \      # Use upstream LLVM/MLIR if present; otherwise standalone C++20 dialect
  -DBT_BUILD_SDK=ON \        # Build libtrium_sdk.so (TriumXBT::SDK) and TriumXBT::Math
  -DBT_SDK_VULKAN=OFF \      # Set ON to enable optional Vulkan compute runtime
  -DBT_BUILD_TESTS=ON \      # Build unit, SDK, and layout conformance tests
  -DBT_ENABLE_ASAN=OFF       # Set ON for ASan/UBSan builds
cmake --build build --target bt-compiler trium_sdk unit_sdk_host unit_sdk_gemm_layout ternary_bench
```

### Track 1 Workflows

```bash
# Inspect environment & toolchains
python3 tools/trium_workflow.py status

# Build bt-compiler + libtrium_sdk.so and run Python + C++ SDK conformance suites
python3 tools/trium_workflow.py track1-check

# Verify bit-exact ternary GEMM kernels with -O3 -march=native
python3 tools/trium_workflow.py track1-bench

# Enforce C / Python / Rust SDK parity
python3 tools/check_sdk_parity.py --check
```

---

## Track 2: Next-Generation Ternary & RISC-V Xbt Hardware Compiler

Track 2 enables zero-compiler-edit onboarding of custom ternary silicon and co-verification of the RISC-V `Xbt` extension from YAML specification down to SystemVerilog RTL, Chipyard RoCC integration, and Yosys synthesis.

### Core Components & Entry Points

- **Declarative Target Descriptions & Code Generation**
  - `schema/target.schema.json`: JSON Schema contract for all machine descriptions.
  - `targets/*.yaml`: declarative machine descriptions (`riscv64-xbt.yaml`, `ternary-torus.yaml`, plus Track 1 descriptions `generic-cpu.yaml` and `portable-gpu.yaml`).
  - `tools/gen_targets.py`: validates `targets/*.yaml` and generates `include/Ternary/Target/GeneratedTargets.hpp`.
  - `tools/gen_backend.py`: scaffolds a compiling backend skeleton + failing conformance suite (`build/backends/<target>/`) from any target YAML.
  - `tools/check_generated_backends.py`: verifies every target package compiles cleanly and surfaces all native-op obligations.
- **RISC-V Xbt RTL, Golden Model & SoC Integration**
  - `hw/riscv_xbt_extension.sv`: RISC-V `Xbt` custom-0 (`0x0B`) ternary ALU extension RTL.
  - `hw/trium_cpu_core.sv` & `hw/fpga_arty_a7_top.sv`: standalone CPU core and Arty A7 FPGA top-level.
  - `hw/tb_riscv_xbt_alu.sv`: SystemVerilog RTL testbench (Verilator / Icarus Verilog).
  - `hw/synth_xbt_yosys.ys`: Yosys CMOS-2 area/cell synthesis script.
  - `verification/chipyard/XbtRoCCAccelerator.scala` & `riscv_xbt_rocc_adapter.sv`: Chipyard RoCC accelerator wrapper.
  - `verification/golden_model/xbt_golden.py`: bit-exact Python golden reference model (`KSADD_v2`, `CSAMAC`, `UNPACK243`/`PACK243`, `LI5`, `DOT64`).
  - `tools/run_hardware_benchmarks.py`: unified runner for golden model regression, POSIX-pipe simulator conformance, RTL simulation, and Yosys synthesis (reports missing EDA tools honestly as `skipped`, or exits `77` with `--require-eda`).

### Track 2 Workflows

```bash
# Validate all YAML target descriptions and generated backend skeletons
python3 tools/trium_workflow.py track2-targets

# Scaffold a new vendor backend package from targets/<name>.yaml
python3 tools/trium_workflow.py track2-scaffold ternary-torus

# Run golden model, simulator pipe, RTL testbench, and Yosys synthesis checks
python3 tools/trium_workflow.py track2-verify
python3 tools/trium_workflow.py track2-verify --require-eda
```

---

## Shared Invariants Across Both Tracks

1. **Single Mathematical Contract**: Balanced ternary dual-rail encoding (`+1` -> P bit, `-1` -> N bit, `0` -> neither; `P & N == 0`) is enforced identically in Track 1 SIMD micro-kernels and Track 2 RTL/golden models.
2. **Explicit Backend Selection**: Neither the C/C++ nor Python SDK performs silent CPU/GPU fallbacks.
3. **Zero-Stub Verification**: Track 1 benchmarks (`--check`) verify bit-exactness before timing; Track 2 backend scaffolds fail their conformance tests by design until native ops are implemented; Track 2 hardware benchmarks never fabricate missing EDA metrics.
