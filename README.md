# Next-Gen Foundation Compiler

A high-performance, hardware-agnostic balanced ternary compiler infrastructure and numerical algebra engine targeting CPUs and GPUs.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![LLVM 7-Layer Test Suite](https://img.shields.io/badge/LLVM_Tests-7_Layers_Passed-brightgreen.svg)]()
[![Sanitizers](https://img.shields.io/badge/Sanitizers-ASan_%2B_UBSan_Clean-success.svg)]()

---

## Overview

The Next-Gen Foundation Compiler provides an end-to-end toolchain for balanced ternary (`{-1, 0, +1}`) computing, dual-rail planar SIMD algebra, and hardware-agnostic compilation:

- **Frontend & AST**: Lexer, recursive-descent parser, and semantic type checker for the `.bt` programming language.
- **LLVM CodeGen & Bare-Metal Clang JIT**: Direct AST-to-LLVM IR lowering with dynamic native library compilation and runtime symbol execution via `dlopen`/`dlsym`.
- **MLIR Dialect & Lowerings**: Production TableGen schema (`bt` dialect) lowering to `linalg.generic` structured loops, `vector.reduction`, and hardware `llvm.intr.ctpop`.
- **IREE Compiler Plugin**: Cross-dispatch bridge targeting Vulkan SPIR-V, Apple Metal MSL, NVIDIA CUDA PTX, and CPU LLVM backends.
- **Numerical Foundations**:
  - **BT-FP32 Floating Point**: Mantissa in balanced ternary ($M \times 3^E$) with 31.7 bits of dynamic precision and zero-mean unbiased truncation rounding ($\mathbb{E}[\text{error}] = 0.000000$).
  - **1-Bit T-CSR Sparse Algebra**: Multiplier-free Sparse Matrix-Vector Multiply (SpMV) storing non-zero values in 1 bit (0 for $-1$, 1 for $+1$) with a 32.0x value memory reduction.
  - **Native Radix-3 Cooley-Tukey FFT**: Decimation-in-time butterflies for $N = 3^k$ sequences with 100% Parseval energy conservation ($<1.3 \times 10^{-15}$ round-trip error).
- **Multi-Target Cross-Compilation**: Direct emission of native assembly (`-S`) and relocatable object code (`-c`) for **ARM64**, **x86_64**, and **RISC-V 64**.

---

## Directory Structure

```
.
├── include/Ternary/              # TableGen dialect & ops schemas (TernaryDialect.td, TernaryOps.td)
├── lib/Ternary/                  # MLIR Dialect & lowering pass implementations
├── tools/bt-opt/                 # Standalone MLIR optimizer tool (bt-opt.cpp)
├── iree/                         # IREE compiler plugin & HAL target backends
├── test/                         # Official 7-layer LLVM test suite
│   ├── Unit/                     # Layer 1: Core, Dense, Memory, AST, MLIR, FP, LibBT, BT-Opt
│   ├── Regression/               # Layer 2: lit + FileCheck regression tests
│   ├── WholeProgram/             # Layer 3: Whole-program pipeline tests
│   ├── Sanitizers/               # Layer 4: AddressSanitizer & UndefinedBehaviorSanitizer
│   ├── CodeGen/                  # Layer 5: Target assembly & IR generation tests
│   ├── Dialect/                  # Layer 6: MLIR dialect & pass verification
│   └── Sema/                     # Layer 7: Diagnostics & front-end syntax tests
├── CMakeLists.txt                # Out-of-tree MLIR & standalone C++20 build configuration
├── run_all_tests.sh              # 7-layer test suite harness (emulates 'ninja check-all')
├── ternary_core.hpp              # BTHA, BTFA, Base-243 reciprocal codecs, cost model
├── ternary_dense_algebra.hpp     # Dual-rail bitplanes, GEMM tiled microkernels
├── ternary_memory_arena.hpp      # Hierarchical 2-level Copy-on-Write zero-arena
├── ternary_frontend_ast.hpp      # Frontend lexer, AST, and parser
├── ternary_llvm_codegen.hpp      # LLVM IR visitor and Native Clang JIT engine
├── ternary_fp.hpp                # BT-FP16 & BT-FP32 floating-point units
├── ternary_sparse.hpp            # 1-bit Ternary Compressed Sparse Row (T-CSR) engine
├── libbt.hpp                     # Standard library (Rational, matrix inversion, Radix-3 FFT)
└── bt_compiler.cpp               # Master compiler CLI driver
```

---

## Quick Start

### 1. Requirements
- C++20 compliant compiler (Apple Clang $\ge 15.0$ or Clang $\ge 16.0$ or GCC $\ge 13.0$).
- Python 3.8+ (for the LLVM lit test runner).
- Optional: Prebuilt LLVM/MLIR installation for out-of-tree CMake dialect compilation.

### 2. Run the Full LLVM 7-Layer Test Suite
```bash
# Set SDK path on macOS if needed:
export SDKROOT=$(xcrun --show-sdk-path)

# Execute the complete 7-layer verification suite:
./run_all_tests.sh
```

### 3. Build Compiler Executables
```bash
# Build bt-compiler
clang++ -O3 -std=c++20 bt_compiler.cpp -o bt-compiler

# Build bt-opt (MLIR modular optimizer)
clang++ -O3 -std=c++20 tools/bt-opt/bt-opt.cpp lib/Ternary/TernaryDialect.cpp lib/Ternary/TernaryPasses.cpp -I. -o bt-opt
```

---

## Usage Examples

### Run a `.bt` Program via Bare-Metal Clang JIT
```bash
./bt-compiler run sample_program.bt --native
```

### Cross-Compile to Target Assembly
```bash
# ARM64 (Apple Silicon / AArch64 Linux)
./bt-compiler sample_program.bt --target=aarch64 -S

# x86_64 (Intel / AMD)
./bt-compiler sample_program.bt --target=x86_64 -S

# RISC-V 64
./bt-compiler sample_program.bt --target=riscv64 -S
```

### Compile to Relocatable Object File (`.o`)
```bash
./bt-compiler sample_program.bt --target=aarch64 -c -o program_arm64.o
```

### Emit GPU Shaders
```bash
# Metal Shading Language
./bt-compiler sample_program.bt --emit-metal

# Vulkan GLSL / SPIR-V
./bt-compiler sample_program.bt --emit-vulkan
```

### Run the MLIR Optimizer Pipeline
```bash
./bt-opt test/Dialect/bt_opt_lowering.mlir --convert-all
```

### Hardware Cycle-Accurate Microbenchmarks
```bash
clang++ -O3 -std=c++20 bench_cycle_accurate.cpp -o bench_cycle_runner
./bench_cycle_runner
```

---

## Verification & Integrity

All commits and builds pass the 7 LLVM testing layers under AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan):

1. **Unit tests**: `test/Unit/` (C++ API verification)
2. **Regression tests**: `test/Regression/` (`lit` + `FileCheck`)
3. **Whole-program tests**: `test/WholeProgram/`
4. **Sanitizers**: ASan + UBSan (0 leaks, 0 out-of-bounds, 0 undefined behavior)
5. **Target CodeGen**: ARM64, x86_64, RISC-V, Metal, Vulkan, LLVM IR
6. **MLIR Dialect**: Syntax, verifiers, and lowering passes
7. **Clang Sema & Diagnostics**: Front-end type checking and error recovery

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for details.
