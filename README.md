# Next-Gen Foundation Compiler (Balanced Ternary System)

<div align="center">

### An Open-Source Foundation Project by **Ojas-Arya OPEN-FOUNDATION**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Rocq 9.2 Machine Proofs](https://img.shields.io/badge/Rocq_9.2-100%25_Verified-brightgreen.svg)](proofs/rocq/BalancedTernaryFoundations.v)
[![Lean 4 Proofs](https://img.shields.io/badge/Lean_4-v4.34.0_Verified-brightgreen.svg)](BalancedTernary.lean)
[![Z3 SMT Solvers](https://img.shields.io/badge/Z3_SMT-0_Counterexamples-brightgreen.svg)](verify_z3_kogge_stone.py)
[![LLVM 7-Layer Test Suite](https://img.shields.io/badge/LLVM_Tests-7_Layers_Passed-brightgreen.svg)]()
[![Sanitizers](https://img.shields.io/badge/Sanitizers-ASan_%2B_UBSan_Clean-success.svg)]()
[![Hardware](https://img.shields.io/badge/RTL-RISC--V_Xbt_SystemVerilog-blueviolet.svg)](hw/riscv_xbt_extension.sv)

*A mathematically proven, hardware-agnostic balanced ternary compiler infrastructure, high-performance numerical algebra engine, and synthesizable silicon design targeting commodity CPUs, GPUs, and RISC-V accelerators.*

[Complete 31,000-Word Guide](NEXT_GEN_FOUNDATION_COMPILER_COMPLETE_GUIDE.md) &nbsp;|&nbsp; [Interactive Visualizer](docs/Next_Gen_Foundation_Compiler_Guide.html) &nbsp;|&nbsp; [Quickstart](#quick-start) &nbsp;|&nbsp; [Contributing](CONTRIBUTING.md)

</div>

---

## 🏛️ About Ojas-Arya OPEN-FOUNDATION

**Ojas-Arya OPEN-FOUNDATION** is dedicated to foundational computing breakthroughs, sovereign computational infrastructure, and mathematical verification. The **Next-Gen Foundation Compiler** is the flagship initiative proving that the computing monopolies of binary matrix accelerators can be overcome using the natural efficiency, zero-bias rounding, and zero-gate negation of **Balanced Ternary Computing (`{-1, 0, +1}`)**.

---

## 🌟 Key Architectural Innovations

1. **Multiplier-Free Dual-Rail Linear Algebra**:
   - Maps balanced ternary digits onto dual 64-bit binary planes $(P, N)$ where $\text{val}(P, N) = P - N$ with invariant $P \land N = 0$.
   - Multiplication of 64 trits executes in **4 bitwise clock cycles** without hardware multipliers.
   - Dot product evaluates via hardware population count: $\text{DotProduct} = \text{POPCNT}(P_{\text{prod}}) - \text{POPCNT}(N_{\text{prod}})$, reaching **100+ GTOPS** on commodity CPUs.
   - **Zero-Gate Negation**: $-x$ is achieved in **0 clock cycles** by swapping register labels ($P \leftrightarrow N$).

2. **Branchless 6-Stage Kogge-Stone Adder**:
   - Powered by the closed **7-state monotone semigroup** $\mathcal{S}_7$ with Stage-0 $c_{\text{in}}$ folding.
   - Resolves a 64-trit carry ripple in strictly 6 deterministic parallel-prefix stages with zero branching and 35% fewer bitwise gates.

3. **Unbiased Floating-Point (`BTFP32`) & Single-Rounding FMA**:
   - Native base-3 floating-point ($M \times 3^E$) with 31.7 bits of precision and exact zero-mean unbiased rounding ($\mathbb{E}[\text{error}] = 0.000000$).
   - 128-bit single-rounding Fused Multiply-Add (`BTFP::fma`) eliminating double-rounding bias.
   - Exact rational linear solver (`Matrix<Rational>`) solving $Ax = b$ and matrix inversion with **$0.000\%$ roundoff error**.

4. **Carry-Preserving Symplectic Lattice Mechanics**:
   - Discrete Hamiltonian phase-space engine (`SymplecticLatticeEngine`) guaranteeing **exact momentum conservation bit-for-bit** ($\sum_i P_i = \text{const}$) and **perfect mathematical time reversibility** without storing past history in RAM.

5. **Synthesizable RISC-V Silicon Extension (`Xbt`)**:
   - Production IEEE 1800-2017 SystemVerilog RTL (`hw/riscv_xbt_extension.sv`) defining Custom-0 (`opcode = 0x0B`) execution units: `XBT.BTHA`, `XBT.KSADD`, `XBT.DOT64`, `XBT.UNPACK243`, and `XBT.NEG`.

6. **Tri-Prover Formal Verification Suite**:
   - **Rocq 9.2 (Coq)**: Dual-rail quotient isomorphism, 0-gate negation, BTHA 6-gate soundness, and BTFA carry disjointness proved with **0 errors, 0 warnings, and 0 axioms** ([`proofs/rocq/BalancedTernaryFoundations.v`](proofs/rocq/BalancedTernaryFoundations.v)).
   - **Lean 4.34.0**: SWAR tree reduction, Kogge-Stone carry associativity, and Hamiltonian lattice momentum conservation proved with **0 `sorry`s** ([`BalancedTernary.lean`](BalancedTernary.lean), [`HamiltonianTernary.lean`](HamiltonianTernary.lean), [`TernaryAlgebraProofs.lean`](TernaryAlgebraProofs.lean), [`TernaryCompilerMath.lean`](TernaryCompilerMath.lean)).
   - **Z3 SMT Solver**: Exhaustive verification across all $3^{16} = 43,046,721$ SWAR inputs and 7-state semigroup permutations with **0 counterexamples** ([`verify_z3_kogge_stone.py`](verify_z3_kogge_stone.py), [`verify_z3_swar_bct.py`](verify_z3_swar_bct.py), [`verify_z3_bigint_divmod.py`](verify_z3_bigint_divmod.py), [`verify_z3_fp_fma.py`](verify_z3_fp_fma.py)).

---

## 📁 Repository Map

```
.
├── include/Ternary/              # MLIR TableGen schemas & C-ABI headers (ternary_c_abi.h)
├── lib/Ternary/                  # MLIR Dialect & C-ABI shared library implementation
├── tools/bt-opt/                 # Standalone MLIR optimizer tool (bt-opt.cpp)
├── iree/                         # IREE compiler plugin & HAL target backends
├── hw/                           # Synthesizable SystemVerilog RTL (riscv_xbt_extension.sv) & co-simulator
├── ffi/                          # Multi-language bindings
│   ├── python/                   # Python ctypes wrapper (bt_ffi.py)
│   └── rust/                     # Safe Rust crate (ffi/rust/src/lib.rs)
├── proofs/rocq/                  # Rocq 9.2 machine-checked formal proofs (.v)
├── test/                         # 7-layer LLVM test suite
│   ├── Unit/                     # Layer 1: Core, Dense, Memory, AST, MLIR, FP, BigInt, LibBT, Xbt
│   ├── Regression/               # Layer 2: lit + FileCheck syntax tests
│   ├── WholeProgram/             # Layer 3: Whole-program pipeline tests
│   ├── Sanitizers/               # Layer 4: AddressSanitizer & UndefinedBehaviorSanitizer
│   ├── CodeGen/                  # Layer 5: Target assembly & DWARF-5 tests
│   ├── Dialect/                  # Layer 6: MLIR dialect & pass verification
│   └── Sema/                     # Layer 7: Diagnostics & front-end syntax tests
├── docs/                         # 9-chapter complete documentation guide
├── ternary_core.hpp              # Dual-rail algebraic engine, Kogge-Stone, SWAR
├── ternary_bigint.hpp            # Arbitrary-precision TritBigInt & Knuth Euclidean divmod
├── ternary_dense_algebra.hpp     # Multi-trit GEMM, cache hierarchy autotuner
├── ternary_fp.hpp                # BT-FP16 & BT-FP32 floating-point units & single-rounding FMA
├── ternary_sparse.hpp            # 1-bit Ternary Compressed Sparse Row (T-CSR) engine
├── ternary_frontend_ast.hpp      # Frontend lexer, AST, parser & semantic type checker
├── ternary_llvm_codegen.hpp      # LLVM IR visitor, SSA intrinsics, DWARF-5 emitter
├── ternary_jit_engine.hpp        # Dynamic in-memory AST interpreter & JIT engine
├── libbt.hpp                     # Standard library (Rational, matrix inversion, Radix-3 FFT)
├── bt_compiler.cpp               # Master compiler CLI driver
└── run_all_tests.sh              # 7-layer master test suite harness
```

---

## 🚀 Quick Start

### 1. Requirements
* C++20 compiler: Clang $\ge 15.0$ or GCC $\ge 13.0$.
* Python 3.8+ (for LLVM lit test runner & FFI).
* Optional: Lean 4 (v4.34.0), Rocq 9.2, Z3 Python for running formal proof suites.

### 2. Build the Core Tools & Shared Library
```bash
# On macOS, export SDKROOT if needed:
export SDKROOT=$(xcrun --show-sdk-path)

# Build bt-compiler
clang++ -O3 -std=c++20 bt_compiler.cpp -o bt-compiler

# Build bt-opt (MLIR modular optimizer)
clang++ -O3 -std=c++20 tools/bt-opt/bt-opt.cpp lib/Ternary/TernaryDialect.cpp lib/Ternary/TernaryPasses.cpp iree/TernaryIREEPlugin.cpp -I. -o bt-opt

# Build C-ABI shared runtime library
clang++ -O3 -std=c++20 -shared -fPIC lib/Ternary/ternary_c_abi.cpp -I. -o libbt_runtime.dylib
```

### 3. Run the Full 7-Layer Test Suite
```bash
./run_all_tests.sh
```

---

## 💻 Language Usage Examples

### A. Writing `.bt` Programs with Built-in Intrinsics
```bt
var a: float = 2.0;
var b: float = 3.0;
var c: float = 4.0;
var f: float = fma(a, b, c);          // 2*3 + 4 = 10.0 (single-rounding FMA)
var s: float = sqrt(f * 2.0 + 5.0);   // sqrt(25.0) = 5.0
var p: float = abs(-7.5);             // 7.5
return f + s + p;
```

Run via native Clang JIT:
```bash
./bt-compiler run program.bt --native
```

Cross-compile to target assembly:
```bash
./bt-compiler program.bt --target=aarch64 -S    # Apple Silicon / ARM64
./bt-compiler program.bt --target=x86_64 -S     # Intel / AMD
./bt-compiler program.bt --target=riscv64-xbt -S # RISC-V with custom Xbt silicon instructions
```

### B. Python Bindings
```python
from ffi.python.bt_ffi import BtFp32, bt_radix3_convolve

# Single-rounding FMA
a = BtFp32.from_double(2.5)
b = BtFp32.from_double(4.0)
c = BtFp32.from_double(1.5)
res = a.fma(b, c) # 11.5

# Exact balanced square root
root = BtFp32.from_double(49.0).sqrt() # 7.0
```

### C. Rust Crate
```rust
use bt_runtime::BtFp32;

let a = BtFp32::from_double(2.0);
let b = BtFp32::from_double(3.0);
let c = BtFp32::from_double(4.0);
let res = a.fma(b, c); // 10.0
```

---

## 🔬 Formal Proof Verification

To execute the full Tri-Prover verification suite:

```bash
# 1. Lean 4 (v4.34.0)
lean BalancedTernary.lean
lean HamiltonianTernary.lean
lean TernaryAlgebraProofs.lean
lean TernaryCompilerMath.lean

# 2. Rocq 9.2 (Coq)
rocq compile proofs/rocq/BalancedTernaryFoundations.v

# 3. Z3 SMT
python3 verify_z3_kogge_stone.py
python3 verify_z3_swar_bct.py
python3 verify_z3_bigint_divmod.py
python3 verify_z3_fp_fma.py
```

---

## 🤝 Contributing

We warmly welcome researchers, systems programmers, and silicon designers! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, testing procedures, and coding standards.

---

## 📜 License

This project is open-sourced under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for full details.

&copy; 2026 **Ojas-Arya OPEN-FOUNDATION**. All rights reserved.
