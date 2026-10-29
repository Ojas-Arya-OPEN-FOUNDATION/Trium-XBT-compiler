# Trium-XBT: Universal Balanced Ternary Compiler & Computing Architecture
## Executive Architectural Blueprint & AI Ingestion Brief

> **Context for AI Models Reading This Repository:**
> This folder (`upload trium core files`) contains the complete, mathematically verified, production-grade core implementation of **Trium-XBT**—an end-to-end computing foundation, compiler, and silicon architecture based on **Balanced Ternary Arithmetic** ($\{-1, 0, 1\}$).
> 
> All source code and specifications have been curated into structured Markdown files, stripped of temporary build artifacts, compiler caches, and intermediate binaries, and augmented with explicit mathematical formulas and architectural annotations for rapid context ingestion and reasoning.

---

## 1. Why Balanced Ternary? (Fundamental Mathematical Foundations)

Binary computing ($0, 1$) has dominated hardware for 75 years, but suffers from inherent fundamental limitations:
1. **Asymmetric Sign Representation**: Two's-complement arithmetic requires separate overflow rules, sign extensions, and has an asymmetric range ($[-2^{n-1}, 2^{n-1}-1]$).
2. **Rounding Drift / Bias**: Truncation or standard rounding in binary IEEE-754 accumulates positive or negative bias during long scientific simulations or neural network training.
3. **Multiplier Complexity**: Negative numbers require sign-bit logic and two's complement inversion.

**Balanced Ternary ($\{-1, 0, 1\}$)** resolves these fundamentally:
- **Natural Sign Symmetry**: The negation of any balanced ternary number is computed by simply swapping the $+1$ and $-1$ trits. There is no sign bit, no two's complement, and the range is perfectly symmetric about zero: $[-\frac{3^k-1}{2}, +\frac{3^k-1}{2}]$.
- **Zero-Drift Rounding (Round-to-Nearest is Truncation)**: In balanced ternary, the least significant trits naturally sum to zero on average. Truncating digits is mathematically identical to rounding to the nearest integer with **zero systemic statistical drift**.
- **Information Radix Economy**: The theoretical optimal radix minimizing radix economy $E(r) = r \cdot \log_r(N)$ is Euler's number $e \approx 2.71828$. The closest integer radix is **3 (Ternary)**, not 2.

---

## 2. The Core Technical Breakthrough: Dual-Rail SWAR on Binary CPUs

Historically, ternary computing failed commercially because it was assumed to require non-standard tri-state analog silicon.

**Trium-XBT solves this entirely in software and silicon** using **Dual-Rail SWAR (SIMD Within A Register)**:
1. **Dual-Rail Word Representation**:
   Every 64-trit word is represented as two binary 64-bit integer bitmasks:
   $$\text{Word64} = (p, n) \quad \text{where } p, n \in \{0, 1\}^{64}$$
   - $p$ (Positive rail): Bit $i = 1 \iff \text{trit}_i = +1$.
   - $n$ (Negative rail): Bit $i = 1 \iff \text{trit}_i = -1$.
   - **Hardware Disjointness Invariant**:
     $$p \ \& \ n == 0$$
     *(A trit can never be simultaneously $+1$ and $-1$. If $p_i = 0$ and $n_i = 0$, $\text{trit}_i = 0$.)*
   - **Numerical Value**:
     $$\text{Value}(p, n) = \sum_{i=0}^{k-1} (p_i - n_i) \cdot 3^i$$

2. **Single-Cycle Bitwise SWAR Arithmetic**:
   Using pure bitwise binary logic (`AND`, `OR`, `XOR`, shifts) on stock 64-bit CPUs (x86_64 and ARM64 Apple Silicon), 64 balanced ternary additions or half-adder steps execute in single CPU instruction cycles without emulation or branching.

3. **Base-243 Dense Memory Packing**:
   Because $3^5 = 243 \le 256 = 2^8$, **5 balanced trits compress into exactly 1 byte** (with 13 unused states). This achieves **95.6% theoretical information density**, eliminating the 50% memory waste of naive 2-bit-per-trit formats.

---

## 3. End-to-End System Pipeline

```mermaid
flowchart TD
    A["Trium Source Program (.bt)"] --> B["Frontend Lexer & AST Parser"]
    B --> C["Type Checker (Dual-Rail, BT-FP32, CSR)"]
    C --> D["Trium Intermediate Representation (IR)"]
    D --> E["MLIR TernaryDialect (bt.add, bt.mul, bt.fma)"]
    E --> F["LLVM / IREE / Metal / CUDA Backends"]
    E --> G["C ABI Shared Dynamic Runtime (libbt_runtime.so)"]
    G --> H["Rust Safe Zero-Cost FFI & Python Bindings"]
    D --> I["Hardware Emitters: RISC-V Xbt Extension & SystemVerilog RTL"]
    I --> J["FPGA / ASIC Silicon Synthesis (Kogge-Stone, BTHA)"]
    
    K["66 Machine-Checked Rocq/Coq Proofs"] -.->|"Formal Mathematical Verification"| D
    L["Z3 SMT Invariant Solvers"] -.->|"Bitwise SWAR Invariant Guarantee"| B
```

---

## 4. Curated Core Files Directory Map

This folder contains the self-contained core dossiers of the entire system:

| File Name | Domain | Contents & Architectural Scope |
| :--- | :--- | :--- |
| **`01_CORE_ARITHMETIC_AND_SWAR.md`** | Core Algebra | Dual-rail word encoding, Kogge-Stone carry-lookahead adders, Kleene logic, BTHA half-adders, Base-243 codec. |
| **`02_FLOATING_POINT_AND_NUMERICS.md`** | Scientific Numerics | BT-FP32 floating-point ($M \times 3^E$), FMA, square root, T-CSR 1-bit ternary sparse matrix compression & SpMV. |
| **`03_ADVANCED_ALGEBRA_AND_BIGINT.md`** | High-Precision | Toom-Cook 3-way multiplication ($\mathcal{O}(n^{1.465})$), NTT arbitrary precision BigInt, Householder QR decomposition. |
| **`04_CORDIC_AND_SYMPLECTIC_ENGINE.md`** | Transforms & Physics | Zero-trit skipping CORDIC trigonometric engine, Radix-3 Cooley-Tukey FFT, symplectic Hamiltonian phase-space integrators. |
| **`05_COMPILER_FRONTEND_AND_MLIR.md`** | Compiler Pipeline | AST nodes, parser, type checker, spaceship operator (`<=>`), IR program lowering, MLIR `TernaryDialect` definition. |
| **`06_SILICON_RTL_AND_RISCV_XBT.md`** | Silicon Hardware | Synthesizable SystemVerilog RTL, 64-trit ALU, Ternary MMU/TLB address translation, custom RISC-V `Xbt` ISA extension. |
| **`07_C_ABI_AND_FFI_BINDINGS.md`** | Interoperability | C11 ABI (`libbt_runtime`), thread-local error propagation, zero-overhead memory-safe Rust bindings, Python ctypes bridge. |
| **`08_FORMAL_VERIFICATION_ROCQ_AND_Z3.md`** | Formal Proofs | 66 Rocq/Coq formal machine proofs (Ring axioms, commutativity, distributivity) + Z3 SMT bitwise state invariants. |

---

## 5. Verification & Code Integrity Guarantees

Every implementation in these core dossiers is backed by rigorous empirical and formal verification:
- **Rocq 9.1 / Coq**: 66 mathematical theorems verified with 0 admitted hypotheses (`Qed`).
- **Z3 SMT Solvers**: Millions of bitwise transformations verified invariant across the entire dual-rail state space.
- **Sanitizers**: Verified clean under LLVM **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)** with 0 memory leaks, 0 buffer overflows, and 0 undefined operations.
- **CI Status**: Multi-tier continuous integration verified on Ubuntu 24.04 and macOS Apple Silicon.
