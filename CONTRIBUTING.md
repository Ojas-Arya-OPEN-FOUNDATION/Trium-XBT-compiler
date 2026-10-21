# Contributing to Next-Gen Foundation Compiler

Thank you for your interest in contributing to the **Next-Gen Foundation Compiler**, hosted by **Ojas-Arya OPEN-FOUNDATION**!

We welcome contributions from mathematicians, compiler engineers, hardware architects, and open-source enthusiasts.

---

## 🛠️ Development Workflow

### 1. Code Standards & Architecture
- **Language**: Modern C++20 (`-std=c++20`).
- **Dependencies**: Header-only core numerics; standard library only (no external boost/abseil dependencies for core arithmetic).
- **Correctness**: Every core algebraic identity must be provable or accompanied by formal verification in Z3, Lean 4, or Rocq.
- **Documentation**: All new primitives must be documented with worked mathematical examples.

### 2. Testing & Verification Gate
Before submitting any pull request or commit, your changes must pass the complete **7-Layer LLVM Verification Suite** and AddressSanitizer/UndefinedBehaviorSanitizer:

```bash
# On macOS:
export SDKROOT=$(xcrun --show-sdk-path)

# Run full test suite:
./run_all_tests.sh
```

All 7 layers must report 100% success:
1. `test/Unit/` (API & microbenchmarks)
2. `test/Regression/` (`lit` + `FileCheck`)
3. `test/WholeProgram/` (End-to-end `.bt` executions)
4. `test/Sanitizers/` (ASan + UBSan: 0 leaks, 0 undefined behavior)
5. `test/CodeGen/` (Target codegen for AArch64, x86_64, RISC-V, Metal, Vulkan)
6. `test/Dialect/` (MLIR `bt` dialect lowers)
7. `test/Sema/` (TypeChecker & front-end diagnostics)

### 3. Formal Proof Verification
If you modify core algebraic algorithms (`ternary_core.hpp`, `ternary_fp.hpp`, `ternary_bigint.hpp`, or `symplectic_ternary_engine.hpp`), please run the formal provers:
```bash
# Lean 4:
lean BalancedTernary.lean
# Rocq 9.2:
rocq compile proofs/rocq/BalancedTernaryFoundations.v
# Z3 SMT:
python3 verify_z3_kogge_stone.py
```

---

## 📬 Reporting Issues
- Use GitHub Issues to report bugs or request new mathematical features.
- Please provide minimal reproducible `.bt` source code or C++ test cases.

---

## 📜 License
By contributing to this repository, you agree that your contributions will be licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
