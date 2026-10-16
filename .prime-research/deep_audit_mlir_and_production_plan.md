# Exhaustive Codebase Audit: MLIR, IREE, Build Infrastructure & Production Generalization Masterplan

**Author:** Antigravity Prime Research & Production Engineering Engine  
**Facet:** Facet 3 (MLIR Dialect, Optimizer Tools, IREE Hardware Dispatch, Test Suite & Production Architecture)  
**Workspace:** `/Users/mohana/Desktop/Balanced ternary system`  
**Date:** October 16, 2026  

---

## 1. Executive Summary & Verification State
The balanced ternary compiler workspace features a comprehensive 7-layer testing hierarchy emulating the LLVM test suite (`ninja check-all`, lit, FileCheck, and compiler-rt sanitizers). All 7 layers currently pass cleanly (0 memory leaks, 0 out-of-bounds accesses, 0 undefined behaviors).

However, a deep structural audit of the MLIR dialect implementation, the `bt-opt` optimizer driver, the IREE hardware abstraction layer plugin, and the code emitters reveals critical bugs, mock stub limitations, and architectural gaps that must be resolved to transition from an educational/prototype system to a battle-tested, 100% generalized, production-ready compiler.

---

## 2. Deep File-by-File Audit & Exact Defect Catalog

### 2.1. `ternary_mlir_lowering.hpp`
* **Defect 1.1: Syntax Formatting Failure in `MLIROperation::to_string()` (Lines 270–288)**
  * *Location:* `ternary_mlir_lowering.hpp:270-288`
  * *Code:*
    ```cpp
    if (!operands.empty()) {
        if (operands.size() > 1 && results.size() > 1) ss << "(";
        for (size_t i = 0; i < operands.size(); ++i) {
            ss << operands[i].type.to_string();
            if (i + 1 < operands.size()) ss << ", ";
        }
        if (operands.size() > 1 && results.size() > 1) ss << ")";
    }
    ```
  * *Root Cause:* The condition `operands.size() > 1 && results.size() > 1` is erroneous. When an operation has multiple operands (e.g. 2 operands) and exactly 1 result (e.g. `bt.planar_dot(a, b) -> i64` or `bt.matmul(A, B, C) -> tensor`), it omits parentheses around operands, generating `i64, i64 -> i64` instead of `(i64, i64) -> i64`. This violates the MLIR specification (MLIR Language Reference § Functional Types) and causes syntax errors in upstream `mlir-opt`.
  * *Fix:* Wrap operand types in parentheses whenever `operands.size() > 1` (or `operands.empty()`), regardless of `results.size()`.

* **Defect 1.2: Tensor Type Dimension Notation Inconsistency (Lines 160–169)**
  * *Location:* `ternary_mlir_lowering.hpp:160-169`
  * *Root Cause:* `MLIRType::to_string()` for `TypeKind::TernaryTensor` emits `!bt.tensor<64x64x64>` where the trailing number is `bitwidth`. In `include/Ternary/TernaryDialect.td`, `BT_TensorType` specifies assembly format ``<` custom<DimensionList>($shape) `,` $width `>``. When parsed by standard MLIR parser, dimension list and parameters conflict.
  * *Fix:* Format explicitly as `!bt.tensor<64x64, 64>` matching TableGen parameter syntax.

* **Defect 1.3: Planar Contraction Shape & Transpose Semantic Mismatch in Linalg Lowering (Lines 383–433)**
  * *Location:* `ternary_mlir_lowering.hpp:383-433`
  * *Root Cause:* `lower_matmul_to_linalg()` defines tensor shapes as `shapeA = {M, K}` and `shapeB = {K, N}` of `i64`. In dual-rail planar encoding, each `i64` represents 64 trits along K. If $K$ is the number of trits, the tensor dimension along the reduction axis must be `K_words = ceil(K / 64)`. Furthermore, accessing $B$ column-wise across words requires $B^T$ (pre-transposed matrix) because in `TernaryMatrix`, rows are contiguous words while columns are bit-sliced across distinct words.

---

### 2.2. `lib/Ternary/TernaryPasses.cpp` & `include/Ternary/TernaryPasses.h`
* **Defect 2.1: Critical Mathematical Correctness Bug in `ConvertCSAReducePass` (Lines 31–42)**
  * *Location:* `lib/Ternary/TernaryPasses.cpp:31-42`
  * *Code:*
    ```cpp
    %s1, %s1_n, %c1_p, %c1_n = func.call @bt_btha_lowering(%sp, %sn, %sh_cp, %sh_cn) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)
    func.return %s1, %s1_n : i64, i64
    ```
  * *Root Cause:* The pass performs only a single BTHA addition step between Sum and Shifted Carry, and then immediately returns `%s1, %s1_n`, discarding `%c1_p, %c1_n`! In balanced ternary arithmetic, adding Sum and Shifted Carry produces a new carry whenever $S_i + C_{i-1} \in \{+2, -2\}$. Discarding the carry produces mathematically incorrect results for all inputs where carries propagate beyond 1 position.
  * *Fix:* Lower `bt.csa_reduce` to either an unrolled 64-level BTHA sequence, a scf.while loop until Carry is zero, or a Douglas W. Jones logarithmic parallel prefix tree.

* **Defect 2.2: High-Order Carry-Out Silent Truncation in `CSAReduceOp::execute` (Lines 44–52)**
  * *Location:* `lib/Ternary/TernaryDialect.cpp:45-46`
  * *Code:* `uint64_t sh_P = curr_C.P << 1; uint64_t sh_N = curr_C.N << 1;`
  * *Root Cause:* Left-shifting 64-bit integer planes silently drops bit 63 without carry-out detection or multi-word propagation. If trit 63 generates a carry, it is lost.
  * *Fix:* Track carry-out bit 63 into an overflow flag or return an extended `(TritWord64, int8_t carry_out)` tuple.

---

### 2.3. `tools/bt-opt/bt-opt.cpp`
* **Defect 3.1: Complete Decoupling of Input MLIR File from Pass Manager Pipeline (Lines 124–209)**
  * *Location:* `tools/bt-opt/bt-opt.cpp:124-209`
  * *Root Cause:* When passes are specified (e.g. `bt-opt --convert-all input.mlir`), the tool reads `inputContent` into memory, but never parses it into `mod`! Instead, `mod` is an empty module, and `pm.run(mod)` executes passes that append static template functions (`bt_gemm_planar_linalg`, `bt_planar_dot_vector_simd`, etc.). When writing output, if `!pm.passes.empty()`, it prints `mod.to_string()` and completely discards `inputContent`!
  * *Architectural Weakness:* The standalone build lacks an MLIR text parser and pattern rewrite engine. It functions as a template generator rather than an optimizing IR transformer.
  * *Fix:* Implement a lightweight recursive-descent MLIR operation scanner for standalone builds, and hook into `mlir::RewritePatternSet` / `mlir::applyPatternsGreedily` when compiled with `BT_HAS_MLIR=1`.

---

### 2.4. `iree/TernaryIREEPlugin.h` & `iree/TernaryIREEPlugin.cpp`
* **Defect 4.1: Vulkan SPIR-V Shader Compilation Failure (`bitCount` on 64-bit integer) (Lines 40–42)**
  * *Location:* `iree/TernaryIREEPlugin.cpp:40-42`
  * *Code:* `acc += int64_t(bitCount(prod_p)) - int64_t(bitCount(prod_n));`
  * *Root Cause:* In Vulkan GLSL / SPIR-V, `bitCount()` does not take `uint64_t` operands. Compiling this shader with `glslangValidator` produces compile error: `no matching overloaded function found`.
  * *Fix:* Add `#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require` and unpack 64-bit integers into 32-bit pairs:
    ```glsl
    uvec2 p_vec = unpackUint2x32(prod_p);
    uvec2 n_vec = unpackUint2x32(prod_n);
    acc += int64_t(bitCount(p_vec.x) + bitCount(p_vec.y)) - int64_t(bitCount(n_vec.x) + bitCount(n_vec.y));
    ```

* **Defect 4.2: Missing Tail-Masking & Integer Division Truncation on Non-Multiples of 64 Trits (Lines 33, 65, 93, 166–169)**
  * *Location:* `iree/TernaryIREEPlugin.cpp:33, 65, 93`
  * *Code:* `for (uint k = 0; k < (K / 64); ++k)` and `row * (K / 64) + k`
  * *Root Cause:* If $K$ is not a multiple of 64:
    1. If $K < 64$ (e.g. $K=16$), `K / 64 == 0`, loop runs 0 times and matrix multiplication returns all zeros!
    2. If $K = 65$, `K / 64 == 1`, the 65th trit is completely ignored.
    3. Row stride calculation `row * (K / 64)` computes corrupted memory offsets for all rows $> 0$!
  * *Fix:* Compute `words_per_row = (K + 63) / 64`. Stride by `words_per_row`. When $k == 	ext{words\_per\_row} - 1$ and $K \% 64 
eq 0$, apply bitmask `(1ULL << (K % 64)) - 1`.

* **Defect 4.3: Empty Stubs in CPU LLVM and Flow/Stream Modules (Lines 111–116, 171–173, 186–188)**
  * *Location:* `iree/TernaryIREEPlugin.cpp:111-116, 171-173, 186-188`
  * *Root Cause:*
    - `CPULLVMBackend::emitShaderKernel` emits an empty function with only `ret void`.
    - `buildFlowDispatchModule` contains an empty dispatch kernel with `return`.
    - `buildStreamDispatchModule` uses `(N/8)` and `(M/8)` which dispatches 0 workgroups if $N < 8$ or $M < 8$.

---

### 2.5. `CMakeLists.txt` & Build Configuration
* **Defect 5.1: TableGen IncGen Missing from C++ Sources (Lines 66–71, 89–108)**
  * *Location:* `CMakeLists.txt:66-71`
  * *Root Cause:* CMake generates `TernaryOpsDialect.h.inc` and `TernaryOps.h.inc` when `BT_HAS_MLIR` is enabled, but neither `TernaryDialect.h` nor `TernaryDialect.cpp` includes these files. `TernaryDialect` does not inherit from `::mlir::Dialect`. If built against upstream LLVM/MLIR, compilation will fail.
  * *Fix:* In `TernaryDialect.h`, conditionally include generated `.inc` files inside `#if defined(BT_HAS_MLIR)` blocks.

* **Defect 5.2: Path Quoting Vulnerability in Sanitizers Test (Line 214)**
  * *Location:* `CMakeLists.txt:214`
  * *Root Cause:* `COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test/Sanitizers/run_sanitizers.sh` lacks quotes. Because the active workspace directory name contains spaces (`Balanced ternary system `), this causes shell path splitting in certain environments.

---

### 2.6. `test/lit_runner.py`
* **Defect 6.1: Pipelining Limitations in FileCheck Parser (Lines 103–125)**
  * *Location:* `test/lit_runner.py:103-125`
  * *Root Cause:* `lit_runner.py` handles `| FileCheck` by splitting on the pipe character once and taking element `[0]`. It cannot handle chained tool pipelines (e.g., `%bt-opt %s --pass1 | %bt-opt - --pass2 | FileCheck %s`).

---

## 3. Production Generalization Architectural Masterplan

To transition this compiler into a production-grade infrastructure, four major architectural systems are required:

### 3.1. Standardized C ABI / Foreign Function Interface (FFI)
To allow C, C++, Rust, Python (`ctypes`/`cffi`), Julia, and Mojo to link and call compiled `.bt` binaries:

1. **System V & AAPCS64 Calling Convention Standardization**:
   - `bt_trit_t`: `int8_t` restricted to `{-1, 0, +1}`.
   - `bt_word64_t`: Packed 128-bit structure:
     ```c
     typedef struct {
         uint64_t P;
         uint64_t N;
     } bt_word64_t;
     ```
     Passed in registers `x0, x1` on ARM64 and `rdi, rsi` on x86_64.
   - `bt_matrix_t`:
     ```c
     typedef struct {
         uint64_t* data_P;
         uint64_t* data_N;
         uint64_t rows;
         uint64_t cols;
         uint64_t words_per_row;
     } bt_matrix_t;
     ```

2. **Core C Entry Points (`extern "C"` Linkage)**:
   - `bt_word64_t bt_make_word64(uint64_t P, uint64_t N);`
   - `bt_word64_t bt_from_int64(int64_t value);`
   - `int64_t bt_to_int64(bt_word64_t word);`
   - `int64_t bt_planar_dot(uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn);`
   - `void bt_btha(uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn, uint64_t* sum_p, uint64_t* sum_n, uint64_t* carry_p, uint64_t* carry_n);`
   - `void bt_gemm(const bt_matrix_t* A, const bt_matrix_t* B, int64_t* C);`
   - `void bt_unpack_base243(uint8_t byte, int8_t out_trits[5]);`
   - `uint8_t bt_pack_base243(const int8_t in_trits[5]);`

3. **FFI Distribution Artifacts**:
   - `include/libbt/ternary_c_abi.h` (C99 / C++ standard header)
   - `bindings/rust/ternary-sys` (`#[repr(C)]` extern blocks)
   - `bindings/python/libbt.py` (ctypes bindings with zero-copy NumPy buffer support)

---

### 3.2. Arbitrary Matrix Dimension Handling & Tail-Masking Engine
Current GEMM implementations assume matrix dimensions are exact multiples of 64 trits.
For general production shapes $(M 	imes K) 	imes (K 	imes N)$ with arbitrary $K$:

1. **Storage Specification**:
   $$	ext{words\_per\_row} = \lceil K / 64 ceil = (K + 63) / 64$$
   $$	ext{tail\_bits} = K \pmod{64}$$

2. **Mask Computation**:
   $$	ext{tail\_mask} = egin{cases} (1	ext{ULL} \ll 	ext{tail\_bits}) - 1 & 	ext{if } 	ext{tail\_bits} > 0 \ \sim 0	ext{ULL} & 	ext{if } 	ext{tail\_bits} == 0 \end{cases}$$

3. **Tail-Masked SIMD Contraction Algorithm**:
   ```cpp
   for (size_t w = 0; w < words_per_row; ++w) {
       uint64_t ap = A.word(i, w).P;
       uint64_t an = A.word(i, w).N;
       uint64_t bp = BT.word(j, w).P;
       uint64_t bn = BT.word(j, w).N;

       if (w + 1 == words_per_row && tail_bits != 0) {
           ap &= tail_mask;
           an &= tail_mask;
           bp &= tail_mask;
           bn &= tail_mask;
       }

       acc += dotProduct64(TritWord64(ap, an), TritWord64(bp, bn));
   }
   ```
4. Applied across all GPU kernels (MSL, Vulkan, CUDA) and MLIR lowerings.

---

### 3.3. DWARF Source-Level Debugging Metadata (`!dbg` in LLVM IR)
To enable native debugging with `lldb` and `gdb` (setting breakpoints at `.bt` line numbers, stepping, variable inspection):

1. **LLVM Metadata Tree**:
   ```llvm
   !llvm.dbg.cu = !{!0}
   !llvm.module.flags = !{!1, !2}

   !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "Universal Balanced Ternary Compiler v2.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
   !3 = !DIFile(filename: "program.bt", directory: "/workspace")
   !1 = !{i32 2, !"Dwarf Version", i32 5}
   !2 = !{i32 2, !"Debug Info Version", i32 3}

   !4 = distinct !DISubprogram(name: "bt_main", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
   !5 = !DISubroutineType(types: !6)
   !6 = !{!7}
   !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
   ```

2. **Source Location Binding**:
   Every AST node in `ternary_frontend_ast.hpp` inherits `SourceLocation loc` (line, column).
   `LLVMCodeGenVisitor` will attach `!dbg !DILocation(line: loc.line, column: loc.col, scope: !4)` to all instructions.

---

### 3.4. Douglas W. Jones SWAR Parallel BCT-to-Binary Tree Reduction
The sequential Horner loop in `Codec::to_int64()` requires 39 sequential `acc = acc * 3 + trit` multiply-add iterations with strict data hazards.

Douglas W. Jones demonstrated that Binary Coded Ternary (2 bits per trit) can be converted to binary integer in $\log_2(N)$ parallel SIMD steps within a register:

1. **Algorithm Structure**:
   - Given 32 packed trits $d_0, \dots, d_{31}$ in dual-rail $(P, N)$ registers:
   - **Level 1 (Pairs $	o$ Base 9):**
     Pairs $(d_{2i+1}, d_{2i})$ form signed values in $\{-4, \dots, +4\}$.
     Evaluated in parallel across sixteen 4-bit fields:
     $$V_1 = 3 \cdot d_{	ext{odd}} + d_{	ext{even}}$$
   - **Level 2 (Quads $	o$ Base 81):**
     Evaluated in parallel across eight 8-bit fields:
     $$V_2 = 9 \cdot V_{1,	ext{odd}} + V_{1,	ext{even}}$$
   - **Level 3 (Octets $	o$ Base 6561):**
     Evaluated in parallel across four 16-bit fields:
     $$V_3 = 81 \cdot V_{2,	ext{odd}} + V_{2,	ext{even}}$$
   - **Level 4 (Hexadecs $	o$ Base $3^{16}$):**
     Evaluated across two 32-bit fields:
     $$V_4 = 6561 \cdot V_{3,	ext{odd}} + V_{3,	ext{even}}$$
   - **Level 5 (Full Word $	o$ 64-bit Binary):**
     $$V_{	ext{final}} = 43046721 \cdot V_{4, 1} + V_{4, 0}$$

2. **Performance Gain**:
   - Sequential Horner: 39 iterations $	imes$ 3 cycles = 117 cycles.
   - SWAR Tree: 5 vector stages $	imes$ 2 cycles = 10 cycles ($11.7	imes$ speedup)!

---

## 4. Prioritized Engineering Action Plan

| Priority | Component | File(s) | Action |
|---|---|---|---|
| **P0 (Critical)** | Correctness | `lib/Ternary/TernaryPasses.cpp` | Fix `ConvertCSAReducePass` to resolve carries iteratively or via parallel prefix tree instead of single HA step. |
| **P0 (Critical)** | Shader Compilation | `iree/TernaryIREEPlugin.cpp` | Replace `bitCount(uint64_t)` with 32-bit unpacked `bitCount()` and enable `GL_EXT_shader_explicit_arithmetic_types_int64`. |
| **P0 (Critical)** | Matrix Dimensions | `iree/TernaryIREEPlugin.cpp`, `ternary_dense_algebra.hpp` | Implement `words_per_row = (K + 63)/64` and tail-masking for all GEMM loops. |
| **P1 (High)** | MLIR Syntax | `ternary_mlir_lowering.hpp` | Fix `MLIROperation::to_string()` operand type parentheses formatting for standard MLIR compliance. |
| **P1 (High)** | TableGen Integration | `include/Ternary/TernaryDialect.h`, `CMakeLists.txt` | Conditionally inherit `mlir::Dialect` and include generated `.inc` files when `BT_HAS_MLIR=1`. |
| **P1 (High)** | C ABI / FFI | `include/libbt/ternary_c_abi.h` (new) | Export `extern "C"` System V / AAPCS64 bindings for C, C++, Rust, and Python. |
| **P2 (Medium)** | Debug Metadata | `ternary_llvm_codegen.hpp` | Generate DWARF 5 metadata (`!dbg`, `!DICompileUnit`, `!DILocation`) from AST source locations. |
| **P2 (Medium)** | Fast Radix Conversion | `libbt.hpp`, `ternary_core.hpp` | Implement Douglas W. Jones logarithmic SWAR BCT-to-binary parallel tree reduction. |
| **P2 (Medium)** | Test Harness | `test/lit_runner.py` | Support multi-pipe commands in test runner (`cmd1 \| cmd2 \| FileCheck`). |

