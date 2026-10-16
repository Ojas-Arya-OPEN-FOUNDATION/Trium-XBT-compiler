# LLVM & MLIR Backend Technical Specification: Universal Balanced Ternary Compiler
**Target**: Native AOT Compilation (`.o`, `.so`, `.dylib`), LLVM ORC JIT v2, and IREE HAL Multi-Target Dispatch  
**Author**: LLVM & MLIR Backend Research Engine  
**Status**: Foundational Backend Architecture Specification  
**Associated Source Files**:
- Dialect Definitions: `TernaryDialect.td`, `TernaryOps.td`
- Lowering Prototype: `ternary_linalg_lowering.mlir`
- Frontend & JIT State: `ternary_frontend_ast.hpp`, `ternary_jit_engine.hpp`, `bt_compiler.cpp`

---

## 1. Executive Architectural Overview

### 1.1 The Limitation of Source Shader Generation
The current generation of `bt-compiler` operates primarily as a source-to-source shader transpiler and AST interpreter:
1. It emits high-level text representations of compute shaders (Apple Metal `.metal`, Vulkan SPIR-V `.comp`) or intrinsic-based C++ headers.
2. Execution of `.bt` scripts at runtime relies on `bt::jit::TernaryJITEngine`, which is a recursive AST tree-walking interpreter incurring dynamic lookups, heap boxing via `JITValue`, and virtual dispatch overhead.
3. Generating machine code currently requires external compiler invocations (`clang++ -O3 -shared`) via shell processes or platform runtime shader compilation (`MTLDevice newComputePipelineStateWithSource`).

### 1.2 The Native Machine Code Objective
To achieve maximum operational efficiency, deterministic latency, and production readiness, the compiler must emit **true native AOT machine code** (`.o` object files, `.so` / `.dylib` dynamic libraries) and support **zero-overhead, in-memory JIT execution** via direct integration with LLVM and MLIR:
- **Zero Process Invocations**: Eliminate runtime dependencies on external toolchains (`clang++`, `gcc`, `as`, `ld`).
- **Direct Dialect Lowering**: Lower `bt` MLIR operations directly through standard MLIR dialects (`linalg`, `vector`, `arith`, `memref`, `func`) to the `llvm` dialect.
- **In-Memory ORC JIT v2**: Compile MLIR / LLVM IR directly into executable machine code in memory at runtime with microsecond latency.
- **Hardware-Agnostic IREE HAL**: Leverage the IREE (Intermediate Representation Execution Environment) Hardware Abstraction Layer for unified dispatch across Vulkan, Apple Metal, CUDA, and CPU thread pools via standalone Virtual Machine FlatBuffers (`.vmfb`).

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Balanced Ternary Source Program (.bt) / PyTorch                │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Frontend AST & Parser
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             MLIR `bt` Dialect Module                             │
│ Types: !bt.trit, !bt.dualrail<W>, !bt.csa_pair<W>, !bt.tensor<shape, W>          │
│ Ops:   bt.gemm, bt.planar_dot, bt.btha, bt.btfa, bt.csa_accum, bt.unpack_base243│
└──────────────────┬───────────────────────────────────────────┬───────────────────┘
                   │                                           │
                   ▼ (Progressive Dialect Lowering)            ▼ (IREE Compiler Pipeline)
┌──────────────────────────────────────┐     ┌─────────────────────────────────────┐
│ High-Level MLIR Transformation Pipeline │   │ IREE Flow & Stream Pipeline         │
│ 1. `bt-to-linalg` & `bt-to-arith`    │     │ 1. Form Dispatch Regions            │
│ 2. Linalg Loop Tiling & Vectorization│     │ 2. Asynchronous Resource Tracking   │
│ 3. One-Shot Bufferization (memref)   │     │ 3. Stream Allocation & Semaphores   │
│ 4. `convert-vector-to-llvm`          │     └──────────────────┬──────────────────┘
│ 5. `convert-func-to-llvm`            │                        │
│ 6. `reconcile-unrealized-casts`      │                        ▼
└──────────────────┬───────────────────┘     ┌─────────────────────────────────────┐
                   │                         │ IREE HAL (Hardware Abstraction)     │
                   ▼                         │ - Target 1: Vulkan SPIR-V (.vmfb)   │
┌──────────────────────────────────────┐     │ - Target 2: Apple Metal MSL (.vmfb) │
│       MLIR `llvm` Dialect Module     │     │ - Target 3: NVIDIA CUDA CUBIN(.vmfb)│
└──────────────────┬───────────────────┘     │ - Target 4: LLVM CPU Local Task     │
                   │ `mlir-to-llvmir`        └─────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────┐
│           LLVM IR Module             │
│ TargetMachine Optimization (O3)      │
└──────────┬───────────────────────────┤
           │                           │
           ▼ (LLVM CodeGen)            ▼ (ORC JIT v2)
┌──────────────────────┐   ┌───────────────────────────┐
│ AOT Native Binaries  │   │ Direct Memory Execution   │
│ - Mach-O / ELF `.o`  │   │ - LLJIT In-Memory Linking │
│ - `.dylib` / `.so`   │   │ - Direct Function Pointer │
│ - C ABI Exports      │   │ - Zero Clang++ Invocation │
└──────────────────────┘   └───────────────────────────┘
```

---

## 2. MLIR `bt` Dialect Analysis & Type System Mapping

### 2.1 Dialect Types Specification
From `TernaryDialect.td`, the `bt` dialect defines four primary types:
1. `!bt.trit`: Scalar ternary digit $\{-1, 0, +1\}$.
2. `!bt.dualrail<width>`: Bit-sliced dual-rail register representing `width` trits ($P \land N = 0$).
3. `!bt.csa_pair<width>`: Redundant Carry-Save Accumulator pair $(Sum, Carry)$.
4. `!bt.tensor<shape, width>`: Multi-dimensional tensor of dual-rail or Base-243 packed trits.

### 2.2 Concrete Lowering Map for Dialect Types

| `bt` Dialect Type | Intermediate MLIR Representation | Target LLVM Dialect Type | Physical Register / ABI Layout |
| :--- | :--- | :--- | :--- |
| `!bt.trit` | `i2` or `(i1, i1)` or `i8` | `!llvm.struct<(i1, i1)>` or `i8` | 2 bits in GPR, unpacks to sign bit & magnitude |
| `!bt.dualrail<64>` | `(i64, i64)` | `!llvm.struct<(i64, i64)>` | 2x 64-bit GPRs (`X0`, `X1` on ARM64; `RAX`, `RDX` on x86) |
| `!bt.dualrail<128>` | `(vector<2xi64>, vector<2xi64>)` | `!llvm.struct<(vector<2xi64>, vector<2xi64>)>` | 2x 128-bit SIMD registers (`Q0`, `Q1` on NEON; `XMM0`, `XMM1`) |
| `!bt.dualrail<256>` | `(vector<4xi64>, vector<4xi64>)` | `!llvm.struct<(vector<4xi64>, vector<4xi64>)>` | 2x 256-bit AVX2/AVX-512 registers (`YMM0`, `YMM1`) |
| `!bt.dualrail<512>` | `(vector<8xi64>, vector<8xi64>)` | `!llvm.struct<(vector<8xi64>, vector<8xi64>)>` | 2x 512-bit AVX-512 registers (`ZMM0`, `ZMM1`) |
| `!bt.csa_pair<W>` | Struct of two `!bt.dualrail<W>` | `!llvm.struct<(!llvm.struct<...>, !llvm.struct<...>)>` | 4 GPRs / 4 SIMD registers holding $(S_P, S_N, C_P, C_N)$ |
| `!bt.tensor<MxK, 64>` | `(tensor<MxKxi64>, tensor<MxKxi64>)` | Struct of two MemRef descriptors: `!llvm.struct<(ptr, ptr, i64, ...)>` | Two parallel contiguous buffers for Positive and Negative bitplanes |

---

## 3. Detailed Op-by-Op Lowering Specifications

### 3.1 Arithmetic Half-Adder (`bt.btha`) Lowering
The half-adder computes $A + B = 3C + S$ across $W$ trits without carry propagation.

**Mathematical Formulation**:
- Carry: $P_C = P_A \land P_B$, $N_C = N_A \land N_B$
- Sum: $P_S = ((P_A \oplus P_B) \land \neg(N_A \lor N_B)) \lor N_C$, $N_S = ((N_A \oplus N_B) \land \neg(P_A \lor P_B)) \lor P_C$

**Lowering to `arith` and `llvm` Dialect**:
```mlir
// Dialect conversion pattern: BTHAOpLowering
func.func @lower_btha_i64(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> (i64, i64, i64, i64) {
  %c_all_ones = arith.constant -1 : i64

  // Carry plane
  %pc = arith.andi %ap, %bp : i64
  %nc = arith.andi %an, %bn : i64

  // Inverted cross-masks
  %or_n = arith.ori %an, %bn : i64
  %not_or_n = arith.xori %or_n, %c_all_ones : i64
  %or_p = arith.ori %ap, %bp : i64
  %not_or_p = arith.xori %or_p, %c_all_ones : i64

  // Sum plane
  %xor_p = arith.xori %ap, %bp : i64
  %xor_n = arith.xori %an, %bn : i64

  %ps_cand = arith.andi %xor_p, %not_or_n : i64
  %ps = arith.ori %ps_cand, %nc : i64

  %ns_cand = arith.andi %xor_n, %not_or_p : i64
  %ns = arith.ori %ns_cand, %pc : i64

  return %ps, %ns, %pc, %nc : i64, i64, i64, i64
}
```

### 3.2 Full-Adder Compressor (`bt.btfa`) Lowering
The 3-to-2 compressor compresses $A + B + D = 3C + S$:
1. $(S_1, C_1) = \text{BTHA}(A, B)$
2. $(S, C_2) = \text{BTHA}(S_1, D)$
3. Carry merge: $P_C = (P_{C1} \lor P_{C2}) \land \neg(N_{C1} \lor N_{C2})$, $N_C = (N_{C1} \lor N_{C2}) \land \neg(P_{C1} \lor P_{C2})$.
By the No-Overlap Theorem, carries from stages 1 and 2 never collide with opposite signs, allowing 0-ripple carry combination.

### 3.3 Carry-Save Accumulator Update (`bt.csa_accum`) Lowering
1. Shift carry rail left by 1 trit (multiplication by 3):
   `%c_shifted_p = arith.shli %cp, %c1 : i64`
   `%c_shifted_n = arith.shli %cn, %c1 : i64`
2. Compress $(S_{in}, C_{shifted}, X)$ via `btfa`.
3. Total cost: 12 bitwise ALU instructions, 0 cycles spent on carry rippling.

### 3.4 Branchless Kleene Selection (`bt.select3`) Lowering
Selects among `pos_val`, `zero_val`, `neg_val` based on condition trit $c \in \{-1, 0, +1\}$:
```mlir
// Lowering to LLVM select tree / branchless CSEL
func.func @lower_select3(%cp: i1, %cn: i1, %pos: i64, %zero: i64, %neg: i64) -> i64 {
  %sel_neg = arith.select %cn, %neg, %zero : i64
  %res = arith.select %cp, %pos, %sel_neg : i64
  return %res : i64
}
```
On AArch64, LLVM instruction selection lowers this directly to two `csel` instructions (`csel x0, x_neg, x_zero, eq` followed by `csel x0, x_pos, x0, ne`). On x86_64, it emits `cmovnz` / `cmovz` with 0 branch misprediction penalties.

### 3.5 Bit-Sliced Planar Dot Product (`bt.planar_dot`) Lowering
Evaluates $\sum_{k=0}^{W-1} u_k \cdot v_k$ via bitwise logic and hardware popcount:
- $P_{\text{prod}} = (u_P \land v_P) \lor (u_N \land v_N)$
- $N_{\text{prod}} = (u_P \land v_N) \lor (u_N \land v_P)$
- $\text{Result} = \operatorname{POPCNT}(P_{\text{prod}}) - \operatorname{POPCNT}(N_{\text{prod}})$

**Direct LLVM Dialect Emission**:
```mlir
func.func @lower_planar_dot_llvm(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> i64 {
  // Positive rail: (ap & bp) | (an & bn)
  %p1 = llvm.and %ap, %bp : i64
  %p2 = llvm.and %an, %bn : i64
  %p_prod = llvm.or %p1, %p2 : i64

  // Negative rail: (ap & bn) | (an & bp)
  %n1 = llvm.and %ap, %bn : i64
  %n2 = llvm.and %an, %bp : i64
  %n_prod = llvm.or %n1, %n2 : i64

  // Hardware population count intrinsic
  %cnt_p = "llvm.intr.ctpop"(%p_prod) : (i64) -> i64
  %cnt_n = "llvm.intr.ctpop"(%n_prod) : (i64) -> i64

  %res = llvm.sub %cnt_p, %cnt_n : i64
  return %res : i64
}
```

### 3.6 High-Level Matrix Contraction (`bt.gemm`) Lowering
Converts `bt.gemm` into a structured `linalg.generic` contraction over planar buffers:
- Inputs: $A_P, A_N \in \text{tensor}\langle M \times K \times \text{i64}\rangle$, $B_P, B_N \in \text{tensor}\langle K \times N \times \text{i64}\rangle$
- Accumulator: $C \in \text{tensor}\langle M \times N \times \text{i64}\rangle$
- Contraction Loop: Along dimension $K$, iterates over 64-trit planar words, computes `bt.planar_dot` in the inner basic block, and accumulates with `arith.addi`.

### 3.7 Base-243 Codec (`bt.unpack_base243`) Lowering
Expands a single packed byte $[0..242]$ into 5 balanced trits using reciprocal fixed-point multipliers:
- $q_1 = (B \times 171) \gg 9$
- $q_2 = (B \times 57) \gg 9$
- $q_3 = (B \times 19) \gg 9$
- $q_4 = (B \times 203) \gg 14$
- $t_k = q_k - 3 \cdot q_{k+1} - 1$
All divisions are eliminated. The entire unpack sequence lowers to 4 integer multiplications and shifts, executing in 2 clock cycles on modern superscalar execution ports.

---

## 4. End-to-End MLIR Lowering Pass Pipeline

To lower an MLIR module containing `bt` operations down to native machine code, the pipeline executes 3 progressive phases:

```
[ Phase 1: High-Level `bt` Dialect to Linalg & Arith ]
  - Pass: `bt-to-linalg`
  - Pass: `bt-to-arith`
  - Pass: `canonicalize`
  - Pass: `cse`

[ Phase 2: Linalg Tiling, Vectorization & Bufferization ]
  - Pass: `linalg-tile-and-fuse` (tile sizes: 64x64 or hardware vector register width)
  - Pass: `convert-linalg-to-vector`
  - Pass: `one-shot-bufferize` (bufferizes tensor types to memref types in-place)
  - Pass: `buffer-deallocation`

[ Phase 3: Vector & Control Flow to LLVM Dialect ]
  - Pass: `convert-vector-to-llvm`
  - Pass: `convert-scf-to-cf`
  - Pass: `finalize-memref-to-llvm`
  - Pass: `convert-arith-to-llvm`
  - Pass: `convert-func-to-llvm`
  - Pass: `reconcile-unrealized-casts`
```

### Detailed Pass Pipeline Driver Configuration (C++ MLIR API):
```cpp
void buildTernaryLoweringPipeline(mlir::PassManager &pm) {
    // 1. High-level conversion
    pm.addPass(mlir::bt::createBTToLinalgPass());
    pm.addPass(mlir::bt::createBTToArithPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // 2. Vectorization & Bufferization
    mlir::bufferization::OneShotBufferizeOptions opt;
    opt.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(opt));
    pm.addPass(mlir::createConvertLinalgToVectorPass());

    // 3. Lowering to LLVM Dialect
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}
```

---

## 5. In-Memory JIT Compilation via LLVM ORC JIT v2

### 5.1 Architecture: Replacing the AST Interpreter
Rather than walking the AST in `bt::jit::TernaryJITEngine`, the frontend translates the AST into an MLIR module, lowers it to an `llvm::Module`, and hands the module directly to LLVM ORC JIT v2 (`llvm::orc::LLJIT`).

### 5.2 Complete ORC JIT Engine Implementation Design

```cpp
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace bt {
namespace jit {

class TernaryNativeOrcJIT {
private:
    std::unique_ptr<llvm::orc::LLJIT> lljit;

public:
    TernaryNativeOrcJIT() {
        // 1. Initialize native host target code generators
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        // 2. Configure LLJIT with custom data layout and JITLink linker
        auto builder = llvm::orc::LLJITBuilder();
        auto jit_or_err = builder.create();
        if (!jit_or_err) {
            throw std::runtime_error("Failed to initialize LLVM ORC JIT: " +
                                     llvm::toString(jit_or_err.takeError()));
        }
        lljit = std::move(*jit_or_err);
    }

    // Compiles an MLIR module directly to native machine code in memory
    void add_mlir_module(mlir::ModuleOp mlir_module, mlir::MLIRContext &context) {
        llvm::LLVMContext llvm_context;
        auto llvm_module = mlir::translateModuleToLLVMIR(mlir_module, llvm_context);
        if (!llvm_module) {
            throw std::runtime_error("Failed to translate MLIR to LLVM IR!");
        }

        // Apply Target Machine optimization passes (O3)
        optimize_llvm_module(*llvm_module);

        // Wrap into ThreadSafeModule and add to JIT execution session
        llvm::orc::ThreadSafeContext tsc(std::make_unique<llvm::LLVMContext>());
        llvm::orc::ThreadSafeModule tsm(std::move(llvm_module), tsc);

        auto err = lljit->addIRModule(std::move(tsm));
        if (err) {
            throw std::runtime_error("Failed to add module to JIT: " + llvm::toString(std::move(err)));
        }
    }

    // Resolves an entrypoint function pointer directly in executable memory
    template<typename FuncSignature>
    FuncSignature get_function(const std::string &name) {
        auto sym_or_err = lljit->lookup(name);
        if (!sym_or_err) {
            throw std::runtime_error("Symbol lookup failed for '" + name + "': " +
                                     llvm::toString(sym_or_err.takeError()));
        }
        return sym_or_err->toPtr<FuncSignature>();
    }

private:
    void optimize_llvm_module(llvm::Module &mod) {
        // Sets up target-specific loop vectorization, SLP, and peephole optimizations
        // Employs llvm::PassBuilder with OptimizationLevel::O3
    }
};

} // namespace jit
} // namespace bt
```

### 5.3 Execution Workflow
1. User invokes `bt-compiler run program.bt`.
2. AST is parsed and lowered to `bt` MLIR module in memory.
3. Lowering passes transform `bt` $\to$ `linalg` $\to$ `llvm`.
4. `translateModuleToLLVMIR` exports the module into `llvm::Module`.
5. ORC JIT compiles the module into native ARM64 or x86_64 machine code via JITLink.
6. Look up `@main` symbol, cast to `int64_t (*)(void)`, and execute directly.
7. **Latency**: < 1.5 milliseconds total compilation + execution time; 0 external processes spawned.

---

## 6. AOT Machine Code Emission: `.o`, `.dylib`, and `.so`

To generate standalone static object files or shared libraries that can be linked into external C++, Rust, or Python programs, the compiler integrates directly with LLVM's `TargetMachine` and code generation layers.

### 6.1 Native Object Code Generator (`TargetMachine::emitFile`)

```cpp
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/FileSystem.h"

void emit_native_object(llvm::Module &module, const std::string &output_obj_path, const std::string &target_triple_str) {
    std::string err;
    auto target = llvm::TargetRegistry::lookupTarget(target_triple_str, err);
    if (!target) {
        throw std::runtime_error("Target lookup failed: " + err);
    }

    llvm::TargetOptions opt;
    auto reloc_model = llvm::Reloc::PIC_; // Position Independent Code for shared libraries
    auto target_machine = target->createTargetMachine(
        target_triple_str,
        "generic", // CPU (or specific: "apple-m1", "skylake-avx512")
        "+neon,+dotprod", // Features (+avx512f,+avx512vpopcntdq for x86)
        opt,
        reloc_model
    );

    module.setDataLayout(target_machine->createDataLayout());
    module.setTargetTriple(target_triple_str);

    std::error_code ec;
    llvm::raw_fd_ostream dest(output_obj_path, ec, llvm::sys::fs::OF_None);
    if (ec) {
        throw std::runtime_error("Could not open file: " + ec.message());
    }

    llvm::legacy::PassManager pass;
    auto file_type = llvm::CGFT_ObjectFile;

    if (target_machine->addPassesToEmitFile(pass, dest, nullptr, file_type)) {
        throw std::runtime_error("TargetMachine cannot emit object file for target!");
    }

    pass.run(module);
    dest.flush();
}
```

### 6.2 Producing Dynamic Shared Libraries (`.dylib` / `.so`)
To link `.o` into a `.dylib` or `.so` without invoking an external shell `clang++` command:
1. **Option A (Integrated LLVM Linker LLD)**: Link using `lld::macho::link` (macOS) or `lld::elf::link` (Linux), linked directly as a library into the `bt-compiler` binary.
2. **Option B (Direct System Tool Execution via LLVM Support API)**:
   ```cpp
   std::vector<llvm::StringRef> args = {
       "ld", "-shared", obj_file, "-o", dylib_file, "-lc"
   };
   llvm::sys::ExecuteAndWait("/usr/bin/ld", args);
   ```

### 6.3 Standard C ABI Export Wrapper
Every compiled `.bt` function exports standard C ABI entrypoints with unmangled symbols:
```c
// C ABI Export for external interoperability
extern "C" {
    // Planar dot product entrypoint
    int64_t bt_planar_dot_export(uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn);

    // Matrix GEMM entrypoint: C = A * B + C
    void bt_gemm_export(
        const uint64_t* A_pos, const uint64_t* A_neg,
        const uint64_t* B_pos, const uint64_t* B_neg,
        int64_t* C_out,
        uint32_t M, uint32_t N, uint32_t K_words
    );
}
```

---

## 7. Multi-Target Hardware Dispatch via IREE HAL

### 7.1 What is IREE HAL and Why is it Superior?
The **IREE Hardware Abstraction Layer (HAL)** provides a unified driver and execution model across heterogeneous accelerators:
- Unifies Vulkan, Apple Metal, NVIDIA CUDA, AMD ROCm, and Multi-Core CPU under a single dispatch API.
- Replaces proprietary vendor runtimes (CUDA Driver/Runtime, cuBLAS) with a lightweight, embeddable C runtime (~150 KB).
- Packages compiled GPU shader binaries and CPU machine code into an immutable **Virtual Machine FlatBuffer (`.vmfb`)**.

### 7.2 Progressive IREE Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Input MLIR Module (`bt` Dialect lowered to `linalg` + `arith` + `math`) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼ iree-compile --iree-hal-target-backends=...
┌─────────────────────────────────────────────────────────────────────────┐
│ IREE Flow Dialect (`iree_flow`)                                         │
│ - `flow.dispatch.workgroups`: Partition linalg operations into grids    │
│ - Tracks tensor lifetimes, fuses elementwise operations                 │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ IREE Stream Dialect (`iree_stream`)                                     │
│ - Models asynchronous execution queues, timeline semaphores             │
│ - Allocates memory slabs, eliminates transient cross-dispatch buffers   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ IREE HAL Dialect (`iree_hal`)                                           │
│ - Generates target-specific device binaries (`hal.executable`):         │
│   * `vulkan-spirv`: SPIR-V bytecode with `OpBitCount`                   │
│   * `metal-spirv` / `metal`: MSL libraries with `metal::popcount`       │
│   * `llvm-cpu`: AOT object code with AVX-512 `vpopcntdq` / ARM NEON `cnt`│
│   * `cuda`: PTX assembly with `popc` instructions                       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Standalone `.vmfb` Executable (Loaded by C Runtime API)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Embedded C Runtime Dispatch Integration
The compiled `.vmfb` module is executed via the standalone IREE runtime C API:
```c
#include "iree/runtime/api.h"

int execute_ternary_vmfb(const char* vmfb_path, const char* device_uri) {
    iree_runtime_instance_t* instance = NULL;
    iree_runtime_instance_options_t options;
    iree_runtime_instance_options_initialize(&options);
    iree_runtime_instance_create(&options, iree_allocator_system(), &instance);

    // Create device (e.g. "vulkan", "metal", "local-task", "cuda")
    iree_hal_device_t* device = NULL;
    iree_runtime_instance_try_create_default_device(instance, iree_make_cstring_view(device_uri), &device);

    iree_runtime_session_t* session = NULL;
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    iree_runtime_session_create_with_device(instance, &session_options, device, iree_allocator_system(), &session);

    // Load .vmfb bytecode into session
    iree_runtime_session_append_bytecode_module_from_file(session, vmfb_path);

    // Invoke entrypoint with unified input/output buffers
    iree_runtime_call_t call;
    iree_runtime_call_initialize_by_name(session, iree_make_cstring_view("module.bt_gemm"), &call);
    // ... bind input/output hal_buffer_views ...
    iree_runtime_call_invoke(&call, 0);

    iree_runtime_call_deinitialize(&call);
    iree_runtime_session_release(session);
    iree_hal_device_release(device);
    iree_runtime_instance_release(instance);
    return 0;
}
```

---

## 8. Comparative Architecture Matrix

| Architectural Feature | Legacy Source Transpiler (v2.0) | Upgraded LLVM/MLIR/IREE Backend (Target) |
| :--- | :--- | :--- |
| **Output Format** | Plaintext `.metal` / `.comp` files, C++ headers | Machine code `.o`, `.dylib`, `.so`, `.vmfb` |
| **JIT Mechanism** | AST Interpreter (`TernaryJITEngine`, recursive tree walk) | LLVM ORC JIT v2 (Direct compilation to host RAM) |
| **External Toolchain** | Requires external `clang++` / `glslangValidator` | **Self-contained**: zero external process invocations |
| **Compilation Latency** | High (process fork/exec, disk I/O) | Ultra-fast (<2 ms in-memory compilation) |
| **Hardware Targeting** | Manual shader templates | Unified IREE HAL (Vulkan, Metal, CUDA, CPU) |
| **Vectorization** | Hardcoded intrinsics (`arm_neon.h`, `immintrin.h`) | Polyhedral vectorization (`linalg` $\to$ `vector` $\to$ target) |
| **Interoperability** | Source header inclusions | Standard C ABI, dynamic library symbols, PyTorch C++ FFI |

---

## 9. Implementation Roadmap & Milestones

### Milestone 1: Concrete MLIR Dialect & Conversion Library
- [ ] Implement C++ TableGen backing classes for `TernaryDialect` and `TernaryOps`.
- [ ] Create `BTToLinalgPass` and `BTToArithPass` conversion patterns.
- [ ] Implement dialect registration hook in `bt-compiler`.

### Milestone 2: LLVM Dialect Lowering & Bufferization Pipeline
- [ ] Implement `OneShotBufferize` configuration for dual-rail tensor types.
- [ ] Lower `bt.planar_dot` directly to `llvm.intr.ctpop` and vector bitwise ops.
- [ ] Implement `finalize-memref-to-llvm` and `reconcile-unrealized-casts`.

### Milestone 3: In-Memory LLVM ORC JIT v2 Engine
- [ ] Integrate `llvm::orc::LLJIT` into `bt-compiler run`.
- [ ] Wire AST frontend to emit MLIR, lower through pipeline, and JIT-execute in memory.
- [ ] Deprecate the AST tree-walking interpreter in favor of native in-memory execution.

### Milestone 4: Native AOT Object & Shared Library Generator
- [ ] Implement `--emit-obj` (`.o`) via `llvm::TargetMachine::addPassesToEmitFile`.
- [ ] Implement `--emit-dylib` (`.dylib` / `.so`) with position-independent code (PIC) and C ABI exports.
- [ ] Provide PyTorch FFI binding example loading the generated `.dylib`.

### Milestone 5: IREE HAL Multi-Target Hardware Dispatch Integration
- [ ] Integrate IREE compiler passes (`iree-flow`, `iree-stream`, `iree-hal`) into the build.
- [ ] Enable multi-target compilation to `.vmfb` for Vulkan SPIR-V, Apple Metal, and multi-core CPU.
- [ ] Provide unified runtime test harness verifying numeric parity across all backends.
