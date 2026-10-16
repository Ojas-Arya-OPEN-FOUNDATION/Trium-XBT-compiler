# MLIR & LLVM Frameworks for Custom Dialects, Types, and Multi-Target Code Generation

**Aspect Deep-Dive**: Custom Non-Standard Types, TableGen Metaprogramming, Dialect Lowering Pipelines, Multi-Target Codegen (LLVM IR / SPIR-V / NVPTX / Metal), Industrial Case Studies, and Concrete Architectural Trade-Offs.  
**Author**: Compiler Research & MLIR Infrastructure Subagent  
**Date**: October 2026  
**Status**: Foundational Research Specification  

---

## 1. Executive Architectural Foundation

### 1.1 The Structural Limitations of Classical Flat Compilers
For decades, compiler design revolved around monolithic, flat intermediate representations—most notably **LLVM IR**. LLVM IR provides a brilliant, universal abstraction for scalar and basic vector operations targeting classical CPU instruction set architectures (x86_64, ARM, RISC-V). However, LLVM IR operates at a single, low level of abstraction (1-dimensional Static Single Assignment over scalar integers, floats, and memory pointers). 

When compiling domain-specific languages (DSLs), tensor computation graphs, high-level synthesis (HLS) hardware pipelines, or non-standard arithmetic representations (such as Balanced Ternary arithmetic, dual-rail bitplanes, or sub-byte quantized types), compiling directly to LLVM IR causes severe structural information loss:
1. **Destruction of Multidimensional Iteration Spaces**: High-level tensor operations and multidimensional loop nests are prematurely decomposed into 1D pointer arithmetic, base offsets, and flat branch instructions (`getelementptr`, conditional branches). Once flattened, reconstructing affine access patterns, polyhedral iteration domains, loop fusion opportunities, and cache tiling schedules is NP-hard or practically impossible.
2. **Premature Type Materialization**: Non-standard types—such as 2-bit trits, 4-bit integers, dual-rail positive/negative register planes, or redundant carry-save accumulator pairs—cannot be represented as first-class citizens in LLVM IR. Compilers targeting LLVM IR directly are forced to immediately lower these semantic concepts into primitive host integers (`i64`, `i128`) and manual bit-manipulation sequences (shifts, masks). This destroys domain-specific algebraic identities (e.g. Kleene logic absorption, ternary half-adder carry annihilation) before high-level optimization passes can exploit them.
3. **Fragmented Heterogeneous Code Generation**: LLVM IR is fundamentally tailored to unified Von Neumann machines with a continuous address space. Emitting cross-vendor GPU kernels (SPIR-V for Vulkan/OpenCL), NVIDIA PTX, Apple Metal Shading Language (MSL), or hardware RTL (SystemVerilog) requires separate, duplicated frontends and disjoint optimization toolchains.

### 1.2 The MLIR Paradigm: Multi-Level Intermediate Representation
Initiated by Chris Lattner and Google in 2019, **MLIR (Multi-Level Intermediate Representation)**, now part of the LLVM Project, resolves this fragmentation by functioning as an **extensible compiler construction kit**. MLIR does not enforce a single fixed opcode set or universal type system. Instead, it provides a meta-infrastructure for defining modular, composable **Dialects**.

A **Dialect** is a self-contained namespace of:
- **Custom Operations (`mlir::Operation`)**: Self-describing nodes in a directed graph representing computational steps, regions, or control flow.
- **Custom Types (`mlir::Type`)**: Domain-specific types with arbitrary verification logic and data layout rules.
- **Custom Attributes (`mlir::Attribute`)**: Compile-time constant metadata attached to operations or types.

The foundational methodology of MLIR is **Progressive Lowering**: high-level operations do not compile to machine code in a single leap. Instead, code traverses multiple dialect levels, incrementally shedding abstraction while applying domain-specific optimizations at each appropriate stratum:

$$	ext{Domain AST / Math DSL} \longrightarrow 	ext{Domain Dialect (`bt`)} \longrightarrow 	ext{Structured Loops (`linalg`, `tensor`)}$$
$$\downarrow$$
$$	ext{Control Flow \& Bit Manipulation (`scf`, `arith`, `vector`)} \longrightarrow 	ext{Target Dialect (`llvm`, `spirv`, `nvvm`)} \longrightarrow 	ext{Native Machine Binary}$$

MLIR has established itself as the de-facto industry standard for next-generation systems programming languages (Mojo/Modular), hardware synthesis (CIRCT), machine learning runtimes (IREE, Torch-MLIR, OpenXLA), and specialized GPU kernel engines (OpenAI Triton).

---

## 2. Defining Custom Non-Standard Types & Operations in MLIR

Modern domain-specific compilers frequently require types that diverge sharply from standard IEEE 754 floating-point or power-of-two two's complement integers. Examples include sub-byte quantization formats (`i1`, `i2`, `i4`), non-power-of-two vector bitplanes, arbitrary-radix integers, and redundant dual-rail balance encodings.

### 2.1 TableGen Metaprogramming: ODS and TypeDef
MLIR enforces declarative compiler engineering via **TableGen (`.td`)** files, compiled by the `mlir-tblgen` tool. This replaces thousands of lines of fragile C++ boilerplate with verifiable declarations.

Custom types inherit from the `TypeDef` class in `mlir/IR/AttrTypeBase.td`:
```tablegen
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

def BT_Dialect : Dialect {
  let name = "bt";
  let cppNamespace = "::mlir::bt";
  let summary = "Balanced Ternary Dialect";
  let useDefaultTypePrinterParser = 1;
}

class BT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// 1. Primitive Scalar Trit: !bt.trit
def BT_TritType : BT_Type<"Trit", "trit"> {
  let summary = "Balanced ternary scalar digit {-1, 0, +1}";
  let description = [{ Represents an individual trit with 3 valid states. }];
}

// 2. Parameterized Dual-Rail Bitplane Register: !bt.dualrail<width>
def BT_DualRailType : BT_Type<"DualRail", "dualrail"> {
  let summary = "Bit-sliced dual-rail register representing width trits";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "unsigned":$width), [{
      return $_get($_ctxt, width);
    }]>
  ];

  let genVerify = 1; // Generates verification hook: verify(width)
}
```

When processed with `mlir-tblgen -gen-typedef-decls` and `mlir-tblgen -gen-typedef-defs`, TableGen generates:
1. **Uniqued Storage Class (`DualRailTypeStorage`)**: Automatically handles interning and garbage collection inside the `mlir::MLIRContext`. Two instances of `!bt.dualrail<64>` share the exact same pointer address in memory.
2. **Type Accessors & Constructors**: Generates `DualRailType::get(MLIRContext *ctx, unsigned width)` and `unsigned getWidth() const`.
3. **Round-Trip Parser & Printer**: The declarative `assemblyFormat = "\`<` $width `>`"` auto-generates recursive-descent parsing and formatting logic, enabling seamless textual MLIR representation (`!bt.dualrail<64>`).
4. **C++ Verification Hooks**: TableGen emits declarations for `LogicalResult DualRailType::verify(...)`, ensuring width constraints (e.g. $width > 0$ and $width \le 512$) are strictly enforced at IR generation time.

### 2.2 Storage Layout and Verification of Custom Non-Standard Types
MLIR provides two primary paradigms for representing non-standard types:

#### Paradigm A: Using Builtin Arbitrary-Width Integers (`IntegerType`)
MLIR's builtin dialect natively supports arbitrary integer bitwidths:
- `IntegerType::get(context, 2)`: Declares a 2-bit scalar (`i2`).
- `IntegerType::get(context, 4)`: Declares a 4-bit scalar (`i4`).
- `IntegerType::get(context, 1)`: Declares a 1-bit boolean/flag (`i1`).

Sub-byte types are fully first-class within the type checker. However, physical hardware ALUs are byte- or word-addressable (8-bit, 32-bit, 64-bit). The compiler must model how these types are laid out in memory and vector registers.

#### Paradigm B: Custom Dialect Type with Bit-Sliced Dual-Rail Invariants
When modeling exotic hardware, such as balanced ternary computing or dual-rail circuits, standard binary scalar types fail to capture algebraic properties. In balanced ternary, a trit $t \in \{-1, 0, +1\}$ is represented as two binary bits:
$$p \in \{0, 1\}, \quad n \in \{0, 1\}, \quad 	ext{subject to the invariant: } p \land n = 0$$

Defining `!bt.dualrail<width>` as a custom dialect type preserves this invariant throughout all high-level compilation passes. The compiler guarantees that operations like `bt.negate` simply swap the positive ($P$) and negative ($N$) rails via zero-cost SSA variable renaming, rather than emitting arithmetic negation instructions.

### 2.3 Memory Layout, DLTI, and Sub-Byte Bufferization
When sub-byte types (`i2`, `i4`) or custom types (`!bt.dualrail<W>`) are grouped into contiguous memory buffers (`memref`), physical addressing constraints arise.
In MLIR, this is handled through the **DLTI (Data Layout and Target Information)** dialect and the `DataLayoutSpec` attribute (`#dlti.dl_spec`):

```mlir
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<i2, 2>,                 // 2-bit size in memory
    #dlti.dl_entry<!bt.trit, 2>,           // 2-bit size for custom trit
    #dlti.dl_entry<!bt.dualrail<64>, 128>  // 128-bit total size (2 x 64-bit bitplanes)
  >
} { ... }
```

#### Byte Offset Calculation Rules:
When calculating the physical byte address for element index $i$ in a sub-byte buffer of element bitwidth $w$:
$$	ext{Byte Offset} = i \cdot \left\lceil rac{w}{8} ightceil \quad 	ext{or for packed elements: } \quad 	ext{Byte Index} = \left\lfloor rac{i \cdot w}{8} ightfloor$$
$$	ext{Bit Shift} = (i \cdot w) \pmod 8$$

MLIR's `memref.load` and `memref.store` operations do not directly load 2-bit or 4-bit quantities from unaligned hardware buses. Instead, lower-level passes (such as `lower-affine`, `convert-memref-to-llvm`, and custom packing passes) synthesize standard word loads (`i32` or `i64`), followed by `arith.shrui` and `arith.andi` masking.

For dual-rail representations (`!bt.dualrail<W>`), the memory layout maps directly to **planar buffers**:
- A tensor `!bt.tensor<MxK, 64>` lowers to two parallel `memref<MxKxi64>` structures: Buffer $P$ (Positive bitplane) and Buffer $N$ (Negative bitplane).
- This structure-of-arrays (SoA) layout guarantees coalesced memory access on GPU SIMT architectures and enables single-instruction bitwise SIMD instructions across 64 trits simultaneously.

### 2.4 Operation Definition Specification (ODS) and Traits
Operations are declared in TableGen using the `Op` class:
```tablegen
def BT_BTHAOp : BT_Op<"btha", [Pure, Commutative]> {
  let summary = "Balanced Ternary Half-Adder";
  let description = [{ Computes A + B = 3*Carry + Sum without carry propagation. }];

  let arguments = (ins BT_DualRailType:$lhs, BT_DualRailType:$rhs);
  let results = (outs BT_DualRailType:$sum, BT_DualRailType:$carry);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}
```

Key Traits and Interfaces:
- `Pure`: Informs CSE (Common Subexpression Elimination) and DCE (Dead Code Elimination) that the operation has no memory side-effects.
- `Commutative`: Allows canonicalizers to normalize operand order ($lhs \le rhs$).
- `InferTypeOpInterface`: Automatically deduces the output type from operand attributes.
- `SameOperandsAndResultType`: Enforces type equality across inputs and outputs.

### 2.5 Dialect Conversion Infrastructure & TypeConverter
Lowering custom types and ops to standard dialects (`arith`, `linalg`, `llvm`) requires MLIR's **Dialect Conversion Framework**:
1. **`TypeConverter`**: Defines recursive rules for transforming custom types to legal types.
2. **`ConversionTarget`**: Declares which dialects and operations are legal, illegal, or dynamically legal.
3. **`RewritePatternSet`**: Contains `OpConversionPattern<OpT>` instances that match illegal high-level ops and emit equivalent sequences of legal lower-level ops.

```cpp
// Example C++ Lowering Pass Infrastructure
class TernaryTypeConverter : public mlir::TypeConverter {
public:
  TernaryTypeConverter() {
    // 1. Pass-through for standard types
    addConversion([](mlir::Type t) { return t; });

    // 2. Lower !bt.trit to i2
    addConversion([](mlir::bt::TritType type) -> mlir::Type {
      return mlir::IntegerType::get(type.getContext(), 2);
    });

    // 3. Lower !bt.dualrail<64> to a struct of two 64-bit integers: !llvm.struct<(i64, i64)>
    addConversion([](mlir::bt::DualRailType type) -> mlir::Type {
      auto ctx = type.getContext();
      auto i64Type = mlir::IntegerType::get(ctx, 64);
      return mlir::LLVM::LLVMStructType::getLiteral(ctx, {i64Type, i64Type});
    });
  }
};

struct BTHALoweringPattern : public mlir::OpConversionPattern<mlir::bt::BTHAOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::bt::BTHAOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Unpack dual-rail struct operands into (P_a, N_a) and (P_b, N_b)
    Location loc = op.getLoc();
    // Emit arith.andi, arith.ori, arith.xori to implement Z3-verified half-adder logic:
    // P_c = P_a & P_b; N_c = N_a & N_b;
    // P_s = ((P_a ^ P_b) & ~(N_a | N_b)) | N_c;
    // N_s = ((N_a ^ N_b) & ~(P_a | P_b)) | P_c;
    // Pack into result struct and replace op...
    return success();
  }
};
```

---

## 3. Multi-Target Code Generation Pipelines (Industry Standard Analysis)

MLIR is the recognized industry standard for generating high-performance native machine code across heterogeneous targets (CPUs, GPUs, and custom accelerators). The foundational architectural pattern is **Progressive Dialect Lowering**.

```
                        ┌──────────────────────────────────────┐
                        │      High-Level Custom Dialect       │
                        │    (!bt.tensor, !bt.dualrail, ops)   │
                        └──────────────────┬───────────────────┘
                                           │
                                           ▼
             ┌────────────────────────────────────────────────────────────┐
             │       Target-Independent Structured Dialects (`linalg`)    │
             │   Tiling, Bufferization, Vectorization, Fusion             │
             └──────┬──────────────────────┬──────────────────────┬───────┘
                    │                      │                      │
        ┌───────────┴──────────┐   ┌───────┴──────────────┐   ┌───┴────────────────┐
        │ CPU CodeGen Path     │   │ GPU CodeGen (SPIR-V) │   │ GPU CodeGen (NVVM) │
        ▼                      ▼   ▼                      ▼   ▼                    ▼
┌──────────────────────────────┐ ┌──────────────────────────┐ ┌────────────────────┐
│ `vector`, `scf`, `arith`     │ │ `gpu`, `spirv` Dialects  │ │ `nvvm` Dialect     │
│ `convert-vector-to-llvm`     │ │ `convert-gpu-to-spirv`   │ │ `convert-gpu-to-   │
│ `convert-func-to-llvm`       │ │ `mlir-to-spirv`          │ │  nvvm`             │
└──────────────┬───────────────┘ └─────────┬────────────────┘ └────────┬───────────┘
               │                           │                           │
               ▼                           ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────┐ ┌────────────────────┐
│      LLVM IR Module          │ │     SPIR-V Binary        │ │  LLVM IR (NVPTX)   │
│  TargetMachine (Opt O3)      │ │  (.spv) Shaders          │ │  PTX CodeGen       │
└──────┬───────────────┬───────┘ └─────────┬────────────────┘ └────────┬───────────┘
       │               │                   │                           │
       ▼               ▼                   ▼                           ▼
┌──────────────┐┌──────────────┐ ┌──────────────────────────┐ ┌────────────────────┐
│ AOT Machine  ││ ORC JIT v2   │ │ Vulkan / Metal Runtime   │ │ CUDA Driver API    │
│ (.o, .dylib) ││ (LLJIT in-mem│ │ (vkCreateShaderModule /  │ │ (cuModuleLoadData /│
│ x86/ARM/RISCV││  Microsec)   │ │  SPIRV-Cross to MSL)     │ │  CUBIN execution)  │
└──────────────┘└──────────────┘ └──────────────────────────┘ └────────────────────┘
```

### 3.1 Target 1: CPU CodeGen via LLVM IR (ARM64, x86_64, RISC-V)
For CPUs, MLIR bypasses manual assembly generation and delegates to LLVM's mature instruction selection, scheduling, and register allocation engines.
1. **Linalg Tiling & Fusion**: High-level tensor contractions (`bt.gemm`) are transformed into tiled `linalg.generic` loops matching CPU cache line boundaries (L1/L2/L3).
2. **One-Shot Bufferization**: Converts value-based `tensor` operations into memory-based `memref` operations with in-place buffer reuse, eliminating heap allocations.
3. **Vectorization**: Transforms scalar loops into `vector` dialect operations (`vector.contract`, `vector.fma`, `vector<4xi64>`).
4. **LLVM Dialect Conversion**:
   - `convert-vector-to-llvm`: Translates MLIR vectors directly to target SIMD intrinsics (AVX-512, AVX2, ARM Neon `llvm.aarch64.neon.*`).
   - `convert-func-to-llvm` & `convert-arith-to-llvm`: Lowers remaining control flow and integers.
   - `reconcile-unrealized-casts`: Eliminates intermediate `unrealized_conversion_cast` operations.
5. **LLVM IR Translation**: Invokes `mlir::translateModuleToLLVMIR()` to synthesize an `llvm::Module`.
6. **Execution Paths**:
   - **AOT (Ahead-of-Time)**: Passed to `llvm::TargetMachine::emitFile()` to produce ELF, Mach-O, or COFF object files (`.o`), linked into shared libraries (`.so`, `.dylib`).
   - **JIT (Just-in-Time)**: Executed dynamically in memory via **LLVM ORC JIT v2 (`llvm::orc::LLJIT`)** with microsecond compilation latency, binding directly to native C ABI function pointers.

### 3.2 Target 2: Cross-Vendor GPUs via SPIR-V (Vulkan, OpenCL, Intel Xe)
1. High-level loops are transformed into `gpu.launch` blocks with grid and block thread dimensions.
2. Dialect conversion translates `gpu` and `arith` operations into the **`spirv` dialect** (`spirv.IAdd`, `spirv.BitwiseAnd`, `spirv.ControlBarrier`).
3. Serialization via `mlir::spirv::serialize()` emits valid binary SPIR-V bytecode (`.spv`).
4. Dispatched cross-platform using standard Vulkan compute pipelines (`vkCreateComputePipelines`) or OpenCL runtimes.

### 3.3 Target 3: NVIDIA GPUs via NVPTX (CUDA)
1. `gpu.launch` kernels are extracted into nested `gpu.module` blocks.
2. Operations lower through the **`nvvm` dialect**, mapping directly to CUDA warp-shuffle primitives, shared memory allocations, and Tensor Core intrinsics (WMMA/WGMMA).
3. The module translates to LLVM IR configured with the `nvptx64-nvidia-cuda` target triple.
4. LLVM’s NVPTX backend compiles LLVM IR into PTX (Parallel Thread Execution) assembly text.
5. Ingested at runtime via the CUDA Driver API (`cuModuleLoadDataEx`) or compiled AOT via `ptxas` into binary `cubin` containers.

### 3.4 Target 4: Apple Silicon GPUs via Metal (MSL)
Apple GPUs (M1/M2/M3/M4) do not expose a native open-source LLVM backend or direct SPIR-V runtime driver. MLIR compilers target Metal through two proven architectural routes:
1. **SPIR-V to MSL Transpilation**:
   - Compile MLIR to standard SPIR-V via the `spirv` dialect.
   - Pass the SPIR-V binary through Khronos **SPIRV-Cross** to generate Apple Metal Shading Language (`.metal`) source code.
   - Compile to binary bytecodes using Apple's command-line tool `metal` / `metallib` or directly at runtime via `MTLDevice newComputePipelineStateWithSource`.
2. **Direct MSL Generation / IREE HAL Metal Driver**:
   - Modern compilers like IREE implement an MLIR-to-MSL serialization pass or leverage the IREE HAL Metal driver to dispatch compute pipelines directly via Metal command buffers.

### 3.5 Unified Multi-Target Dispatch: The IREE Paradigm
**IREE (Intermediate Representation Execution Environment)** demonstrates the most complete multi-target architecture built on MLIR:
- **`flow` Dialect**: Analyzes dataflow graphs and partitions global tensor operations into discrete **dispatches** (executable kernels) with deterministic scheduling.
- **`stream` Dialect**: Schedules asynchronous execution, resource lifecycles, and synchronization fences/semaphores across concurrent queues.
- **`hal` (Hardware Abstraction Layer) Dialect**: Eliminates vendor API fragmentation by providing a unified driver interface modeled after modern low-level graphics architectures.
- **Virtual Machine FlatBuffer (`.vmfb`)**: The compiled artifact packages host VM bytecode and target-specific device binaries (embedded ELF for CPU, SPIR-V for Vulkan, PTX for CUDA, MSL for Metal). The runtime is written in minimal ANSI C99 with zero heavy dependencies.

---

## 4. Deep-Dive Case Studies of Open-Source MLIR Architectures

| Project | Primary Domain | Core Dialects | Target Execution Backends | Architecture Repository Link |
| :--- | :--- | :--- | :--- | :--- |
| **CIRCT** | EDA & Silicon Hardware Synthesis | `firrtl`, `hw`, `comb`, `seq`, `sv`, `llhd` | SystemVerilog (`.sv`), VHDL, Verilator | [github.com/llvm/circt](https://github.com/llvm/circt) |
| **IREE** | Heterogeneous ML & Tensor Codegen | `flow`, `stream`, `hal`, `vm`, `vmvx` | Vulkan/SPIR-V, CUDA/PTX, ROCm, Metal, CPU | [github.com/iree-org/iree](https://github.com/iree-org/iree) |
| **Torch-MLIR** | PyTorch Deep Learning Ingestion | `torch`, `torchc` | Linalg, TOSA, StableHLO, IREE | [github.com/llvm/torch-mlir](https://github.com/llvm/torch-mlir) |
| **OpenAI Triton**| High-Performance GPU Kernel Authoring | `tt` (TTIR), `ttg` (TTGIR) | NVIDIA PTX/CUBIN, AMD HSACO | [github.com/triton-lang/triton](https://github.com/triton-lang/triton) |
| **Modular / Mojo**| Systems Programming & AI Infrastructure | `hlcf`, `kgen`, custom vector dialects | Native CPU (LLVM), NVIDIA GPU (PTX), MAX Engine | [modular.com/mojo](https://www.modular.com/mojo) |

### 4.1 Case Study 1: CIRCT (Circuit IR Compilers and Tools)
CIRCT is an official LLVM Incubator project designed to replace aging, monolithic Electronic Design Automation (EDA) frameworks.
- **The Problem**: Chisel and Verilog compilers relied on the Scala-based SFC (Scala FIRRTL Compiler), which suffered from high memory consumption and slow multi-hour compilation runs on large ASIC netlists.
- **The MLIR Solution**: CIRCT decomposes hardware compilation into progressive dialects:
  $$	ext{Chisel / `.fir`} \longrightarrow 	ext{`firrtl` Dialect} \overset{-lower-to-hw}{\longrightarrow} 	ext{`hw` + `comb` + `seq` Dialects} \overset{-lower-to-sv}{\longrightarrow} 	ext{`sv` Dialect} \longrightarrow 	ext{SystemVerilog}$$
- **Architectural Breakthrough**: Combinational optimizations (`comb.add`, `comb.mux`, `comb.and`) use standard MLIR canonicalization passes (CSE, DCE, constant folding). Abstract sequential registers (`seq.compreg`) are lowered into explicit clock-edge procedural blocks (`sv.always_ff`), achieving over $10	imes$ compilation speedups over legacy Scala toolchains.

### 4.2 Case Study 2: IREE (Intermediate Representation Execution Environment)
Originally developed at Google and now an independent open-source foundation project, IREE is designed for ubiquitous deployment from microcontrollers to server clusters.
- **Core Abstraction**: Unlike PyTorch or TensorFlow, which bundle multi-gigabyte Python runtimes and heavyweight C++ runtime libraries, IREE produces a single, compact **`.vmfb` (Virtual Machine FlatBuffer)**.
- **Separation of Scheduling and Codegen**: High-level execution flow is optimized in the device-agnostic `flow` and `stream` dialects. When lowering to concrete hardware, the IREE compiler invokes target-specific MLIR codegen pipelines (emitting embedded ELF for CPU or SPIR-V for Vulkan).
- **The C99 Runtime**: The IREE runtime loader is pure C99, requiring only a few kilobytes of RAM. It interprets host bytecode and dispatches kernels asynchronously via operating system primitives or GPU command streams.

### 4.3 Case Study 3: Torch-MLIR
Torch-MLIR serves as the foundational compiler bridge for the PyTorch ecosystem.
- **The Ingestion Pipeline**: Extracts computational graphs from PyTorch via TorchScript or PyTorch 2.0 `torch.export` / FX graphs, translating them into the **`torch` dialect**.
- **Backend Contract**: Normalizes high-level PyTorch operator variants (in-place mutations, optional dtypes, dynamic rank tensors) into a strict, verifiable subset.
- **Lowering Fans**: From the `torch` backend contract, the compiler can branch to:
  1. `linalg-on-tensors` (for custom loop tiling, vectorization, and IREE execution).
  2. `tosa` (Tensor Operator Set Architecture, for lightweight edge NPUs and embedded microcontrollers).
  3. `stablehlo` (for distributed training and OpenXLA accelerators).

### 4.4 Case Study 4: OpenAI Triton
Triton revolutionized deep learning kernel development by enabling engineers to write highly competitive GPU kernels in Python rather than CUDA/C++.
- **Dialect Pipeline**:
  $$	ext{Python AST} \longrightarrow 	ext{`tt` (TTIR)} \longrightarrow 	ext{`ttg` (TTGIR)} \longrightarrow 	ext{`nvvm` / `rocdl`} \longrightarrow 	ext{LLVM IR} \longrightarrow 	ext{PTX / HSACO}$$
- **Key Architectural Pattern**: The transition from `tt` to `ttg` (Triton GPU IR) is where hardware-specific layout distribution occurs. Abstract multi-dimensional block operations are assigned physical thread layouts, shared memory (SRAM) allocations, asynchronous copy pipelining (`cp.async`), and Tensor Core dot layouts. This decouples algorithm design from hardware microarchitecture.

### 4.5 Case Study 5: Modular / Mojo (MAX Engine)
Created by Chris Lattner and Modular, Mojo is designed as a Python-superset systems programming language built natively on MLIR.
- **KGEN Subsystem**: The Kernel Generator (KGEN) evaluates compile-time metaprogramming constructs (`@parameter`), specializing types and unrolling loops before lowering to MLIR.
- **Custom Dialects**: Mojo utilizes custom dialects for high-level control flow (`hlcf`) and explicit memory borrow-checking/affine types.
- **MAX Engine**: The graph runtime executes automatic kernel fusion and schedules tasks across CPUs and GPUs using MLIR-generated kernels without intermediate C++ translation.

---

## 5. Concrete Architectural Trade-Off Analysis

When building a domain-specific compiler (such as for Balanced Ternary computation), architects face four distinct development strategies:
1. **Option A: Hard-Forking the LLVM/MLIR Monorepo**
2. **Option B: In-Tree MLIR Dialect (Upstream Contribution to LLVM)**
3. **Option C: Out-of-Tree Standalone MLIR Project (The Industry Standard: CIRCT, IREE, Triton pattern)**
4. **Option D: Standalone Compiler Emitting LLVM IR Directly (Pre-MLIR Classical Pattern: Clang, Rustc, Zig)**

### 5.1 Comprehensive Trade-Off Comparison Matrix

| Evaluation Dimension | Option A: Hard-Fork LLVM/MLIR | Option B: In-Tree MLIR Dialect | Option C: Out-of-Tree Standalone Dialect | Option D: Standalone Emitting LLVM IR |
| :--- | :--- | :--- | :--- | :--- |
| **Build & CI Iteration Time** | **Extremely Slow**: Must compile multi-GB LLVM monorepo from source on every clean build. | **Extremely Slow**: Bound to LLVM monorepo build times and test suites. | **Very Fast**: Consumes pre-built LLVM/MLIR libraries via CMake `find_package(MLIR)`. | **Fast**: Links only against `llvm-core` / `LLVMSupport` libraries. |
| **Upstream Maintenance Burden** | **Catastrophic ("Rebase Hell")**: Diverges rapidly; merging upstream updates requires massive manual conflict resolution. | **Zero Maintenance Tax**: Upstream community refactorings automatically update in-tree callers. | **Moderate**: Must track pinned LLVM commits; periodic API updates needed, but isolated to public interfaces. | **Low**: LLVM C/C++ IRBuilder API is relatively stable compared to MLIR C++ internals. |
| **Multi-Dimensional Semantic Retention** | **Full**: Full access to all MLIR dialects and internal passes. | **Full**: Seamless integration with standard MLIR dialect hierarchy. | **Full**: Retains high-level tensor, affine, and custom dialect semantics until lowering. | **None (Severe Loss)**: Must flatten loops, tensors, and multidimensional arrays to 1D pointers immediately. |
| **Custom Non-Standard Type Modeling** | **High**: Can modify core LLVM/MLIR types directly. | **High**: Can add custom types via TableGen, subject to upstream review. | **High**: Declarative `TypeDef`, custom verifiers, and storage classes fully supported. | **Poor**: Must map all domain types prematurely to standard host scalars (`i64`, `i128`). |
| **Multi-Target Portability (GPU, CPU, RTL)**| **Complete**: Direct access to LLVM, SPIR-V, NVVM, ROCDL backends. | **Complete**: Native access to all upstream target pipelines. | **Complete**: Lowers cleanly to standard `llvm`, `spirv`, and `gpu` dialects for universal dispatch. | **Fragmented**: Strictly targets LLVM backends; cannot easily emit SPIR-V, MSL, or RTL. |
| **Contribution & Governance Overhead** | **Zero Upstream Governance**: Total autonomous control over source code. | **Extremely High**: Must satisfy LLVM Foundation RFC process, coding standards, and community reviews. | **Zero Governance Friction**: Complete autonomy over releases, features, CI/CD, and language design. | **Zero Governance Friction**: Completely autonomous standalone executable. |
| **IP Protection & Commercial Isolation** | **Moderate**: Fork can be private, but merging public upstream requires vigilance. | **Zero**: Code must be public under Apache-2.0 with LLVM Exceptions. | **Complete**: Proprietary hardware algorithms and dialect passes remain 100% private. | **Complete**: Entire codebase is private. |
| **Tooling & Infrastructure Reuse** | **Full**: Reuses `FileCheck`, `mlir-opt`, TableGen ODS/DRR. | **Full**: Direct upstream integration. | **Full**: Generates standalone `*-opt`, `*-translate`, and custom test runners using LLVM tooling. | **Minimal**: Must build custom optimizer, custom AST printer, and manual verification passes. |

### 5.2 Concrete Architectural Analysis & Recommendation

#### Why Option A (Hard Fork) is an Anti-Pattern:
Hard-forking LLVM should be avoided at all costs unless modifying foundational C++ ABI rules in `clang` or rewriting core LLVM code generation algorithms. Upstream LLVM sees hundreds of commits daily. Within 3–6 months, a hard fork becomes an unmaintainable software island, permanently locking the compiler out of upstream performance improvements, new target architectures (e.g. newer Apple Silicon cores, RISC-V extensions), and upstream bug fixes.

#### Why Option D (Direct LLVM IR Emission) is Obsolete for Custom Domains:
While direct LLVM IR emission remains viable for standard procedural languages like Zig or Rust, it is completely inadequate for non-standard algebras (e.g. Balanced Ternary), custom hardware architectures, and AI tensors. In Balanced Ternary, an operation like `bt.planar_dot` evaluates an inner product of 64-trit dual-rail vectors using 4 bitwise ops and two hardware popcounts:
$$	ext{Dot}(u, v) = 	ext{popcount}((P_u \land P_v) \lor (N_u \land N_v)) - 	ext{popcount}((P_u \land N_v) \lor (N_u \land P_v))$$
In MLIR, this can be retained as a high-level primitive operation, optimized via term rewriting (e.g. canceling symmetric terms in dual-rail subtraction), and vectorized across SIMD lanes before emitting LLVM IR. Compiling directly to LLVM IR destroys this mathematical identity at the frontend, preventing vectorization passes from understanding the high-level intent.

#### The Winning Industry Standard: Option C (Out-of-Tree Standalone MLIR Project)
Every major successful MLIR-based compiler in production—**CIRCT, IREE, Torch-MLIR, and Triton**—operates as an **Out-of-Tree Standalone MLIR Project**. 
- It uses CMake's `find_package(MLIR REQUIRED CONFIG)` to link against prebuilt, release-mode LLVM/MLIR toolchains, keeping build times under 2 minutes rather than 90 minutes.
- It pins a specific upstream LLVM git commit hash via a `CMakeLists.txt` or git submodule, updating periodically in disciplined synchronization cycles.
- It provides custom standalone binaries (`bt-opt`, `bt-translate`) alongside C++ and Python API bindings.

---

## 6. Concrete TableGen & Pass Architecture for Balanced Ternary (`bt`)

Applying these principles to the Balanced Ternary Compiler yields a production-grade MLIR architecture.

### 6.1 Complete TableGen Dialect Definition (`TernaryDialect.td`)
```tablegen
#ifndef BT_DIALECT_TD
#define BT_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def BT_Dialect : Dialect {
  let name = "bt";
  let summary = "Balanced Ternary Dialect for Native Machine Code and Accelerators";
  let description = [{
    First-class representation of balanced ternary computing over digit set {-1, 0, +1}.
    Preserves dual-rail bitplane invariants, carry-save representations, and Base-243 encodings.
  }];
  let cppNamespace = "::mlir::bt";
  let useDefaultTypePrinterParser = 1;
}

class BT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def BT_DualRailType : BT_Type<"DualRail", "dualrail"> {
  let summary = "Bit-sliced dual-rail register representing width trits";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "unsigned":$width), [{
      return $_get($_ctxt, width);
    }]>
  ];
}

def BT_CSAPairType : BT_Type<"CSAPair", "csa_pair"> {
  let summary = "Redundant Carry-Save Accumulator (Sum, Carry) pair";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "unsigned":$width), [{
      return $_get($_ctxt, width);
    }]>
  ];
}
#endif // BT_DIALECT_TD
```

### 6.2 TableGen Operations Definition (`TernaryOps.td`)
```tablegen
#ifndef BT_OPS_TD
#define BT_OPS_TD

include "TernaryDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class BT_Op<string mnemonic, list<Trait> traits = []> :
    Op<BT_Dialect, mnemonic, traits>;

def BT_PlanarDotOp : BT_Op<"planar_dot", [Pure]> {
  let summary = "Bit-sliced planar dual-rail dot product via population count";
  let arguments = (ins BT_DualRailType:$a, BT_DualRailType:$b);
  let results = (outs AnyInteger:$res);
  let assemblyFormat = "$a `,` $b attr-dict `:` functional-type(operands, results)";
}

def BT_BTHAOp : BT_Op<"btha", [Pure, Commutative]> {
  let summary = "Balanced Ternary Half-Adder";
  let arguments = (ins BT_DualRailType:$lhs, BT_DualRailType:$rhs);
  let results = (outs BT_DualRailType:$sum, BT_DualRailType:$carry);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}
#endif // BT_OPS_TD
```

### 6.3 Lowering Pipeline to Standard MLIR & LLVM
The complete pass pipeline lowers `bt` operations down to the `llvm` dialect:
```
mlir-opt   --convert-bt-to-arith   --convert-bt-to-llvm   --convert-vector-to-llvm   --convert-func-to-llvm   --reconcile-unrealized-casts
```

### 6.4 In-Memory Zero-Latency Execution via LLVM ORC JIT v2
```cpp
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

class NativeTernaryJIT {
  std::unique_ptr<llvm::orc::LLJIT> jit;

public:
  NativeTernaryJIT() {
    auto jitOrErr = llvm::orc::LLJITBuilder().create();
    if (!jitOrErr) llvm::report_fatal_error("Failed to initialize LLJIT");
    jit = std::move(*jitOrErr);
  }

  void compileAndExecute(mlir::ModuleOp mlirModule) {
    llvm::LLVMContext llvmCtx;
    auto llvmModule = mlir::translateModuleToLLVMIR(mlirModule, llvmCtx);
    
    llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), 
                                    std::make_unique<llvm::LLVMContext>());
    cantFail(jit->addIRModule(std::move(tsm)));

    auto sym = cantFail(jit->lookup("ternary_main"));
    auto *funcPtr = sym.toPtr<int64_t(*)()>();
    int64_t result = funcPtr();
    // Direct zero-overhead execution complete!
  }
};
```

---

## 7. Authoritative References & Source URLs

1. **MLIR Official Architectural Documentation**:
   - Defining Dialects: <https://mlir.llvm.org/docs/DefiningDialects/>
   - TableGen ODS Language: <https://mlir.llvm.org/docs/OpDefinitions/>
   - Defining Dialect Attributes and Types: <https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/>
   - Dialect Conversion Infrastructure: <https://mlir.llvm.org/docs/DialectConversion/>
   - Declarative Rewrite Rules (DRR): <https://mlir.llvm.org/docs/DeclarativeRewriteFormulas/>
   - Target and Data Layout (DLTI): <https://mlir.llvm.org/docs/DataLayout/>
   - Out-of-Tree Standalone Dialect Example: <https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone>

2. **CIRCT (Circuit IR Compilers and Tools)**:
   - CIRCT Architecture Overview: <https://circt.llvm.org/>
   - Core CIRCT Dialects (`hw`, `comb`, `seq`, `sv`): <https://circt.llvm.org/docs/Dialects/>
   - GitHub Repository: <https://github.com/llvm/circt>

3. **IREE (Intermediate Representation Execution Environment)**:
   - IREE Architecture Specification: <https://iree.dev/developers/design-roadmap/>
   - IREE Hardware Abstraction Layer (HAL): <https://iree.dev/reference/hal/>
   - GitHub Repository: <https://github.com/iree-org/iree>

4. **Torch-MLIR**:
   - Torch-MLIR Architecture and Backend Contract: <https://github.com/llvm/torch-mlir>
   - Lowering Paths (Linalg, TOSA, StableHLO): <https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md>

5. **OpenAI Triton**:
   - Triton Compiler Architecture: <https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html>
   - TTIR & TTGIR Dialects: <https://github.com/triton-lang/triton>

6. **Modular / Mojo**:
   - Mojo Architecture & MLIR Integration: <https://docs.modular.com/mojo/notebooks/MojoArchitecture>
   - Modular MAX Engine: <https://www.modular.com/max>

7. **LLVM ORC JIT v2**:
   - LLJIT and Execution Engine APIs: <https://llvm.org/docs/ORCv2.html>
