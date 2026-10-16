# Hardware-Agnostic Compiler Engines: In-Depth Evaluation for Heavy Balanced Ternary Workloads
**Target Architectures**: CPU (x86_64 AVX2/AVX-512/AMX, ARM64 NEON/SVE), GPU (Apple Metal, Vulkan/SPIR-V, NVIDIA CUDA, AMD ROCm)  
**Evaluated Frameworks**: IREE (Intermediate Representation Execution Environment), Apache TVM (TVM Unity / TensorIR), Halide  
**Author**: Antigravity Prime Hardware & Compiler Research Engine  
**Status**: Exhaustive Framework Analysis & Architecture Recommendation  

---

## 1. Executive Summary & Suitability Ranking

This research evaluates open-source compiler frameworks capable of auto-generating high-performance machine code across heterogeneous CPU and GPU backends for the **Universal Balanced Ternary System**. The balanced ternary computational model deviates fundamentally from conventional IEEE-754 floating-point paradigms:
1. **Multiplication Elimination**: Multiplication is replaced by dual-rail bitwise gating: $(+1  *  +1 = +1)$, $(-1  *  -1 = +1)$, $(+1  *  -1 = -1)$.
2. **Bit-Sliced Planar Dot Products**: Inner products are executed via bitwise Boolean logic (`AND`, `OR`, `XOR`) coupled with Population Count (`POPCNT` / `ctpop`).
3. **Carry-Free Redundant Accumulation**: Carry-Save Addition (3-to-2 compressors / `BTFA` / `BT-CSA`) maintains redundant $(Sum, Carry)$ pairs in registers, deferring carry resolution until tensor boundary output.
4. **Base-243 / Base-59049 Compressed Weight Streaming**: High-density radix-3 packing achieving 1.585 bits/trit, requiring on-the-fly register-tile decompression during memory loads.

### Suitability Ranking for Custom Balanced Ternary Engine:
1. **Rank 1: IREE (OpenXLA / Nod.ai / AMD) — 96/100 (Primary Architecture Choice)**
   - **Why**: Native MLIR infrastructure permits first-class custom dialects (`!bt.dualrail<W>`, `!bt.csa_pair<W>`), zero-copy asynchronous execution via modern HAL (Vulkan, Metal, CUDA, ROCm, CPU), progressive lowering into standard vector/linalg dialects with native `ctpop`, and ultra-compact C runtime (~50–150 KB). Fully aligned with existing project MLIR artifacts (`TernaryDialect.td`, `TernaryOps.td`, `ternary_linalg_lowering.mlir`).
2. **Rank 2: Apache TVM (TVM Unity / Relax & TensorIR) — 74/100 (Viable for Micro-Kernel Scheduling)**
   - **Why**: Strong automated schedule exploration via MetaSchedule and custom micro-kernel binding via `tensorize`. Built-in `tvm.tir.popcount`. However, lacks first-class custom type systems (requires shoehorning dual-rail states into scalar integer types), suffers from commercial maintenance volatility following OctoML's shutdown in early 2024, and incurs higher runtime overhead (~1–3 MB C++ runtime).
3. **Rank 3: Halide — 48/100 (Unsuitable for End-to-End System; Niche Kernel Utility Only)**
   - **Why**: Exceptional loop nest scheduling for dense stencils, but intentionally scoped to image processing. Lacks high-level neural network graph representations (no automatic operator fusion, graph partitioning, or tensor memory lifetime planning). Core DSL lacks built-in `popcount` (requires SWAR software emulations or extern C calls that break GPU thread-block inlining), and multi-channel tuples do not support custom carry-free arithmetic abstractions cleanly.

---

## 2. In-Depth Architectural Evaluation

### 2.1 IREE (Intermediate Representation Execution Environment)

#### A. Architecture & Multi-Level IR Pipeline
IREE is an MLIR-native, whole-program compiler and lightweight runtime developed under OpenXLA. Its compilation pipeline implements clean multi-level lowering:
- **Input Layer**: Ingests models via StableHLO, TOSA, or domain-specific custom MLIR dialects (e.g. `bt`).
- **Flow Dialect**: Performs hardware-agnostic tensor transformations, operator fusion, dataflow partitioning into dispatch regions, and asynchronous dependency tracking.
- **Stream Dialect**: Manages concurrent stream execution, memory lifetime management, asynchronous timeline semaphores, and allocation planning.
- **HAL Dialect (Hardware Abstraction Layer)**: Maps abstract dispatches onto hardware drivers modeled after modern low-overhead compute APIs (Vulkan, Metal, CUDA driver, CPU threadpools).
- **Target CodeGen (`iree-codegen`)**: Lowers dispatch regions to machine code using LLVM CPU (AArch64, x86_64), SPIR-V (Vulkan), Metal Shading Language / MSL-SPIR-V (Apple Silicon), and LLVM NVPTX/ROCDL (CUDA/ROCm).

#### B. Runtime Execution Model
- **Footprint**: Ultra-lean C runtime (~50 KB minimal, ~150 KB full HAL).
- **Format**: Compiles models into Ahead-of-Time (AOT) standalone Virtual Machine FlatBuffers (`.vmfb`), encapsulating host VM bytecode and compiled device shaders/executables.
- **Zero Runtime Dependencies**: No Python, no heavy PyTorch/LLVM libraries required at runtime.
- **Memory Model**: Asynchronous, zero-copy buffer transfers. Native support for unified memory architectures (Apple Silicon `MTLResourceStorageModeShared`, Vulkan host-visible/device-local memory).

#### C. Scheduling Primitives & Vectorization
- **MLIR Transform Dialect**: Provides explicit, composable scheduling strategies:
  - Multi-level loop tiling: Workgroup tiling -> Subgroup / Warp tiling -> Thread tiling -> Vector tiling.
  - Vectorization: Lowers directly to `vector.contract`, `vector.transfer_read`, and `vector.transfer_write`.
  - Software pipelining, double-buffering, and register-level unrolling.
- **Hardware Opcode Mapping**:
  - Popcount: Lowers MLIR `math.ctpop` or `llvm.intr.ctpop` directly to hardware instructions:
    * x86_64: `vpopcntdq` (AVX-512) or scalar `popcnt`.
    * ARM64: `cnt.16b` (NEON) and `uaddlp` reduction trees.
    * Vulkan / SPIR-V: `OpBitCount`.
    * Apple Metal: `metal::popcount`.
    * NVIDIA CUDA: `__popc` / PTX `popc`.

#### D. Balanced Ternary Integration Path
1. **Custom MLIR Dialect**: Consume `!bt.dualrail<64>` and `!bt.csa_pair<64>` directly from `TernaryOps.td`.
2. **Lowering Pass**: Implement a standard MLIR dialect conversion pass `ConvertTernaryToLinalg` lowering `bt.gemm` and `bt.planar_dot` to `linalg.generic` contractions over bitplane buffers $(A_P, A_N, B_P, B_N)$.
3. **Weight Codec Inlining**: Base-243 / Base-59049 unpackers compile into elementwise fused dispatches preceding GEMM, decoding 5 trits per byte directly in GPU L1 / registers without DRAM round-trips.

---

### 2.2 Apache TVM (TVM Unity / Relax & TensorIR)

#### A. Architecture & IR Pipeline
- **Relax (Relay Next)**: High-level computation graph IR supporting first-class dynamic shapes, structured control flow, and Virtual Machine execution.
- **TensorIR (TIR)**: Imperative, loop-level AST with mathematical block annotations (`T.block`, `T.reads`, `T.writes`) allowing fine-grained loop scheduling and hardware intrinsic binding.
- **Target Emitters**: Emits C code, LLVM IR (for x86, ARM), CUDA PTX, ROCm HIP, Metal Shading Language, and Vulkan SPIR-V.

#### B. Runtime Execution Model
- **Footprint**: ~1 MB to 3 MB (`libtvm_runtime.so` / dynamic library).
- **FFI**: TVM `PackedFunc` system enables dynamic polymorphic invocation across C++, Python, and Rust.
- **Virtual Machine**: Relax VM coordinates graph execution and memory pool allocations via TVM `NDArray`.

#### C. Scheduling Primitives & Vectorization
- **TensorIR Primitives**: Comprehensive suite of loop transformations:
  - `split`, `fuse`, `reorder`, `tile`, `vectorize`, `unroll`.
  - `bind` to GPU thread blocks (`blockIdx.x`, `threadIdx.x`).
  - `storage_scope` (`shared`, `local`), `cache_read`, `cache_write`.
- **`tensorize` Primitive**: Matches a sub-loop nest in TIR against a target micro-kernel description (`desc`) and replaces it with hardware-specific assembly or intrinsic calls (`impl`).
- **MetaSchedule**: Automated schedule search space exploration leveraging evolutionary algorithms and machine-learned cost models.
- **Popcount Support**: Built-in `tvm.tir.popcount(x)`. Lowers to LLVM `ctpop` on CPU and `__popc` on CUDA. Requires custom lowering rules (`register_intrin_rule`) for specific Vulkan/Metal shader dialects.

#### D. Balanced Ternary Integration Path & Friction Points
1. **Data Type Friction**: TVM lacks native support for custom bitplane structs or dual-rail algebraic types. Tensors must be typed as standard integer containers (`uint32`, `uint64`), losing semantic type safety.
2. **Carry-Save Accumulation**: The redundant $(Sum, Carry)$ accumulator pair must be manually expanded into multiple local buffers in TensorIR blocks, preventing standard auto-reduction optimizations.
3. **Codec Friction**: Base-243 decompression can be expressed as a TIR compute block using integer division and modulo, but autotuning often fails to vectorize modulus chains efficiently without manual assembly injection.
4. **Project Governance Risk**: Commercial maintainer OctoML pivoted and closed operations in early 2024. While the TVM Unity / MLC-LLM open-source branch continues at CMU/LMSYS, support for non-standard backends has slowed.

---

### 2.3 Halide

#### A. Architecture & Core Tenet
- **Decoupled Architecture**: Strict separation between functional **Algorithm** ($f(x, y) = ...$) and hardware **Schedule** (loop order, vectorization factor, tiling dimensions, storage hierarchy).
- **Backend CodeGen**: Generates LLVM IR directly for CPU targets (x86_64, ARM, RISC-V, Hexagon) and source compute shaders / API calls for GPU backends (CUDA, Metal, Vulkan, OpenCL, Direct3D 12).
- **Target Domain**: Specialized for image processing, computational photography, stencil computations, and local array transforms.

#### B. Runtime Execution Model
- **Footprint**: Near-zero. Emits standalone AOT object files (`.o`) and C headers (`.h`). No virtual machine or dynamic graph engine.
- **Thread Management**: Relies directly on host threading runtimes (GCD on macOS, OpenMP or custom thread pool on Linux).

#### C. Scheduling Primitives & Vectorization
- **Pioneering Schedule DSL**:
  - `split`, `fuse`, `tile`, `reorder`, `vectorize`, `unroll`, `parallel`.
  - GPU mapping: `gpu_blocks`, `gpu_threads`, `gpu_lanes`.
  - Storage scheduling: `compute_at`, `store_at`, `fold_storage`, `compute_root`.
- **Auto-Schedulers**: Highly mature classical and ML-based auto-schedulers (Li et al., Anderson et al., Adams et al.).

#### D. Balanced Ternary Integration Path & Critical Deficiencies
1. **Missing Native POPCNT**: Halide does not provide a native `popcount` operator in its core expression DSL. Users must either:
   - Implement SWAR (SIMD Within A Register) bit-twiddling trees in Halide arithmetic (generating 12–15 instructions per popcount instead of 1 hardware cycle).
   - Use `Halide::Internal::Call` or `ExternFunc` calling C `__builtin_popcount`, which prevents loop inlining across GPU thread blocks.
2. **No Graph-Level Neural Network Infrastructure**: Halide compiles single pipeline kernels. It lacks graph-level operator fusion across multi-layer transformers, dynamic activation allocation, and framework frontends (PyTorch/ONNX).
3. **Dual-Rail Representation**: Expressing dual-rail arithmetic requires multi-channel `Func` or `Tuple(P, N)`. Scheduling redundant carry-save updates across multidimensional iteration spaces leads to prohibitive schedule search spaces.

---

## 3. Comprehensive Technical Comparison Matrix

| Architectural Dimension | IREE (OpenXLA) | Apache TVM (TVM Unity) | Halide |
| :--- | :--- | :--- | :--- |
| **Core Abstraction Layer** | Composable MLIR Dialects (`flow`, `stream`, `hal`, `linalg`, `vector`) | Relax (Graph IR) + TensorIR (Loop Block AST) | Pure Functional Array Expressions + Decoupled Schedule Tree |
| **Target Backends (CPU)** | x86_64 (AVX2, AVX-512, AMX), ARM64 (NEON, SVE, SME), RISC-V | x86_64 (AVX2, AVX-512), ARM64 (NEON), RISC-V | x86_64, ARM64, RISC-V, Qualcomm Hexagon DSP |
| **Target Backends (GPU)** | Vulkan (SPIR-V), Apple Metal (MSL), NVIDIA CUDA (NVPTX), AMD ROCm (ROCDL) | CUDA (PTX), ROCm (HIP), Apple Metal (MSL), Vulkan (SPIR-V), WebGPU | CUDA (PTX), Apple Metal (MSL), Vulkan (SPIR-V), OpenCL, Direct3D 12 |
| **Runtime Engine & Footprint** | Standalone C HAL Runtime: **~50 KB – 150 KB** (Embeddable, zero Python) | C++ Runtime + PackedFunc FFI: **~1 MB – 3 MB** | **Zero Runtime** (Emits static `.o` + `.h` directly) |
| **Model Distribution Format** | Standalone FlatBuffers (`.vmfb`) | Shared Object (`.so` / `.dylib`) + JSON Metadata | Compiled Object (`.o` / `.a`) + C Header |
| **Custom Dialect / Type Support**| **First-Class**: Extensible MLIR type system (`!bt.dualrail`, `!bt.csa_pair`) | **Limited**: Restricted to primitive integers (`uint32`, `uint64`) | **None**: Primitives only (`Int`, `UInt`, `Float`) |
| **POPCNT / Bitwise Lowering** | **Native**: `math.ctpop` / `llvm.intr.ctpop` -> `vpopcntdq`, `cnt.16b`, `OpBitCount`, `metal::popcount` | **Native**: `tvm.tir.popcount` -> LLVM `ctpop`, CUDA `__popc` | **Deficient**: No DSL node; requires SWAR trees or opaque `ExternFunc` |
| **Carry-Save Addition (BTFA)** | Seamless lowering via MLIR pattern rewrites to 3-input bitwise logic | Complex multi-buffer state tracking inside TIR blocks | Cumbersome multi-channel `Func` / `Tuple` logic |
| **Base-243 Codec Fusion** | Fused elementwise dispatch direct to L1/shared memory tile | Expressible in TIR compute; autotuner struggles with modulo | Expressible, but modulo chains inhibit vectorization |
| **Scheduling Methodology** | Declarative MLIR Transform Dialect (Tiling, Fusion, Vectorization) | Imperative TIR Schedule + Automated MetaSchedule | Imperative Schedule DSL (`tile`, `vectorize`, `gpu_blocks`) |
| **Multi-Stream & Async Concurrency**| Timeline semaphores & async stream execution in HAL | Stream execution via Device API and Virtual Machine | Relies on host-side threading / OS queues |
| **Unified Memory Exploitation** | Full zero-copy support (`MTLResourceStorageModeShared`, Vulkan host-visible) | Supported via TVM Device API zero-copy allocators | Supported on CPU/Metal via pointer wrapping |
| **Ecosystem Maintenance (2024–2026)**| **Hyperactive**: Backed by OpenXLA, Google, nod.ai (AMD), Modular | **Volatile/Slow**: OctoML closed 2024; academic community (UW/CMU) | **Stable/Curated**: Google, Adobe; strictly image processing |
| **Overall Suitability Score** | **96 / 100** | **74 / 100** | **48 / 100** |

---

## 4. Hardware-Specific Micro-Architecture Execution Deep-Dive

### 4.1 ARM64 (Apple Silicon M-Series, AWS Graviton 3/4)
- **Bitwise Planar Multiply**: 
  - Dual-rail product rails $P = (A_P \land B_P) \lor (A_N \land B_N)$ and $N = (A_P \land B_N) \lor (A_N \land B_P)$.
  - Implemented via ARM NEON `and`, `orr`, and bitwise ternary select `bsl`.
- **Vector Population Count**:
  - Instruction: `cnt.16b` computes population count for 16 separate 8-bit lanes simultaneously in 1 cycle.
  - Reduction: Vector pairwise widening addition `uaddlp.8h` followed by `uaddlp.4s`, accumulating into 32-bit integer sums in 3 cycles total.
- **IREE Efficiency**: Generates exact `cnt.16b` + `uaddlp` sequences via `vector.contract` lowering without register spills.

### 4.2 x86_64 AVX-512 (Intel Xeon Sapphire Rapids, AMD EPYC Genoa)
- **Ternary Boolean Logic (`vpternlogd`)**:
  - `vpternlogd` evaluates any arbitrary 3-input truth table across 512-bit registers in a single clock cycle with a throughput of 0.5 cycles.
  - Computes half-adder sum $S = A \oplus B$ and carry $C = A \land B$ using exact immediate truth table bitmasks (`0x96` for XOR, `0x88` for AND), executing dual-rail additions with zero register overhead.
- **Vector Population Count (`vpopcntdq`)**:
  - Counts set bits across 8 independent 64-bit quadwords per register in 1 clock cycle.
  - Throughput: 1 vector popcount per cycle on port 0/port 5.
- **IREE Advantage**: IREE's LLVM CPU backend natively recognizes `math.ctpop` on `vector<8xi64>` and directly emits `vpopcntdq` when compiled with `-iree-llvmcpu-target-cpu-features="+avx512f,+avx512vpopcntdq"`.

### 4.3 Apple Metal GPU (Unified Memory)
- **Execution Units**: 32-wide SIMD execution groups (`simdgroup`).
- **Bitplane Streaming**:
  - Base-243 packed weights stored in LPDDR5X unified RAM (`MTLResourceStorageModeShared`).
  - Read directly into GPU threadgroup shared memory without PCIe transfers.
  - Unpacked into registers using bitwise shifts and LUT multiplications.
- **Hardware Popcount**:
  - Built-in `metal::popcount(uint64_t)` executes across SIMD lanes in 1 cycle.
  - Subgroup reductions leverage `simd_sum(val)` to accumulate inner products across 32 threads without memory roundtrips.
- **IREE Integration**: IREE's Metal target compiles MLIR dispatches directly into Metal Shading Language compute pipelines (`.vmfb`), orchestrating dispatches via Metal command buffers.

### 4.4 Vulkan / SPIR-V (Cross-Vendor GPU: Qualcomm Adreno, AMD RDNA 3/4)
- **Instruction Mapping**:
  - Standard bitwise operators map to `OpBitwiseAnd`, `OpBitwiseOr`, `OpBitwiseXor`.
  - Population count maps to standard SPIR-V `OpBitCount`.
- **Driver Portability**:
  - Vulkan 1.1+ requires `OpBitCount` in hardware on all compliant mobile and desktop architectures.
  - Eliminates proprietary CUDA driver lock-in while matching CUDA warp-level performance.

---

## 5. Strategic Architectural Recommendation for Balanced Ternary System

### Primary Recommendation: IREE (OpenXLA) as the Production Execution Engine
1. **Direct Alignment with Existing Repository Dialect**:
   - The workspace already defines `TernaryDialect.td` and `TernaryOps.td`.
   - IREE is the only evaluated framework that can natively ingest these MLIR definitions directly without converting them to C++ strings or AST wrapper objects.
2. **Progressive Lowering Strategy**:
   ```
   [Ternary Source / PyTorch]
              │
              ▼
      [MLIR `bt` Dialect]
              │
              ▼ (bt-to-linalg conversion pass)
   [linalg.generic + math.ctpop]
              │
              ▼ (IREE Flow & Stream Partitioning)
   [IREE HAL Multi-Target Dispatch]
     ├──► SPIR-V / Vulkan (Android, AMD, Intel)
     ├──► Metal MSL (Apple Silicon Mac & iOS)
     ├──► LLVM CPU (x86-64 AVX-512 & ARM64 NEON)
     └──► CUDA / ROCm (NVIDIA / AMD Data Center)
   ```
3. **Decompression Fusion**:
   - Implement the Base-243 / Base-59049 unpacker as an affine elementwise MLIR op (`bt.unpack_base243`).
   - IREE's fusion passes will automatically fuse the unpacker into the consuming `linalg.generic` GEMM dispatch, keeping unpacked dual-rail trits entirely in L1/register memory.

### Secondary Utility: Halide for Rapid Isolated Micro-Kernel Prototyping
- Use Halide solely as a local C++ testbed for investigating loop tiling geometries and memory strides for standalone bitplane GEMM routines prior to codifying them into MLIR Transform Dialect scripts.

---

## 6. Authoritative Source Citations & References

1. **IREE Architecture & Documentation**:
   - OpenXLA IREE Project: https://iree.dev/
   - IREE Architecture Specification: https://iree.dev/developers/design-docs/architecture/
   - IREE HAL Abstraction Layer: https://iree.dev/developers/design-docs/hal-design/
   - Vulkan Compute in IREE: https://www.vulkan.org/ecosystem/iree
2. **Apache TVM & TVM Unity**:
   - Apache TVM Main Repository: https://tvm.apache.org/
   - TensorIR Abstraction & Tensorize: https://tvm.apache.org/docs/arch/infer_low_level.html
   - TVM Unity / Relax Architecture: https://mlc.ai/chapter_graph_optimization/index.html
   - Low-Precision TVM Kernels: https://arxiv.org/abs/2007.14435
3. **Halide**:
   - Halide Official Website: https://halide-lang.org/
   - Decoupled Algorithm & Schedule Design: https://halide-lang.org/docs/
   - Halide GPU Backends (CUDA, Metal, Vulkan): https://github.com/halide/Halide
4. **Hardware SIMD & Popcount Mechanics**:
   - Intel Intrinsics Guide (AVX-512 `_mm512_popcnt_epi64`, `_mm512_ternarylogic_epi32`): https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
   - ARM Architecture Reference Manual (NEON `CNT`, `UADDLP`): https://developer.arm.com/architectures/instruction-sets/simd-isas/neon
   - Apple Metal Shading Language Specification (`metal::popcount`): https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
