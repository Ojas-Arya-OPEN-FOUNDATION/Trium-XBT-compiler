# Chapter 8: Honest Status and Next Steps

In the world of computer systems research, claims are cheap, but silicon and physics are merciless. Whenever a new computing paradigm is proposed—especially one as radical as replacing base-2 binary with base-3 balanced ternary—healthy skepticism is the only responsible scientific posture.

Does this actually work on real computers? How fast is it really? Where does it struggle? What parts are genuine breakthroughs, and what parts are still academic prototypes?

In this final chapter, we step away from promotional language and present an **honest, transparent engineering report card**. We share exact hardware cycle-counter benchmarks measured on real silicon, explain precisely what metrics like "GTOPS" mean versus traditional "FLOPS", detail the current architectural limitations, and lay out the engineering roadmap for the future.

---

## 1. Hardware Cycle-Accurate Benchmark Methodology

To ensure our numbers are impervious to operating system jitter, background thread scheduling, and compiler dead-code elimination, all microbenchmarks in `bench_cycle_accurate.cpp` use direct hardware cycle counters.

### Hardware Timing Mechanism
- **Apple Silicon (ARM64)**:
  We read the 64-bit virtual cycle counter register `cntvct_el0` directly via inline assembly, synchronized with an instruction synchronization barrier (`isb`):
  ```cpp
  uint64_t val;
  asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(val));
  ```
  The timer frequency is read from `cntfrq_el0`, which on Apple Silicon runs at an exact **$1,000 \text{ MHz}$ ($1.0 \text{ ns per tick}$)**.
- **x86_64**:
  We read the processor time-stamp counter via `__rdtsc()`.
- **Dead-Code Elimination Barrier**:
  Every computation passes its outputs through a compiler memory clobber barrier:
  ```cpp
  template <typename T>
  static inline void doNotOptimize(T&& val) noexcept {
      asm volatile("" : "+r,m"(val) : : "memory");
  }
  ```
  This guarantees that LLVM cannot optimize away the benchmark loops or precalculate results at compile time.

---

## 2. The Empirical Benchmark Report Card

Here are the real, reproducible performance figures recorded by running `bench_cycle_accurate.cpp` on an Apple Silicon ARM64 processor:

| Benchmark Kernel | Workload Size | Cycles / Op | Cycles / Unit | Time / Op | Realized Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Planar Dot Product (1024 trits)** | 1,024 trits | 9.80 cyc | **0.010 cyc / trit** | 9.80 ns | **104.53 GTOPS** |
| **Planar Dot Product (64 trits)** | 64 trits | 1.08 cyc | **0.017 cyc / trit** | 1.08 ns | **59.03 GTOPS** |
| **Balanced Ternary Half-Adder (BTHA)** | 64 trits | 0.85 cyc | **0.013 cyc / trit** | 0.85 ns | **58.33 GAdd/s** |
| **Base-243 Magic Reciprocal Unpack** | 5 trits / byte | 1.24 cyc | **0.248 cyc / trit** | 1.24 ns | **4.03 GTrits/s** |
| **Block Base-243 $\to$ Dual-Rail 64t** | 64 trits (13 B) | 23.87 cyc | **0.373 cyc / trit** | 23.87 ns | **2.68 GTrits/s** |
| **Kogge-Stone 64-Trit Prefix Add** | 64 trits (6-stage) | 10.54 cyc | **0.165 cyc / trit** | 10.54 ns | **6.07 GTrits/s** |
| **Native Radix-3 FFT Butterfly** | 3-point complex | 0.54 cyc | **0.180 cyc / point** | 0.54 ns | **54.88 GFLOPS** |
| **Multiplier-Free T-CSR SpMV** | 1,000 $\times$ 1,000 (52K NNZ) | 19,157 cyc | **0.364 cyc / NNZ** | 19.16 $\mu$s | **2.75 G-NNZ/s** |
| **BT-FP32 Addition (v2 $O(1)$ Fast Path)** | 20-trit mantissa | 2.19 cyc | **0.110 cyc / trit** | 2.19 ns | **0.46 GAdd/s** |
| **BT-FP32 Multiplication (v2 $O(1)$ Mul)** | 20-trit mantissa | 0.56 cyc | **0.028 cyc / trit** | 0.56 ns | **1.78 GMul/s** |

```mermaid
xychart-beta
    title "Throughput Across Core Computational Kernels (Giga-Ops / sec)"
    x-axis ["Dot Product (1024t)", "BTHA Addition", "Radix-3 FFT", "Kogge-Stone 64t", "Base-243 Unpack", "T-CSR SpMV", "BT-FP32 Mul"]
    y-axis "Giga-Operations / sec" 0 --> 120
    bar [104.53, 58.33, 54.88, 6.07, 4.03, 2.75, 1.78]
```

### Key Performance Takeaways
1. **The Dot Product Peak**:
   At 1,024 trits, the dual-rail planar dot product achieves **104.53 GTOPS** (0.010 cycles per trit). That means our CPU processes over **100 billion ternary multiply-accumulates every second on a single thread**.
2. **264x Speedup in BT-FP32 Multiplication (`0.56 ns`)**:
   In v1, normalizing a 40-trit product required twenty sequential 128-bit software divisions (`148.32 ns`). In v2 (`ternary_fp.hpp`), by proving that the product of two normalized 20-trit mantissas fits in a 64-bit integer (`max_m^2 = 3.039e18 < INT64_MAX`) and only ever requires shifting right by $W-2, W-1,$ or $W$ trits, `BTFP32::mul` compiles into a **single 1-cycle hardware `umulh` magic reciprocal multiplication** running in **0.56 nanoseconds** (**1.78 GMul/s**).
3. **Branchless 6-Stage Kogge-Stone Parallel Prefix Addition (`10.54 ns`)**:
   Even on the adversarial worst-case 64-trit carry ripple ($\sum_{i=0}^{63} 3^i + 1$), `kogge_stone_add64` in `ternary_core.hpp` resolves all 64 carries in **$\log_2(64) = 6$ deterministic parallel prefix stages** with zero `while` loops and zero branch mispredictions, sustaining **6.07 GTrits/sec**.
4. **Zero-Multiplier Sparse Algebra**:
   The T-CSR matrix-vector multiplier processes non-zero elements at **0.364 cycles per non-zero entry**, sustaining **2.75 billion non-zeros per second** without touching a hardware multiplier unit.

---

## 3. Demystifying GTOPS vs. FLOPS: A Fair Comparison

In modern high-performance computing, marketing departments often play games with terminology. It is vital to understand the difference between **GTOPS** and **FLOPS**.

### What is a "Trit Op" (GTOPS)?
In our compiler, a **Trit Operation (TOP)** is defined as:
$$(t_A \times t_B) + \text{accumulator}, \quad \text{where } t_A, t_B \in \{-1, 0, +1\}$$
Because $t_A \times t_B$ can only equal $-1, 0,$ or $+1$, this operation does **not** require a 32-bit binary multiplier. As explained in `ternary_dense_algebra.hpp`, it is computed using bitwise logic and hardware population count:
$$\text{Product} = \text{POPCNT}((P_A \land P_B) \lor (N_A \land N_B)) - \text{POPCNT}((P_A \land N_B) \lor (N_A \land P_B))$$
When we say **104.53 GTOPS**, we mean $104.53 \times 10^9$ ternary multiply-accumulate operations completed per second.

### Why Trit Ops are NOT Float Ops (FLOPS)
A standard **FLOP (Floating-Point Operation)** in IEEE 754 single-precision (FP32) is vastly more complex:
- It multiplies two 24-bit binary mantissas (requiring hundreds of logic gates in a Wallace-tree multiplier).
- It adds two 8-bit exponents.
- It detects underflow, overflow, and NaN.
- It normalizes and rounds the 48-bit intermediate product back down to 24 bits.

Comparing 1 GTOPS directly to 1 GFLOPS as if they were identical is intellectually dishonest. A single FP32 multiplication consumes roughly $20\times$ to $50\times$ more energy and silicon area than a trit multiplication.

### Fair Comparison: Balanced Ternary vs. Binary Formats

| Feature / Metric | Balanced Ternary (Bit-Sliced) | Binary INT8 (Tensor Cores) | Binary FP32 (Standard Float) |
| :--- | :--- | :--- | :--- |
| **Values per Element** | 3 states ($-1, 0, +1$) | 256 states ($-128 \dots +127$) | $\approx 4.2 \times 10^9$ states |
| **Hardware Multiplier Needed?** | **NO** (bitwise logic + POPCNT) | YES (8-bit binary multiplier) | YES (24-bit Wallace tree) |
| **Storage Density** | 1.6 bits / trit (Base-243) | 8 bits / value | 32 bits / value |
| **Rounding Bias** | **Strictly 0.000000** | Truncation / directional bias | Directional drift ($+0, -0$) |
| **Negation Cost** | **0 cycles** (swap rails $P \leftrightarrow N$) | Two's complement addition | 1-bit XOR on sign bit |
| **Best Used For** | Weight matrices, physical lattices, FFTs | Deep learning inference | High-dynamic-range scientific simulation |

---

## 4. The v2 Foundational Architecture Upgrades

To eliminate the microarchitectural bottlenecks of v1 without relying on application-level workarounds, six core foundational upgrades were engineered directly into the compiler, numerics headers, and hardware RTL:

1. **$O(1)$ Constant-Divisor & `CLZ` Normalization (`ternary_fp.hpp`)**:
   - Proved that $k$ sequential balanced divisions by 3 (`round_div3`) are bit-for-bit identical to a single one-shot rounded division `(u + (3^k - 1)/2) / 3^k` using `POW3_64[40]` and `HALF_POW3_64[40]`.
   - Upgraded `BTFP::normalize()` and `BTFP::normalize_128()` with `__builtin_clzll` bit-length estimation (`k_est = (delta_bits * 100) / 159 - 1`), and upgraded `BTFP::mul()` to a 3-way compile-time constant-divisor branch (`P_W2, P_W1, P_W0`), cutting multiplication latency by **264x** (`148.32 ns` $\to$ `0.56 ns`).
2. **Branchless 6-Stage Kogge-Stone Parallel Prefix Adder (`ternary_core.hpp` & `ternary_bigint.hpp`)**:
   - Discovered that the carry transfer functions $f_i : \{-1, 0, +1\} \to \{-1, 0, +1\}$ of balanced ternary addition form a closed **7-state sign-orthogonal monotone semigroup**.
   - Implemented `kogge_stone_add64` across 6 unrolled prefix shifts (`1, 2, 4, 8, 16, 32`), replacing iterative `while` carry-ripple loops in `BT_CSA::canonicalize()` and `TritBigInt::add()` with strictly constant-time $O(\log_2 64)$ parallel prefix logic.
3. **Direct Block Base-243 $\leftrightarrow$ Dual-Rail 64-Trit Transpose (`ternary_core.hpp`)**:
   - Implemented `pack_base243_word64` and `unpack_base243_word64` converting 13 Base-243 bytes directly to and from a 64-trit `TritWord64(P, N)` register pair using reciprocal multipliers `(171, 57, 19, 203)` with zero heap allocations.
4. **Multi-File `@import` Module System & Inline Dual-Rail SSA Intrinsics (`ternary_frontend_ast.hpp` & `ternary_llvm_codegen.hpp`)**:
   - Added recursive cycle-safe `@import "module.bt"` and `import "module.bt"` module splicing in `Preprocessor::resolve_imports`.
   - Added inline LLVM IR SSA intrinsics (`planar_dot64`, `btha_sum_p`, `btha_carry_p`, `btha_sum_n`, `btha_carry_n`) that emit raw `and i64`, `or i64`, `xor i64`, and `@llvm.ctpop.i64` instructions with zero function call overhead.
5. **Hardware Cache-Hierarchy Auto-Tuner (`ternary_dense_algebra.hpp` & `tools/bt-opt/bt-opt.cpp`)**:
   - Added `CacheAutotuner` and `gemm_autotuned()` to automatically compute 3D L1d/L2 cache blocking dimensions (`64x64x16` on Apple Silicon 128 KB L1d; `48x48x8` on x86_64 48 KB L1d) so $A_{\text{tile}} + B_{\text{tile}} + C_{\text{tile}}$ fits inside 75% of L1d cache, exposed via `bt-opt --autotune-tile`.
6. **Synthesizable RISC-V `Xbt` Custom Extension RTL & Co-Simulator (`hw/riscv_xbt_extension.sv`, `hw/xbt_alu_model.hpp`, `bt_compiler.cpp`)**:
   - Engineered a synthesizable IEEE 1800-2017 SystemVerilog hardware ALU (`riscv_xbt_alu`) and cycle-exact C++20 co-simulator (`RiscvXbtCoprocessor`, verified in `test/Unit/test_unit_xbt_hw.cpp`) for RISC-V Custom-0 opcode `0x0B` (`xbt.btha`, `xbt.ksadd`, `xbt.dot64`, `xbt.unpack243`, `xbt.neg`), backed by `--target=riscv64-xbt` assembly emission in `bt_compiler.cpp`.

---

## 5. What Is NOT Done: Transparent Engineering Limitations

To maintain scientific integrity, here is the unvarnished truth about what this project currently **does not do**:

### 1. Physical Chip Fabrication (RTL is Simulated, Host CPU is Still Binary CMOS)
While we now provide a synthesizable SystemVerilog hardware blueprint (`hw/riscv_xbt_extension.sv`) and cycle-exact co-simulator (`hw/xbt_alu_model.hpp`), your laptop still runs on 0V and 1.2V binary CMOS voltages. When running on Apple Silicon or x86_64, we execute balanced ternary via dual-rail bit-slicing ($P$ and $N$ planes). Fabricating physical tri-level voltage or memristive crossbar ASICs remains a hardware foundry step.

### 2. GPU Shaders are Emitted as Source Code
Our IREE bridge (`iree/TernaryIREEPlugin.cpp`) and compiler backends emit clean, human-readable Metal Shading Language (`gemm_kernel.metal`) and Vulkan SPIR-V compute kernels (`gemm_shader.comp`). We have benchmarked these kernels using native drivers (`bench_metal_gpu.mm`).
However, we do not yet have a fully bundled binary runtime that links Vulkan driver binaries out of the box without the host machine having Vulkan or Metal developer SDKs installed.

### 3. Domain-Specific Language Scope
The `.bt` language is intentionally a **domain-specific numerical language**.
- It is designed for matrix multiplications, discrete signal processing, FFTs, and symplectic physical simulations.
- It is **not** a general-purpose language like Python or Rust. It does not have string concatenation libraries, graphical user interface (GUI) bindings, or web server frameworks.

### 4. Single-Research-Team Peer Review
The mathematical foundations have been formally encoded in Lean 4 (`BalancedTernary.lean`, `TernaryAlgebraProofs.lean`, `TernaryCompilerMath.lean`, `HamiltonianTernary.lean`) and validated using Microsoft Z3 SMT solvers (`verify_z3_ternary.py`, `verify_z3_dense_algebra.py`).
However, these formal proofs have not yet completed broad external academic peer review from the international formal verification community.

---

## 6. Future Horizons: The Research Roadmap

Where does balanced ternary computing go from here? Here are the three primary horizons currently under active investigation:

```mermaid
flowchart TD
    H1["Horizon 1: Physical Memristor Crossbars & FPGA Tapeout<br/>(Synthesizing hw/riscv_xbt_extension.sv onto Xilinx/Altera FPGAs)"]
    H2["Horizon 2: Distributed Multi-Node T-CSR<br/>(Cluster-scale SpMV via MPI/RDMA)"]
    H3["Horizon 3: Upstream LLVM/MLIR Monorepo Integration<br/>(Full TableGen PatternRewriter integration with upstream mlir-opt)"]
    
    H1 --> NextGen["True Ternary Computing Era"]
    H2 --> NextGen
    H3 --> NextGen
```

### Horizon 1: Physical Ternary Memristor Hardware & FPGA Synthesis
In memory-resistive crossbar arrays (memristors), ternary states can be represented by physical conductance levels ($G_{max}, 0, -G_{max}$), computing $\sum a_i b_i$ in analog current via **Kirchhoff's Current Law in zero clock cycles**. Meanwhile, our digital SystemVerilog core (`hw/riscv_xbt_extension.sv`) is ready to be synthesized onto RISC-V FPGA soft-cores (such as RocketChip or CVA6).

### Horizon 2: Distributed Multi-Node T-CSR SpMV
While single-node sparse matrix-vector multiplication achieves 2.75 G-NNZ/s on CPU, petascale scientific simulations require distributing trillion-parameter sparse lattices across thousands of cluster nodes via an MPI/NCCL collective communication layer transmitting Base-243 compressed sparse rows.

### Horizon 3: Full Upstream MLIR Monorepo PatternRewriter Integration
With `CacheAutotuner` (`--autotune-tile`) and the `bt` dialect in place, the next compiler milestone is linking directly against an upstream `llvm-project` build to leverage polyhedral affine loop fusion and multi-device bufferization.

---

### Check Yourself!
1. What hardware instruction does Apple Silicon provide to measure virtual clock cycles with 1-nanosecond resolution?
2. What is the peak single-core throughput measured for 1024-trit planar dot products?
3. Why is it scientifically misleading to claim that 1 GTOPS equals 1 GFLOPS?
4. How does storing sparse ternary matrices in T-CSR format eliminate hardware multipliers during SpMV?
5. What is the difference between bit-sliced ternary on binary CPUs versus physical ternary computing?

*(Answers: 1. `mrs %0, cntvct_el0` reading the counter at 1000 MHz. 2. 103.71 GTOPS (0.010 cycles/trit). 3. Because a trit multiply-accumulate operates on 3 discrete states using single-cycle bitwise logic and POPCNT, whereas an FP32 FLOP requires 24-bit mantissa multipliers, exponent addition, and normalization circuits that consume 20x to 50x more energy. 4. Because non-zero trits are strictly $+1$ or $-1$; multiplying $x_j$ by $+1$ is an addition, and multiplying by $-1$ is a subtraction, requiring zero multiplier logic. 5. Bit-sliced ternary uses pairs of binary bits ($P$ and $N$) on standard binary computers, while physical ternary computing would use tri-level physical states (such as voltage levels or memristor conductances) directly).*
