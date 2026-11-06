# High-Performance Ternary GEMM, Register-Blocked Microkernels & LLM Inference
## 743 GTOPS AVX-512/NEON Microkernel, Operand Memory Layouts, and Prefill vs Decode Architecture

> **AI Context Brief:**
> This document details the high-performance linear algebra compute engine of Trium-XBT:
> 1. **743 GTOPS Microkernel (`include/Ternary/Blas/TernaryMicroKernel.hpp`)**: Hand-tuned, register-blocked ($MR \times NR$) GEMM engine utilizing AVX-512 `VPOPCNTDQ` and ARM NEON intrinsics.
> 2. **Operand Memory Layout Contract**: Exact specifications for row-major $A$ and transposed word-plane-major $B$, including mathematical prevention of the $K > 64$ layout trap.
> 3. **Hardware Benchmarks**: Empirical validation on real silicon comparing Balanced Ternary against INT8 and BF16.
> 4. **LLM Inference Workload Architecture**: Comprehensive analysis distinguishing **Prefill ($M > 1$ GEMM)** from **Decode ($M = 1$ GEMV / MatVec)**.

---

## 1. Mathematical Formulation of the Dual-Rail GEMM

The fundamental matrix multiplication is:
$$C[M, N] = A[M, K] \times B[K, N] \quad \text{where } A_{i,k}, B_{k,j} \in \{-1, 0, 1\}$$

In dual-rail balanced ternary, every 64 trits are encoded into two 64-bit unsigned integers: Positive Rail ($P$) and Negative Rail ($N$).

### Single-Cycle Planar Dot Product
For two 64-trit words $A = (A_P, A_N)$ and $B = (B_P, B_N)$:
$$\text{Product}_P = (A_P \land B_P) \lor (A_N \land B_N)$$
$$\text{Product}_N = (A_P \land B_N) \lor (A_N \land B_P)$$
$$\text{Dot64}(A, B) = \text{popcount64}(\text{Product}_P) - \text{popcount64}(\text{Product}_N)$$

On x86_64 CPUs with **AVX-512F + AVX-512VPOPCNTDQ**, this is computed without branching:
```cpp
// AVX-512 vectorized 512-trit dot product step
__m512i prod_p = _mm512_ternarylogic_epi64(ap, bp, an_bn, 0x82); // (ap & bp) | (an & bn)
__m512i prod_n = _mm512_ternarylogic_epi64(ap, bn, an_bp, 0x82); // (ap & bn) | (an & bp)
__m512i count_p = _mm512_popcnt_epi64(prod_p);
__m512i count_n = _mm512_popcnt_epi64(prod_n);
__m512i diff = _mm512_sub_epi64(count_p, count_n);
```

---

## 2. Register Blocking ($MR \times NR$) & Cache Hierarchy

To achieve peak memory bandwidth and compute efficiency, `bt::blas::ternary_gemm` employs an outer-product register-tiled microkernel:

```
  Matrix A (M x K)                    Matrix B (K x N)
  [ Row-Major ]                       [ Transposed + Word-Plane-Major ]
  
     K trits (kw words)                     N columns
  ┌───────────────────────┐            ┌─────────────────┐
M │                       │          K │ Word-Plane 0:   │
  │  MR rows in registers │          │   bp[0*N + j]   │
  │                       │            ├─────────────────┤
  └───────────────────────┘            │ Word-Plane 1:   │
                                       │   bp[1*N + j]   │
                                       └─────────────────┘
```

### Tile Dimensions
- **AVX-512**: $MR = 8$, $NR = 4$. Accumulates 32 independent output cells across 32 zmm vector registers.
- **ARM NEON**: $MR = 8$, $NR = 4$. Unrolls across 16 NEON 128-bit registers.
- **Portable Scalar**: $MR = 4$, $NR = 4$. Uses SWAR popcount with 64-bit GPRs.

---

## 3. The Operand Memory Layout Contract & The $K > 64$ Solution

A critical engineering lesson from hardware validation: **Matrix $B$ must NOT be stored row-major.**

### The Layout Trap ($K \le 64$ vs $K > 64$)
When $K \le 64$ ($kw = 1$), row-major ordering and word-plane-major ordering are mathematically identical. A naive row-major packing works by coincidence:
- $K = 64$: 0 errors.
- $K = 128$: 3,885 out of 4,096 entries silently corrupted (status 0).
- $K = 256$: 3,956 out of 4,096 entries corrupted.

### The Correct Formats
1. **Matrix $A$ ($M \times K$)**: Row-Major.
   - Word $w$ of row $i$: `ap[i * kw + w]`, where $kw = \lceil K / 64 \rceil$.
2. **Matrix $B$ ($K \times N$)**: Transposed + Word-Plane-Major.
   - Word $w$ of column $j$: `bp[w * N + j]`.
   - All column values for word slice $w$ are contiguous in memory. This allows the microkernel to broadcast one word of $A$ and perform continuous vector FMA with consecutive columns of $B$.

### Official Layout Packing Functions
Available across Python, C++, and Rust (`pack_matrix_a` and `pack_matrix_b`):
```cpp
DualRail pack_matrix_a(const int64_t* a, uint64_t m, uint64_t k);
DualRail pack_matrix_b(const int64_t* b, uint64_t k, uint64_t n);
```

---

## 4. Empirical Silicon Benchmarks ($512 \times 512 \times 512$)

Independent benchmarks executed on AVX-512 hardware with native `vpopcntdq`:

| Engine | Latency | Compute Work | Memory Density | Comparison to Ternary |
| :--- | :--- | :--- | :--- | :--- |
| **Trium `ternary_gemm`** | **0.722 ms** | **743 GTOPS** (185.9 GMAC/s) | **1.58 bits/trit** | **Baseline** |
| Native INT8 (`_mm512_dpbusd`) | 0.420 ms | 1,278 GTOPS (319.5 GMAC/s) | 8.00 bits/val | 1.63× faster, **5.06× LESS dense** |
| Floating-Point BF16 | 8.770 ms | 61 GFLOPS (15.3 GMAC/s) | 16.00 bits/val | **12.1× SLOWER**, **10.1× LESS dense** |

### The Information Density Advantage
At equal bytes of memory, Balanced Ternary performs **3.1× more mathematical operations** than INT8. For large language models (70B, 120B) where memory bandwidth is the primary operational bottleneck, ternary density delivers superior throughput per memory channel.

---

## 5. Workload Architecture: Prefill ($M > 1$) vs Decode ($M = 1$)

Understanding LLM execution is essential for deploying ternary acceleration:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   LLM INFERENCE WORKLOAD DIVISION                      │
├──────────────────────────────────┬─────────────────────────────────────┤
│ 1. PREFILL PHASE (Prompt Batch)  │ 2. DECODE PHASE (Token Generation)  │
│ - Batch size / sequence M > 1    │ - Single token per step: M = 1      │
│ - Matrix x Matrix (GEMM)         │ - Vector x Matrix (GEMV / MatVec)   │
│ - Compute-Bound                  │ - Memory-Bandwidth Bound            │
│ - Operands: Ternary x Ternary    │ - Operands: INT8 (Act) x Ternary (W)│
│ -> RUNS ON: Trium ternary_gemm   │ -> REQUIRES: Dedicated GEMV Kernel  │
│    (743 GTOPS Peak Throughput)   │    (Zero-Allocation Activation FMA) │
└──────────────────────────────────┴─────────────────────────────────────┘
```

### The BitNet Decode Engine Roadmap
In autoregressive token generation ($M=1$), input activations are generated dynamically by RMSNorm/LayerNorm in INT8 precision, while weights $W$ reside in Ternary.
- Standard naive execution re-slices INT8 activations into 8 bit planes on every token, incurring latency overhead.
- **The Production Solution**: A fused $M=1$ GEMV kernel that directly accumulates INT8 vector elements against dual-rail weight columns without intermediate bit-plane packing.
