# Trium XBT as a Foundation Compiler for Next-Generation Hardware

**Status: direction, not accomplishment.** This document states what the project
is for, what already exists that supports it, and what a new hardware backend
must implement. Claims are labelled `[verified]`, `[built]` or `[planned]`.
Nothing here should be quoted as a result until it carries a measurement.

---

## 1. The thesis

Binary won by accident, not by physics. Two stable states were the easiest thing
to build in silicon, so the entire stack above it — Boolean algebra, two's
complement, IEEE-754, the LLVM IR — was shaped by that one constraint.

Balanced ternary (values −1, 0, +1) is the better fit for what comes next, for
three reasons that compound:

**Information density.** A trit carries log₂(3) = 1.585 bits. For the same
information you need ~37% fewer wires, and in nanoscale fabrics the interconnect
*is* the cost — in area, in energy, and in yield.

**Arithmetic that is cheaper than binary's.** In balanced ternary there is no
sign bit and negation is free: swap the positive and negative rails. Rounding to
nearest is truncation. There is no carry-propagation asymmetry between addition
and subtraction, which is why the CSA trees in `ternary_core.hpp` collapse so
cleanly. Every partial product is in {−1, 0, +1}, so a ternary dot product
accumulates in exact integer arithmetic with no rounding error at any length —
a property no floating-point format has.

**Many nanoscale devices are natively three-state.** This is the part that makes
it more than a curiosity. Candidate post-CMOS substrates tend to present three
distinguishable states rather than two:

| Substrate | Where the third state comes from |
|---|---|
| CNTFET / dual-gate FET | Ambipolar conduction: n-type, p-type, and off |
| Memristive / ReRAM | Intermediate resistance between HRS and LRS |
| Quantum-dot cellular automata | Three-state cell polarisation |
| Spin / domain-wall logic | Three stable magnetisation or wall configurations |
| Photonic | Phase or amplitude with a null state |

A compiler that already thinks in three states can target these directly. A
binary compiler must first throw the third state away, then emulate it with two
wires and a lookup — paying back the density advantage it was supposed to gain.

That is the niche: **not a faster binary compiler, but the layer that lets
ternary-native hardware be programmed at all.**

---

## 2. What actually exists today

Being precise here matters, because the gap between "runs" and "is a foundation"
is where projects usually mislead themselves.

| Layer | State | Evidence |
|---|---|---|
| Frontend → IR → execution | **works** | 14/14 `.bt` corpus programs produce correct output; `--test` 4/4 |
| Ternary core library | **works** | 341 files, 1,175 assertions, zero `Admitted`/`Axiom`/`sorry` |
| Formal verification | **partial** | Lean 4 and Z3 proofs present; no Lean job in CI `[planned]` |
| CPU BLAS | **fast, measured** | 213 G MAC/s at n=1024, verified bit-exact vs naive O(n³) |
| Packed storage | **works** | 1.6 bits/trit; 20× smaller than FP32 weights |
| GPU backend | **algorithm verified, dispatch not built** | Shaders + emulator pass; no Vulkan dispatch |
| RTL / FPGA | **exists, unmeasured** | `hw/` present; no numbers on real hardware |
| MLIR dialect | **exists** | `include/Ternary/TernaryDialect.td`, `include/Ternary/TernaryOps.td` |
| SDK | **usable** | C ABI 1.3 + C++ + Python; Rust wrapper uncompiled |

The honest summary: **the CPU side is real and measured. Everything device-side
is scaffolding.** That is the correct order — a compiler should be trustworthy on
the machine you can actually test before it targets machines you cannot.

---

## 3. The Target Abstraction Contract

This is the concrete answer to "how does new hardware plug in". A backend
implements the following. It is written against what is already in the tree, so
it is a description of a seam, not a wish.

### 3.1 A backend declares its capabilities

```cpp
struct TargetCaps {
    bool     ternary_native;     // device computes on trits, not emulated in binary
    unsigned trits_per_word;     // e.g. 64 for our packed word, 5 for base-243
    bool     shader_int64;       // drives native vs portable GLSL selection
    bool     packed_native;      // can the device unpack base-243 itself?
    size_t   shared_memory_bytes;
    size_t   max_workgroup[3];
    bool     exact_int64_mac;    // if false, results are NOT bit-identical
};
```

`exact_int64_mac` is the important one. It is the difference between a backend
that preserves the library's central guarantee and one that does not, and it
must be declared rather than assumed.

### 3.2 A backend implements one primitive

Everything else can be composed from it:

```
mac_ternary(A : packed trits [M,K],
            B : packed trits [K,N],
            C : int64 [M,N]) -> exact, or error
```

That is the whole contract. `blas::ternary_gemm` is the CPU implementation;
`ternary_gemm_glsl_native` / `_portable` / `_packed` are the device
implementations `[built, algorithm verified, dispatch not built]`.

### 3.3 The existing seam

- `bt::blas::Device` (`TernaryMicroKernel.hpp`) — Host / Vulkan / Cuda / Metal.
  Today only Host is implemented; the other three are the dispatch points.
- `lib/SDK/VulkanRuntime.cpp` — the one real device runtime. It already shells
  out to `glslangValidator` and `spirv-val` and reports `native64` from
  `VkPhysicalDeviceFeatures::shaderInt64`. **Caveat:** `VulkanProgram::State`
  allocates only 3 descriptor bindings and a 4-byte push-constant range, so it
  cannot host the 5-binding / 20-byte GEMM shader as written. That must be
  widened before the GEMM shader can run through it.
- `ternary_xbt_blas.hpp`, `hw/`, `iree/` — further lowering surfaces.

### 3.4 Non-negotiable: validate against the CPU

Every backend must pass `validate_gpu_vs_cpu()` before any number is published.
Because the ternary MAC is exact, device and host results must be **bit-identical**.
Any mismatch is a bug, never floating-point tolerance. A backend that cannot
pass this does not get a benchmark number; it gets an investigation.

---

## 4. Where the advantage actually is

The single most important strategic point, and the one most likely to be
misunderstood:

> **Ternary does not win on raw compute. It wins on memory bandwidth.**

A ternary weight is 1.6 bits packed, 2.0 dual-rail, versus 32 for FP32. That is
a **20× reduction in bytes moved**, and modern accelerators are bandwidth-bound,
not FLOP-bound. So target large `n`, where the operands no longer fit in cache
and the memory system is the bottleneck. At small `n` the packed path is
measurably *slower* than dual-rail on CPU (49 vs 213 G MAC/s at n=1024), because
decompression costs more than the traffic it saves. Anyone quoting ternary
speedups at n=256 is measuring the wrong regime.

---

## 5. Milestones

Ordered so that each one is testable before the next is started.

**M1 — Make the CPU path unarguable.** *Mostly done.*
Measured, verified bit-exact, benchmark harness that cannot lie in place.
Remaining: Lean job in CI; fix the `GemmStats::gmacs`/`ms` fields that are
declared but never populated.

**M2 — One real device backend.** *In progress.*
Shaders written and algorithm-verified by emulation. Needs: Vulkan headers,
widened `VulkanProgram` bindings, `glslangValidator`, and a GPU. Until the
emulator's algorithm runs on silicon, GPU numbers stay unpublished.

**M3 — Packed weights end-to-end on device.** *Design complete.*
1.6 bits/trit in HBM, unpacked in shared memory. Verified by emulation at
20× smaller than FP32. Needs M2.

**M4 — BitNet-class inference.**
Ternary weights × INT8 activations is the near-term application that matters,
because it runs on today's hardware. Currently 2.03× faster than the original
kernel, verified.

**M5 — A target backend that is not Vulkan.**
The real test of the abstraction in §3. If a second backend plugs in without
touching the IR, the architecture is right. If it needs IR changes, it is not.

**M6 — Translation validation.**
`CoreLowering.v` currently admits to its own unverified status. Closing that is
what makes the compiler trustworthy enough to put above a fab run.

---

## 6. What this project is deliberately not

- **Not a faster binary compiler.** If ternary has to be emulated in binary, the
  density argument evaporates. Where that is unavoidable, say so.
- **Not a benchmark project.** Numbers exist to catch regressions and to choose
  between designs, not to win comparisons. The harness refuses to print a number
  it has not verified.
- **Not claiming hardware it has not measured.** `hw/` exists; no claim about it
  should be made until it runs on a device.

---

## 7. One-paragraph summary

Trium XBT aims to be the compiler layer for ternary-native and three-state
nanoscale hardware. The argument is not that ternary arithmetic is faster in the
abstract — on binary silicon it often is not — but that post-CMOS substrates
present three states natively, that ternary representation is ~37% denser on the
wire and 20× denser in memory than FP32, and that no mainstream compiler can
express any of this. The immediate work is unglamorous and deliberate: make the
CPU path measured and trustworthy first, add one real device backend second, and
let the abstraction be judged by whether a *second* backend can be added without
touching the IR.
