# Existing-hardware portability, integration maturity, and the separate hardware track

Date: 2026-10-29. User-directed priority clarification.

## Standing project direction

**Stage 6 is a separate, long-term native ternary/mixed-radix hardware track.**
Keep and develop it: executable ISA/memory specifications, simulation, RTL,
co-simulation, synthesis and eventual physical hardware qualification. A working
software compiler backend is not proof of a synthesizable processor, timing/area/
power properties, or fabricated hardware correctness.

Before making that track the main implementation focus, mature the mathematical
foundation, compiler integration and execution on existing binary CPUs/GPUs.
Native ternary hardware is not a prerequisite for useful software on present-day
machines. Preserve shared semantics between the tracks without conflating their
readiness or requiring hardware completion for a software release.

This is a direction/acceptance note, **not newly implemented GPU functionality**.
Current executable evidence remains in [STEPS_1_TO_5_STATUS.md](STEPS_1_TO_5_STATUS.md).
README changes remain deferred.

The next detailed proposal is [SDK_MATH_GPU_IMPLEMENTATION_PLAN.md](SDK_MATH_GPU_IMPLEMENTATION_PLAN.md),
supported by [SDK_GPU_RESEARCH.md](SDK_GPU_RESEARCH.md). The user approved R1/R2. Implementation and current evidence are now recorded
in [R1_R2_DELIVERY.md](R1_R2_DELIVERY.md); the maturity tables below describe the
pre-implementation planning baseline.

## What the existing-hardware project can become

A reusable mathematical compiler/runtime stack for declared balanced-ternary and
numeric workloads, usable through both its own language and host applications:

```text
.bt source / explicit host or framework adapter
  → checked shared semantics and core IR
  → portable kernel representation and target legalization
  → CPU or supported GPU backend
  → capability-aware runtime, diagnostics and measured optimization
```

Its useful scope can grow from scalar/map kernels to reductions, tensor storage,
dot products and sparse operations. Each extension needs explicit shape/layout,
arithmetic, ordering, ownership and failure contracts. Mathematical library
coverage alone does not establish source-language or GPU coverage.

This does not automatically make it a replacement for GCC/Clang, an upstream
LLVM backend, a Python compiler, or a drop-in compiler for arbitrary applications.
Each external language/framework needs an adapter and tested semantics. A workload
must fit the declared profile; unsupported behavior must be rejected or handled
through an explicitly selected, semantically compatible fallback.

## Present maturity versus intended maturity

| Area | Current evidence | Needed for broader integration |
|---|---|---|
| Scalar CPU compilation | Real Linux x86-64 execution, shared scalar LLVM route and test/proof evidence | More named toolchains/physical platforms; wider fuzzing and regression campaigns |
| Mathematical library | Checked integer/rational/BTFP APIs with documented exactness/capacity boundaries | Versioned packaging, client qualification and maintained compatibility policy |
| Application embedding | C runtime/Python integration tested; Rust checked APIs/CI authored | Execute Rust CI; define a supported compiler/runtime SDK, ownership/lifetimes, loading and concurrency contracts |
| GPU execution | Restricted CUDA map emission and real-driver runtime implemented; offline tests pass | Actual device qualification, then a vendor-neutral backend/runtime and per-device support matrix |
| Optimization | Conservative opt-in passes; matched-boundary measurements include mixed outcomes | Representative application benchmarks, cost models and correctness-checked tuning |
| Larger workflows | Some aggregate/library compatibility functionality exists | Shared-core aggregate/tensor lowering, layout/aliasing contracts, reductions and explicit framework adapters |
| Production operation | Bounded frontend, diagnostics and sanitizer coverage | Fuller debugging, packaging/deployment, release policy, long-running reliability and resource testing |

Today the project is an executable, tested foundation for scoped use and
integration experiments. It is not yet a broadly qualified general-purpose
production compiler or stable universal backend SDK.

## Three different GPU goals

### 1. Cross-vendor portability

Run the **same supported workload** on different GPU vendors. The next proposed
route is Vulkan compute/SPIR-V, retaining CUDA as an optional specialized backend.
Apple support may require a translation layer or a dedicated Metal route and its
own feature qualification. Actual compatibility depends on device features,
operating system, drivers and backend implementation.

First extract the selected pure int39 map subset into a portable kernel contract;
then implement SPIR-V lowering, buffer/dispatch/runtime ownership and capability
checks. Do not start by promising unrestricted scalar programs on every GPU.

### 2. Capability-aware optimization

Discover supported integer widths, subgroup/workgroup limits and memory features.
Choose only legal representations and kernel configurations. For example, int39
needs suitable 64-bit operations or an explicitly implemented checked multiword
representation; a 32-bit device must not silently truncate it.

Optional autotuning may benchmark valid alternatives and cache the best observed
choice by workload, device, driver and compiler/profile version. Validate results
against the reference before accepting a candidate. Preserve overflow, rounding,
selection laziness and ordering rules. Never imply that automatic tuning proves
optimality or that it can be fully qualified without actual device measurements.

### 3. Multiple GPUs executing one workload

This is distinct from cross-vendor portability. It needs partitioning, per-device
capabilities and compilation, placement/transfers, scheduling, synchronization,
result collection and failure handling. Cross-vendor memory/peer-transfer behavior
cannot be assumed. Reductions need an explicit order and numerical policy.

First qualify single-device portability. Then add explicit multi-device execution,
including heterogeneous-device tests. Distributed multi-machine execution is a
further project, not an automatic consequence of multiple local GPUs.

## Acceptance and evidence boundaries

- Describe support as **all devices meeting a named profile with qualified
  backend/driver combinations**, not literally every GPU.
- Report compilation, software execution and physical GPU execution separately.
  Software Vulkan can exercise an API/runtime path but cannot certify a physical
  GPU's behavior, throughput, memory limits or performance.
- Publish a device/driver/feature matrix and reproducible reference comparisons.
- Missing features produce a clear diagnostic or an explicitly reported compatible
  fallback; never label CPU execution as GPU execution.
- Measure compilation, transfers, synchronization and kernel time separately.
- Treat open-source licensing and dependency compliance separately from hardware
  compatibility; an open compiler can still depend on proprietary device drivers.

## Development order before the main Stage 6 hardware push

1. Qualify the current runtime/client paths and define a versioned embedding SDK.
2. Extract a portable GPU map representation from the existing shared semantics.
3. Implement and validate Vulkan/SPIR-V plus capability-driven legalization.
4. Qualify actual single devices as hardware becomes available; keep unavailable
   configurations explicitly unqualified.
5. Add reference-checked tuning, representative workloads and deployment tooling.
6. Expand aggregate/reduction/framework integration according to concrete use cases.
7. Add explicit multi-device scheduling after the single-device foundation works.

Continue preserving/developing the independent Stage 6 specification/simulation/
RTL/synthesis work, with its own milestones and evidence. Neither software GPU
portability nor hardware synthesis should be reported as completion of the other.
