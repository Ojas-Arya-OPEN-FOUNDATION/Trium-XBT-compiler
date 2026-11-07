# Scaling Plan — vector/tensor IR, memory, scalar multiply, and two-track hardware support

**Status: PLANNING DOCUMENT. Nothing here is implemented yet.**

This is the detailed design for the work requested on 2026-11-07:

1. Scalar multiply (currently refused by the Xbt selector)
2. Vector and tensor types in the core IR
3. A memory management system
4. **Two tracks** — Track A for present-day CPU/GPU silicon, Track B for
   next-generation hardware (not only new RISC-V: any future target, plus
   simulators and emitted backends)
5. Scale-up: parallel development, so new hardware can adopt this compiler
   without editing it
6. Existing hardware: expose *advanced* capabilities that match modern
   workflows
7. SDK development

Each improvement below is specified with: the problem as it stands today, the
design, the concrete changes, how it will be verified, and its exit criteria.

---

## 0. Where the project stands (ground truth, verified)

| Area | State |
|---|---|
| CI | 8/8 jobs green (`mlir-dialect`, `debug`, `release`, `sanitizers`, `sdk-vulkan`, `xbt-verification`, `rust-client`, `native-rocq`) |
| Core IR types | `Type { Int39, Trit, Positional39, Packed64, Float64, Void }` — **scalar only** |
| Core IR ops | 26 ops; no vector/tensor/memory op |
| Xbt ISA | 12 ops: `BTHA BTFA KSADD DOT64 UNPACK243 NEG CSAMAC MUL PACK243 LI5 SIGN CMOV` |
| Xbt selector | Rule table; handles constants, params, `IntCast`, neg, add, sub, single-trit mul, 6 comparisons, `<=>`, `select`. **Refuses general multiply.** |
| Target abstraction | `TargetDescription` (capability fields) + `Backend` virtual interface + `BackendRegistry` — the plug-in seam exists |
| Selector checks | 332 (lowered code executed on the hardware model, compared to the interpreter) |
| Z3 proofs | 19/19, incl. `SIGN` exhaustive ≤10 trits + symbolic at 64 trits |
| Layouts | 4 specified (dual-rail, base-243, T-CSR, BTFP); **zero conformance vectors** |
| Memory | Ad-hoc `std::vector` allocation in 6+ places; no ownership model |
| Hardware | **Nothing has run on real hardware.** GLSL validated, not executed; no FPGA bring-up |

---

## 1. The one architectural idea everything else follows from

**Extensions must be data, not edits.**

The selector rewrite in this session is the template. It was a `switch` with 6
cases; it is now a rule table (`SelectionRule{op, name, cost, matches, emit,
reason}`). Adding `<=>` and `select` meant *appending entries*, not touching the
driver. That is also why the six comparisons are produced by a single factory
rather than written six times — which is precisely what allowed one bug
(inverted `!=`) to exist in one place and be fixed in one place.

Every item below is designed so that adding a type, op, layout, target, or
backend is **appending a declaration**, with the driver never edited.

---

## 2. Two tracks, one IR

The request is explicit that these are two *parallel* efforts. They are not two
compilers — they are two backends (plus two SDK runtimes) over one IR.

```
                    .bt source
                         │
                 core IR (scalar + vector + tensor + memory)
                         │
              ┌──────────┴───────────┐
              │                      │
     ┌────────▼─────────┐   ┌────────▼──────────────┐
     │  TRACK A         │   │  TRACK B              │
     │  present silicon │   │  next-gen hardware    │
     ├──────────────────┤   ├───────────────────────┤
     │ x86_64 AVX-512/2 │   │ riscv64-xbt (today)   │
     │ aarch64 NEON     │   │ capability-contract   │
     │ scalar fallback  │   │   backends (future)   │
     │ Vulkan / Metal   │   │ simulator/emulator    │
     │ CUDA (offline)   │   │   plug-in interface   │
     │LLVM IR / obj/asm │   │ emitted backend       │
     └──────────────────┘   │   skeletons           │
                            └───────────────────────┘
              │                      │
              └──────────┬───────────┘
                         │
              shared: layout conversion, memory model,
                      translation validation, Z3 specs
```

**Track A** answers: *how do I make ternary workloads fast on the silicon
everyone already owns, in the workflows they already use?* Everything here is
binary, runnable today, and measurable.

**Track B** answers: *how does a hardware team — for any future device, not
just another RISC-V — adopt this compiler in a day rather than a quarter?*
Everything here is a contract plus generated scaffolding, so a new target needs
**zero compiler edits**.

Both tracks consume the same type system, memory model, layout conversions, and
proof infrastructure. That shared spine is built first.

---

## 3. The shared spine

Both tracks are blocked on the same three foundations. These come first.

### 3.1 Vector and tensor types in the core IR

**Problem.** `Type` is a flat scalar enum. There is no way to express "256
trits", "a 4×8 matrix of Int39", or "a batched tensor in base-243 layout". This
is, per the existing audit, *the prerequisite for expressing accelerators
generically* — it gates Track B entirely, and it gates any serious GEMM work in
Track A.

**Design.** Do **not** turn `Type` into a class — that would touch every
`switch` in the tree. Keep `Type` as the *element* kind and add a parallel
shape-carrying type:

```cpp
// include/Ternary/IR/Core.hpp
struct TensorType {
    Type element{Type::Int39};
    std::vector<int64_t> dims;   // empty = scalar, {n} = vector, {m,n,...} = tensor
    Layout layout{Layout::DualRail};   // storage format, see 3.4

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    int64_t num_elements() const;      // product of dims, -1 if dynamic
    bool is_scalar() const { return dims.empty(); }
    bool conforms_to(const TensorType&) const;  // same element + dims
};
```

`Expr` gains a `TensorType type2` (or the existing `type` is widened via a
`Shape` side-table indexed by `Expr*`, if ABI churn matters — **decide during
implementation**, preferring the side-table only if profiling shows the struct
growth is costly; correctness is identical).

New op family, registered as data in the rule tables rather than as driver
cases:

| Op | Semantics |
|---|---|
| `Broadcast` | scalar → shaped, or shape promotion |
| `Elemwise{Add,Sub,Mul,Neg,...}` | elementwise over existing scalar ops |
| `Reduce{Add,Mul,Max}` | reduce along axes |
| `Matmul` | `TensorType` GEMM |
| `Transpose` / `Reshape` | view/permutation; must specify whether it copies |
| `LayoutConvert` | explicit, checked format change (see 3.4) |

**Verification.**
- Z3: shape/broadcast/transpose algebra — ranks, index arithmetic, and
  `reshape ∘ reshape` composition.
- Conformance: elementwise and reduce against a naive reference, differential,
  across all 4 layouts.
- Round-trip: `LayoutConvert(a→b)` then `LayoutConvert(b→a)` must be identity
  for every layout pair — **exhaustive for small shapes**, which also finally
  delivers the "zero conformance vectors" the audit flags.

**Risks.** Dynamic shapes (data-dependent dims) are a large scope increase.
**Recommendation: static shapes only in phase 1**, with dynamic shapes deferred
and explicitly out of scope for now.

**Exit criteria.** A `.bt` program can declare and compute on a vector and a
2-D tensor; it lowers through the dialect; layout round-trips are proved; the
scalar path is unchanged (68/68 still green, no regressions).

### 3.2 Memory management system

**Problem.** Allocation is ad-hoc `std::vector` in at least six translation
units (`Blas/TernaryGemmGpu.hpp`, `IR/GLSL.hpp`, `IR/LLVM.hpp`,
`IR/Program.hpp`, `Math/Interchange.hpp`, the Xbt selector). There is no
ownership model, no device awareness, and no lifetime information for a
lowering to schedule against. Track A cannot do async GPU transfers or
zero-copy without it; Track B cannot target an accelerator with a scratchpad.

**Design.** Three layers, introduced bottom-up.

*(a) Host allocator* — arena/pool with explicit lifecycles, so lowering can
reuse scratch instead of allocating per op:

```cpp
namespace bt::mem {
class Arena {                      // bump allocator, reset() not free()
    void* allocate(size_t bytes, size_t align);
    void reset();                  // O(1) bulk reclaim
};
class Pool { ... };                // fixed-size block pool for hot temporaries
}
```

*(b) Buffer abstraction* — the thing both tracks share:

```cpp
struct BufferDesc {
    uint64_t     handle;
    TensorType   type;             // element, dims, layout
    Device       device;           // Host / GPU / Accelerator(id)
    size_t       bytes;
    Lifetime     lifetime{ ... };  // first-use .. last-use, for reuse
};
```

*(c) IR memory ops* — made explicit so lowering can schedule and prove:

| Op | Purpose |
|---|---|
| `Alloc` | create a buffer on a device |
| `Dealloc` / lifetime-end | release |
| `Copy` | device→device (incl. host↔GPU staging) |
| `View` | non-owning re-interpretation |
| `LayoutConvert` | conversion that may or may not allocate |

**Verification.**
- Leak/double-free: run the whole suite under ASan **and** LSan; assert the
  arena returns to zero outstanding bytes at scope exit.
- Aliasing: a Z3 model of `View` aliasing rules; prove no two mutable views
  alias unless explicitly permitted.
- Reuse correctness: differential test — results must be bit-identical with
  reuse enabled and disabled.
- Track A: transfer correctness across devices (host↔GPU), verified by
  readback comparison.

**Risks.** Making memory explicit in the IR is invasive; it touches every
lowering. Mitigate by introducing `Alloc`/`Copy` as **optional** metadata first
(lowering may ignore them), then making them mandatory once all backends honour
them.

**Exit criteria.** Every backend allocates through the shared allocator; the
suite is ASan/LSan-clean; host↔GPU transfer verified by readback; no
performance regression on the GEMM benchmarks.

### 3.3 Translation validation

**Problem.** Each lowering is tested end-to-end, but no pass *proves* its
output equals its input. Test cost grows quadratically with passes and targets.

**Why it is in the spine.** This is the highest-leverage investment in the
whole plan. Without it, every new op, pass, layout, and target needs hand-written
tests — which is exactly the cost that makes "scale up" fail. With it, coverage
stops being the bottleneck, and it reuses the Z3 infrastructure that already
exists.

**Design.** Each pass emits a proof obligation:

```
pass P:  IR_before  ──P──▶  IR_after
obligation:  ∀ inputs. eval(IR_before)(inputs) == eval(IR_after)(inputs)
```

Discharged by Z3 where decidable, and by exhaustive/differential testing where
not. Obligations are emitted automatically by the pass harness, not written by
hand.

**Verification.** Start with the passes where the obligation is decidable
(layout conversion, constant folding, select-to-cmov). Report coverage: *N of M
passes have discharged obligations*, tracked in CI.

**Exit criteria.** Every pass in the default pipeline carries either a
discharged obligation or an explicit, reviewed "not yet provable" marker.

### 3.4 Layouts as a first-class, proved concept

Four layouts exist (dual-rail 2 bits/trit, base-243 1.6 bits/trit, T-CSR,
BTFP). The audit notes **zero conformance vectors**. Layout is the contract
between Track A (bandwidth-bound on real silicon) and Track B (native formats
differ per device), so it belongs in the spine.

**Design.** A single `Layout` enum + a `LayoutTraits` table (bits/trit, native
ops, alignment, conversion cost). `LayoutConvert` is an explicit IR op. Every
conversion pair gets a Z3 spec and generated vectors.

**Exit criteria.** Conformance vectors exist for all 4 layouts and all
conversion pairs; round-trip identity proved for every pair.

---

## 4. Improvement P1 — Scalar multiply

**Problem.** `xbt.mul` is lane-wise: it drops every carry crossing a lane
boundary. The selector *correctly refuses* general multiplication rather than
emit wrong code — but that leaves a real op unimplemented.

**Design — two paths, chosen from `TargetDescription`.**

`TargetDescription::has_scalar_mul` already exists and is `false` today. That
field becomes the switch:

- **Hardware has true multiply** (`has_scalar_mul == true`) → emit it directly.
- **Otherwise** → lower to a shift-add sequence over balanced ternary.

Shift-add needs a shift primitive, which the ISA lacks. Add:

```
SHL3 rd, rs, imm      rd = value(rs) * 3^imm      (shift by trit positions)
```

Balanced-ternary multiplication then decomposes as

    a * b = Σ_i  (trit_i of b) * (a << i)        for each nonzero trit i

with the sum accumulated by `ksadd`. Using **non-adjacent form (NAF)** —
digits in {-1,0,+1} with no two adjacent nonzeros, which balanced ternary
gives naturally — halves the number of partial products versus plain ternary
recoding and removes the need for a separate subtraction pass.

This is a new selector rule (appended, per the rule-table design), plus a new
ISA op with a Z3 spec, plus a hardware-model implementation.

**Concrete changes.**
- `docs/XBT_ISA_SPEC.md` — specify `SHL3` (and note `has_scalar_mul` routing)
- `hw/xbt_alu_model.hpp` — `XbtGroup2::SHL3` + execution semantics
- `verification/xbt_isa.py` — `SHL3` spec; `verification/verify_isa.py` — proof
- `include/Ternary/Target/XbtInstructionSelector.hpp` — one new rule, with a
  `reason` explaining the refusal case
- `test/Unit/test_xbt_selector.cpp` — execute on the model vs the interpreter

**Verification.**
- Z3: `SHL3` is exact multiplication by 3^k within range; the shift-add
  composition equals integer product — symbolic over full width, exhaustive for
  small widths.
- Selector: differential against the interpreter across a sweep including
  negative, zero, and boundary (Int39 min/max) operands.
- Overflow: must **fail loudly**, not wrap — consistent with the existing
  `__int128` overflow-check philosophy in the reference `matmul`.

**Risks.** 40-trit shift-add is ~40× slower than a hardware multiplier; that is
acceptable for correctness but must be *reported*, and is exactly why
`has_scalar_mul` routing matters — a target with real multiply never pays it.

**Exit criteria.** General multiply compiles and is proved correct on the
model; overflow is detected; targets with `has_scalar_mul` emit the native
instruction.

---

## 5. Improvement P2 — Track A: present-day CPU and GPU

Goal: *make ternary workloads fast on silicon people already own, in workflows
they already use, with advanced capabilities.*

### 5.1 Runtime dispatch on CPU

**Today:** AVX-512 / AVX2 / NEON / scalar paths exist as separate emission
targets chosen by flag.

**Design:** one binary that detects capabilities at runtime and dispatches, with
a scalar reference always present as the fallback and as the differential oracle.

**Verification:** every SIMD path differentially tested against the scalar
reference; CI runs all paths (and a forced-scalar run to keep the oracle
honest).

### 5.2 Real GPU execution

**Today:** GLSL is *validated, not executed*. `SDK_Vulkan_Execution` is the only
GPU test that runs, and there is no GPU on a hosted runner.

**Design:** close the loop in three steps, in order of increasing evidence:
1. **Offline** (exists): validate SPIR-V/GLSL.
2. **Software execution**: run the generated compute shader on a software
   rasteriser/compute emulator so CI can execute it on a CPU-only runner.
3. **Device**: real-device qualification, gated and reported honestly (never
   claimed as passing when it was skipped).

**Verification:** readback comparison against the CPU reference; numerical
identity (ternary GEMM is exact, so results must match **bit-for-bit**, which
is a much stronger check than floating-point tolerance).

### 5.3 Advanced capabilities on binary hardware

This is the part of the request about *"advanced capabilities that match modern
workflows"*. Concretely:

| Capability | Why it matters |
|---|---|
| Bit-packing at 1.6 bits/trit (base-243) vs 2 bits/trit (dual-rail) | Memory-bound inference: this is a **bandwidth** win, not just compute |
| Fused ops (GEMM + activation + quantisation in one kernel) | Avoids materialising intermediates |
| Batched / strided GEMM | Real serving workloads |
| Async execution + explicit stream ordering | Overlap transfer and compute |
| Zero-copy / pinned buffers | Removes staging copies |
| Multi-device | Scale beyond one GPU |
| Quantisation-aware paths | Ternary is a quantisation target; the path in must be first-class |
| Profiling hooks | Users need to see where time goes |

### 5.4 Modern-workflow integration

Meet users where they are rather than asking them to adopt a bespoke toolchain:
- Stable C ABI (exists — keep it versioned and frozen)
- Python and Rust SDK parity (Python now has auto-routing `matmul` and pack
  helpers; Rust has `pack_matrix_*`; **close the remaining gaps so all three
  languages expose the same surface**)
- A framework dispatch layer, so ternary kernels can be selected underneath an
  existing tensor API
- Docs with runnable examples and honest benchmark numbers

**Exit criteria for Track A.** Runtime dispatch shipping; GPU path executed
(software runner in CI, device qualification reported honestly); batched,
async, and fused paths available from all three SDK languages; published
benchmarks with methodology attached.

---

## 6. Improvement P3 — Track B: next-generation hardware

Goal: *"not just new RISC-V, but compatibility with next-generation hardware,
simulators, and emitting backends."*

### 6.1 The capability contract

`TargetDescription` already holds the right instincts (`native_trit_width`,
`has_word_add`, `has_lane_mul`, `has_scalar_mul`, `has_popcount_dot`,
`native_layouts`, register counts). The move is to make it **complete** and
**machine-readable**, so a target is a file, not C++.

Target authors write a description (YAML/JSON); the compiler loads it. What it
must capture:

- Datapath: native trit width, which ops exist natively, their latencies
- Storage: native layouts, memory hierarchy sizes, bandwidth/level
- Registers: counts, whether a zero register is hardwired
- Encoding: instruction formats (arbitrary — **not** assumed RISC-V)
- Cost model: per-op cost, for the tiling/lowering decisions
- Verification hooks: which simulator/emulator validates this target

This directly generalises beyond RISC-V: a custom VLIW, an FPGA soft core, a
systolic array, or a non-von-Neumann device is a different description, not a
different compiler.

### 6.2 Backend generator

From a target description, **emit a backend skeleton**: the `Backend`
subclass, a `TargetDescription` loader, a `matches()` implementation, and —
critically — **a conformance test harness pre-wired to the shared vector
suite**. A hardware team gets a compiling backend and a failing test suite on
day one, then fills in instruction selection.

### 6.3 Simulator / emulator plug-in interface

Define a small interface any simulator implements:

```
load(program) → exec(trace) → read_state() / read_memory()
```

Then the *same* conformance vectors drive: the host C++ model (today), an
instruction-set simulator, a cycle-accurate RTL simulation, and eventually real
silicon. This is what makes the "nothing has run on real hardware" gap
closable, and it is how `hw/riscv_xbt_extension.sv` finally gets exercised in
CI (currently it is not simulated at all, and its known CSAMAC bug is unfixed).

### 6.4 Emitted backends

For targets that want a compiler of their own, emit a standalone backend
package (description + lowering rules + tests) rather than requiring them to
live in this tree. This is the "emitting backends" part of the request and is
what makes adoption non-invasive for both sides.

**Exit criteria for Track B.** A target can be added by writing a description
file with **zero compiler edits**; the generator produces a skeleton that
compiles and fails its conformance suite; at least one simulator other than the
host model is driven by the shared vectors; `--list-targets` shows descriptions
loaded from files.

---

## 7. Improvement P4 — Parallel development and vendor onboarding

The request is explicit that this should be *parallel*: new hardware teams
working concurrently without colliding.

**What makes that possible:**
1. **Zero-edit target addition** (§6.1–6.2) — no shared file to conflict on
2. **Shared conformance suite as the contract** — a vendor's deliverable is
   "my target passes the suite"; CI runs per-target jobs independently
3. **Frozen, versioned interfaces** — `TargetDescription`, `Backend`, the C
   ABI, and the simulator interface get SemVer; breaking changes require a
   major bump
4. **Per-target CI isolation** — adding a target adds a job; it cannot break
   another target's job
5. **A target authoring guide** — description file → generator → conformance →
   upstreaming checklist

**Deliverable:** `docs/TARGET_AUTHORING.md` plus a worked example target that
is deliberately trivial, so the path is demonstrable end to end.

---

## 8. Improvement P5 — SDK development

Current surface: C ABI, Python (`trium`) and Rust (`bt_runtime`) bindings.
Python gained `pack_matrix_a/b`, `is_ternary`, and auto-routing `matmul` this
session; Rust gained `pack_matrix_a/b` and now re-exports `encode_gemm`.

**Plan:**
1. **Parity matrix** — one table in the docs listing every capability ×
   {C, Python, Rust}, with gaps filled. Treat asymmetry as a bug.
2. **Buffer/device APIs** — expose the §3.2 memory model through all three
   languages: allocate, transfer, async, lifetime.
3. **Error handling** — consistent, typed errors with actionable messages
   (the selector's new per-rule `reason` is the model to follow).
4. **ABI stability** — versioned headers, deprecation policy, and an ABI
   regression test in CI (one exists; extend it).
5. **Docs + examples** — every public API has a runnable example that CI
   executes, so docs cannot rot.

**Verification:** each SDK function has a differential test against a
reference; the parity matrix is enforced by a CI check that fails when a
capability exists in one language but not another.

---

## 9. Phasing and dependency graph

Work is sequenced so value lands incrementally and the two tracks can proceed
genuinely in parallel once the spine exists.

```
Phase 0 — SPINE (blocks everything)
  3.1 vector/tensor types ─┐
  3.2 memory system       ─┼─▶ both tracks unblocked
  3.4 layout conformance  ─┘
  3.3 translation validation (start here; pays off continuously)

Phase 1 — P1 scalar multiply (self-contained; exercises the rule table;
          closes a known refusal; independent of the spine)

Phase 2 — TRACK A ∥ TRACK B  (parallel, once Phase 0 lands)
  A: dispatch → GPU execution → advanced caps → workflows
  B: capability contract → generator → simulator interface → emitted backends

Phase 3 — P4 vendor onboarding + P5 SDK maturation
          (depends on both tracks having stabilised interfaces)

Phase 4 — Hardware reality: real-device qualification, RTL simulation in CI
```

**Suggested start order** (each item independently shippable):

| # | Item | Why here |
|---|---|---|
| 1 | Translation validation scaffolding | Highest leverage; cheaper if it exists before the pass count grows |
| 2 | Scalar multiply | Self-contained win, proves the rule-table design scales |
| 3 | Tensor types | Gates Track B; large but well-understood |
| 4 | Memory model | Gates GPU async and accelerator scratchpads |
| 5 | Track A dispatch + GPU execution | Closes the "validated, not executed" gap |
| 6 | Track B contract + generator | The parallel-development unlock |
| 7 | SDK parity + docs | Adoption |

---

## 10. Verification standard (applies to every item)

Non-negotiable, carried over from the standing requirements on this project:

- **Proofs are exhaustive per case where decidable**, not sampled — across trit
  mathematics, language semantics, and hardware behaviour *simultaneously*.
- **Differential testing against a reference**, never self-consistency.
- **Negative controls**: a test that cannot fail is not a test. (The GEMM
  layout tests assert the wrong layout *is* detected as wrong; every new
  conformance suite needs the same.)
- **No speculative hardware claims.** A green CI job means host C++ and proofs
  are sound. It says nothing about `hw/`, the dialect, or any device. Say so.
- **Honest skipping.** Tests that cannot run in an environment return skip
  codes and are *reported as skipped*, never as passing.

---

## 11. Risks and explicit non-goals

**Risks**

| Risk | Mitigation |
|---|---|
| Tensor types touch every `switch` in the tree | Element/shape split (§3.1) keeps `Type` intact |
| Dynamic shapes explode scope | Static shapes only in phase 1; deferred explicitly |
| Memory ops invasive across all backends | Introduce as optional metadata first, mandatory later |
| Translation validation undecidable for some passes | Report coverage; allow reviewed "not yet provable" markers |
| Track B over-engineered ahead of any real target | Ship the contract + generator + one trivial example target, then let real targets drive it |
| Verification cost grows with the plan | This is exactly what §3.3 exists to prevent |

**Explicit non-goals (say them now to keep scope honest)**

- No claim of hardware support until something runs on hardware.
- No dynamic-shape support in phase 1.
- No new precision formats beyond what exists, until BTFP conformance lands.
- No performance numbers without methodology attached.
- Not fixing the known `CSAMAC` bug in `hw/riscv_xbt_extension.sv` as part of
  this plan — but §6.3 (RTL simulation in CI) is what would catch it, and it is
  the reason that item is in the plan.

---

## 12. Open questions to settle before implementation

1. **Shape representation**: new field on `Expr` vs a side-table — decide by
   measuring struct growth and compile-time impact.
2. **Static vs dynamic shapes**: confirm static-only for phase 1.
3. **Shift-add vs NAF only**: confirm NAF is worth the extra ISA surface, or
   start with plain shift-add and add NAF as an optimisation.
4. **Target description format**: YAML vs JSON vs TableGen — TableGen is
   already a dependency for MLIR builds and would unify the two, but adds a
   build-time dependency for non-MLIR builds.
5. **Simulator interface scope**: how much state must a simulator expose
   (registers only, or memory + traces)?

---

*Plan written 2026-11-07. Implementation begins on your go-ahead, in the order
in §9 unless you redirect it.*
