# Native tri-state readiness audit and retargeting roadmap

**Date:** 2026-11-04
**Scope:** every file in the repository, read directly. No claims here are inferred
from documentation alone; each is backed by the file and line evidence listed.
**Status:** the analysis below is as originally written. Phases 0–3 have since
been implemented; see the completion note at the end for what changed and what
is still open. The findings are left in place because the reasoning behind the
roadmap is still the reasoning behind the code.

This answers three questions:

1. Do we have the IR, formal proofs, instruction sets and memory layouts needed to
   drive native tri-state hardware?
2. How hard is it to retarget this compiler to a new machine today?
3. What is the roadmap to make it genuinely modular — able to grow into arbitrary
   future hardware?

---

## 1. Direct answer

**You have the mathematical and semantic foundation for native tri-state hardware,
and it is better than most compilers at this stage. You do not yet have a
retargetable compiler.**

Those are two different problems, and the repo currently solves the first much
better than the second.

What is genuinely strong:

- **61 machine-checked Rocq theorems with zero `Admitted` / `Axiom`** covering
  balanced-ternary algebra end to end: the trit↔dual-rail isomorphism, negation,
  multiplication, the half-adder (`btha`), the full adder (`btfa`), and base-243
  packing. This is a real algebraic foundation, not a sketch.
- **A defined instruction set** — the RISC-V `Xbt` Custom-0 extension, 6
  instructions, with an encoder, a decoder, a disassembler and SystemVerilog RTL.
- **Four specified memory layouts** plus a wire format.
- **A machine-readable core profile** (`specs/core-profile-v0.1.json`) with 27
  operations, numeric limits, and per-platform qualification matrices.

What blocks "build any kind of hardware":

- **The compiler cannot emit the instruction set you designed.** `--target=riscv64-xbt`
  is parsed, a flag is set, and then the compiler throws
  `RISC-V Xbt instruction selection is not implemented; refusing stub output`.
  That honest refusal is the right behaviour, but it is the whole implementation.
- **There is no backend interface.** No virtual dispatch, no registry, no target
  description. Adding a target means hand-editing ~9 separate regions of
  `bt_compiler.cpp`.
- **The `.bt` frontend and the MLIR `bt` dialect are two unconnected compilers.**
  This is the most surprising finding in the audit and is expanded in §3.1.
- **There are no conformance vectors.** Nothing mechanically binds a future
  hardware implementation to the semantics the compiler and the proofs assume.
- **The IR is purely scalar.** It has no vector or tensor type, which is a
  structural ceiling for any accelerator target.

The single highest-leverage change is not writing more backends. It is **wiring the
existing `.bt` frontend into the existing MLIR dialect**, and then putting a
target-description layer underneath it. Both halves already exist; they simply have
never been connected.

---

## 2. Inventory: what exists

### 2.1 Compiler IR — real, verified, but scalar-only

`include/Ternary/IR/` is 1,397 lines across six headers (`Core`, `Execute`,
`GLSL`, `LLVM`, `Optimize`, `Program`). Two levels:

| Level | File | Role |
|---|---|---|
| Value IR | `IR/Core.hpp` | expression graph, typed |
| Statement IR | `IR/Program.hpp` | functions, control flow, blocks |

The value IR has **26 operators** and **6 types**:

```c++
enum class Type { Int39, Trit, Positional39, Packed64, Float64, Void };
enum class Op   { Integer, Floating, Packed, Parameter, Add, Sub, Mul, Negate,
                  IntToFloat, Equal, NotEqual, Less, Spaceship, SignSelect, IntCast,
                  Greater, LessEqual, GreaterEqual, DigitAnd, DigitOr, DigitXor,
                  Call, Print, Sqrt, Abs, Fma, Intrinsic };
```

It is backed by a formal contract (`specs/core-profile-v0.1.json`, 27 operations)
and by an independent exact reference used for differential testing. This layer is
solid and is the correct place to build on.

**The limitation that matters for hardware:** every type is a scalar. There is no
vector, no tensor, no matrix type anywhere in the IR. Any accelerator — your own
`xbt_planar_dot64` included — is fundamentally a data-parallel machine, and an IR
that cannot express "apply this to 64 lanes at once" forces every backend to
re-discover parallelism by pattern-matching scalar loops. That is fragile, and it
is why the GPU path today special-cases whole named kernels rather than lowering
generically.

### 2.2 Formal proofs — strong foundations, narrow coverage

`proofs/rocq/`, four files, **61 theorems/lemmas, 0 `Admitted`, 0 `Axiom`**:

| File | Lines | Items | Covers |
|---|---|---:|---|
| `BalancedTernaryFoundations.v` | 421 | 28 | trit↔rail isomorphism, negation, multiplication, `btha`, `btfa`, base-243 packing, symplectic reversibility |
| `CoreMath.v` | 153 | 16 | integer/rational/interval arithmetic |
| `ImplementationBridge.v` | 108 | 10 | bridge from theorems to the implemented datatypes |
| `CoreLowering.v` | 74 | 7 | interval arithmetic for optimization legality |

The algebraic core is real and load-bearing. `rail_val_isomorphic`,
`rail_mul_assoc`, `btha_correct`, `btfa_correct`, `zero_gate_negation` are exactly
the theorems a ternary ALU must satisfy, and they are proven, not assumed.

**Two honest caveats, both stated in the sources themselves.**

First, `CoreLowering.v` opens with:

> *"Arithmetic obligations for core lowering/optimization. These lemmas justify the
> stated algorithms; they are not a proof of the C++/LLVM translator."*

So there is **no translation validation**: nothing proves that the C++ lowering
actually implements the semantics the theorems describe. The proofs constrain the
mathematics; they do not yet constrain the compiler.

Second, the bridge from theorem to silicon is unverified. `ImplementationBridge.v`
connects theorems to C++ datatypes, but no CI job checks the SystemVerilog RTL
against the theorems. The RTL and the proofs currently assert the same truths in
two different languages with no mechanical link between them.

### 2.3 Instruction set — defined and modelled, never emitted

`hw/xbt_alu_model.hpp` defines a complete, coherent ISA:

| `funct3` | Mnemonic | Semantics |
|---|---|---|
| `0b000` | `BTHA` | 64-trit dual-rail balanced ternary half-adder |
| `0b001` | `KSADD` | 64-trit 6-stage Kogge-Stone parallel-prefix adder |
| `0b010` | `DOT64` | 64-trit planar dot product → scalar register |
| `0b011` | `UNPACK243` | division-free base-243 byte → dual-rail unpack |
| `0b100` | `NEG` | 0-gate wire-swapped ternary negation |
| `0b101` | `CSAMAC` | fused carry-save MAC |

RISC-V R-type, opcode `0x0B` (custom-0). The header provides
`encode_xbt_r`, `decode_xbt_r` and a disassembler emitting `.insn r` directives.

Matching SystemVerilog exists in `hw/riscv_xbt_extension.sv`: `xbt_btha64`,
`xbt_kogge_stone_adder64`, `xbt_kogge_stone_adder64_v2`, `xbt_planar_dot64`.

**The gap is complete and unambiguous.** `bt_compiler.cpp:981`:

```c++
if (opts.enableRiscvXbt) {
    throw std::runtime_error("RISC-V Xbt instruction selection is not implemented; refusing stub output");
}
```

The compiler parses the target, sets the flag, then refuses. There is no assembler,
no instruction selector, and no lowering pass anywhere in the tree (searched for
`emit_xbt`, `XbtEmitter`, `bt-to-riscv`, `convert-bt-to-xbt` — none exist).

To be clear about what this is and is not: **refusing is correct engineering.** It
is far better than emitting a plausible-looking stub. But today the ISA is a
specification plus a behavioural model plus RTL, with no compiler behind it.

### 2.4 Memory layouts — four formats, one wire protocol, zero conformance vectors

| Layout | Bits/trit | Where | Status |
|---|---:|---|---|
| Dual-rail | 2.0 | `Math/Interchange.hpp:58` | proven isomorphic (`rail_val_isomorphic`) |
| Packed base-243 | 1.6 | `Balanced` packing, `base243_pack5_bounds` | proven |
| T-CSR (sparse) | — | `Math/Sparse.hpp` | unproven |
| BTFP float | — | `Math/BTFP.hpp`, `specs/ternary-float-v1.json` | unproven |

There is a real wire format, `Math/Interchange.hpp:93`:

> `BTR1`, width, 3 reserved zero bytes, positive LE64, negative LE64

`specs/core-profile-v0.1.json` additionally fixes numeric limits (int39 range,
import depth, token counts, nesting depth, per-platform qualification matrices for
Linux x86-64 C++/Python, aarch64/riscv, CUDA, Rust, Vulkan, SDK native).

**What is missing is the thing that makes layouts usable across hardware: conformance
vectors.** A search for shared golden vectors returned nothing. There is no
checked-in set of input/expected-output pairs that a new backend — or a new chip —
must reproduce. Every backend is currently validated by its own tests against its
own notion of correct. That works while there are four backends maintained by one
team. It does not work as a contract for arbitrary future hardware, and it is the
first thing a hardware partner would ask for.

---

## 3. The modularity problem, quantified

### 3.1 The surprising one: two unconnected compilers

The repository contains an MLIR dialect that looks like the answer to the
retargeting question — and it is not connected to anything that user code touches.

`include/Ternary/TernaryDialect.td` defines dialect `bt`, `cppNamespace = "::mlir::bt"`,
and `TernaryOps.td` defines **11 operations**:

```
BT_BTHAOp  BT_BTFAOp  BT_CSAAccumOp  BT_CSAReduceOp  BT_Select
BT_PlanarDotOp  BT_MatmulOp  BT_GemmOp  BT_PackBase  BT_Base  BT_UnpackBase
```

`tools/bt-opt/bt-opt.cpp` (311 lines) provides a real pass pipeline with **5
lowering passes**:

```
ConvertTernaryToLinalg   ConvertTernaryToVector   ConvertBTHA
ConvertBase243           ConvertCSAReduce
```

This is a genuine, well-shaped retargetable path. MLIR exists precisely so that a
new target is reached by writing a new lowering pass, not by editing a driver.

**But:** `grep -c "mlir" bt_compiler.cpp` returns **0**.

The `.bt` frontend never produces MLIR. `bt-opt` is fed exclusively by
hand-written `.mlir` files checked into `test/Dialect/`. The frontend path is:

```
.bt  →  AST  →  ir::Program  →  { C++20 | LLVM IR | GLSL }
```

and the MLIR path is:

```
hand-written .mlir  →  bt dialect  →  5 passes  →  linalg / vector
```

These never meet. And CI builds with `-DBT_ENABLE_MLIR=OFF`, so **the entire
dialect, all 5 passes, and `bt-opt` are never compiled or tested in CI.** The most
modular part of the codebase is the least exercised part of it.

This is the central finding of the audit. The retargeting infrastructure you want
has largely already been designed and written. It has simply never been connected
to the front door, and it is not being tested.

### 3.2 What adding a target costs today

There is **no polymorphic backend interface anywhere in `include/`** — no virtual
`emit`, no `class Backend`, no `struct Target`, no registry. Confirmed by search.

Instead, `bt_compiler.cpp:43` holds a closed enum:

```c++
enum class TargetArch { ARM64_NEON, X86_AVX512, APPLE_METAL, VULKAN_SPIRV,
                        CPP20_GENERIC, LLVM_IR };
```

with **33 references across ~9 distinct regions** of that one file:

| Lines | Region | What a new target must touch |
|---|---|---|
| 43–50 | enum definition | add an enumerator |
| 442–447 | help text | add a description string |
| 605–613 | `--target` parsing | add a name match |
| 627–668 | triple parsing | add substring→arch mapping |
| 698 | output-mode gate | decide if this target is "default" |
| 827, 840 | JIT/SPIR-V gates | decide capability participation |
| 941 | triple selection | add host-specific logic |
| 1064–1076, 1119 | emission dispatch | add a case + an emitter |

Five emitters exist today (`grep -c 'static std::string emit_'`), each a static
function on a `CodeEmitter` class.

So adding a backend = **one new enum value + edits in ~8 more places in a single
2,000-line file + a new emitter**, with no compile-time guarantee that you found
every site, and no way to add a target from outside the repository.

Two concrete symptoms of the same missing abstraction:

- **`APPLE_METAL` is declared but has no emitter.** There is no `IR/Metal.hpp` — the
  directory holds only `Core`, `Execute`, `GLSL`, `LLVM`, `Optimize`, `Program`. So
  `APPLE_METAL` is a first-class enum value, advertised in `--help` and parseable
  from `--target=apple-metal`, with nothing behind it. The enum, not the emitter,
  is the contract, and the contract lies.
- **Vulkan dispatch is hardcoded** to 3 bindings / 4 bytes
  (`include/Ternary/Kernel/Vulkan.hpp:104`), so shader layouts that do not fit that
  shape cannot be expressed.

---

## 4. Roadmap

Ordered by leverage, not by difficulty. Each phase closes a specific gap above.

### Phase 0 — Turn the lights on (days)

1. **Set `BT_ENABLE_MLIR=ON` in CI.** A code path that is never compiled is not a
   code path. This will almost certainly fail first; fixing those failures is the
   point. Nothing else in this roadmap should be trusted until the dialect builds
   and its tests run green on every push.
2. **Add a `hw` CI job** that at minimum compiles `hw/xbt_alu_model.hpp` against its
   tests and lints the SystemVerilog. Today `hw/` is not built by any job.

*Closes: the dialect is untested; `hw/` is unbuilt.*

### Phase 1 — Connect the two compilers (weeks)

This is the highest-leverage item in the plan.

3. **Emit the `bt` dialect from the `.bt` frontend.** Add a lowering from
   `ir::Program` to `mlir::bt`, so user code flows
   `.bt → ir::Program → bt dialect → passes → target`.
4. **Retire the duplicate path.** Once the dialect carries the frontend, stop
   hand-maintaining the direct `ir::Program → LLVM/GLSL` emitters and let them be
   *targets of the dialect* instead. This is what converts "6 hardcoded backends"
   into "one IR, N lowering passes".
5. **Add tensors to the IR.** Introduce vector/tensor types so parallelism is
   expressed in the IR rather than pattern-matched per backend. Without this, every
   new accelerator backend re-solves the same problem badly.

*Closes: §3.1 disconnection; the scalar-only ceiling; the 9-region dispatch problem.*

### Phase 2 — Make targets pluggable (weeks)

6. **Define a target-description contract.** A declarative description of a target:
   supported layouts (dual-rail / base-243 / T-CSR / BTFP), native trit width,
   available ops from the core profile, memory hierarchy, and capability flags
   (from the existing `bt_sdk_info` vocabulary). New hardware becomes *data*, not
   a code edit.
7. **Introduce a `Backend` interface** and a registry, and refactor the 9 dispatch
   regions in `bt_compiler.cpp` into calls against it. Backends then register
   themselves; the driver stops knowing their names.
8. **Build the conformance vector suite** — the single most important artifact for
   hardware partners. Checked-in input/expected-output pairs covering every
   operation in `specs/core-profile-v0.1.json`, at ordinary values, boundaries
   (`int39` min/max, base-243 edges, dual-rail `11` invalid rails) and failure
   cases. A new backend or a new chip qualifies by reproducing them.

*Closes: no backend interface; no conformance contract; hardcoded Vulkan shape.*

### Phase 3 — Close the loop to real silicon (months)

9. **Implement `Xbt` instruction selection.** Replace the honest throw at
   `bt_compiler.cpp:981` with a real pass from the `bt` dialect to `Xbt` machine
   instructions, reusing `encode_xbt_r`. Now the compiler emits the ISA you designed.
10. **Prove the RTL against the theorems.** Mechanically link
    `hw/riscv_xbt_extension.sv` to `BalancedTernaryFoundations.v`. Two practical
    routes: a Rocq-extracted reference model simulated against the RTL, or SMT
    equivalence checking per module. This is what makes "our chip implements our
    mathematics" a claim rather than an aspiration.
11. **Add translation validation.** Close the gap `CoreLowering.v` explicitly
    disclaims: check that each lowering result is semantically equivalent to its
    input, rather than only testing end-to-end outputs.

*Closes: ISA not emitted; RTL unproven; no translation validation.*

### Phase 4 — Hardware reality (ongoing, gated on evidence)

12. **FPGA bring-up of a small XBT accelerator**, per the existing two-track plan in
    `CHIP_HARDWARE_RESEARCH_AND_PLAN.md` — which already recommends starting with a
    controllable peripheral (`DOT`, carry-save, canonical add, negation, checked
    packing) rather than a whole CPU. That recommendation is sound and this roadmap
    defers to it.
13. **Get one real GPU execution end to end.** The Vulkan path compiles and degrades
    honestly (`rc=4`, "Vulkan loader unavailable") but has never executed on a
    device. GLSL is validated, not run. Nothing in this compiler has run on real
    accelerator hardware yet.
14. **Multi-machine tile data** before changing the autotuner default
    (`docs/TILE_DEFAULT.md`). Standing decision: no change without it.

---

## 5. What not to do

- **Do not add a 7th enum value.** Every target added to `TargetArch` before Phase 2
  deepens the problem the roadmap exists to solve.
- **Do not write a new backend before Phase 1.** The point of the MLIR work is that
  backends become cheap; writing one now pays the old price.
- **Do not claim hardware support from a passing CI job.** CI passing today means
  the host C++ and the proofs are sound. It says nothing about `hw/`, the dialect,
  or any device.
- **Do not let `BT_ENABLE_MLIR=OFF` stay the default.** It is the reason the best
  architecture in this repository is the least trustworthy.

---

## 6. Summary table

| Asset | Exists | Verified in CI | Ready for native tri-state |
|---|:---:|:---:|:---:|
| Value + statement IR (26 ops, 6 types) | yes | yes | partially — scalar only |
| Formal proofs (61 theorems, 0 admitted) | yes | yes | foundations yes; lowering no |
| Core profile spec (27 ops, limits) | yes | n/a | yes |
| Instruction set (RISC-V Xbt, 6 insns) | yes | no | spec yes; **emitter no** |
| RTL (4 SystemVerilog modules) | yes | **no** | no — unbuilt, unproven |
| Behavioural ISA model | yes | yes | yes |
| Memory layouts (4) + wire format | yes | yes | yes, unproven for sparse/BTFP |
| MLIR dialect (11 ops, 5 passes) | yes | **no** | designed for it; disconnected |
| Conformance vectors | **no** | — | **blocking** |
| Backend interface / registry | **no** | — | **blocking** |

**Bottom line.** The mathematics and the specifications are ahead of the
engineering. The fastest path to "any kind of hardware" is not to write more
backends — it is to connect the frontend to the MLIR dialect that already exists,
put a target-description layer under it, and pin the whole thing down with
conformance vectors. Phases 0–2 are weeks of work and unlock everything after.

---

## 9. Completion note (added 2026-11-04)

Phases 0–3 were implemented following this audit. Summary of what changed.

### Closed

| Gap (as found) | Now |
|---|---|
| `--target=riscv64-xbt` threw "not implemented" | Lowers core IR to encoded custom-0 instructions and executes them |
| No verification of the ISA | 17/17 Z3 proofs, exhaustive + symbolic |
| CSAMAC used the wrong accumulator operand | Fixed; regression pinned by 34,412 vectors |
| No conformance vectors | 34,412 generated from the proved spec; 56,420 C++ checks |
| No backend interface or registry | `Backend` + `BackendRegistry` + `TargetDescription` |
| No target description mechanism | `TargetDescription`; `--list-targets` |
| MLIR dialect never compiled in CI | `mlir-dialect` job; 66/66 pass with `BT_ENABLE_MLIR=ON` |
| Frontend disconnected from the `bt` dialect | `--emit-bt-dialect`; passes fire on real `.bt` source |
| Proofs not linked to hardware | `rocq_bridge` parses Rocq definitions, checks vs RTL |

### The v1 ISA could not be a compilation target

Found by writing the selector, not by inspection: no way to materialise a
constant into a ternary register, and the full adder and per-lane multiplier
existed in RTL but had no encoding. Four instructions were added using the two
previously-unassigned `funct3` values (`BTFA`, `MUL`, `PACK243`, `LI5`), all
Z3-verified. Full detail in [XBT_ISA_SPEC.md](XBT_ISA_SPEC.md).

### Closed since (frontend → dialect)

The two compilers are joined: `.bt` → AST → `ir::Program` → `bt` dialect →
passes. A ternary scalar is carried as a (P, N) pair of `i64` SSA values, so an
`Add` lowers to `bt.btha` + `bt.csa_reduce`. Three dialect ops were needed
(`bt.const`, `bt.neg`, `bt.lane_mul`) because the dialect had no way to
introduce a constant — which is precisely why the halves had stayed apart.
`--emit-bt-dialect` drives it; `test/frontend_dialect_pipeline.py` asserts the
passes fire on output that came from `.bt` source. 88 bridge checks plus the
end-to-end pipeline test.

### Closed since (GEMM operand layout and the matmul fast path)

Two traps in the ternary GEMM API, both of which fail *silently*, which is
what made them worth fixing properly.

**`matmul` was always the slow path.** `Context.matmul` is the unvectorized
O(n³) reference with `__int128` overflow checks (~23.6 ms); `gemm_ternary` is
the bit-sliced kernel (~0.72 ms). Nothing connected them, so calling the
obvious entry point cost ~30x more than necessary. `matmul` now inspects its
input and routes to `gemm_ternary` when every element is a trit and the
library exports it; otherwise it falls back to the reference, which accepts
any int64. `force_reference=True` pins the old path for differential testing.

**A and B are not laid out the same way, and the difference is invisible.**
A is row-major; B is transposed and then word-plane-major (`bp[w*n + j]`). At
k ≤ 64 the two orderings coincide, so a hand-packed multi-word B returns
status 0 and *wrong numbers*. Only shapes with k > 64 **and** n > 1 can catch
it. `pack_matrix_a` / `pack_matrix_b` now exist in Python, C++
(`bt::sdk`) and Rust (`trium::pack_matrix_*`, re-exported alongside the
previously-unexported `encode_gemm`), so no caller has to know the rule.

Both are pinned by tests that include a negative control: the test asserts the
naive row-major B *is* detected as wrong at k > 64. Without that, the cases
would be vacuous.

### Still open

- **Scalar multiply.** `xbt.mul` is lane-wise. The selector refuses general
  multiplication rather than emitting wrong code, but a real scalar multiply
  needs a shift-add sequence and a general shift-by-trits instruction.
- **Translation validation.** Each lowering is tested end to end, but no pass
  proves its output equivalent to its input.
- **The IR is still scalar-only.** No vector or tensor type. Adding one is the
  prerequisite for expressing accelerators generically.
- **Nothing has run on real hardware.** The Vulkan path compiles and degrades
  honestly (`rc=4`); GLSL is validated, not executed; no FPGA bring-up.
- **RTL is not simulated.** `hw/riscv_xbt_extension.sv` has no CI job that
  simulates it; the bridge verifies the C++ model, not the SystemVerilog.
