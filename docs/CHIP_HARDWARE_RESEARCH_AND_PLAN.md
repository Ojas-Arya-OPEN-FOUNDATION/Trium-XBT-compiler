# Trium-XBT chip-side research and two-track development plan

**Date:** 2026-10-31  
**Status:** researched architecture proposal; implementation requires approval.  
**Decision requested:** prioritize a shared hardware contract, real RTL qualification and a small controlled FPGA accelerator, while starting a separate physical-ternary circuit feasibility track.

This responds to both requested directions:

- **Track A — binary hardware optimized for ternary execution:** existing CPUs/GPUs, then purpose-built FPGA and binary-CMOS ASIC blocks.
- **Track B — genuinely physical ternary hardware:** three distinguishable physical logic states, restored/composable ternary circuits and eventually ternary storage/datapaths. This is not merely a new binary chip running dual-rail code.

The objective is an ecosystem for **controlling hardware and building hardware**, not just adding instructions or mathematical functions. Both tracks should implement the same specified mathematics without pretending that their physical designs or qualification methods are interchangeable.

No new numerical routines, HDL changes, toolchain installation or hardware execution were performed for this report. README and licenses remain unchanged.

## 1. Recommendation in brief

Build the shared semantic and control foundation first, then advance both tracks through different evidence gates:

```text
 Existing mathematical specifications + independent exact references
                                |
              Hardware operation / encoding contracts
                 + capability and fault vocabulary
                                |
             Typed compiler operations + SDK dispatch
                      /                       \
             TRACK A                         TRACK B
   Binary implementation routes      Physical-ternary implementation
   CPU/GPU kernels                   Device/circuit models + process route
   Verified digital RTL              Three-state gates + restoration
   MMIO-controlled FPGA              Characterized arithmetic + storage
   Optional custom ISA + DMA         Binary/ternary bridges + test island
   Binary-CMOS ASIC feasibility      Mixed-signal prototype, then expansion
                      \                       /
                 Shared conformance vectors and workloads
             Separate digital, analog and physical evidence
```

**The first useful hardware product should be a small, controllable XBT accelerator, not a complete new CPU.** Begin with `DOT`, lane multiplication, carry-save arithmetic, canonical addition, negation and checked packing. A binary controller with a well-tested peripheral interface is the shortest path to exercising our own hardware reliably. A later physical-ternary compute island can reuse its host-side submission and test infrastructure.

This does **not** abandon a native ternary processor. It creates the prerequisite operation model, register/memory semantics, circuitry, tools and evidence needed to design one intelligently.

## 2. What the primary-source research establishes

Sources are pinned by commit in the [research inventory](research/CHIP_HARDWARE_SOURCES_2026_10_31.md) and its [JSON manifest](research/chip-hardware-sources-2026-10-31.json). These are documentation/artifact findings, not independently reproduced upstream benchmarks or circuit measurements. The research used accessible GitHub primary sources; it is not a systematic survey of all device literature or a foundry-access assessment.

| Finding | Consequence for Trium-XBT |
|---|---|
| RISC-V explicitly allocates custom major opcodes; `custom-0` is an appropriate allocation category. [S1] | Our `0x0B` choice is sensible, but does not establish a standard extension, register ABI or working CPU attachment. |
| CV-X-IF specifies instruction issue, register transfer, commit/kill and completion. Widths and optional features are constrained/configured. [S2] | A combinational ALU must gain a real protocol adapter; speculative operations cannot change architectural state early. |
| CV32E40X offers a concrete external instruction-extension route; its extension interface does not support control-transfer instructions. [S3] | Good later integration candidate, not a way to replace the CPU instruction/control architecture with a ternary one. Match compatible core/interface revisions. |
| CFU Playground supports small FPGA custom-function experiments; LiteX provides SoC, bus, BIOS, CSR and debug infrastructure. [S4][S5] | Reuse infrastructure rather than building DDR, UART, debug and a CPU from scratch. Choose one first integration, not three simultaneous frameworks. |
| Chipyard supports larger RISC-V SoC/custom accelerator development. [S6] | A later Rocket/RoCC route is possible, but distinct from CV-X-IF. Do not label an unattached block as both protocols. |
| BitNet provides CPU/GPU kernels for low-bit/ternary-weight inference on binary machines. [S7] | Relevant workload and baseline family. Its advertised speedups do not transfer to this project, and it is not evidence of physical three-state logic. |
| Verilator/cocotb, SBY and EQY address actual HDL simulation, properties and equivalence. [S8][S9][S10][S11] | Extend our mathematics verification into the RTL. C++ tests and Z3 arithmetic formulas alone do not validate compiled HDL or its control protocol. |
| ORFS and LibreLane provide digital implementation flows. The reviewed OpenLane 2 README redirects development to LibreLane. [S12][S13] | Use pinned supported tool/PDK combinations. Producing GDS is not fabrication or automatic signoff, and ordinary binary cell mapping remains Track A. |
| The reviewed SKY130 open PDK explicitly warns that its public release is an experimental preview/alpha. [S14][S15] | Suitable process candidates require current, revision-specific suitability checks and a fabrication partner. Do not infer production readiness from an open repository. |
| Authors publish memristor-CMOS ternary SPICE examples, with model/history-dependent simulation cautions. [S17] | Physical ternary has concrete circuit research to investigate; a netlist/model is not a manufacturable, qualified cell library. |
| T-STA releases CNTFET cell netlists and extended timing data covering **six ternary transitions**. Its external compact model and license status are material constraints. [S18] | Native ternary needs device-aware characterization and timing support, not simply a binary Liberty file with renamed pins. |
| MixedRadixCircuitSynthesis distinguishes binary-CMOS mapping from native CNTFET circuit generation. [S19] | This is a useful precedent for our two physical mappings behind shared logical semantics. Its linked designs do not prove our hardware or a generally accessible CNTFET fabrication route. |

**Key distinction:** a physically ternary circuit can be constructed from conventional transistors if its information-bearing nodes genuinely use and restore three logic levels. A novel three-state transistor is not a prerequisite. Conversely, a dual-rail ASIC made with ordinary binary cells is still binary hardware, however ternary its mathematics is.

## 3. What we already have, and what the hardware audit found

### Reusable assets

- Specified balanced-ternary arithmetic, dual-rail operations, codecs, exact references and formal mathematical checks.
- A compiler/SDK foundation that can host capability-based lowering and CPU fallback.
- Existing candidate RTL in `hw/riscv_xbt_extension.sv`: half-adder, two prefix-adder designs, dot units, register file, a pipelined compressor unit, base-243 decoder and top ALU.
- A C++ instruction/behavioral model in `hw/xbt_alu_model.hpp`, plus tests of its behavior and encodings.
- Completed numerical work documented separately in [SDK 1.2](ADVANCED_NUMERICS_SDK_1_2.md). That software remains useful as an oracle, not proof that every operation belongs in a first chip.

### Important qualification and specification gaps

| Audited observation | Required action |
|---|---|
| `test_systemverilog_rtl_presence()` reads source text and checks module names. The other tests exercise C++, not a running HDL simulator. | Keep source-presence checks as smoke tests, but add real elaboration and RTL-versus-independent-reference simulation. Do not count the existing output text as completed co-simulation. |
| The top `riscv_xbt_alu` is combinational. It does not implement CPU issue/ready/commit/kill/retirement or a bus interface. | Design a clocked transaction shell first; CPU-specific integration comes later. |
| Each 64-trit rail operand is 128 binary bits. There are rail, auxiliary-rail and scalar outputs. | Specify operand transfer, result ownership and multi-result atomicity. These do not fit automatically into a normal two-source, one-destination RV32/RV64 R-type instruction. |
| The model contains a separate 32-entry ternary register array; the top ALU does not itself establish that architectural state or its load/store/context-switch rules. | Treat that register file as an architectural proposal, not an existing CPU feature. Initially prefer explicitly managed accelerator buffers. |
| Top-level `funct7` is declared but not used by the combinational execution logic; the model uses it for some auxiliary register destinations. | Resolve the distinction between arithmetic outputs and instruction-level writeback. Specify supported encodings and reject reserved ones in a future adapter. |
| The current `CSAMAC` branch forms lane-wise `rs1 * rs2` and combines it with `rs1`; there is no independently supplied accumulator operand in that top-level interface. | Define the intended operation before exposing it as a general MAC. Separate lane product/compression from positional whole-word multiplication and independently accumulated matrix MAC. |
| Base-243 RTL maps bytes `243..255` to zero outputs without an error port. | Make malformed-input handling explicit and versioned; do not silently change a legacy codec's behavior while claiming identical compatibility. |
| Comments/model counters describe one-cycle execution, zero-cycle negation and gate savings. | Measure latency, initiation interval, mapped area and timing. A wire swap can remove arithmetic gates; it does not remove instruction issue, routing, register or transfer costs. |

These are **source-audit findings**, not results of executing or formally disproving the RTL. No `verilator`, `iverilog`, `yosys`, `sby`, `ngspice` or `Xyce` was found on PATH during this audit. No new HDL simulation, synthesis, FPGA or silicon result is claimed.

## 4. Shared foundation: define this once for both tracks

### 4.1 Separate mathematical types that happen to use trits

The hardware profile must distinguish:

1. **Trit lane vector:** independent values in `{-1,0,+1}`; lane multiply and dot reduce these values.
2. **Positional balanced word:** `V = sum(t[i] * 3^i)`; carries connect positions. Lane multiplication is **not** whole-word integer multiplication.
3. **Carry-save pair:** two vectors with a declared weighting/shift convention. For an unshifted compressor carry, `V(A)+V(B)+V(C) = V(S)+3*V(Carry)`, including the specified boundary carry behavior.
4. **Ternary floating point:** the existing significand/exponent, rounding, exceptional and resource contracts. Not equivalent to a trit lane array or to an ordinary hardware binary floating-point unit.
5. **Quantized AI tensor:** values plus scale, layout and accumulation contract. Ternary weights do not require ternary activations, and weight quantization is not a lossless conversion of arbitrary models.

Use separate typed operations and explicit conversions. This prevents a fast lane product from being incorrectly substituted for integer or floating multiplication.

### 4.2 Encodings and conversion contracts

| Representation | Proposed role and checks |
|---|---|
| Dual rail `P,N` | Compute representation: `+1=(1,0)`, `0=(0,0)`, `-1=(0,1)`, invalid when `P & N != 0`. Define this as an ordered pair; do not rely on ambiguous printed bit ordering. |
| Packed five-trit byte | Storage/transport candidate: `b = sum((t[i]+1)*3^i)`, `i=0..4`. Valid byte range `0..242`; five zero trits encode as `121`, not byte zero. |
| Existing public records | Preserve the current canonical records and ABI. A hardware payload is not permission to reinterpret floating records or BTR1/BTQ1 layouts as rail arrays. |
| Three-level physical signal | A candidate mapping is `-1 -> GND`, `0 -> VDD/2`, `+1 -> VDD`; no negative supply is inherently required. Specify receiver thresholds, timing and invalid/uncertain regions separately. |

Required decisions:

- Trit zero is the least-significant **lane/position index**, not necessarily a byte boundary. Define lane order, word order, endianness, alignment, strides and tail masks.
- For partially filled five-trit bytes, require unused digits to encode zero trits (`digit=1`) in a canonical profile. Buffer length and logical trit count must both be explicit.
- Propose a **checked hardware profile** that rejects invalid rails, malformed packed bytes and noncanonical tails before accepting work. Preserve legacy behavior through a separately identified compatibility codec where needed; do not change SDK 1.2 implicitly.
- Exact integer/trit conversions must round-trip; float/quantizer conversions must declare rounding and any scale/clipping. Reuse existing verified conversion semantics wherever applicable.
- Carry-out, overflow, saturation, wrapping and invalid encodings are distinct outcomes. No silent saturation or truncation in an exact profile.

**Derived storage facts, not benchmark claims:** `log2(3) ~= 1.585` information bits/trit; dual rail uses two stored bits/trit. Five trits/byte uses 1.6 bits/trit for complete groups. A 64-trit vector occupies 16 rail bytes versus 13 packed bytes, before metadata/alignment. Packing can reduce traffic but decoding, padding and staging can erase the gain. None of this proves a density, energy or speed advantage in fabricated hardware.

### 4.3 Initial operation profile

Start with a small complete set, not an unqualified copy of every existing mnemonic:

| Operation family | Required semantics |
|---|---|
| Validate / pack / unpack / negate | Checked encodings, deterministic tail handling, exact round trips. Negation swaps the two rails logically. |
| Lane product | `out[i] = a[i]*b[i]`, no cross-position carry. |
| Dot of ternary lanes | `sum(a[i]*b[i])`; for 64 lanes the exact range is `[-64,+64]`. Keep the external result width explicit even if the internal tree is narrow. |
| Half/full compressor | Independent arithmetic identity, explicit carry weighting and boundary outputs. Exhaustive per-lane truth tables. |
| Canonical positional addition | Exact represented-word sum with explicit most-significant carry/overflow policy. Compare ripple, prefix and compressor-based alternatives. |
| Batched/tiled reduction | Add only after the primitives and transaction protocol pass; specify accumulator width, overflow and reduction order. |

For valid dual rails, a dot can be expressed as:

```text
positive = (A.P & B.P) | (A.N & B.N)
negative = (A.P & B.N) | (A.N & B.P)
dot      = popcount(positive) - popcount(negative)
```

This directly connects the CPU bit-parallel implementation to FPGA/ASIC gates and a reference for physical-ternary alternatives. It does not imply that all implementations must use popcount internally.

For ternary weights times signed int8 activations, use add/subtract/select with a sufficiently wide accumulator. **Widen before negating `-128`**; its positive counterpart does not fit int8. Derive accumulator bounds from reduction length rather than choosing a width from a nominal model name.

### 4.4 Versioned hardware/software boundary

Proposed capability information includes:

- Device/implementation identity, hardware ABI version and operation-profile version.
- Implementation class: software, simulated RTL, binary FPGA, binary ASIC, experimental physical-ternary island, etc.
- Supported operations, lane widths, result widths, layouts, rounding/exactness profiles and maximum transfer/reduction sizes.
- Alignment, memory ownership, queue limits, coherency model and available counters.
- Reset generation, health/fault status and features actually enabled.

Define command/completion layouts with fixed-width fields and explicit byte order. Specify size/version/reserved-field validation and a collision policy for identifiers. Do not freeze the final offsets or allocate SDK ABI values in a research document.

Compiler operations, device ABI and the public SDK ABI are separate versioned contracts. Initially call a host library/intrinsic wrapper; capability discovery and deterministic CPU fallback are required. A full LLVM target or new operating system is not the first dependency.

## 5. Track A: binary hardware optimized for ternary execution

### A0 — CPU/GPU execution remains the reference and practical baseline

Continue using bit-parallel rails, popcount, vector operations and tiled memory access on available binary hardware. Select implementations from detected features, not CPU/GPU brand names alone. Compare packed and unpacked storage, SIMD and scalar paths, and real conversion costs.

BitNet is a useful comparator for supported ternary-weight inference cases, not a universal comparator for our positional arithmetic or finite ternary FP. Match shapes, input precision, scaling, accuracy, threading and data residency before comparing. [S7]

Keep the cross-vendor GPU goal. Existing GPU qualification tools remain a separate physical-device task for the user; this hardware plan does not claim a new GPU run or silently make CUDA mandatory. GPU software portability and FPGA board portability are different problems.

### A1 — qualify a standalone digital block

Proposed first implementation target:

- Reuse/audit existing RTL operations rather than rewriting the entire math foundation.
- A registered request/response shell with operation, operands, tag, valid/ready, result/status and reset.
- Start with one outstanding command. No speculative CPU coupling, host-memory DMA, coherent cache interface or hidden architectural register file.
- Expose all arithmetic outputs explicitly; reject unsupported operations/encodings before acceptance.
- Parameterize useful lane widths where practical, but require a tested 64-trit configuration to retain the existing candidate's scope.

First prove correctness, then choose pipelining. An initial synchronous implementation may take multiple cycles; an honest measured design is more valuable than a forced one-cycle claim.

**Exit:** independent arithmetic agreement, control properties and reproducible elaboration/simulation/formal results. No performance claim yet.

### A2 — first controllable FPGA accelerator

**Recommended first host: LiteX peripheral with a thin bus adapter.** Its documented SoC/CSR/BIOS/debug infrastructure reduces infrastructure work. A CFU Playground experiment is an alternative if instruction-level ML exploration becomes the first priority; it is not an additional mandatory platform. [S4][S5]

First system:

```text
 Binary RISC-V soft CPU / host bridge
                 |
       MMIO control and scratchpad
                 |
       XBT registered command shell
                 |
       64-trit arithmetic / dot block
                 |
       Stable result bank + counters
```

Use one chosen bus integration first—native Wishbone/LiteX CSR is a reasonable starting point. Keep the arithmetic core bus-neutral. Add AXI-Lite only when a selected host requires it, rather than developing two untested control paths.

An ECP5-class board with an open-tool flow is a candidate; an Arty-class board is an alternative where access/tool support favors it. CFU's board examples demonstrate ecosystem options, not qualification of our design on those boards. Choose the actual board only after checking LUT/BRAM/DSP capacity, clocking, memory, I/O and tool versions. [S4]

**Exit:** bitstream and firmware provenance, known-answer and randomized hardware comparisons, reset/fault tests, measured cycles/timing/resources and a reproducible host runner. A successful digital FPGA prototype is Track A, not a physical three-level ternary computer.

### A3 — optimize useful compute and transfers

After A2, choose from measured bottlenecks:

- Batch dot commands and keep operands on-device.
- Add a small tiled matrix/reduction engine with explicit scratchpad reuse.
- Compare dual-rail transport against packed input with local decoding.
- Explore sparsity skipping only with a defined mask/layout and measured benefit.
- Introduce DMA only after the memory protection and ownership model is reviewed.
- Benchmark end-to-end execution and energy, not merely an isolated combinational unit.

**AI target:** ternary-weight/low-bit dot and matrix tiles with preserved scales and wide integer accumulation. This is a plausible first specialized workload; full binary64 or ternary-FP GEMM acceleration does not follow automatically.

**Simulation target:** exact integer/trit reductions, structured state updates and selected stencils with real reuse. Preserve general numerical workflows through CPU fallback; do not force arbitrary simulation values into one-trit approximations.

### A4 — optional RISC-V custom instructions

Make this an evidence-based choice, not the starting bottleneck:

| Integration choice | Appropriate use | Cost/constraint |
|---|---|---|
| MMIO/batched accelerator | First FPGA; wide operands; independent device buffers | Submission overhead, amortized by batching |
| CV-X-IF + compatible CV32E40X | Tight scalar/instruction integration | Must implement acceptance, dependencies, speculation/kill, completion and actual supported widths/features [S2][S3] |
| Rocket/RoCC through Chipyard | Larger SoC/cache/memory-system research | Different adapter and substantially broader integration surface [S6] |

The existing 128-bit-per-operand rail interface cannot simply be attached to a normal RV32 register port. Options include smaller lane operations, explicitly staged accelerator registers, supported register-pair mechanisms, or pointer/descriptor-based submission. Decide from measured workloads and a specific host configuration.

A dedicated ternary register file would additionally require load/store instructions, hazards/bypass, context save/restore, privilege rules, debugger visibility and multi-result retirement. Defer it unless it has a measurable advantage over buffers. Unsupported instructions must trap/reject rather than execute an unintended default.

### A5 — binary-CMOS ASIC feasibility, then a separate fabrication decision

Once an architecture is stable, use a pinned digital flow such as ORFS or LibreLane to explore synthesis and placed/routed implementation on an appropriate PDK. [S12][S13]

Require clock/reset constraints, macro models, timing corners, routing/parasitics, power-grid considerations, DRC/LVS, equivalence and test access. Retain both the unconstrained assumptions and constrained results in the report. Do not compare a LUT count to transistor area or different process nodes as though they were the same measurement.

This is a **binary CMOS ASIC implementing ternary semantics**. Tapeout additionally needs actual foundry/shuttle acceptance, IP/model rights, manufacturability/signoff review, packaging, board/test plans and funded fabrication. A GDS file alone closes none of those procurement gates.

## 6. Hardware-level control ecosystem

The arithmetic unit is only one layer. The following is the proposed control model shared by Track A and the host side of Track B.

### 6.1 Minimal first device

| Area | First contract |
|---|---|
| Discovery | Read-only identity, ABI/profile versions, supported operations/widths and buffer sizes. Reject unknown incompatible versions. |
| Operands | CPU/host writes a private staging bank or scratchpad. Multiword input writes are not live execution inputs. Define byte strobes and which fields must be fully initialized. |
| Submission | A single doorbell validates and atomically snapshots a complete command. Busy/invalid submissions return a defined failure and do not start partial work. |
| Ownership | One command in flight; accepted inputs cannot change under execution. Specify whether busy input writes are rejected or held in a separate next-command bank; choose one implementation, not accidental behavior. |
| Completion | Tag, status, all result fields and valid/ack ownership. Results stay stable until acknowledged. No partial sum/carry result becomes visible as a completed result. |
| Polling/interrupts | Bounded polling first. Interrupts later with documented enable/status/ack semantics and no lost-completion races. |
| Reset/timeout | Reset discards in-flight work, invalidates stale results and advances a generation visible to the host. The driver returns an aborted/device-reset outcome; no old completion may satisfy a new request. |
| Faults | Unsupported operation, encoding/range failure, bus error and device fault remain distinguishable. Error results must not overwrite a previously valid application output as if successful. |
| Debug | Accepted/completed counts, busy/stall cycles, operation cycles, bytes moved and fault counts; define counter widths, rollover and snapshot consistency. |

The minimal engine should have a watchdog/recovery policy. A host timeout does not itself prove the device stopped; the driver must quiesce/reset before reusing buffers.

### 6.2 Firmware, driver, SDK and compiler responsibilities

1. **Firmware:** startup/self-test, clock/reset sequencing, capability exposure, known-answer diagnostics, polling/interrupt servicing and safe fault recovery.
2. **Host driver/transport:** discover the device, validate bounds, stage data, submit, wait, acknowledge and clean up resources. Start with a user-run bare-metal/host-bridge runner; do not invent a Linux kernel driver before there is a host bus requiring one.
3. **SDK device layer:** explicit device/context/buffer lifetimes, blocking calls initially, structured status, CPU fallback and provenance. Preserve ABI 1/API 1.2 behavior; introduce hardware discovery as a separately reviewed additive API.
4. **Compiler lowering:** lower only supported typed operations; insert explicit layout conversions and overflow/rounding checks. Use a cost model that includes launch and transfer overhead. Keep unsupported FP/numerical operations on the qualified CPU implementation.
5. **Debug/observability:** reproducible traces linking command IDs to compiler operations, counters, firmware version, RTL/bitstream hash and device identity. Do not claim physical device support from code-generation success alone.

### 6.3 DMA, concurrency and security are a later reviewed slice

Before unrestricted host-memory access or multi-process use, require:

- Registered/pinned buffers or appropriate OS-managed DMA mappings; descriptor handles/offsets rather than arbitrary user pointers.
- Checked `offset + length`, alignment, stride, size, tail and overlap arithmetic, including overflow and descriptor-ring wraparound.
- Device-visible addresses, cache clean/invalidate or proven coherent mappings, release/acquire ordering and bus-specific barriers. A C++ memory fence alone is not a DMA coherency design.
- A real IOMMU/translation mechanism where available, or explicitly bounded device memory/physical windows. Never assume the CPU's PMP automatically constrains an independent DMA master.
- Queue ownership, sequence/generation tags, completion ordering, cancellation, reset and teardown that cannot write into freed buffers.
- Per-context isolation or a documented exclusive-access single-client mode. Clear retained state before reassignment.
- Ready/valid stability, backpressure, burst boundaries, error responses, clock-domain crossing and reset-domain crossing checks.

CV-X-IF memory operations using the host's load/store units have a different protection path from an autonomous DMA master; the specification describes host memory checks for that optional interface. [S2] Review the actual chosen implementation rather than borrowing its security properties by name.

Debug access must be restrictable outside development. Constant-time/confidential-computing behavior is not implied by ternary arithmetic; it would need its own threat model and testing.

## 7. Track B: genuinely new-generation physical ternary hardware

### 7.1 Define the milestone honestly

| Demonstration | Classification |
|---|---|
| Ternary truth tables on binary FPGA LUTs | Logical/architectural prototype, Track A |
| Dual-rail gates mapped to binary standard cells | Binary-encoded ternary ASIC, Track A |
| Three-level wire driven into an ADC, followed only by binary computation | A ternary interface experiment; not evidence of a native ternary arithmetic datapath |
| Restored three-state gates carrying out arithmetic with three-level internal nodes | Physical-ternary compute primitive, Track B |
| Such a datapath under binary control with binary SRAM | Physical-ternary compute island, not an all-ternary processor |
| Characterized three-state storage plus compute/control system | A stronger native ternary subsystem; qualify each component separately |

Begin with a compute island under binary supervision. A binary controller is a practical bridge, not a reason to call the entire system physically ternary. A fully native ternary CPU remains a later architecture/device-system milestone.

### 7.2 Investigate two routes, but choose one initial circuit experiment

**Route B-CMOS: physical multi-valued circuits in an accessible CMOS process**

- Investigate whether available devices, thresholds, references and supplies can support restored three-level gates with acceptable static power and noise margins.
- Model middle-level generation and sensing, fanout, loading and cascades—not just an unloaded waveform from one gate.
- Use the actual PDK's legal device options and voltage/reliability limits. Do not invent a threshold device or change process layers beyond what the fabrication route permits.
- If an accessible process supports a viable cell, this is a plausible initial process route. It is **not yet established** that the reviewed SKY130 release gives us a ready-made ternary library. [S14][S15]

**Route B-emerging: devices/circuits needing a research process or model**

- CNTFET threshold engineering and memristor-CMOS logic are concrete research candidates, supported here by author-supplied artifacts. [S17][S18]
- Treat compact-model availability, redistribution rights, parameter validity and a real fabrication collaborator as entry gates.
- Device history, variation, retention, endurance and convergence assumptions may matter; the memristor supplement specifically cautions against an ordinary DC-sweep interpretation of its time-dependent model. [S17]
- Do not treat a 32 nm CNTFET model as a drop-in replacement for a silicon CMOS PDK, or a simulated cell as a foundry-supported component.

Ferroelectric, other resistive/current-mode or compute-in-memory approaches can enter a later literature/device review if a collaborator has credible models/process access. They are not selected dependencies or verified options in this report.

**Decision rule:** choose the first experiment based on obtainable model rights, reproducible simulation, restoration/noise behavior and a plausible fabrication route—not a paper's isolated gate-count claim. Track B can start its model/process work in parallel with A1; it need not wait for a binary ASIC tapeout.

### 7.3 Build a circuit library, not a single attractive waveform

Start with this ladder:

1. Binary-to-ternary driver and ternary-to-binary receiver with explicit invalid regions.
2. Restoring ternary buffer and inverter; demonstrate loaded cascades.
3. Minimal arithmetic cell: lane product or half/full adder with exhaustively enumerated logical inputs.
4. A small multi-trit arithmetic block, then a controlled test island.
5. Separately, a ternary latch/storage element if genuinely native storage is a goal.

For every candidate, record:

- Logic-level windows and worst-case noise margins at the receiver, including the middle level.
- All **six directed transitions**: low↔middle, middle↔high, low↔high; delay and output slew across input slew/load grids.
- Static current for each stable input state, short-circuit and switching energy; include reference and bridge overhead.
- Fanout, restoration, coupling, interconnect parasitics and repeatable cascaded behavior.
- Process/voltage/temperature corners, available mismatch/Monte Carlo models and their validity; do not manufacture statistical yield claims without a calibrated distribution.
- Sequential setup/hold, metastability, clock/reset and retention behavior where storage exists.
- Device aging, endurance or state-history effects when applicable, and a defined power-up/recovery sequence.

For ideal equally spaced voltage levels, adjacent separation is `VDD/2` instead of binary `VDD`; the real allowable noise is determined by driver/receiver transfer characteristics, not this spacing alone. Middle-level generation, sensing and static current can consume the apparent information-density advantage. Measure the whole circuit.

### 7.4 Tooling required beyond ordinary digital RTL

A SPICE-capable simulator such as Xyce is a candidate for device/circuit work, subject to model compatibility and licensing. Simulator feature support does not establish that a particular compact model or translated PDK is correct. [S16]

Required artifacts for a useful ternary cell library include:

- Circuit netlists and legal, versioned device models.
- Logical truth tables and transition-aware characterization scripts.
- Timing/power/noise data including middle-level transitions.
- Physical layout, extraction and process-specific design-rule/connectivity checks.
- Abstract views/technology mappings and a demonstrated way to preserve ternary behavior through the chosen flow.

T-STA's extended `ternary.lib` contains transition-specific tables and a timing-arc table beyond ordinary binary rising/falling arcs. [S18] **Do not assume a conventional Liberty parser/STA tool understands those extensions.** First test parsing and propagation against the circuit simulations; otherwise retain a dedicated characterization/timing checker. Binary synthesis can lower a logical ternary operation into binary gates and thereby erase the intended physical-ternary implementation.

Maintain two distinct mappings from the common operation model:

- A binary Boolean/rail implementation for FPGA/binary-cell synthesis.
- A technology-specific physical-ternary cell/circuit implementation, possibly using protected macros and dedicated analysis until tool support exists.

Share truth tables and conformance vectors; do not pretend a binary Boolean equivalence proof verifies analog margins or transient behavior.

### 7.5 Physical prototype and eventual native architecture

A first test island should contain a small ternary arithmetic block, bridges, programmable input patterns, observable internal/output nodes, bypass/reference paths and self-test/counters under a binary supervisor. Include pads/package/board parasitics in feasibility analysis. Do not send an uncertain `VDD/2` level directly into an FPGA's ordinary binary input and call it a valid receiver.

A small board-level three-level demonstrator may help validate sensing/control, but is not evidence of integrated-chip area or power. Silicon claims require fabricated samples, calibrated measurement, declared sample size/conditions and comparisons to the corresponding model. Clear entry criteria are needed before paying for tapeout.

A later native ternary processor needs its own specification for instruction encoding, program counter, branch/control logic, word/address widths, storage organization, privilege/interrupts, atomics, boot, debug and binary interoperability. Candidate word sizes can be explored, but **do not freeze a 9/27/81-trit ISA before cell, memory and workload evidence**. RISC-V remains a useful binary host; it does not automatically supply those native ternary semantics.

Physical reliability is separate from arithmetic rounding. Even an exact logical operation can fail electrically. Error detection, calibration and possibly ECC must be designed for the actual fault model; invalid-code detection cannot catch every incorrect but valid trit. Approximate analog/CIM operation would need a separate opt-in accuracy profile, not silent weakening of the exact profile.

## 8. Verification and measurable acceptance gates

All counts/targets below are **proposed acceptance work**, not tests already run.

| Gate | Required evidence before the next claim |
|---|---|
| G0 — contract | Reviewed operation/encoding/transaction specs; independent reference; explicit legal/illegal cases; traceability to current software contracts and resolved legacy behavior. |
| G1 — arithmetic RTL | Clean elaboration/lint under pinned tools; exhaustive one-trit input tables (9 half-adder pairs, 27 full-adder triples); all 256 codec inputs under the selected profile; exhaustive small widths where feasible (e.g. 4-trit two-input addition: 6,561 pairs); deterministic seeded/random full-width comparisons and adversarial boundaries. |
| G2 — protocol/formal | Reset, valid/ready stability, atomic results, no double execution, no stale completion, unsupported-command rejection and fault containment. If CPU-integrated: killed instructions have no architectural effects and dependencies/retirement follow the chosen protocol. Prove safety; state fairness/bus-response assumptions for liveness. |
| G3 — digital implementation | Report actual pipeline latency/initiation interval and constrained timing/resource use. Check transformations/netlist equivalence where supported; retain tool/PDK/clock/seed/constraint hashes. No universal one-cycle or area-saving claim from source structure. |
| G4 — FPGA system | Physical known-answer/randomized tests, repeated reset and boundary transfers, long-run stability, firmware/driver/SDK failure tests, actual timing/counters and user-reproducible runner. Record board/device and bitstream provenance. |
| G5 — usefulness | Same-workload end-to-end time and, where instrumented, energy versus tuned binary baselines; include conversion/launch/transfers and resident-data cases separately. A correct but slower design can be published as a correctness prototype, not as a speedup. |
| G6 — physical-ternary circuit | Restored three-state cascades, full transition characterization, declared PVT/load/noise operating envelope, model rights and plausible process route. Simulation evidence remains labeled simulation. |
| G7 — fabrication/measurement | Accepted fabrication package, signoff/test plan, funded packaging/board, then measured samples with operating envelope and reliability limitations. A test island does not qualify an entire native CPU. |

Additional arithmetic vectors must include zero, all positive/all negative, alternating rails, carries through the full word, carry at the top position, dot extrema, malformed rails, malformed/tail-packed inputs and parameter boundaries. For finite ternary FP, separately require the existing exact rounding/exception reference suite before advertising an FP accelerator.

Use Z3 for mathematical/encoding obligations and small reference models; SBY/other supported RTL formal flows for the **actual elaborated hardware**; EQY where appropriate for implementation equivalence. [S10][S11] Record proof scope and assumptions, counterexamples and timeouts. A timed-out proof is not a pass; a bounded proof is not an unbounded theorem.

Analog gates require transient/noise/corner analysis in addition to digital truth-table agreement. Layout parasitics and silicon measurements are later, distinct evidence layers.

## 9. Performance and usefulness: what to measure

Measure three views independently:

1. **Kernel/device:** cycles, latency, initiation interval, supported lane operations per cycle and resource use.
2. **Application:** complete dot/matrix/stencil execution with the same accuracy and layout semantics, including launch/conversion/transfers and CPU fallback.
3. **Physical:** total power/energy at a declared measurement boundary; report idle subtraction and instrumentation assumptions. FPGA board power is not ASIC cell energy.

A useful model is:

```text
T_total = T_convert + T_submit + T_transfer + T_compute + T_sync
```

For a specified operation count and representation, compare sustainable compute with actual data-bandwidth limits. Benchmark tiny/large and memory-resident/streaming cases; report where CPU execution wins and use that crossover in dispatch.

Candidate workloads:

- 64-lane dot microbenchmarks plus batched dots that amortize control overhead.
- Ternary-weight × int8/other explicitly specified activation matrix tiles, including scales and model-level accuracy where inference is claimed.
- Checked balanced-word arithmetic and compressor reductions.
- Selected exact simulation/state-update kernels and structured reductions.

Do not count “64 trits” as automatically 64 full-precision MACs. State exactly which logical operation a throughput number counts. Compare equal outputs/accuracy and tuned baselines, not a naive scalar loop against an accelerated low-bit kernel.

A speed/energy win is an acceptance condition for a **performance claim**, not an assumption underpinning the mathematics. If one primitive has no advantage, change representation/pipeline/workload or keep it as a correctness/reference component.

## 10. Ordered work packages, dependencies and stop rules

These are engineering planning ranges, **not quotes or calendar commitments**. They assume relevant RTL/verification experience and working tool access; external reviews, board procurement, fabrication and model rights are excluded. Packages overlap and should not be added as a promised delivery date.

| Package | Deliverable | Dependency / rough effort |
|---|---|---|
| P0 — hardware contracts and audit closure | `specs/hardware-profile-v0.1.*`, operation truth tables, encoding/transaction/compatibility decisions, corrected qualification vocabulary | Start first; roughly 1–2 engineer-weeks |
| P1 — real RTL verification | Reproducible simulator/lint/formal setup, independent vectors, actual HDL tests, failures resolved | P0; roughly 2–4 engineer-weeks, depending on audit findings |
| P2 — registered shell and control model | Request/response core, MMIO snapshot/result ownership, counters/reset/fault tests | P0/P1; roughly 2–4 engineer-weeks |
| P3 — first FPGA/control ecosystem | One LiteX board target, firmware, host runner, additive SDK integration and evidence | P2; roughly 3–6 engineer-weeks plus board/tool access |
| P4 — measured specialization | Select tiled compute, batching, custom ISA or DMA from bottleneck evidence; do not automatically implement all | P3/G5; roughly 4–8 engineer-weeks for a selected slice |
| P5 — binary ASIC feasibility | One pinned PDK/flow, physical reports, signoff/procurement gap list | Stable P1–P4 design; roughly 4–12 engineer-weeks; excludes tapeout |
| R0 — physical-ternary entry study | Model/license/process inventory, simulator reproduction and candidate selection | Parallel with P1; roughly 2–4 engineer-weeks for study, not for a qualified cell |
| R1 — restored cell/arithmetic characterization | Legal models, loaded cascades, transition/noise/power/corner data and a go/no-go decision | R0; several design iterations; schedule only after model/process access is established |
| R2 — test island/layout/fabrication proposal | Bridges, small arithmetic block, test architecture, physical views and partner-reviewed route | R1/G6; separate funded project, not a software-only completion promise |
| R3 — native subsystem/processor architecture | Memory/control/ISA prototypes grounded in measured cells | Only after arithmetic and storage feasibility; no present completion date |

Suggested ownership, if people are available: digital RTL/control, verification/formal, compiler/SDK/benchmarks, and a device/analog collaborator for Track B. These are work streams, not a claim that multiple agents or engineers have already been assigned.

### Go/no-go rules

- **No custom ISA freeze** until operand/state/result semantics and the specific host protocol are reviewed.
- **No DMA expansion** until bounds, coherency, reset and ownership tests pass.
- **No performance advertising** without equal-workload measurements.
- **No binary ASIC tapeout** merely because a synthesis/physical tool produced output.
- **No physical-ternary library claim** without restoration, transition and operating-envelope evidence.
- **No emerging-device commitment** without legal usable models and a plausible process collaborator.
- If three-state cascades, noise margins or static power fail, document that result and revise the circuit/technology. Track A remains useful and independently deliverable.

### Procurement and cost dependencies

Before approving spending, obtain current quotes/access confirmation for:

- FPGA board, programmer/cables, suitable power measurement and a supported host/tool environment.
- Digital EDA compute/storage, board tool licenses if a proprietary route is selected, and versioned PDK availability.
- Device/analog modeling expertise, model rights, simulator compatibility or commercial licenses where needed.
- Mixed-signal measurement access: controllable supplies, suitable oscilloscope/measurement fixtures, environmental testing and calibration as required by the proposed envelope.
- Foundry/shuttle eligibility and slot, NRE, die area, packaging, test PCB, assembly, shipping and possible respins.

No board price, shuttle schedule, fabrication cost or CNTFET fabrication access was verified in this research. Public open-source availability is not a quotation or a right to manufacture every device model. Parallel progress is practical for contracts/digital verification and circuit feasibility; a fabrication timeline cannot be inferred from software task estimates.

## 11. License, evidence and release boundaries

The existing [license-boundary note](LICENSE_BOUNDARIES.md) applies: the root contains AGPLv3 text; no new permissive SDK/runtime or generated-output exception has been granted. This report does not change licenses or decide the legal scope of HDL, generated netlists, bitstreams or layouts.

Before distributing integrated hardware/software artifacts:

- Inventory ownership and per-file terms for our RTL, firmware, drivers, SDK, generated helpers/netlists and third-party IP.
- Separately review the selected RISC-V core, interface definitions, bus IP, EDA tools, PDK/device models and generated-artifact obligations. Tool licensing and design licensing are not interchangeable.
- Preserve notices and avoid claiming that an open tool makes all its inputs/outputs permissively licensed.
- Treat research artifacts as references until reuse rights are resolved. The memristor supplement notes third-party model copyrights; T-STA announces a future license file and excludes the external compact model from that grant. [S17][S18]
- Have the rights holders clarify any desired HDL/SDK/generated-artifact boundary; do not edit a license or add an exception without authorization.

Archive evidence by milestone: specs, independent vectors, tool/frontend/solver versions, seeds, commands, logs, source hashes, assumptions, timing/resource reports, board/bitstream/firmware identity, measurement method and failures/skips. Keep large build/PDK/generated artifacts outside Git under existing ignore/external-storage conventions.

## 12. The immediate next approved slice

**Recommended next implementation: P0 + P1, followed by the minimal P2 shell—not another elementary math function and not a fabricated chip claim.**

Concrete outputs:

1. A reviewed hardware profile separating trit lanes, positional words, carry-save values and FP; strict encoding and legacy compatibility behavior.
2. A runnable HDL test/formal harness for the actual RTL, with independent references and adversarial cases.
3. A registered transaction shell and minimal MMIO specification with observable reset/failure behavior.
4. In parallel, an R0 model/process/license feasibility record selecting the first genuinely three-state circuit experiment, without importing unlicensed models.
5. After correctness gates, a separately scoped first-board proposal and user-runnable FPGA qualification plan.

That sequence turns Trium-XBT from a mathematical/compiler foundation with candidate RTL into the beginnings of a **controlled, verifiable hardware platform**, while keeping a credible path toward **physically ternary hardware** rather than confusing it with binary dual-rail implementation.

## Sources

The [pinned research inventory](research/CHIP_HARDWARE_SOURCES_2026_10_31.md) explains what was inspected, what it supports and what remains unverified. Reference links below resolve to the exact reviewed upstream commits, not moving branch heads.

[S1]: https://github.com/riscv/riscv-isa-manual/blob/8fb3965fe4b501a6bfa00c34affffcdbdb2c552c/src/unpriv/rv-32-64g.adoc
[S2]: https://github.com/openhwfoundation/core-v-xif/blob/d26eeab305270dd6462126b775cdbe28fb989477/docs/source/x_ext.rst
[S3]: https://github.com/openhwfoundation/cv32e40x/blob/d952cd63bc1b4eb58cd893c28ef8283c781e345e/docs/user_manual/source/x_ext.rst
[S4]: https://github.com/google/CFU-Playground/blob/7f8ec4147b39db98534f8a10f9ddb1c26aca5648/README.md
[S5]: https://github.com/enjoy-digital/litex/blob/57dc334007b56f9b3b74e2a4ac97882d96bd4260/README.md
[S6]: https://github.com/ucb-bar/chipyard/blob/157dd69e49b9866be2bf8db390ee78bc38cd4438/README.md
[S7]: https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/README.md
[S8]: https://github.com/verilator/verilator/blob/92ac4051b66ab409351361910b865b9db98b816e/README.rst
[S9]: https://github.com/cocotb/cocotb/blob/3ec55d81a04b73663e7b0fc5f829f81e2882907f/README.md
[S10]: https://github.com/YosysHQ/sby/blob/79fc9eddca5e33a28cc2f2cd19c9825efd9a7255/README.md
[S11]: https://github.com/YosysHQ/eqy/blob/33015166c47398dff1ac72a699f4212891bb0618/README.md
[S12]: https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/202c7fccbce32e86db899c54b0a0f0484f869cc3/README.md
[S13]: https://github.com/librelane/librelane/blob/64ef80802d3d189b476636d2a6b17888c03304ee/Readme.md
[S14]: https://github.com/google/skywater-pdk/blob/7198cf647113f56041e02abf3eb623692820c5e1/README.rst
[S15]: https://github.com/RTimothyEdwards/open_pdks/blob/f55b79336b96b4d0a90abec0bdebdee4f4c34ab3/README.md
[S16]: https://github.com/Xyce/Xyce/blob/e5b7b9bb5b221630c66654c059fc2d33524f5b21/README.md
[S17]: https://github.com/jeshraghian/mr-ternary/blob/57a401031674b37d99f58fb47389fb49db4f0212/README.md
[S18]: https://github.com/jeyeong-park/T-STA/blob/c7d9557f92a4eb187bb3d49513635acc16966aa7/README.md
[S19]: https://github.com/aiunderstand/MixedRadixCircuitSynthesis/blob/0865cee1f49fe9a803b4910b570f4e96e514c61b/README.md
