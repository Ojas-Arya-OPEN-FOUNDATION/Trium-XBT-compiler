# Chip-side research: pinned primary sources

Research date: **2026-10-31**. Companion to [the two-track hardware plan](../CHIP_HARDWARE_RESEARCH_AND_PLAN.md).

Primary-source repository documentation and selected interface/circuit/library files read through GitHub CLI. Sources pinned to commits; no third-party source vendored. Not a systematic literature review or experimental replication.

These are upstream statements and inspected artifacts, not Trium-XBT integration or performance results. The pinned branch heads are research snapshots, not a claim of a ratified or mutually compatible release. No external paper full text was retrieved; the circuit-paper observations below concern author-supplied repositories. Prices, shuttle availability, foundry quotes and laboratory access were not established.

The [machine-readable manifest](chip-hardware-sources-2026-10-31.json) records file URLs, content SHA-256 values and local audit hashes. External source copies remain scratch files, not vendored deliverables.

## S1: Custom instruction allocation

**Repository:** `riscv/riscv-isa-manual`  
**Pinned commit:** `8fb3965fe4b501a6bfa00c34affffcdbdb2c552c`

Custom-0 through custom-3 are the recommended custom major opcodes; reserved encodings are not a safe substitute. This is not a registered Xbt extension or a CPU integration.

Files reviewed:
- [src/unpriv/rv-32-64g.adoc](https://github.com/riscv/riscv-isa-manual/blob/8fb3965fe4b501a6bfa00c34affffcdbdb2c552c/src/unpriv/rv-32-64g.adoc)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S2: Coprocessor protocol

**Repository:** `openhwfoundation/core-v-xif`  
**Pinned commit:** `d26eeab305270dd6462126b775cdbe28fb989477`

Issue/register/commit/result channels, speculative kill/commit and optional memory interfaces are explicit. Register widths/read counts are configured, not arbitrary 128-bit ternary registers. Match a specific core and specification revision.

Files reviewed:
- [docs/source/x_ext.rst](https://github.com/openhwfoundation/core-v-xif/blob/d26eeab305270dd6462126b775cdbe28fb989477/docs/source/x_ext.rst)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S3: A concrete RISC-V integration candidate

**Repository:** `openhwfoundation/cv32e40x`  
**Pinned commit:** `d952cd63bc1b4eb58cd893c28ef8283c781e345e`

The core exposes an extension interface; control-transfer instructions are not supported through it. Separate core verification infrastructure and compatible interface revisions remain required.

Files reviewed:
- [docs/user_manual/source/x_ext.rst](https://github.com/openhwfoundation/cv32e40x/blob/d952cd63bc1b4eb58cd893c28ef8283c781e345e/docs/user_manual/source/x_ext.rst)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S4: A small FPGA custom-function exploration route

**Repository:** `google/CFU-Playground`  
**Pinned commit:** `7f8ec4147b39db98534f8a10f9ddb1c26aca5648`

The README describes instruction/kernel/FPGA iteration and no-board simulation, supported board examples, and its prototype/not-officially-supported disclaimer. It is an alternative experiment, not a second mandatory SoC integration.

Files reviewed:
- [README.md](https://github.com/google/CFU-Playground/blob/7f8ec4147b39db98534f8a10f9ddb1c26aca5648/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S5: FPGA SoC and control infrastructure

**Repository:** `enjoy-digital/litex`  
**Pinned commit:** `57dc334007b56f9b3b74e2a4ac97882d96bd4260`

Buses, memory/peripheral generation, BIOS and CSR headers, host bridges and LiteScope are described. Proposed first control-plane host; none is integrated into Trium-XBT in this research.

Files reviewed:
- [README.md](https://github.com/enjoy-digital/litex/blob/57dc334007b56f9b3b74e2a4ac97882d96bd4260/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S6: Larger SoC route

**Repository:** `ucb-bar/chipyard`  
**Pinned commit:** `157dd69e49b9866be2bf8db390ee78bc38cd4438`

The framework covers RISC-V cores, MMIO/custom accelerators, simulation, FPGA-assisted simulation and VLSI flows. Consider later if Rocket/RoCC is chosen; not the same interface as CV-X-IF.

Files reviewed:
- [README.md](https://github.com/ucb-bar/chipyard/blob/157dd69e49b9866be2bf8db390ee78bc38cd4438/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S7: A relevant binary-hardware ternary-weight baseline

**Repository:** `microsoft/BitNet`  
**Pinned commit:** `0b341e582afbf9e1011f24744b554c96a3477eb5`

Official inference framework describing CPU/GPU low-bit kernels. It demonstrates the category of binary-hardware optimization; upstream advertised performance is not a Trium measurement or an equal-workload comparison.

Files reviewed:
- [README.md](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S8: Actual HDL compilation and simulation

**Repository:** `verilator/verilator`  
**Pinned commit:** `92ac4051b66ab409351361910b865b9db98b816e`

Accepts SystemVerilog, performs lint and compiles simulation models into C++/SystemC. Reading an RTL file or incrementing a C++ cycle counter is not this process.

Files reviewed:
- [README.rst](https://github.com/verilator/verilator/blob/92ac4051b66ab409351361910b865b9db98b816e/README.rst)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S9: Independent Python-driven HDL tests

**Repository:** `cocotb/cocotb`  
**Pinned commit:** `3ec55d81a04b73663e7b0fc5f829f81e2882907f`

Python testbenches for Verilog/VHDL; requires a simulator. A verification harness does not synthesize a chip or prove analog behavior.

Files reviewed:
- [README.md](https://github.com/cocotb/cocotb/blob/3ec55d81a04b73663e7b0fc5f829f81e2882907f/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S10: RTL property checking

**Repository:** `YosysHQ/sby`  
**Pinned commit:** `79fc9eddca5e33a28cc2f2cd19c9825efd9a7255`

Front end for Yosys-based formal flows; engines/components have separate terms and syntax support differs between open and commercial suites. A completed proof must report its assumptions and supported frontend.

Files reviewed:
- [README.md](https://github.com/YosysHQ/sby/blob/79fc9eddca5e33a28cc2f2cd19c9825efd9a7255/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S11: RTL/netlist equivalence

**Repository:** `YosysHQ/eqy`  
**Pinned commit:** `33015166c47398dff1ac72a699f4212891bb0618`

Yosys-based equivalence checking frontend. Use to check transformations where supported; not a substitute for an independently specified arithmetic reference.

Files reviewed:
- [README.md](https://github.com/YosysHQ/eqy/blob/33015166c47398dff1ac72a699f4212891bb0618/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S12: Binary digital RTL-to-GDS flow

**Repository:** `The-OpenROAD-Project/OpenROAD-flow-scripts`  
**Pinned commit:** `202c7fccbce32e86db899c54b0a0f0484f869cc3`

Documents synthesis, floorplan, placement, clock tree, routing and finishing stages. Flow completion is not a foundry reservation, production signoff or physical ternary-cell implementation.

Files reviewed:
- [README.md](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/202c7fccbce32e86db899c54b0a0f0484f869cc3/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S13: Current modular digital implementation option

**Repository:** `librelane/librelane`  
**Pinned commit:** `64ef80802d3d189b476636d2a6b17888c03304ee`

Current LibreLane README describes configurable reproducible implementation flows. The separately reviewed chipfoundry/openlane2 README explicitly points development here; use a pinned supported flow rather than assuming the old brand/repository is current.

Files reviewed:
- [Readme.md](https://github.com/librelane/librelane/blob/64ef80802d3d189b476636d2a6b17888c03304ee/Readme.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S14: Open process data and explicit maturity warning

**Repository:** `google/skywater-pdk`  
**Pinned commit:** `7198cf647113f56041e02abf3eb623692820c5e1`

README calls this open release an experimental preview/alpha, intended for test chips and initial verification rather than production. A mature process does not make every public PDK release or ternary design production-qualified.

Files reviewed:
- [README.rst](https://github.com/google/skywater-pdk/blob/7198cf647113f56041e02abf3eb623692820c5e1/README.rst)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S15: PDK preparation is a separate dependency

**Repository:** `RTimothyEdwards/open_pdks`  
**Pinned commit:** `f55b79336b96b4d0a90abec0bdebdee4f4c34ab3`

Builds foundry data for open tools; sky130/gf180 support and prebuilt IHP distinction are documented. A PDK builder is not a ternary cell library, a fabrication contract or a generic advanced-node model.

Files reviewed:
- [README.md](https://github.com/RTimothyEdwards/open_pdks/blob/f55b79336b96b4d0a90abec0bdebdee4f4c34ab3/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S16: Circuit/device simulation capability

**Repository:** `Xyce/Xyce`  
**Pinned commit:** `e5b7b9bb5b221630c66654c059fc2d33524f5b21`

SPICE-compatible DC/transient/noise analyses and compact/custom models are described. Simulator capability does not validate a particular device model, PDK translation or fabrication process.

Files reviewed:
- [README.md](https://github.com/Xyce/Xyce/blob/e5b7b9bb5b221630c66654c059fc2d33524f5b21/README.md)

**License boundary:** Upstream terms remain separate; this research is not a full per-file license audit.

## S17: Author-supplied physical-ternary circuit research

**Repository:** `jeshraghian/mr-ternary`  
**Pinned commit:** `57a401031674b37d99f58fb47389fb49db4f0212`

SPICE supplement to High-Density Memristor-CMOS Ternary Logic Family, DOI 10.1109/TCSI.2020.3027693. Includes circuit/model references and warns ordinary DC sweeps can be inaccurate for its time-dependent memristor model. Not independently simulated or treated as fabricated Trium hardware.

Files reviewed:
- [README.md](https://github.com/jeshraghian/mr-ternary/blob/57a401031674b37d99f58fb47389fb49db4f0212/README.md)

**License boundary:** No repository-wide license grant was identified in the reviewed README/metadata; it explicitly notes original authors’ model copyrights. Review all files and obtain rights before reuse.

## S18: Six-transition ternary timing research

**Repository:** `jeyeong-park/T-STA`  
**Pinned commit:** `c7d9557f92a4eb187bb3d49513635acc16966aa7`

Research cells and extended Ternary Liberty data cover six ternary transitions; the actual std_cells.sp and ternary.lib headers/tables were inspected. Requires an external 32 nm Stanford CNTFET model; characterized with HSPICE per README. No independent reproduction, silicon qualification, or ordinary binary-STA parser compatibility is established.

Files reviewed:
- [README.md](https://github.com/jeyeong-park/T-STA/blob/c7d9557f92a4eb187bb3d49513635acc16966aa7/README.md)
- [std_cells.sp](https://github.com/jeyeong-park/T-STA/blob/c7d9557f92a4eb187bb3d49513635acc16966aa7/std_cells.sp)
- [ternary.lib](https://github.com/jeyeong-park/T-STA/blob/c7d9557f92a4eb187bb3d49513635acc16966aa7/ternary.lib)

**License boundary:** README says a CC BY 4.0 license file will be added on publication; do not treat this announced future grant as resolved. The external CNTFET compact model is not redistributed and has separate terms.

## S19: Research precedent for separating logical and physical mappings

**Repository:** `aiunderstand/MixedRadixCircuitSynthesis`  
**Pinned commit:** `0865cee1f49fe9a803b4910b570f4e96e514c61b`

Author describes binary-CMOS mapping versus CNTFET-native SPICE generation and links design examples. Useful architectural precedent; README performance/fabrication/service availability claims were not independently checked.

Files reviewed:
- [README.md](https://github.com/aiunderstand/MixedRadixCircuitSynthesis/blob/0865cee1f49fe9a803b4910b570f4e96e514c61b/README.md)

**License boundary:** Repository metadata identifies GPL-3.0; model dependencies and generated-output licensing require their own file-specific review.

## Additional cross-checks

- `chipfoundry/openlane2` at `a7b0e6dba75ee7e891ff3d7824b29473d9cad289`: README explicitly redirects development to LibreLane; no old OpenLane service price/schedule was adopted.
- RISC-V extension naming source at the S1 commit (`src/unpriv/naming.adoc`) was also read; a project-local Xbt name is not standards approval.
- VA-Models and an ngspice repository README were read as background only; no simulator/model compatibility was tested or recommended solely from these README files.
- Community search results were discovery aids, not evidence that a device was fabricated.

## Local source-audit boundary

Audited: `hw/riscv_xbt_extension.sv`, `hw/xbt_alu_model.hpp`, `test/Unit/test_unit_xbt_hw.cpp`, plus the existing license-boundary note. Local hashes and the base commit are in the manifest. Observations are detailed in section 3 of the plan.

None of `verilator`, `iverilog`, `yosys`, `sby`, `ngspice`, or `Xyce` was on PATH during this audit. No new RTL/circuit execution, FPGA/ASIC result or hardware performance figure is claimed. Previously completed CPU numerical tests are a separate milestone, not a substitute for hardware qualification.
