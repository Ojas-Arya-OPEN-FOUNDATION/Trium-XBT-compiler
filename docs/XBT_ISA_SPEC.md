# RISC-V "Xbt" custom-0 balanced ternary extension — ISA specification

**Date:** 2026-11-04
**Status:** v1 instructions are specified and machine-verified. v2 instructions
are **proposed** (the encodings 0b110 / 0b111 were previously unassigned).
**Verification:** `python3 -m verification.verify_isa` — 16/16 proofs.
**Conformance:** `python3 -m verification.gen_vectors` — 34,412 vectors.

This document is the contract between the mathematics, the compiler, and the
hardware. Every instruction here is proved equivalent to balanced-ternary
arithmetic in Z3; a future RTL implementation, an FPGA bring-up, or a physical
ternary chip qualifies by reproducing the generated conformance vectors.

---

## 1. Encoding

Custom-0 opcode `0x0B`. Two formats, distinguished by `funct3`:

**R-type** (`funct3` = 0b000–0b110)

```
[31:25] funct7   [24:20] rs2   [19:15] rs1   [14:12] funct3   [11:7] rd   [6:0] 0x0B
```

**I-type** (`funct3` = 0b111)

```
[31:20] imm[11:0]   [19:15] rs1   [14:12] funct3   [11:7] rd   [6:0] 0x0B
```

Register file: 32 dual-rail ternary registers `tr0..tr31`. `tr0` is hardwired
to zero. A dual-rail word is a pair of 64-bit masks `(P, N)`; bit *i* of `P`
means trit *i* is +1, bit *i* of `N` means trit *i* is −1. `P = N = 1` is
invalid — there is no fourth trit.

The value of a word is `Σ trit(i) · 3ⁱ`.

---

## 2. v1 instruction set (existing, verified)

| funct3 | Mnemonic | Semantics | Verification |
|---|---|---|---|
| `0b000` | `xbt.btha rd, rs1, rs2` | `a + b == sum + 3·carry` per lane | exhaustive 9 + symbolic 2¹²⁸ |
| `0b001` | `xbt.ksadd rd, rs1, rs2` | `value(rs1) + value(rs2) == value(sum) + cout·3⁶⁴` | symbolic vs ripple-carry, 64 lanes |
| `0b010` | `xbt.dot64 rd, rs1, rs2` | `x[rd] = Σᵢ rs1[i]·rs2[i]` | per-lane BV lemma + LIA popcount |
| `0b011` | `xbt.unpack243 rd, rs1` | `value == (byte − 121)`, rejects byte ≥ 243 | exhaustive 256 bytes |
| `0b100` | `xbt.neg rd, rs1` | `rd[i] = −rs1[i]` (rail swap, zero gates) | exhaustive 3 + symbolic |
| `0b101` | `xbt.csamac rd, rs1, rs2` | `rd + (rs1·rs2) == sum + 3·carry` per lane | exhaustive 27 + symbolic |

### 2.1 Bug found and fixed: CSAMAC

The v1 model computed `rs1 + rs1·rs2`, using **rs1** as the accumulator while
the documentation specified *"rs1 \* rs2 accumulated into **rd**"*. The
accumulator operand was wrong.

Z3 caught it in both strategies (`exhaustive`: `a=-1 b=-1 rd=0` expected
`sum=1`, got `sum=0`; `symbolic`: solver `failed` with a concrete
counterexample). Fixed in `hw/xbt_alu_model.hpp`; the regression is now pinned
by 34,412 conformance vectors and would produce **5,149 failures** if
reintroduced.

This is the concrete argument for machine verification: the pre-existing test
only asserted `isValid()` — that the output rails did not overlap — and never
checked a value. The bug was invisible to it.

---

## 3. Why v1 alone cannot be a compilation target

Three gaps, all found by attempting to write the instruction selector rather
than by inspection:

1. **No constant materialisation.** There is no way to put a literal into a
   ternary register. A compiler cannot emit a single statement.
2. **The full adder is unreachable.** `BTFA` exists in the RTL and is used by
   the systolic compressor, but has no `funct3`.
3. **The multiplier is unreachable.** The AOI22 product network exists inside
   `DOT64`, but there is no instruction exposing per-lane multiply.

The v2 extension closes exactly these three gaps and nothing more.

---

## 4. v2 proposed extension

`funct3 = 0b110` is an R-type group with `funct7` as an extended opcode;
`funct3 = 0b111` is a new I-type.

| funct3 | funct7 | Mnemonic | Semantics |
|---|---|---|---|
| `0b110` | `0x00` | `xbt.btfa rd, rs1, rs2` | `rd = sum(rs1, rs2, rd)`; carry latched |
| `0b110` | `0x01` | `xbt.mul rd, rs1, rs2` | `rd[i] = rs1[i] · rs2[i]` (**lane-wise**) |
| `0b110` | `0x02` | `xbt.pack243 rd, rs1` | `x[rd]` = base-243 byte from `rs1[4:0]` |
| `0b111` | — | `xbt.li5 rd, byte, chunk` | `rd = (byte − 121) · 3^(5·chunk)` |

### 4.1 LI5 immediate field

```
imm[7:0]   base-243 byte, 5 trits (digit = trit + 1)
imm[10:8]  chunk index k ∈ [0,7]; the trit shift is 5·k
```

A chunk index rather than a raw trit shift: 3 bits cover 8 chunks × 5 trits =
40 trits, one more than an `int39` needs. A raw shift field would have needed
6 bits and would not have fit alongside the byte.

An `int39` constant is materialised in `ceil(39/5) = 8` `LI5` instructions
accumulated with `KSADD`. Verified exhaustively over all 256 byte values ×
8 chunks, including rejection of the 13 invalid bytes.

### 4.2 PACK243 / UNPACK243 round-trip

The base-243 codec is only usable as a *storage format* if the two directions
are inverses. Both directions are verified exhaustively over the whole
243-value domain:

- `pack(unpack(b)) == b` for all 243 valid bytes
- `unpack(pack(v)) == v` for all 243 representable values
- `pack(v) ∈ [0, 242]` always

---

## 5. Semantics that are deliberately NOT provided

**There is no scalar multiply.** `xbt.mul` is lane-wise: it drops every
carry between lanes. Using it as a scalar multiply silently produces wrong
answers as soon as either operand needs more than one trit.

The instruction selector therefore **refuses** `ir::Op::Mul` unless both
operands are single-trit values (|v| ≤ 1), raising a `runtime_error` rather
than emitting incorrect code. A true scalar multiply needs a shift-add
sequence (`a·b = Σᵢ bᵢ·(a ≪ i)`), which requires a general shift-by-trits
instruction that is not proposed here.

This is the same principle the rest of the compiler follows: refuse honestly
rather than emit a plausible-looking stub.

---

## 6. What the compiler can target today

`bt-compiler --target=riscv64-xbt -S file.bt` lowers the backend-neutral core
IR to encoded custom-0 instructions. Supported:

| IR construct | Lowers to |
|---|---|
| integer constant | `li5` chain + `ksadd` |
| `Add` | `ksadd` |
| `Sub` | `neg` + `ksadd` |
| `Negate` | `neg` |
| `Mul` (both operands single-trit) | `mul` |
| `Mul` (otherwise) | **refused** |
| everything else (float, comparison, control flow) | **refused** |

Incomplete lowering is reported by name, never emitted silently.

---

## 7. Verification summary

```
BTHA      exhaustive  9 combinations                        PASS
          symbolic    all 2^128 dual-rail input pairs       PASS
BTFA      exhaustive  27 combinations                       PASS
          symbolic    all 2^192 dual-rail input triples     PASS
KSADD     symbolic    matches ripple-carry on all 64 lanes   PASS
DOT64     symbolic    per-lane lemma (BV) + popcount (LIA)   PASS
UNPACK243 exhaustive  all 256 byte values                    PASS
          symbolic    byte-121 contract                      PASS
NEG       exhaustive  3 combinations                         PASS
          symbolic    involution on all lanes                PASS
CSAMAC    exhaustive  27 combinations                        PASS
          symbolic    accumulates rs1*rs2 into rd            PASS
MUL       symbolic    per-lane product, output rails valid   PASS
PACK/UNPACK  exhaustive round-trip, 243-value domain         PASS
LI5       exhaustive  256 bytes x 8 chunks                   PASS
                                                    16/16 total
```

Plus, in C++:

- `unit_xbt_conformance` — 56,420 checks over 34,412 vectors
- `unit_xbt_selector` — IR → machine code → execution on the model

---

## 8. Adding a new instruction

The verification stack is data-driven, so a new instruction is added in one
place and verified by the existing engine:

1. Add an `Operation` in `verification/xbt_isa.py` with `spec` (trit-level
   mathematics) and `impl` (rail-level hardware behaviour).
2. Run `python3 -m verification.verify_isa`. The engine runs every enabled
   strategy automatically — exhaustive enumeration for small input spaces,
   symbolic bitvector proof for wide ones.
3. Add vector generation in `verification/gen_vectors.py` if the instruction
   should be part of the hardware contract.
4. Implement it in `hw/xbt_alu_model.hpp` and the RTL.

A divergence between `spec` and `impl` fails the build.
