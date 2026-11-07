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
| `0b110` | `0x03` | `xbt.sign rd, rc` | `rd = sign(value(rc)) ∈ {−1, 0, +1}` |
| `0b110` | `0x04` | `xbt.cmovz rd, rc, ra` | `if sign(rc) == 0: rd = ra` |
| `0b110` | `0x05` | `xbt.cmovp rd, rc, ra` | `if sign(rc) > 0: rd = ra` |
| `0b110` | `0x06` | `xbt.cmovn rd, rc, ra` | `if sign(rc) < 0: rd = ra` |
| `0b110` | `0x07` | `xbt.shl3 rd, rs1, shamt` | `rd = value(rs1) · 3^shamt`; **throws on overflow** |
| `0b111` | — | `xbt.li5 rd, byte, chunk` | `rd = (byte − 121) · 3^(5·chunk)` |

#### 4.x SHL3 — shift by trit positions

`xbt.shl3 rd, rs1, shamt` multiplies a word by `3^shamt`, i.e. moves each trit
up `shamt` places. The shift amount rides in the **rs2 field** (bits `[24:20]`),
exactly as a RISC-V shift-immediate does, so no new instruction format is
needed.

This is the primitive **P1 scalar multiply** is built on. The ISA had no shift
at all, so a general multiply could not be lowered: `a · b` decomposes as
`Σ_i (trit_i of b) · (a << i)`, and every term needs a shift.

**Overflow is refused, not wrapped.** Trits that would move past bit 63 are
lost, so `shl3` signals overflow rather than silently returning a wrong word.
This matches the `__int128` overflow-check philosophy of the reference
`matmul`. Note the check is against the **64-trit word**, not the logical width
of the value: shifting a 12-trit value by 12 is legal. The scalar-range check
belongs to the lowering, which must refuse a product exceeding `int39`.

Verified in `verification/verify_scalar_mul.py`:
- **S1** exactness — exhaustive over 59 049 (word, shift) pairs, plus symbolic
  at w = 10.
- **S2** overflow is detected *exactly* (iff a set trit leaves the word).
- **S7** out-of-range products are refused; none wrap.

#### 4.y Scalar multiply routing

Whether a multiply compiles to `xbt.mul` or to a shift-add sequence is decided
by `has_scalar_mul` in `targets/*.yaml` — a data-driven decision, not an
`#ifdef`. Xbt declares `false` (its `xbt.mul` is lane-wise and drops
inter-lane carries), so it gets the shift-add; `ternary-torus` declares `true`
and emits the native instruction.

Only the **constant**-multiplier case is lowered: decomposing a runtime
multiplier needs a trit-extract instruction the ISA lacks. See
`docs/SCALING_PLAN.md` §4 for the scope limit and for a correction to the
partial-product estimate.

#### 3.x SIGN and the conditional moves

`SIGN` is a priority encoder over the trit word: `rd` receives the **most
significant non-zero trit** of `rc`, or zero if `rc` is zero.

Correctness rests on the balanced-ternary dominance lemma — the magnitude of
any block of lower trits is strictly less than the next trit's weight:

    Σ_{i<j} 3^i = (3^j − 1) / 2 < 3^j

so no combination of trits below `j` can change the sign established by trit
`j`. Verified exhaustively for every trit word up to 10 trits and symbolically
in Z3 at the full 64-trit width.

The three conditional moves test the **sign** of `rc` (not its value) and leave
`rd` untouched when the test fails. Together with `SIGN` they make every
comparison and every three-way select branch-free.

> **Note on `CMOV` and eager evaluation.** The core `SignSelect` node carries
> three *thunks* and the interpreter evaluates exactly one of them. A
> conditional-move lowering necessarily evaluates all three arms. The selector
> therefore only applies the cmov rule when every arm is provably
> **pure** (`ir::verify(...) == ir::Effect::Pure`); a `SignSelect` with an
> impure arm is refused rather than miscompiled.

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
| `IntCast` (integral) | move (value-preserving) |
| `Equal`, `NotEqual`, `Less`, `Greater`, `LessEqual`, `GreaterEqual` | `sign(a−b)` + one cmov per true sign |
| `Spaceship` (`<=>`) | `sign(a−b)` |
| `SignSelect` (all arms pure) | `cmovz` / `cmovn` over `sign(cond)` |
| `SignSelect` (any arm impure) | **refused** |
| float, control flow | **refused** |

Every relational op is lowered with one uniform shape, which is why the six
rules are generated from a single `comparison_rule` factory rather than
written out six times:

    d  <- (-1)                       // the FALSE value, always
    tv <- (+1)                       // the TRUE value, always
    cmov<mode> d, sign(a-b), tv      // one per sign where the predicate is TRUE

Seeding with the *false* value and moving in the *true* value is essential.
An earlier revision seeded with whichever value was in the majority and moved
in the other; that is only correct when the seed happens to be the false
value, and it silently inverted `!=`. The uniform shape costs at most one
extra `cmov` and removes the trap.

| Relational op | Signs where the predicate is true |
|---|---|
| `>` | positive |
| `==` | zero |
| `<` | negative |
| `!=` | positive, negative |
| `>=` | positive, zero |
| `<=` | negative, zero |

Recall that core comparisons yield **1 when true and −1 when false** (never
zero), while `<=>` yields −1, 0 or +1.

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
