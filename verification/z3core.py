"""
z3core -- the trit mathematics language and dual-rail encoding.

This module is the *specification* layer of the Xbt verification stack. It
defines balanced-ternary arithmetic in two independent ways that must agree:

  1. TRIT LEVEL  -- values in {-1, 0, +1}, the mathematical semantics.
  2. RAIL LEVEL  -- dual-rail pairs (P, N) of 64-bit words, the hardware
                    representation used by hw/riscv_xbt_extension.sv and
                    hw/xbt_alu_model.hpp.

Every hardware block is verified by proving, in Z3, that the rail-level
implementation agrees with the trit-level specification for *every* input.

Nothing here is target-specific: adding a new instruction or a new hardware
block means adding a description, not changing this file.

Conventions
-----------
A "trit word" is ``width`` trits, trit i having place value 3**i.
A dual-rail word is a pair (P, N) of ``width``-bit integers where
bit i of P means trit i is +1 and bit i of N means trit i is -1.
The rail combination P=1, N=1 is INVALID (there is no fourth trit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

# --------------------------------------------------------------------------
# Trit level -- the mathematics
# --------------------------------------------------------------------------

TRITS: tuple[int, ...] = (-1, 0, 1)


def trit_add(a: int, b: int) -> tuple[int, int]:
    """Balanced ternary half-add: return (sum, carry) with a + b == sum + 3*carry."""
    t = a + b
    if t == 2:            # 1 + 1  -> -1 with carry +1
        return (-1, 1)
    if t == -2:           # -1 + -1 -> +1 with carry -1
        return (1, -1)
    return (t, 0)


def trit_add3(a: int, b: int, d: int) -> tuple[int, int]:
    """Balanced ternary full-add: a + b + d == sum + 3*carry."""
    t = a + b + d
    # t in [-3, 3]; reduce into [-1, 1] with a carry in {-1, 0, 1}
    if t >= 2:
        return (t - 3, 1)
    if t <= -2:
        return (t + 3, -1)
    return (t, 0)


def trit_neg(a: int) -> int:
    return -a


def trit_mul(a: int, b: int) -> int:
    return a * b


def word_value(trits: Sequence[int]) -> int:
    """Value of a trit word: sum of trits[i] * 3**i."""
    return sum(t * (3 ** i) for i, t in enumerate(trits))


def word_neg(trits: Sequence[int]) -> list[int]:
    return [-t for t in trits]


def encode_word(value: int, width: int) -> list[int]:
    """Encode an integer into a balanced ternary word of ``width`` trits."""
    trits: list[int] = []
    v = value
    for _ in range(width):
        r = v % 3
        if r == 0:
            trits.append(0)
            v //= 3
        elif r == 1:
            trits.append(1)
            v = (v - 1) // 3
        else:                     # r == 2  ->  trit -1, carry +1
            trits.append(-1)
            v = (v + 1) // 3
    return trits


def balanced_range(width: int) -> tuple[int, int]:
    """Closed interval representable by ``width`` balanced ternary trits."""
    h = (3 ** width - 1) // 2
    return (-h, h)


# --------------------------------------------------------------------------
# Rail level -- the hardware representation
# --------------------------------------------------------------------------

def rail_to_trit(p: int, n: int) -> int:
    """Decode one dual-rail pair. Raises on the invalid 11 combination."""
    if p and n:
        raise ValueError("invalid dual-rail encoding: P=1 and N=1 (no fourth trit)")
    return 1 if p else (-1 if n else 0)


def trit_to_rail(t: int) -> tuple[int, int]:
    return (1, 0) if t == 1 else ((0, 1) if t == -1 else (0, 0))


def rails_to_trits(P: int, N: int, width: int) -> list[int]:
    if P & N:
        raise ValueError("invalid dual-rail word: overlapping P and N rails")
    return [
        1 if (P >> i) & 1 else (-1 if (N >> i) & 1 else 0)
        for i in range(width)
    ]


def trits_to_rails(trits: Sequence[int]) -> tuple[int, int]:
    P = N = 0
    for i, t in enumerate(trits):
        if t == 1:
            P |= 1 << i
        elif t == -1:
            N |= 1 << i
    return (P, N)


# --------------------------------------------------------------------------
# Declarative description of a hardware operation
# --------------------------------------------------------------------------

@dataclass
class Operand:
    """One input or output of a hardware operation."""
    name: str
    kind: str          # "trit"        : a single trit
                       # "trit_word"   : width trits (dual-rail word)
                       # "byte"        : an 8-bit packed base-243 value
                       # "int"         : a machine integer result
    width: int = 64


@dataclass
class Operation:
    """
    A verified hardware operation.

    ``spec``  is the mathematical meaning, written over trit values.
    ``impl``  is the hardware behaviour, written over rail words.

    The engine proves they agree. Keeping them separate is the point: it is
    what makes "the hardware implements the mathematics" a checkable claim
    rather than an assertion.
    """
    name: str
    mnemonic: str
    funct3: int
    operands: list[Operand] = field(default_factory=list)
    spec: Callable | None = None      # trit-level  : (*trit args) -> result
    impl: Callable | None = None      # rail-level  : (*rail args) -> result
    doc: str = ""
    # Exhaustive verification enumerates every trit combination; symbolic
    # verification proves the rail-level function for all widths at once.
    exhaustive: bool = True
    symbolic: bool = True
    # Invalid inputs that must be rejected rather than silently computed.
    rejects: Callable | None = None


class VerifiedISA:
    """A collection of operations forming an instruction set."""

    def __init__(self, name: str, width: int = 64):
        self.name = name
        self.width = width
        self.ops: list[Operation] = []

    def add(self, op: Operation) -> Operation:
        self.ops.append(op)
        return op

    def __iter__(self) -> Iterable[Operation]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)
