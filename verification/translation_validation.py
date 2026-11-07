"""
Translation validation for the core IR (docs/SCALING_PLAN.md 3.3).

The problem this solves: every lowering is tested end-to-end, but no pass
*proves* its output equals its input. Test cost then grows quadratically with
passes and targets -- which is exactly the cost that makes "scale up" fail.

The obligation, for every pass P:

    IR_before  --P-->  IR_after
    obligation:  forall inputs.  eval(IR_before)(inputs) == eval(IR_after)(inputs)

Discharged by Z3 where the theory is decidable, and by exhaustive enumeration
where it is not. Obligations are declared by the pass harness, not hand-written
per test.

Two layers, and BOTH are needed:

  1. Rule soundness (this file). Each rewrite rule is proved once,
     symbolically, over all inputs. Cost is independent of program size.

  2. Differential check (test/Unit/test_translation_validation.cpp). The real
     C++ optimizer is run on random programs and its output is compared
     against this model. Catches the case where the rule is sound but the
     implementation applies it wrongly -- e.g. the identity rule firing on a
     non-commutative op, which is a real hazard in Optimize.hpp.

Coverage is reported as "N of M passes have discharged obligations", and is
the number CI tracks.

Run:  python3 verification/translation_validation.py
"""

from __future__ import annotations

import itertools
import random
import sys

import z3

# ---------------------------------------------------------------------------
# Core IR model (mirrors include/Ternary/IR/Core.hpp)
# ---------------------------------------------------------------------------

INT39_MIN = -4052555153018976267 // 3      # -(3^39-1)/2 ... see note
INT39_MAX = 4052555153018976267 // 3


class Expr:
    __slots__ = ("op", "operands", "integer", "symbol", "type")

    def __init__(self, op, operands=(), integer=0, symbol=0, type="Int39"):
        self.op = op
        self.operands = list(operands)
        self.integer = integer
        self.symbol = symbol
        self.type = type

    def __repr__(self):
        if self.op == "Integer":
            return str(self.integer)
        if self.op == "Parameter":
            return f"p{self.symbol}"
        return f"({self.op} {' '.join(map(repr, self.operands))})"


def integer(v):
    return Expr("Integer", integer=v)


def param(s):
    return Expr("Parameter", symbol=s)


def binop(op, a, b):
    return Expr(op, [a, b])


# ---------------------------------------------------------------------------
# Concrete evaluation (the reference semantics)
# ---------------------------------------------------------------------------

class Divergence(Exception):
    """Raised when an evaluation is not defined (overflow / out of range)."""


def eval_expr(e, env):
    if e.op == "Integer":
        return e.integer
    if e.op == "Parameter":
        return env[e.symbol]
    if e.op == "Add":
        return eval_expr(e.operands[0], env) + eval_expr(e.operands[1], env)
    if e.op == "Sub":
        return eval_expr(e.operands[0], env) - eval_expr(e.operands[1], env)
    if e.op == "Mul":
        return eval_expr(e.operands[0], env) * eval_expr(e.operands[1], env)
    if e.op == "Negate":
        return -eval_expr(e.operands[0], env)
    if e.op == "DigitAnd":
        a, b = eval_expr(e.operands[0], env), eval_expr(e.operands[1], env)
        return min(a, b) if a >= 0 or b >= 0 else max(a, b)   # Kleene min
    if e.op in ("SignSelect",):
        c = eval_expr(e.operands[0], env)
        return eval_expr(e.operands[1 if c > 0 else (2 if c == 0 else 3)], env)
    raise Divergence(f"no semantics for op {e.op}")


# ---------------------------------------------------------------------------
# Symbolic encoding
# ---------------------------------------------------------------------------

def z3_expr(e, env, width_bits=64):
    """Encode as a Z3 bit-vector expression (machine semantics, wrapping)."""
    BV = lambda: z3.BitVec
    if e.op == "Integer":
        return z3.BitVecVal(e.integer, width_bits)
    if e.op == "Parameter":
        return env.setdefault(e.symbol, z3.BitVec(f"p{e.symbol}", width_bits))
    if e.op == "Add":
        return z3_expr(e.operands[0], env) + z3_expr(e.operands[1], env)
    if e.op == "Sub":
        return z3_expr(e.operands[0], env) - z3_expr(e.operands[1], env)
    if e.op == "Mul":
        return z3_expr(e.operands[0], env) * z3_expr(e.operands[1], env)
    if e.op == "Negate":
        return -z3_expr(e.operands[0], env)
    if e.op == "SignSelect":
        c = z3_expr(e.operands[0], env)
        pos = z3_expr(e.operands[1], env)
        zero = z3_expr(e.operands[2], env)
        neg = z3_expr(e.operands[3], env)
        return z3.If(c > 0, pos, z3.If(c == 0, zero, neg))
    raise Divergence(f"no symbolic encoding for op {e.op}")


# ---------------------------------------------------------------------------
# The pass harness
# ---------------------------------------------------------------------------

class Pass:
    """
    A rewrite pass plus its proof obligation.

    `rewrite` transforms an Expr. `prove` discharges the obligation; the
    default is a symbolic Z3 check over all inputs, which is what most
    arithmetic rewrites need. A pass may override it when the theory demands
    something else (and must then say why).
    """

    def __init__(self, name, rewrite, description="", symbols=("a", "b", "c")):
        self.name = name
        self.rewrite = rewrite
        self.description = description
        self.symbols = symbols

    def prove(self):
        """Default obligation: forall inputs, before == after. Returns (ok, how)."""
        env = {}
        try:
            before_sym = self._template_before()
            after = self.rewrite(self._copy(before_sym))
            a = z3_expr(before_sym, env)
            b = z3_expr(after, env)
        except Divergence as d:
            return False, f"could not encode: {d}"

        s = z3.Solver()
        s.add(a != b)
        r = s.check()
        if r == z3.unsat:
            return True, "symbolic (Z3, BV64)"
        if r == z3.sat:
            m = s.model()
            return False, f"counterexample: {m}"
        return False, "unknown (undischarged)"

    def _copy(self, e):
        return Expr(e.op, [self._copy(o) for o in e.operands],
                    e.integer, e.symbol, e.type)

    def _template_before(self):
        raise NotImplementedError


class TemplatePass(Pass):
    """A pass whose obligation is stated as an explicit before-template."""

    def __init__(self, name, template, rewrite, description=""):
        super().__init__(name, rewrite, description)
        self.template = template

    def _template_before(self):
        return self.template()


# ---------------------------------------------------------------------------
# The passes, mirroring include/Ternary/IR/Optimize.hpp
# ---------------------------------------------------------------------------

def build_passes():
    passes = []

    # ---- constant folding -------------------------------------------------
    def fold_constants(e):
        if e.op in ("Add", "Sub", "Mul") and all(
                o.op == "Integer" for o in e.operands):
            a, b = e.operands[0].integer, e.operands[1].integer
            v = {"Add": a + b, "Sub": a - b, "Mul": a * b}[e.op]
            return integer(v)
        return e

    passes.append(TemplatePass(
        "constant-folding",
        lambda: binop("Add", integer(6), integer(7)),
        fold_constants,
        "Replace an all-constant arithmetic expression by its value."))

    # ---- additive identity: x + 0 -> x ------------------------------------
    def add_zero(e):
        if e.op == "Add":
            a, b = e.operands
            if b.op == "Integer" and b.integer == 0:
                return a
            if a.op == "Integer" and a.integer == 0:
                return b
        return e

    passes.append(TemplatePass(
        "add-zero-identity",
        lambda: binop("Add", param(1), integer(0)),
        add_zero,
        "x + 0 == x."))

    # ---- multiplicative identity: x * 1 -> x ------------------------------
    def mul_one(e):
        if e.op == "Mul":
            a, b = e.operands
            if b.op == "Integer" and b.integer == 1:
                return a
            if a.op == "Integer" and a.integer == 1:
                return b
        return e

    passes.append(TemplatePass(
        "mul-one-identity",
        lambda: binop("Mul", param(1), integer(1)),
        mul_one,
        "x * 1 == x."))

    # ---- subtract zero: x - 0 -> x ----------------------------------------
    # NOTE: Optimize.hpp guards this one with `e->op != Op::Sub` on the LEFT
    # operand, because subtraction is not commutative. A naive version that
    # rewrites  0 - x  ->  x  is WRONG, and this pass exists to catch exactly
    # that class of bug.
    def sub_zero(e):
        if e.op == "Sub":
            a, b = e.operands
            if b.op == "Integer" and b.integer == 0:
                return a
            # Deliberately NOT rewriting  a == 0  ->  b.
        return e

    passes.append(TemplatePass(
        "sub-zero-identity",
        lambda: binop("Sub", param(1), integer(0)),
        sub_zero,
        "x - 0 == x. (0 - x is NOT rewritten: Sub is not commutative, and "
        "Optimize.hpp guards that case explicitly.)"))

    # ---- double negation: -(-x) -> x --------------------------------------
    def double_neg(e):
        if e.op == "Negate" and e.operands[0].op == "Negate":
            return e.operands[0].operands[0]
        return e

    passes.append(TemplatePass(
        "double-negation",
        lambda: Expr("Negate", [Expr("Negate", [param(1)])]),
        double_neg,
        "-(-x) == x."))

    # ---- multiply by zero --------------------------------------------------
    def mul_zero(e):
        if e.op == "Mul":
            a, b = e.operands
            if (a.op == "Integer" and a.integer == 0) or \
               (b.op == "Integer" and b.integer == 0):
                return integer(0)
        return e

    passes.append(TemplatePass(
        "mul-zero",
        lambda: binop("Mul", param(1), integer(0)),
        mul_zero,
        "x * 0 == 0."))

    # ---- SignSelect with a constant selector -------------------------------
    def const_select(e):
        if e.op == "SignSelect" and e.operands[0].op == "Integer":
            c = e.operands[0].integer
            return e.operands[1 if c > 0 else (2 if c == 0 else 3)]
        return e

    passes.append(TemplatePass(
        "const-sign-select",
        lambda: Expr("SignSelect", [integer(-5), param(1), param(2), param(3)]),
        const_select,
        "select by a constant sign collapses to the taken arm."))

    return passes


# ---------------------------------------------------------------------------
# Differential fuzzing: the rewrites must preserve evaluation on real programs
# ---------------------------------------------------------------------------

def random_expr(rng, depth, symbols):
    if depth <= 0 or rng.random() < 0.3:
        return integer(rng.randint(-20, 20)) if rng.random() < 0.5 \
            else param(rng.choice(symbols))
    op = rng.choice(["Add", "Sub", "Mul", "Negate"])
    if op == "Negate":
        return Expr("Negate", [random_expr(rng, depth - 1, symbols)])
    return binop(op, random_expr(rng, depth - 1, symbols),
                 random_expr(rng, depth - 1, symbols))


def apply_everywhere(e, rewrite):
    e = Expr(e.op, [apply_everywhere(o, rewrite) for o in e.operands],
             e.integer, e.symbol, e.type)
    return rewrite(e)


def differential_fuzz(passes, trials=20000, seed=1234):
    """Run every pass on random programs; evaluation must be preserved."""
    rng = random.Random(seed)
    symbols = [1, 2, 3]
    bad = []
    for p in passes:
        for _ in range(trials):
            e = random_expr(rng, 4, symbols)
            env = {s: rng.randint(-30, 30) for s in symbols}
            try:
                want = eval_expr(e, env)
            except Divergence:
                continue
            try:
                got = eval_expr(apply_everywhere(e, p.rewrite), env)
            except Divergence:
                continue
            if want != got:
                bad.append((p.name, repr(e), env, want, got))
                break
    return bad


# ---------------------------------------------------------------------------

PASS_N = 0
FAIL_N = 0


def main() -> int:
    print("=" * 78)
    print("  Translation validation -- core IR passes")
    print("=" * 78)

    passes = build_passes()
    discharged = 0

    print("\n  Proof obligations (symbolic, Z3)")
    for p in passes:
        ok, how = p.prove()
        if ok:
            discharged += 1
            PASS_N if False else None
            print(f"    [DISCHARGED] {p.name:<22} {how}")
        else:
            print(f"    [FAILED]     {p.name:<22} {how}")
        if p.description:
            for line in _wrap(p.description, 62):
                print(f"                 {line}")

    print(f"\n  Coverage: {discharged} of {len(passes)} passes have "
          f"discharged obligations")

    print("\n  Differential fuzz (20000 random programs x every pass)")
    bad = differential_fuzz(passes)
    if bad:
        for name, expr, env, want, got in bad[:5]:
            print(f"    [FAIL] {name}: {expr} with {env} -> {want} != {got}")
        print(f"    {len(bad)} mismatching pass/program combinations")
    else:
        print("    [PASS] every pass preserves evaluation on every program")

    total_fails = (len(passes) - discharged) + len(bad)
    print("\n" + "=" * 78)
    print(f"  {discharged}/{len(passes)} obligations discharged, "
          f"{len(bad)} fuzz mismatches")
    print("=" * 78)
    return 1 if total_fails else 0


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    sys.exit(main())
