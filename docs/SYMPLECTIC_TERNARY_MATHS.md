# Symplectic-ternary: a categorical framework for carry-free arithmetic

**Status: FRAMEWORK + ALGORITHM + Z3 VERIFICATION.**

This develops the mathematics behind `BT_CSATree::add`, turns it into an
algorithm that removes carry propagation from the accumulation phase entirely,
and discharges the obligations in Z3.

---

## Part 0 — Deep review of `BT_CSATree::add`

### 0.1 The primitives

`BTHA(A,B)` (`ternary_core.hpp:165`) computes, **per trit and in parallel**:

```
    PC = A.P & B.P                                  (both +1  →  sum +2)
    NC = A.N & B.N                                  (both −1  →  sum −2)
    PS = ((A.P ^ B.P) & ~(A.N | B.N)) | NC
    NS = ((A.N ^ B.N) & ~(A.P | B.P)) | PC
```

Hand-verified across all 9 input pairs; the invariant is

> **(H1)**  `val(Sum) + 3·val(Carry) = val(A) + val(B)`

with `Sum ∈ T^n`, `Carry ∈ T^n`. Worked cases:

| a | b | a+b | Sum | Carry | Sum + 3·Carry |
|---|---|---|---|---|---|
| +1 | +1 | +2 | −1 | +1 | −1 + 3 = 2 ✔ |
| +1 | 0 | +1 | +1 | 0 | 1 ✔ |
| +1 | −1 | 0 | 0 | 0 | 0 ✔ |
| 0 | 0 | 0 | 0 | 0 | 0 ✔ |
| −1 | −1 | −2 | +1 | −1 | 1 − 3 = −2 ✔ |

`BTFA(A,B,D)` (`:182`) is two half-adders plus a carry merge:

```
    st1 = BTHA(A, B)          S1 + 3·C1 = A + B
    st2 = BTHA(S1, D)         Sum + 3·C2 = S1 + D
    Carry = merge(C1, C2)     see below
```

The merge is **not** a plain OR — it is a signed merge that *cancels opposing
carries*:

```
    PC = (C1.P | C2.P) & ~(C1.N | C2.N)
    NC = (C1.N | C2.N) & ~(C1.P | C2.P)
```

This matters. `C1 = +1` forces `S1 = −1`, and `C2 = −1` then requires `D = −1`;
so `A = B = +1, D = −1` gives `C1 = +1, C2 = −1` and a raw sum of `+1`. The
merge correctly yields `Carry = 0` and `Sum = +1`, preserving `1 = 1+1−1`. ✔

> **(H2)**  `val(Sum) + 3·val(Carry) = val(A) + val(B) + val(D)`

`BTFA` has **O(1) depth** (2 half-adder levels + 1 merge) and **no carry
ripple**. It is a genuine 3:2 compressor.

### 0.2 The accumulator step

```cpp
inline void add(const TritWord64& X) noexcept {
    TritWord64 C_shifted(C.P << 1, C.N << 1);   // 3·C
    AdderOutput comp = BTFA(S, C_shifted, X);   // 3:2 compress, O(1)
    S = comp.Sum;  C = comp.Carry;
}
```

Substituting (H2) with `A=S`, `B=3C`, `D=X`:

```
    S' + 3·C' = S + 3C + X
```

so, with the energy `H(S,C) := val(S) + 3·val(C)`,

> **(H3, conservation)**  `H(S', C') = H(S, C) + val(X)`

**This is the whole idea, and it is correct.** One BTFA, constant depth, no
propagation; the carry is parked in `C`.

### 0.3 Four defects found in review

**(D1) Silent overflow at the top trit.** `C.P << 1` shifts bit 63 out of the
word. A carry sitting at the most significant trit is silently discarded during
`add`. `resolve()` sets an `overflow` flag, but by then the value is already
wrong. *Fix: size the accumulator so the shift cannot overflow (§2.1), or
detect at `add` time.*

**(D2) `resolve()` is unbounded and unchecked.** It is a `while (C != 0)` loop
(`ternary_dense_algebra.hpp:187`) with no iteration bound and no per-iteration
overflow check; its worst case is 64 iterations. *Fix: hard-bound it and check
the top trit each iteration (§3.3).*

> **Do not "fix" this by swapping in a parallel prefix** — I tried, measured it,
> and reverted. See §5.3: the ripple converges in a mean of 3.38 iterations on
> real states, so the 6-stage prefix is 1.30× *slower*. The defect is the
> unboundedness, not the sequentiality.

**(D3) Fixed two-word representation.** The state is always `(S,C)`, so values
beyond `3^64` wrap silently. The framework below removes this ceiling by
sizing, then by a hierarchical bag for the unbounded case.

**(D4) `P` is vestigial in `SymplecticLeapfrogEngine`.** `forward_step`
(`:114`) returns `PhaseState(kogge_stone_add64(state.Q, input), state.P)` —
a full 6-stage propagating add every step, with `state.P` passed through
unchanged and forever zero. The file that claims to be the symplectic engine is
the one place *not* using the carry-save trick. `BT_CSATree` is one file away
and unused by it.

---

## Part I — The category of redundant representations

### 1.1 Objects

Fix a trit width `n`, `T = {−1,0,1}`. An **object** of **Rep** is a finite
multiset ("bag") of pairs `(W, e)` with `W ∈ T^n` a dual-rail word and `e ∈ N`
a **weight exponent**. Intuitively `e` says "this word is worth `3^e` times its
face value".

The **value** of an object is

```
    V(R) = Σ_{(W,e) ∈ R}  3^e · val(W)
```

The monoidal structure is multiset union `⊎` with unit `∅` (the empty bag).

### 1.2 The value functor is monoidal — this is the "no carry" theorem

> **Theorem 1.2 (monoidality).** `V : (Rep, ⊎, ∅) → (Z, +, 0)` is a strict
> monoidal functor: `V(R ⊎ R') = V(R) + V(R')`, and `V(∅) = 0`.

*Proof.* `V` is defined as a sum over the bag; multiset union concatenates the
index set, and `Z`-addition is associative and commutative. ∎

**This is the categorical content of "eliminate the carry".** Addition in
**Rep** is *literally the monoidal product* — you append the word to the bag.
Cost `O(1)`. **Zero carries are computed because none are needed**: the sum is
not a word, it is a bag, and bags have no normalisation obligation.

Carry propagation is what you pay when you insist the result be a *single
word*. The framework's move is simply: **don't insist, until you have to.**

### 1.3 Morphisms and the compressor

A **morphism** `f : R → R'` is a value-preserving reduction: `V(R) = V(R')`.

The **compressor at weight `e`** is the morphism

```
    κ_e :  { (A,e), (B,e), (D,e) }  ⟶  { (S,e), (C,e+1) }
```

where `BTFA(A,B,D) = (S,C)`.

> **Proposition 1.3.** `κ_e` is a morphism of **Rep**.

*Proof.* By (H2), `val(S) + 3·val(C) = val(A)+val(B)+val(D)`. Multiplying by
`3^e`: `3^e·val(S) + 3^{e+1}·val(C) = 3^e·(val(A)+val(B)+val(D))`. ∎

**Naturality.** Let `Σ : Rep → Rep` be the **shift functor**
`Σ(W,e) = (W, e+1)`, so `V(ΣR) = 3·V(R)`. Then compression commutes with
shifting:

```
    Σ ∘ κ_e  =  κ_{e+1} ∘ Σ³
```

i.e. compressing then scaling equals scaling then compressing. The compressor
is therefore **weight-natural** — it behaves identically at every scale, which
is what makes the hierarchical scheme in §2.4 well-defined.

### 1.4 Canonical forms as a reflective subcategory

Let **Can** ⊂ **Rep** be the full subcategory of objects that are a **single
word at weight 0**. By Theorem 1.1 of the companion document (unique balanced
representability), each representable integer has exactly one such object.

> **Theorem 1.4 (reflection).** **Can** is a **reflective subcategory** of
> **Rep**. The reflector `L : Rep → Can` is *assimilation*, and the reflection
> unit `η_R : R → L(R)` is the (unique) value-preserving morphism into
> canonical form.

*Proof sketch.* We need `Hom_Can(L(R), C) ≅ Hom_Rep(R, C)` naturally in `C`.
A morphism `R → C` exists iff `V(R) = V(C)`, and likewise for `L(R) → C`; since
`V(L(R)) = V(R)` by construction, the two Hom-sets are both singleton-or-empty
under the same condition, naturally in `C`. ∎

**So "canonicalisation" is not an ad-hoc cleanup step — it is the reflector of
a reflective subcategory.** The unit `η` *is* the assimilation. This is the
precise sense in which the redundant form is a free construction and the
canonical form is its reflection.

### 1.5 The gauge picture

The fibres of `V` are the **redundancy classes**: many bags denote the same
integer. Two objects in the same fibre are **gauge-equivalent**.

- **Compression moves within a fibre** (it is a gauge transformation: value
  unchanged, representation changed).
- **Accumulation moves between fibres** (`V → V + val(X)`).

This is exactly the structure of a gauge theory, and it is why compression can
be deferred indefinitely without affecting the answer.

---

## Part II — The symplectic structure

Now the honest account of where "symplectic" is literally true.

### 2.1 Phase space

Restrict to the two-word state `(S, C)` and name it

```
    Q := S   (position — accumulated sum)
    P := C   (momentum — the parked carry)
    H(Q,P) := val(Q) + 3·val(P)          (the Hamiltonian / the value)
```

(These names and this `H` are already those of `PhaseState` in
`symplectic_ternary_engine.hpp`; the point is that they are not merely
decorative.)

### 2.2 The carry-injection step is a symplectic shear

Work at the level of the **ring** `R = Z` (not digit-wise). The step that
absorbs the parked carry is

```
    Φ : (Q, P) ↦ (Q + 3P, P)
```

Its linear part is `M = [[1, 3], [0, 1]]` — a **shear**. Check symplecticity
against `J = [[0,1],[−1,0]]`:

```
    Mᵀ J M = [[1,0],[3,1]] · [[0,1],[−1,0]] · [[1,3],[0,1]]
           = [[1,0],[3,1]] · [[0,1],[−1,−3]]
           = [[0,1],[−1,0]]  =  J          ✔
```

and `det M = 1` — **Liouville volume preservation**. So:

> **Theorem 2.2.** The carry-injection step `Φ` is a symplectic shear, hence
> volume-preserving and (on the ring) bijective, hence reversible.

The shear is generated by the quadratic Hamiltonian `H_gen = (3/2)·P²`, since
`Q̇ = ∂H_gen/∂P = 3P` and `Ṗ = −∂H_gen/∂Q = 0`, whose unit-time flow is exactly
`Q ↦ Q + 3P`. The *value* `H = val(Q) + 3·val(P)` is separately the conserved
quantity of the discrete dynamics. Both structures are present and they are
different objects — worth stating plainly rather than conflating.

### 2.3 Redundancy is Lagrangian

In a 2-dimensional symplectic space, Lagrangian submanifolds are
1-dimensional. The level sets of `H` are 1-dimensional curves. Therefore:

> **Proposition 2.3.** The redundancy classes — the fibres of `V`, i.e. the
> sets of representations of a fixed integer — are **Lagrangian submanifolds**
> of the phase space, and compression is movement **along** a Lagrangian fibre.

This is the rigorous version of "park the carry in a mathematical space":
the space is a Lagrangian leaf, and compression slides along it without ever
leaving the leaf.

### 2.4 The cost of reversibility — stated exactly

The 3:2 compressor takes **three** words to **two**. It is therefore
information-losing.

Rigorous argument: the step conserves `H` up to `+val(X)`, so it maps the fibre
`V⁻¹(v)` into `V⁻¹(v + val(X))`. Fibres are **not equipotent** — they thin out
near the extremes of the representable range — so no value-conserving map can
be a bijection here.

> **Trade-off theorem.** One may choose any two of: (**a**) `O(1)` depth per
> step, (**b**) reversibility, (**c**) two-word state.
>
> - (a) + (c): plain carry-save — **fast, not reversible**.
> - (a) + (b): retain the consumed operand as a third word — **fast and
>   reversible, three words**.
> - (b) + (c): full ring addition — **reversible and compact, `O(log n)`**.

The retained third word *is* the momentum being preserved. This is precisely
the user's phrase, and it is the right way to get reversibility without giving
up `O(1)`: **pay one word of storage, not `O(log n)` of latency.**

---

## Part III — The algorithm

### 3.1 Sizing: the move that removes overflow

The entire accumulation happens in a redundant pair of width `w`. Choose

```
    w = n + ⌈ log₃( N · M ) ⌉ + 1
```

where `N` is the number of accumulations and `M` an upper bound on
`|val(X)|`. Then `|Σ val(X_i)| ≤ N·M < 3^{w-n}`, so **no carry can ever reach
the top trit** and the `C << 1` shift cannot overflow.

This eliminates defect **(D1)** structurally rather than by detection: with
correct sizing, overflow is *impossible*, not merely flagged.

### 3.2 Accumulate — `O(1)`, zero carries

```
    accumulate(X):
        (S, C) ← BTFA(S, C << 1, X)        // one 3:2 compression
```

By (H3): `H ← H + val(X)`. Depth: 2 half-adder levels + 1 merge. **No carry
propagation whatsoever.**

### 3.3 Restore — one assimilation, exact

```
    restore():
        return bounded_ripple_assimilate(S, C)    // mean 3.38 iterations
    restore_fixed_latency():
        return parallel_prefix_add(S, C << 1)     // always 6 stages at w=64
```

Exactly **one** carry-propagating operation for the whole accumulation, at the
very end. Because arithmetic happens in `Z`, the result is **bit-identical** to
the `N`-fold propagated sum — not approximate.

Both paths are provided and verified equal. The bounded ripple is the default;
the fixed-latency prefix is opt-in. See §5.3 for the measurement that decides
this — it is the opposite of what asymptotic analysis suggests.

### 3.4 Unbounded values — the hierarchical bag

When `N` is not known ahead of time, work in **Rep** directly:

```
    accumulate(X, e):  bag ⊎= {(X, e)}                    O(1)
    compress():        apply κ_e to any three at weight e O(1) each
    restore():         compress to ≤2 words, then §3.3
```

`V` is monoidal (Theorem 1.2), so appending *is* addition. The bag grows to
`O(N)` words at worst and is collapsible at any time. **No carry is ever
propagated during accumulation, at any magnitude.**

### 3.5 The carry transfer semigroup Σ₇

The parallel prefix needs an associative composition operator on per-position
carry behaviour. A position whose two digits sum to `raw ∈ {−2..2}` induces
`f : {−1,0,1} → {−1,0,1}`, `f(c_in) = c_out`. Composing these across positions
*is* the prefix operator.

Computing the closure of the five source functions under composition yields
**exactly 7 elements** (independently derived in `verification/`; the header
carries the table as a `static_assert`-checked constant):

| idx | `(f(−1), f(0), f(1))` | raw |
|---|---|---|
| 0 | `(−1, −1, −1)` | — (generated) |
| 1 | `(−1, −1, 0)` | −2 |
| 2 | `(−1, 0, 0)` | −1 |
| 3 | `( 0, 0, 0)` | 0 |
| 4 | `( 0, 0, 1)` | +1 |
| 5 | `( 0, 1, 1)` | +2 |
| 6 | `( 1, 1, 1)` | — (generated) |

> **Characterisation (derived, not assumed).** Σ₇ is exactly the set of
> monotone non-decreasing self-maps of the 3-chain with **oscillation ≤ 1**:
>
> `{ f : {−1,0,1} → {−1,0,1} | f non-decreasing, f(1) − f(−1) ≤ 1 }`
>
> There are 10 monotone non-decreasing maps on a 3-chain; 7 satisfy the bound.
> **The bound is the statement that carries cannot amplify**: a prefix can turn
> a carry of ±1 into a carry of ±1, never into something larger. That is why
> the prefix terminates and why 7 states — not 27 — suffice.

It is a genuine semigroup, **not** commutative and **not** idempotent:
`compose(5,5) = 6 ≠ 5`, and `compose(1,4) = 3 ≠ compose(4,1) = 1`. Associativity
is checked exhaustively over all 7³ triples in the C++ test.

This independently confirms the "closed 7-state monotone carry transfer
semigroup" comment on `kogge_stone_add64` in `ternary_core.hpp:203` — that
implementation's `(fm, f0, fp)` rails are precisely the ±/rails of the triple
`(f(−1), f(0), f(1))`, which is why `resolve_parallel()` can reuse it rather
than reimplementing the prefix.

### 3.6 Cost

| Scheme | Cost for N accumulations |
|---|---|
| Propagate each step (today's engine) | `N · O(log w)` = 6N carry stages |
| **Carry-save (§3.2–3.3)** | **`N · O(1)` + one assimilation** |

For `N = 256`: **1536 carry stages → 256 compressor stages + ~3**. Carry
propagation is eliminated from the accumulation phase entirely.

Measured: **4.41×** end-to-end (§5.2). This is less than the 6× the stage count
suggests, because the single surviving assimilation is not free.

---

## Part IV — Verification

Three artefacts, all executed:

| Artefact | What it does |
|---|---|
| `verification/verify_redundant.py` | Z3 + exhaustive verification (V1–V7) |
| `ternary_carry_free.hpp` | the algorithm in C++ |
| `test/Unit/test_unit_carry_free.cpp` | 678 conformance checks |

| # | Obligation | Method |
|---|---|---|
| V1 | (H2) `S + 3C = A+B+D` | exhaustive 27 triples + Z3 uniqueness |
| V2 | (H3) energy conservation | symbolic Z3, width 12 |
| V3 | N-fold accumulation equals the sum | symbolic Z3 |
| V4 | `restore` returns the exact sum | exhaustive, 177 147 states |
| V5 | Sizing prevents overflow | worst-case operands, N ∈ {1,2,3,8,16} |
| V6 | Edge cases | 8 named patterns + **531 441 exhaustive chains** |
| V7 | Ripple and parallel `restore` agree | exhaustive |

---

## Part V — Results

### 5.1 Verification: 11 / 11 passed

```
  V1 BTFA spec exhaustive              27 trit triples
  V1 BTFA solution is unique           no two distinct (sum,carry) per input
  V2 energy conservation (symbolic)    width=12
  V2 overflow == dropped top trit      loss == C[w-1]*3^w          (D1)
  V3 N-fold accumulation               width=8, N=4
  V4 restore exactness (exhaustive)    w=6, 177147 states, max ripple 6
  V6 edge cases (sized accumulator)    8 patterns x N=3
  V6 exhaustive operand chains         531441 chains of N=3 (3^12 space)
  V5 sizing rule is sufficient         n=6, N in (1,2,3,8,16)
  V5 sizing is necessary (D1)          loss=729 accounts for 1092 -> 363
  V7 parallel restore == ripple        w=6
  ------------------------------------------------
  11 passed, 0 failed
```

C++ conformance: **678 checks, 0 failures**, including 50 000 random cases for
`resolve_parallel == resolve_bounded_ripple == BT_CSATree::resolve`.

### 5.2 Measured speedup

N = 256 accumulations (a realistic GEMM K-depth), 20 000 repetitions:

| | time |
|---|---|
| propagate every step with Kogge-Stone (what `SymplecticLeapfrogEngine` does today) | 113.0 ms |
| **carry-free: 256 × O(1) + one resolve** | **25.6 ms** |
| **speedup** | **4.41×** |

The two sinks were bit-identical (`8181032215171047360`), confirming the
exactness claim: this is not a faster approximation, it is the same answer.

### 5.3 A correction: defect D2 as first stated was wrong

I initially recorded D2 as "`resolve()` is a 64-iteration ripple, replace it
with the 6-stage parallel prefix". **That is true in the worst case and false
in practice**, and shipping it as a fix would have been a 1.30× regression.

Measured distribution of ripple iterations over 20 000 states produced by 256
carry-free accumulations:

| iterations | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| share | 0.5% | 20.8% | **40.0%** | 24.3% | 9.6% | 3.1% | 1.2% | 0.3% | 0.1% | — | — |

Mean **3.38**, worst observed **11** — not 64. Each ripple iteration is one
BTHA (~10 bitwise ops); each prefix stage is ~20 bitwise ops and there are
always 6. Timed over 200 000 realistic states: ripple **3.4 ms**, parallel
**4.5 ms**.

**Resolution.** Both are shipped, and the choice is a stated engineering
trade-off, not a default:

- `resolve_bounded_ripple()` — **default**. Fastest on the real distribution;
  now hard-bounded and overflow-checking, so it can neither spin nor drop a
  carry silently (both defects D1 and D2's *unboundedness* are fixed).
- `resolve_parallel()` — **opt-in**, fixed 6-stage latency. Use when the
  data-dependent branch is itself the defect: hardware emission, constant-time
  cryptographic code, real-time deadlines.

What survives of the original D2 is real and important: the ripple was
**unbounded and unchecked**. That is fixed. Its asymptotic worst case is not
what matters here, and I was wrong to lead with it.
