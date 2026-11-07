# Mathematical framework: tensors, layouts, and memory-aware management

**Status: THEORETICAL FOUNDATION. No implementation.**

This is the mathematical substrate for the vector/tensor types and the
memory-aware memory manager, and the bridge from both up to P3 (next-generation
hardware targets). Everything is stated so it can be (a) implemented directly
and (b) discharged in Z3.

The organising idea, stated up front:

> **A layout is an injective map from a logical index space to a physical
> address space.** Every question about "how is this stored", "is this access
> fast", "do these two formats agree", "how much data moves" is a question about
> the properties of that map.

---

## Part I — Trit algebra

### 1.1 Balanced ternary

Let `T = {-1, 0, 1}`. An **n-trit word** is a tuple `t = (t_0, …, t_{n-1}) ∈ T^n`
interpreted by the evaluation map

```
    val(t) = Σ_{i=0}^{n-1} t_i · 3^i
```

**Theorem 1.1 (unique representability).** `val : T^n → Z` is injective, and

```
    val(T^n) = [ -(3^n - 1)/2 ,  +(3^n - 1)/2 ] ∩ Z
```

*Proof.* Injectivity: if `t ≠ t'`, let `j` be the largest index with `t_j ≠ t'_j`.
Then

```
    |val(t) - val(t')| = |(t_j - t'_j)·3^j + Σ_{i<j} (t_i - t'_i)·3^i|
                       ≥ 3^j - Σ_{i<j} 2·3^i
                        = 3^j - (3^j - 1)
                        = 1 > 0
```

since `Σ_{i<j} 3^i = (3^j-1)/2`. This is the **dominance lemma**: no combination
of lower trits can reach the next trit's weight. It is also exactly what makes
`SIGN` correct — the sign of a word is the sign of its most significant nonzero
trit — and it is the lemma already proved in Z3 in `verification/xbt_isa.py`.

Surjectivity onto that interval: the count `|T^n| = 3^n` equals the count of
integers in the interval, and injectivity plus equal cardinality gives the
result. ∎

For `n = 40` (Int39) the range is `±(3^40-1)/2 ≈ ±6.08 × 10^18`, which is why
Int39 fits an `int64` with room for the sign.

### 1.2 Dual-rail encoding

A **dual-rail word** is a pair `(P, N) ∈ {0,1}^n × {0,1}^n` with the
*disjointness invariant* `P ∧ N = 0` (no trit is both +1 and −1).

The encoding is the isomorphism

```
    φ : (P, N) ↦ Σ_i (P_i - N_i) · 3^i
```

with inverse `φ^{-1}(v)_i = (+1 ↦ P_i=1, −1 ↦ N_i=1, 0 ↦ both 0)`.

### 1.3 Which operations are homomorphisms — the cost model's origin

This is the single most important structural fact for both the compiler and the
hardware: **dual-rail is not a ring.** Addition is *not* computed by bitwise
operations on the rails; it needs carry propagation. But negation and
lane-wise multiplication *are* bitwise.

| Operation | Rail formula | Homomorphism? | Cost |
|---|---|---|---|
| Negation | `P' = N`, `N' = P` | Yes — exact isomorphism | **0 gates** (wire permutation) |
| Lane-wise mul | `P' = (P_A∧P_B) ∨ (N_A∧N_B)`, `N' = (P_A∧N_B) ∨ (N_A∧P_B)` | Yes, on disjoint-rail words | 1 AOI22 gate/lane |
| Addition | — | **No** | carry chain (`BTHA`/`BTFA`) |

*Lane-wise multiplication correctness.* Per lane, with `a, b ∈ {-1,0,1}`:
`a·b = +1` iff `(a,b) = (+1,+1)` or `(−1,−1)` — i.e. `(P_A∧P_B) ∨ (N_A∧N_B)`.
`a·b = −1` iff `(a,b) = (+1,−1)` or `(−1,+1)` — i.e. `(P_A∧N_B) ∨ (N_A∧P_B)`.
Otherwise 0. ∎

**Consequence for scalar multiply (P1).** Because lane-wise mul has no carry,
`xbt.mul` cannot compute a general product — that is *why* the selector refuses
it. A true product must be built from shifts and adds, i.e. it must leave the
bitwise world and re-enter the carry-propagating one.

### 1.4 Exactness of ternary arithmetic

**Theorem 1.4 (exactness).** Let `A ∈ T^{m×k}`, `B ∈ T^{k×n}`. Then every
partial product `A[i,t]·B[t,j]` lies in `{-1,0,1}` and

```
    C[i,j] = Σ_t A[i,t] · B[t,j]   ∈   [-k, +k] ∩ Z
```

is computed **exactly in Z**, with no rounding at any `k` that fits the
accumulator.

*Proof.* Each partial product is in `T`, and `Z` is a ring, so the finite sum is
exact. There is no rounding step because there is no representation error: the
operands and partial products are all exact integers. ∎

**This is the decisive advantage over floating point.** Floating-point GEMM
suffers representation and rounding error; ternary GEMM is a *homomorphism into
Z*. The practical consequences are:

- Results are **bit-identical** across CPU, GPU, and any future target — so a
  cross-target differential test is an *equality* check, not a tolerance check.
  This is a far stronger CI property than anyone gets with floating point.
- The only failure mode is **overflow**, which is detectable and must be
  reported loudly (consistent with the existing `__int128` checks in the
  reference `matmul`).

---

## Part II — Tensors

### 2.1 Tensors as elements of a free module

Let the **shape** be a tuple `d = (d_0, …, d_{r-1}) ∈ N^r`, `r` the rank. The
**logical index space** is the product of finite ordinals

```
    I(d) = Z_{d_0} × Z_{d_1} × … × Z_{d_{r-1}}
```

A **tensor** of shape `d` over `T` is a function

```
    A : I(d) → T
```

i.e. an element of the free module `T^{I(d)}` of rank `|I(d)| = ∏ d_i`.

A vector is `r = 1`; a matrix is `r = 2`; a scalar is `r = 0` (so `I = {()}`, a
singleton — scalars fall out of the same definition, which is why we do not need
a separate scalar type).

### 2.2 Layouts as index morphisms

**Definition 2.1 (layout).** Let `M` be a physical address space (a finite set
of storage units — bits, bytes, or words, depending on level). A **layout** for
shape `d` is an injective map

```
    λ : I(d) → M
```

Two tensors are *the same tensor in different layouts* iff they are equal as
functions on `I(d)`; the layout only says where each entry lives.

**Definition 2.2 (strided / affine layout).** `λ` is **affine** if there is an
offset `o` and a stride vector `(s_0, …, s_{r-1})` with

```
    λ(i_0, …, i_{r-1}) = o + Σ_k s_k · i_k
```

This covers row-major (`s = (d_1, 1)`), column-major (`s = (1, d_0)`), and every
permutation and slice. Affine layouts are closed under composition with
transpose, slice, and reshape-when-contiguous.

### 2.3 Dual-rail as a layout — and the exact statement of the "K ≤ 64 trap"

For a matrix `X` with flat row-major index `e = i·c + j`, the dual-rail layout
maps into **(word, bit) space** rather than a flat address:

```
    λ_DR(e) = ( e div 64 , e mod 64 )
```

Now take the GEMM `C[m,n] = A[m,k] · B[k,n]`.

- **A** is stored row-major with `kw = ceil(k/64)` words per row:
  ```
      λ_A(i, t) = ( i·kw + (t div 64) ,  t mod 64 )
  ```
- **B** is supplied **transposed** as `B^T[n,k]` and then **word-plane-major**:
  ```
      λ_B(j, t) = ( (t div 64)·n + j ,  t mod 64 )
  ```

The "obvious" thing a caller writes instead is row-major on `B^T`:

```
      λ_naive(j, t) = ( j·kw + (t div 64) ,  t mod 64 )
```

**Theorem 2.3 (coincidence of the two B layouts).** `λ_B = λ_naive` **iff**
`kw = 1` (i.e. `k ≤ 64`) **or** `n = 1`.

*Proof.* The bit components are identical (`t mod 64`), so equality reduces to
the word components. Put `u = t div 64 ∈ [0, kw)`. We need

```
    u·n + j  =  j·kw + u        for all j ∈ [0,n), u ∈ [0,kw)
⟺  u·(n − 1) = j·(kw − 1)      for all j, u
```

- If `kw = 1`: then `u = 0` always, so LHS `= 0` and RHS `= j·0 = 0`. ✔
- If `n = 1`: then `j = 0` always, so RHS `= 0` and LHS `= u·0 = 0`. ✔
- If `kw ≥ 2` **and** `n ≥ 2`: take `u = 1`, `j = 0`. Then LHS `= n − 1 ≥ 1`
  but RHS `= 0`. ✘

∎

This is exactly the condition the GEMM conformance tests are built around, and
it is now a *theorem* rather than a warning in a docstring. It also shows why the
negative control in the test suite is essential: for `k ≤ 64` **or** `n = 1` the
two layouts are provably indistinguishable, so any test in that region is
vacuous. Only `k > 64 ∧ n > 1` has discriminating power.

**Design consequence.** `pack_matrix_b` exists because `λ_B` is not the
composition of "transpose" and "the A layout" in the naive order — it is a
**stride permutation**: the roles of `j` and `t div 64` are swapped between the
`n`-multiple and the `kw`-multiple. Any correct API must expose the map, not
describe it in prose.

### 2.4 Broadcasting is stride zero

**Definition 2.4 (broadcast).** Shapes `d` and `d'` are *compatible* if,
aligning from the right, each pair of dims is equal or one of them is 1. The
**broadcast** of `d` to `d'` is the map

```
    β : I(d') → I(d),     β(i)_k = i_k  if d_k > 1,  else 0
```

Given a strided layout with strides `s` for `d`, the broadcast tensor has the
**same** layout with stride `s_k = 0` substituted on every axis where
`d_k = 1`.

**Theorem 2.4.** Broadcasting is exactly the substitution of zero strides.
Hence broadcast is *free* — it allocates nothing and copies nothing.

This unifies broadcast, transpose, and slicing into one mechanism: all three are
stride manipulations on the same buffer.

| View | Stride operation |
|---|---|
| Transpose | permute `(s_0, …, s_{r-1})` |
| Slice | add offset, keep strides |
| Broadcast | set `s_k = 0` on size-1 axes |
| Reshape | refactor index — **requires contiguity** |

**Reshape contiguity criterion.** A reshape from `d` to `d'` is copy-free iff
there is a stride vector consistent with both, i.e. iff the flat orders agree.
Concretely, grouping consecutive axes: `(d_0,d_1)` → `(d_0·d_1)` is free iff
`s_0 = d_1 · s_1`. Otherwise it is a genuine permutation requiring a copy.

### 2.5 The view groupoid

A **view** is `(buffer, offset, shape, strides)`. Views compose when the result
is again affine; since the composition of affine maps is affine, and inversion
is not always possible (a broadcast forgets information), the structure is a
**groupoid** — a category in which the invertible morphisms are exactly the
stride permutations.

This is the right abstraction for the memory manager: a "copy" is needed
precisely when an op is *not* a groupoid morphism.

### 2.6 Contractions

The general tensor op is **contraction** (Einstein summation). For the GEMM:

```
    C[i,j] = Σ_{t} A[i,t] · B[t,j]
```

This is the contraction of `A ⊗ B` over the `t` axis. In balanced ternary it is
exact (Theorem 1.4).

**The bit-sliced formulation.** Write `A[i,t] = P_A[i,t] − N_A[i,t]` and let `X`
be int8 activations. Then

```
    C[i,j] = Σ_t (P_A[i,t] − N_A[i,t]) · X[t,j]
           = Σ_t P_A[i,t]·X[t,j]  −  Σ_t N_A[i,t]·X[t,j]
```

Each sum is a **masked accumulation**, i.e. a popcount-like operation over the
rails — which is precisely what `DOT64` computes in hardware. This is the
mathematical reason the GEMM achieves its throughput: the multiply is replaced
by a bit-mask and an add.

**Popcount identity (dual-rail dot of two ternary vectors).** For `a, b ∈ T^n`:

```
    Σ_t a_t · b_t  =  |{ t : a_t = b_t ≠ 0 }|  −  |{ t : a_t = −b_t ≠ 0 }|
```

In rails: `= popcount(P_a ∧ P_b) + popcount(N_a ∧ N_b)
             − popcount(P_a ∧ N_b) − popcount(N_a ∧ P_b)`.

This is a pure bitwise formula — **no carries** — and is the exact specification
`DOT64` must satisfy. It is the natural target for a Z3 proof.

---

## Part III — Memory-aware memory management

The layout map `λ` is what connects the algebra to the machine. Everything in
this part is derived from it.

### 3.1 Locality is a property of λ

Fix a memory level with line size `L` (in the same units as `M`). The **line
map** is

```
    ℓ(i) = λ(i) div L
```

Two logical indices `i, i'` **share a line** iff `ℓ(i) = ℓ(i')`. An access
sequence `i^{(1)}, i^{(2)}, …` is *cache-friendly* iff the image sequence
`ℓ(i^{(k)})` has high temporal locality.

**This is why layout choice is not cosmetic.** For a strided layout with stride
`s`, consecutive logical steps along axis `k` map to addresses `s_k` apart. If
`s_k · sizeof(elem) ≥ L`, every step along that axis is a **new cache line** —
a worst-case access pattern. The layout map makes that a computable predicate.

### 3.2 Tiling as an optimisation over the index space

A **tile** is a sub-block `J ⊆ I(d)`, typically a product of intervals.
Tiling is the choice of `J` so that:

1. `|λ(J)|` fits the level's capacity `S`, and
2. the ratio (useful work)/(bytes moved) is maximised.

**Theorem 3.2 (GEMM working set).** For a tile `(b_m, b_n, b_k)` of
`C = A·B`, the working set at a level in units of elements is

```
    W = b_m·b_k  +  b_k·b_n  +  b_m·b_n
```

and the arithmetic intensity (MACs per element loaded) is

```
    I(b_m, b_n, b_k) = (b_m · b_n · b_k) / W
```

*Proof.* The tile of `C` is `b_m·b_n`, requiring `b_m·b_k` of `A` and `b_k·b_n`
of `B`; the work is `b_m·b_n·b_k` MACs. ∎

*Optimisation.* Maximise `I` subject to `W ≤ S`. By AM–GM, `W` is minimised
relative to the product `b_m·b_n·b_k` when the three terms are balanced, giving
the classical guidance: choose `b_m ≈ b_n ≈ √(S/3)` (with `b_k` then set by the
remaining budget and by register pressure). The point is that this is now
**derived**, not hand-tuned.

**Crucially, `S` is measured in elements — and elements are trits.** So bits per
trit enters the capacity constraint directly:

```
    S_elements = S_bytes · 8 / bits_per_trit
```

At 1.6 bits/trit (base-243) versus 2 bits/trit (dual-rail) the same cache holds
**25 % more elements**, which propagates straight through Theorem 3.2 into a
larger tile and higher intensity. This is the quantitative statement of the
bandwidth claim — it is a layout-dependent cache-capacity effect, not a slogan.

### 3.3 The roofline, and a decision procedure for layout

Machine balance at a level with peak compute `F` (MAC/s) and bandwidth `Bw`
(bytes/s):

```
    B_machine = F / Bw        (MACs per byte)
```

A kernel is **compute-bound** iff `I ≥ B_machine`, else **memory-bound**.

**Decision procedure (which layout to use when):**

1. Compute `bits_per_trit(layout)` → element capacity → optimal tile (§3.2) →
   intensity `I`.
2. If `I < B_machine`: the kernel is memory-bound. **Choose the denser layout**
   (base-243 at 1.6 bits/trit) — the capacity gain converts directly into
   intensity.
3. If `I ≥ B_machine`: compute-bound. **Choose the layout with the cheaper
   native ops** (dual-rail: negation is free, lane-mul is one gate), since
   unpacking costs would now dominate.

This replaces "we benchmarked it once" with a per-target, per-shape decision
computed from the target's own `F`, `Bw`, and capacities.

### 3.4 The memory hierarchy as a tower of layout maps

Model level `ℓ` as `(S_ℓ, Bw_ℓ, L_ℓ, NativeLayouts_ℓ)` — capacity, bandwidth,
line size, and the set of layouts it can hold natively.

**Transfer cost.** Moving a block `J` between levels `ℓ → ℓ'` costs

```
    cost(J, ℓ → ℓ') = |J| · elemsize(λ_source) / Bw_{ℓ,ℓ'}
                    + [ λ_target ∉ NativeLayouts_{ℓ'} ] · convert(J, λ_source → λ_target)
```

The bracket is 1 if the destination cannot hold the format natively, in which
case a conversion is forced. **Layout choice and data placement are therefore a
single joint optimisation**, not two independent decisions.

### 3.5 Allocation as interval graph colouring

Given liveness intervals for buffers (from the IR), two buffers may share
storage iff their intervals are disjoint. This is exactly **graph colouring on
an interval graph**, which is solvable greedily and optimally in linear time by
left-edge (linear scan) ordering — the same algorithm as register allocation,
generalised from registers to buffers.

```
    reuseable(b1, b2)  ⟺  live(b1) ∩ live(b2) = ∅
    arena_size         =  max over time t of  Σ_{b live at t} size(b)
```

That maximum is the arena's high-water mark, and is the exact number the
allocator must reserve. This makes the memory manager's footprint **provably
minimal** for a greedy policy, and gives an assertable invariant: at every
program point, outstanding bytes ≤ the computed high-water mark.

### 3.6 Track A — implicit hierarchy (caches)

Present-day CPUs/GPUs manage levels implicitly. The compiler's job is therefore
**only to choose tiles and layouts** (§3.2–3.3) and to order accesses so that
`ℓ(i)` has good locality. No explicit movement instructions are emitted.

Consequences for the plan:
- Runtime dispatch by detected capability (AVX-512 / AVX2 / NEON / scalar).
- Tile defaults **computed** from detected cache sizes, not hardcoded.
- Because ternary GEMM is exact (§1.4), every SIMD path is verified by
  **bit-equality** against the scalar reference.

### 3.7 Track B — explicit hierarchy (scratchpads, accelerators)

Next-generation targets expose levels explicitly. Now the compiler must also:

1. **Insert the movement** (DMA / scratchpad transfers) — the tile choice from
   §3.2 becomes an actual instruction schedule, not just a loop-nest order.
2. **Honour per-level native layouts** (§3.4) — including conversions forced at
   boundaries.
3. **Pipeline** transfer with compute (double buffering) so bandwidth and
   compute overlap; the tile parameters then come from a *two-constraint*
   system: the working set must fit **twice** for double buffering.

The same `S_ℓ` and `Bw_ℓ` numbers drive both tracks. **This is the unification:**
Track A and Track B differ in who issues the movement, not in the mathematics of
choosing it.

---

## Part IV — How this builds up to P3

P3 (next-generation hardware) needs to *describe* a target. The framework above
says precisely which numbers a description must supply, because each one is a
parameter in a theorem in Part III:

| Concept | Parameter | Origin |
|---|---|---|
| Cache/scratchpad tiling | `S_ℓ` per level | Theorem 3.2 |
| Bandwidth, transfer cost | `Bw_ℓ` per level | §3.4 |
| Line size, locality predicate | `L_ℓ` per level | §3.1 |
| Layout choice & conversions | `NativeLayouts_ℓ` | §3.3, §3.4 |
| Trit width → `kw` | `native_trit_width` | §2.3 |
| Whether `xbt.mul` suffices | `has_scalar_mul` | §1.3 |
| Free ops (neg, lane-mul) | op capability table | §1.3 |
| Exactness / overflow bound | accumulator width, `max |C| ≤ k` | Theorem 1.4 |

So the capability contract is not an ad-hoc wishlist — **it is the parameter
list of the memory and layout cost model.** That is what makes a target
description sufficient to drive code generation, and it is why the mathematics
has to come first.

---

## Part V — What a "target description format" is

You asked what this means, so: **plainly, then concretely.**

### The problem

Right now, to support a new chip you write C++ inside this repository — a
`Backend` subclass, a `TargetDescription` filled in by hand, instruction
selection rules. That means:

- you must edit the compiler's source to add a target;
- two hardware teams working at once collide in the same files;
- the only way to know a description is complete is to compile and see.

### The idea

A **target description format** is a *machine-readable file* that states what a
target is and can do. The compiler reads the file at build or run time instead
of being edited. Adding a target becomes *adding a file*.

The `TargetDescription` struct in `include/Ternary/Target/TargetDescription.hpp`
is already a proto-version of this — it has fields like:

```cpp
unsigned native_trit_width{64};   // trits per native ternary word
bool     has_scalar_mul{false};   // true integer multiply in hardware
bool     has_lane_mul{true};      // native per-lane (SIMD) multiply
std::set<Layout> native_layouts;  // layouts held natively
unsigned ternary_registers{32};
```

The question is only: **what language should that file be written in?**

### Sketch of one

```
target: example-trit-accel
triple: tritv1-unknown-none

datapath:
  native_trit_width: 128
  max_immediate_trits: 5
  ops: [ksadd, btha, btfa, sign, cmov, dot64, lane_mul]
  has_scalar_mul: false

registers: { ternary: 64, scalar: 32, zero_hardwired: true }

memory:
  - level: scratchpad    capacity: 256KiB   bandwidth: 512 GiB/s  line: 64B
  - level: main          capacity: 8GiB     bandwidth: 200 GiB/s  line: 128B
  native_layouts: [dual_rail, base243]

costs: { ksadd: 1, lane_mul: 1, dot64: 8, shl3: 2 }
```

A backend generator reads that and emits a compiling skeleton plus a
pre-wired conformance harness.

### The three realistic options

**1. YAML** — the sketch above.
- *Pros:* by far the most readable; comments; the natural choice for a file
  humans author and review; trivial parsers in C++, Python, Rust.
- *Cons:* needs a parser dependency (yaml-cpp, or a vendored minimal one);
  not validated by the compiler, so a typo is a runtime error unless we add
  schema checking.

**2. JSON** — same content, brace syntax.
- *Pros:* parsers everywhere, including in CMake and Python with zero
  dependencies; trivially machine-generated and machine-checked; JSON Schema
  gives free validation.
- *Cons:* no comments; noticeably noisier to hand-write for large
  descriptions.

**3. TableGen (`.td`)** — the LLVM/MLIR dialect already uses it
(`include/Ternary/TernaryOps.td`).
- *Pros:* already a dependency for MLIR builds; description and op definitions
  would share one language and one generator; compile-time checking, so errors
  surface at build time.
- *Cons:* **adds a build dependency on an LLVM tool (`mlir-tblgen`) for
  non-MLIR builds too** — and most of this project's builds are non-MLIR. That
  is a real cost for a project that currently builds standalone with just a C++
  compiler. Steeper learning curve for hardware contributors.

### Where I land

I'd lean **YAML, with a JSON Schema for validation**, and generate C++ from it —
keeping TableGen strictly for the MLIR dialect definitions where it already
lives. Rationale: hardware teams are the authors of these files, YAML is the
most writable by non-compiler people, and it avoids dragging an LLVM tool into
the standalone build.

But it's your call — the trade-off hinges on whether you'd rather have one
language for descriptions *and* dialect ops (TableGen), at the price of the
build dependency.

**Tell me which of YAML / JSON / TableGen you want and I'll proceed to P3 on
that basis.**

---

## Verification plan for all of the above

Every theorem here is a proof obligation:

| Result | Method |
|---|---|
| Theorem 1.1 (dominance / unique representability) | **already proved in Z3** (`SIGN`) |
| §1.3 rail homomorphisms (neg, lane-mul) | **already proved in Z3** |
| Theorem 1.4 (exactness) | Z3: `Σ` over rails equals integer product, symbolic width |
| §2.3 Theorem (K≤64 coincidence) | Z3 over the index arithmetic; plus exhaustive for small `m,n,k` |
| §2.4 broadcast = zero stride | Z3 on index maps |
| §2.6 popcount dot identity | Z3 per-lane bitvector lemma |
| §3.2 working set / intensity | Analytic (proved above) + validated by measured bytes-moved |
| §3.5 allocation bound | Runtime assertion: outstanding ≤ high-water mark; ASan/LSan clean |

---

*Written 2026-11-07. No implementation yet, and **no pull request opened** —
waiting on your choice of target description format before proceeding to P3.*
