# Should the GEMM tile default change?

**Status: no change made.** The measurement below does not support changing
`GemmOptions::mr`, and it found that the comparison which prompted the question
was comparing two variables at once. This document records the evidence and the
protocol for settling it on real hardware.

Reproduce with:

```
g++ -O3 -march=native -std=c++20 -I. -Iinclude tools/tile_survey.cpp -o tile_survey
./tile_survey --sizes=256,512,1024 --reps=11
```

Output is CSV so results from several machines can be concatenated.

## The original comparison changed two things at once

The claim was "scalar 4x4 beats AVX-512 8x4 (213 vs 179), so `Auto` should not
pick 8x4." Those two rows differ in **both** the ISA and the tile, so they
cannot justify a tile choice. `Isa::Auto` picks AVX-512 here; the tile is a
separate field (`GemmOptions::mr`, default 4).

Measuring the two independently is the whole point of `tools/tile_survey.cpp`.

## Results: one machine (this one)

2-core shared Xeon @ 2.6 GHz, AVX-512 + VPOPCNTDQ, GCC 12, `-O3 -march=native`.
n = m = k = 1024. Five independent invocations, 11 in-process reps each,
median of each run reported below. Every cell verified bit-exact against a
naive O(n^3) reference before and after timing.

G MAC/s, `2*m*n*k` convention (the older figures in this repo use `m*n*k`,
so halve these to compare).

| config        | median | min  | max  | run-to-run spread |
|---------------|--------|------|------|-------------------|
| AVX-512 4x4   | 334    | 317  | 344  | 8.5%              |
| AVX-512 8x4   | 363    | 354  | 374  | 5.6%              |
| AVX2 4x4      | 161    | 155  | 163  | 5.2%              |
| AVX2 8x4      | 169    | 163  | 172  | 5.5%              |
| scalar 4x4    | 421    | 405  | 429  | 5.9%              |
| scalar 8x4    | 403    | 391  | 404  | 3.3%              |

Within an ISA — the comparison that actually decides the tile:

- **AVX-512: 8x4 wins by 8.7%**
- **AVX2: 8x4 wins by 5.0%**
- **scalar: 4x4 wins by 4.5%**

So for the ISA that `Auto` actually selects on this machine, 8x4 is *faster*,
by more than the spread. The original reading pointed the wrong way.

## Why that is still not enough to change the default

### 1. The scalar path is a compiler artifact, not an algorithm property

The scalar row is the fastest overall, which is backwards for a portable path.
It is GCC auto-vectorizing `ScalarLane` into AVX-512. Disabling that:

| config      | `-O3` | `-O3 -fno-tree-vectorize` | change |
|-------------|-------|---------------------------|--------|
| scalar 4x4  | 421   | 110                       | -74%   |
| scalar 8x4  | 403   | 91                        | -77%   |
| AVX-512 4x4 | 334   | 337                       | +1%    |
| AVX-512 8x4 | 363   | 314                       | -13%   |

The scalar advantage is ~4x of **auto-vectorizer output**, which depends on the
compiler and its version far more than on the CPU. The intrinsics paths barely
move. Any conclusion that rests on the scalar row does not transfer between
toolchains, let alone machines.

### 2. The answer depends on how you measure it, on the same machine

Three ways of measuring the AVX-512 tile question at n=1024, same binary, same
box, same afternoon:

| measurement                                | 4x4  | 8x4  | winner |
|--------------------------------------------|------|------|--------|
| 5 isolated runs of n=1024 only, 11 reps     | 334  | 363  | 8x4 by 8.7% |
| one full 18-cell grid, 11 reps              | 328  | 325  | tie |
| one full 18-cell grid, 15 reps (earlier)    | 349  | 338  | 4x4 by 3.3% |

Running all 18 cells back to back changes the answer. Two shared vCPUs do not
hold a steady clock while 18 benchmarks stream through them, and the AVX-512
paths are the ones that move — consistent with frequency behaviour under
512-bit loads, not with anything about the tiles.

So even the sign of the effect here depends on the measurement protocol. That
is a property of this box, and it is exactly why a dispatch default should not
be moved on the strength of it.

### 3. It is one machine, and a virtualised one

A shared 2-vCPU VM with AVX-512 is close to the worst case for a dispatch
default: frequency behaviour under 512-bit loads on a shared core is not
representative of the hardware anyone would deploy on.

## Recommendation

Leave `GemmOptions::mr` at 4 for now. The 8.7% is real but was measured on one
noisy VM, and it is **in the same direction as the current default only for
scalar** — for the ISA `Auto` picks, changing to 8x4 would be the change.

To settle it, collect `tile_survey --sizes=256,512,1024 --reps=11` output from
at least three machines with different microarchitectures (Intel AVX-512,
AMD AVX2/AVX-512, ARM NEON) and record them in `docs/tile_results/`. Change the
default only if a tile wins on **every** machine for the ISA that `Auto`
selects there.

If the answer turns out to be ISA-dependent — plausible, given 8x4 wins for
both AVX paths but 4x4 wins for scalar — the right fix is not a different
constant but a per-ISA tile in the dispatch table, chosen once at
`detect_isa()` time. That is a small change once the data exists.

## Incidentally: a documentation defect this uncovered

Building the survey tripped over the same operand-layout trap documented in
`sdk.h`. `ternary_gemm`'s docstring said "A and B are supplied as dual-rail
word arrays, row-major." That is wrong for B, which is word-plane-major
(`Bp[w*n + j]`). The two orderings coincide when k <= 64, so it passes small
tests and silently returns wrong values for larger K.

The survey reported **UNVERIFIED for all 18 cells** rather than printing
numbers, which is what caught it. The docstring in
`include/Ternary/Blas/TernaryMicroKernel.hpp` is now correct.
