# Foundation progress and measured CPU performance

> Historical foundation-pass record. For the later shared LLVM/CUDA/optimization
> implementation and current validation, see [STEPS_1_TO_5_STATUS.md](STEPS_1_TO_5_STATUS.md).
> Counts and measurements below describe the earlier pass, not the current build.

Date: **2026-10-29**. README remains unchanged.

Trium-XBT now has a working checked scalar CPU compiler, a separately executable
whole-program typed core IR, and independent mathematical validation. This pass
moves the foundation beyond an expression schema: source programs can be lowered,
verified and run entirely through the new scalar IR. It also strengthens the
numeric library before expanding hardware backends.

## Implemented in this pass

### 1. Exact mathematics and rounding

- New reusable `bt::math::Rational<W>` library, widths 1..40: exact reduced
  fractions, positive denominators, canonical zero, checked arithmetic, comparison
  and nearest rounding. Reduction precedes final component range checks.
- BTFP normalization/multiplication now account for the irregular gap between
  adjacent normalized mantissa binades. For example, `BTFP<1>(5)` is 3, not 9;
  the exact halfway value 6 rounds to 9, away from zero.
- Direct exact-ratio rounding removes intermediate division rounding and supports
  division at **all 40 supported mantissa widths**. Full-int64 ratio inputs,
  including INT64_MIN, have independent reference coverage.
- Square-root rounding compares the original radicand with the final mantissa
  midpoint, avoiding an intermediate rounded-root decision.
- FMA retains the exact aligned product/addend on accepted paths. Alignments
  beyond the explicit i128 capacity now raise an error rather than silently
  pre-rounding an addend. Wide-accumulator/transcendental closure remains a
  distinct development item; this does not change `.bt float` from binary64.
- New algorithm-level Rocq bridge covers Euclidean correction, equivalence of the
  actual C++ quotient-rounding formula to its mathematical model, tie/error
  properties, rational accumulator capacity and binade midpoint comparisons.

### 2. Executable whole-program core IR

The new headers `IR/Program.hpp` and `IR/Execute.hpp`, together with
`ternary_ir_lowering.hpp`, provide:

- Stable symbol IDs and lexical scopes; typed bindings and assignments.
- Function signatures, forward calls, recursion and checked return paths.
- Blocks, loops, conditionals, returns and lazy three-way selection.
- Explicit casts/promotions, exact checked int39 arithmetic, binary64 math,
  digitwise ternary logic and supported planar/BTHA intrinsics.
- Conservative `Pure` / `MayFail` / `Observable` effects: printing and calls
  cannot accidentally become speculatable pure operations.
- IR verification before execution, a default 10-million-operation budget and a
  512-call depth limit. The C++ API permits a different operation budget.

Run an existing supported scalar source file with:

```sh
build/phase1-release/bt-compiler --core-run program.bt
```

This route executes the IR, **not the AST**. It is an opt-in scalar execution
profile; the established interpreter still supplies its broader aggregate path.
LLVM generation currently consumes the checked AST. Routing LLVM generation
through this shared core is the next backend integration step.

### 3. Native execution hardening

Benchmarking exposed host-library symbol interposition for a source function
named `step`. Source functions now use private mangled LLVM names; generated
runtime helpers also have internal linkage. Regression programs exercise legal
source names such as `step`, `puts`, `malloc` and `sin` without calling host
functions accidentally. Existing reserved-name checks remain intact.

## What the project can do now

| Area | Current capability |
|---|---|
| Source-language scalars | Checked 39-trit integers, constrained trits/positional values, binary64, functions, lexical scopes, loops and lazy selection |
| Mathematical library | Checked balanced integers, Euclidean division, nearest quotient rounding, exact reduced rationals, normalized ternary floating operations with explicit capacity limits |
| CPU execution | Existing AST interpreter, new verified scalar core interpreter, actual LLVM machine-code execution on this x86-64 host |
| Matrix/sparse library | Validated dimensions/CSR structure and checked integer operations in the exercised CPU paths |
| Cross-target output | Actual AArch64 and standard RISC-V assembly emission tested; device execution is not inferred from emission |
| Future hardware | Software instruction/ALU models and mathematical building blocks; physical ternary hardware is not a claimed deliverable |
| GPU | Existing templates are not a qualified program-dependent GPU compiler/runtime; shared IR-to-backend lowering comes first |

**Development stage:** a functioning, validated compiler foundation suitable for
controlled CPU experiments and continued backend development. The mathematical
library and scalar execution paths have substantially stronger evidence now.
Production qualification is a subsequent milestone: shared core-to-LLVM lowering,
continuous fuzzing, native Rocq CI, ABI/platform qualification and real GPU/device
execution are concrete remaining gates, not reasons to discount the working core.

## Executed validation

| Check | Result |
|---|---|
| Debug CTest | **28 passed, 1 native-Rocq dependency skip, 29 registered** |
| Release `check-all` | **28 passed, 1 skip, 29 registered** |
| ASan + UBSan CTest | **28 passed, 1 skip, 29 registered**; no findings on exercised instrumented paths |
| Baseline differential | **525 expected values** across native LLVM / AST / core IR / Python, plus errors and overflow propagation |
| Seeded whole-program differential | **211 expected outputs**, including loops, shadowing, calls, recursion, promotions, block selection and symbol isolation |
| Exact rational/BTFP reference | **15,072 cases** against independent Fraction/candidate-distance calculations |
| Existing integer reference | **3,289 cases** against Python bigint/Fraction |
| Z3 | **All 9 scripts passed**, including new width-1..40 capacity obligations |
| Coq-kernel proof checks | **57 results**, all assumption-free: 28 foundations + 18 core + 11 algorithm bridge |

The local proof engine is explicitly pinned **jsCoq 0.17.1**, using only a
`Stdlib`→`Coq` import-prefix adapter. This is Coq-kernel checking, not a claim that
native Rocq 9 ran locally or that all C++/LLVM code has been formally refined.
The native Rocq GitHub workflow exists but has not run in this session.
Sanitizers cover C++ compiler/runtime paths, not automatic instrumentation of
all generated LLVM code.

## Actual CPU performance

**Environment:** GNU C++ 12.2 Release build, Linux x86-64 sandbox,
Intel Xeon processor reported at 2.60 GHz. Each workload performs **100,000 loop
iterations**, checks its checksum, warms up once, and records five samples.
Figures below are medians; raw samples and the measured binary hash are saved in
[`benchmarks/2026-10-29-foundation.json`](benchmarks/2026-10-29-foundation.json).

| Workload | Existing AST CLI | New core-IR CLI | Core CLI improvement | Warm native kernel |
|---|---:|---:|---:|---:|
| Checked integer accumulation | 73.670 ms | **7.615 ms** | **9.67×** | 0.643 ms |
| Function calls + checked recurrence | 692.683 ms | **263.039 ms** | **2.63×** | 3.168 ms |

The **CLI-to-CLI ratios** compare the same source/checksum and include process
startup, parsing, checking and execution. They demonstrate useful gains on these
two workloads, not a universal speedup claim.

The native column uses the actual emitted LLVM through the independent llvmlite
host adapter, **target-machine opt level 0 with no IR optimization pipeline**.
It excludes compile/link/load costs, so it is **not** an apples-to-apples native
speedup over the CLI columns. Single-sample emit + object generation + link costs
were about **33–43 ms** per workload. This is a baseline measurement, not a claim
of fully optimized native code or GPU throughput. Sandbox scheduling affects
absolute timings; the raw JSON retains sample ranges.

### Reproduce

```sh
cmake -S . -B build/phase1-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBT_ENABLE_MLIR=OFF
cmake --build build/phase1-release --target check-all -j2
python3 test/benchmark_foundation.py build/phase1-release/bt-compiler \
  --iterations 100000 --repeats 5 --output build/foundation-performance.json
for script in verify_z3*.py; do python3 "$script" || exit 1; done
python3 test/check_rocq.py \
  --jscoq build/proof-tools/node_modules/jscoq/dist-cli/cli.cjs
```

Install the pinned Python verification dependencies in `requirements/verification.txt`
and the documented optional jsCoq tool before invoking those checks.
See [CORE_SEMANTICS_V0_1.md](CORE_SEMANTICS_V0_1.md) and
[MATHEMATICAL_CONTRACT.md](MATHEMATICAL_CONTRACT.md) for the precise contracts.
