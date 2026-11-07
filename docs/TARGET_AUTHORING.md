# Target authoring guide

How to teach this compiler about a new machine.

The design constraint behind everything here: **adding a target requires zero
edits to any file another team might be editing at the same time.** You write
one YAML file. Nothing else in the tree changes. That is what makes parallel
development by several hardware teams actually work rather than just being
aspirational.

---

## The path

```
1. copy examples/minimal-target/minimal.yaml -> targets/<yours>.yaml
2. python3 tools/gen_targets.py --check        validate the description
3. python3 tools/gen_targets.py                regenerate the C++ header
4. python3 tools/gen_backend.py <yours>        emit your backend package
5. cmake --build build/v && ctest --test-dir build/v
```

Step 4 gives you a directory that **compiles** and whose conformance suite
**fails**. That is the deliverable: a list of obligations derived from your own
description file, not a green checkmark hiding the fact that nothing is
implemented.

---

## 1. The description file

A target description is the complete contract between your hardware and the
compiler. It is validated against `schema/target.schema.json`, so a typo is a
build failure rather than a silently ignored field.

### `datapath` — what your silicon does

```yaml
datapath:
  native_trit_width: 64      # how wide a word is, in trits
  max_immediate_trits: 5     # widest constant encodable in an instruction
  has_word_add: true         # native word-level addition
  has_lane_mul: true         # native per-lane (elementwise) multiply
  has_scalar_mul: false      # native scalar x word multiply
  has_popcount_dot: true     # native dot product / popcount reduction
```

**Declare only what exists.** The compiler does not assume anything not listed.
If you omit `has_scalar_mul`, the compiler emits shift-and-add instead of an
instruction that would trap. Over-declaring is worse than under-declaring: it
produces code that fails on real silicon, and it will pass every test that runs
on a model.

### `storage` — what formats you can hold

```yaml
storage:
  native_layouts: [dual-rail, base-243]
  preferred_layout: dual-rail
```

The four layouts are `dual-rail`, `base-243`, `tcsr`, `btfp`. A layout you do
not list will never be produced for your target. All four have conformance
vectors in `specs/layout_conformance.json`; conversions that are not defined
**refuse** rather than approximate, because the GEMM path depends on exactness.

### `registers`

```yaml
registers:
  ternary: 32
  scalar: 32
  zero_hardwired: true
```

`zero_hardwired` is worth getting right — it is a genuine optimisation and the
compiler cannot infer it.

### `memory`

```yaml
memory:
  cache_line_bytes: 64
  scratchpad_bytes: 49152
  global_bytes: 1073741824
  explicit_movement: true
```

`explicit_movement` is the single most consequential field in the file.

- `false` — hardware manages movement (a cache). The compiler emits loads and
  stores and lets the hierarchy work.
- `true` — **you** issue every transfer. The compiler must plan the movement,
  and getting this wrong is a correctness bug, not a performance bug: data
  that was never copied is silently stale.

GPUs and most accelerators want `true`. CPUs want `false`.

### `costs` and `default_cost`

Per-op cycle counts used for tiling and lowering decisions. Ops you omit cost
`default_cost`.

### `native_ops`

Optional. If you list them, the generated conformance suite produces **one
failing obligation per op** — so the suite you receive is exactly as demanding
as the capabilities you claimed. If you omit the field, the suite falls back to
a generic three-op check.

### `simulator`

```yaml
simulator:
  command: bt-myhw-sim
  args: ["--cycles"]
```

If you supply a simulator, the same conformance vectors that drive the host
model will drive yours. See `include/Ternary/Target/Simulator.hpp` for the
four-operation interface (`load`/`exec`/`read_state`/`read_memory`). Real
hardware has no simulator block, and that is correct.

---

## 2. The contract your backend must honour

Generated backends come with one hard rule, stated in the emitted README:

> **Never emit code you do not believe is correct.**

A construct you cannot lower goes into `BackendResult::unsupported` with a
reason. Emitting a silent stub is the one forbidden behaviour — a stub that
compiles and returns zero is indistinguishable from correct code until it is
running on your hardware.

---

## 3. Failed vs Skipped

When your simulator is driven by the shared vectors:

- **Failed** means it ran and gave the wrong answer. That is a bug in your
  implementation.
- **Skipped** means it could not run the vector at all — an instruction you
  have not implemented yet. That is normal and expected during development.

The two are reported separately, and **Skipped still blocks a clean summary.**
This is deliberate. If "nothing ran" counted as "everything passed", then
"nothing has run on real hardware" would quietly become "everything passes",
which is precisely the gap this project is trying to close.

---

## 4. Upstreaming checklist

Before opening a pull request:

- [ ] `python3 tools/gen_targets.py --check` passes
- [ ] The generated backend **compiles**
- [ ] The generated conformance suite runs (it is expected to fail)
- [ ] Every declared native op has a corresponding obligation in the suite
- [ ] `bt-compiler --list-targets` shows your target with the facts you wrote
- [ ] `ctest --test-dir build/v` is green — your addition broke nothing
- [ ] `explicit_movement` is set deliberately, not left at the default
- [ ] No field is declared optimistically; every capability is real

---

## 5. Adding a target must not break anyone

Adding a `targets/*.yaml` adds:

- one CTest check (`Check_Target_Descriptions` validates every description
  against the schema)
- one backend package under `build/backends/`
- one entry in `--list-targets`

It cannot break another target's tests, because descriptions are independent
data validated against a shared schema. If you find yourself editing a shared
file to add a target, that is a bug in this design — please report it.

---

## Worked example

`examples/minimal-target/minimal.yaml` is the smallest file that produces a
real backend: 9 trits, one accumulator, no memory hierarchy. Run

```sh
cp examples/minimal-target/minimal.yaml targets/minimal.yaml
python3 tools/gen_targets.py
python3 tools/gen_backend.py minimal
```

and you will have `build/backends/minimal/` with a compiling backend and a
failing suite. Read the emitted `README.md` — it restates your own description
back to you, which is the fastest way to catch a field you got wrong.
