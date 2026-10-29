# Steps 1–5: implementation and validation status

Date: **2026-10-29**. Branch: `arena/f3c30e8e-trium-xbt-compiler`.
README is unchanged. This report supersedes the earlier foundation-pass status,
not its historical measurements.

## 1. Delivered implementation

The compiler now has a **shared, executable scalar core-to-LLVM pipeline**, a
conservative optimizer, versioned runtime/client integration, and a restricted
**program-dependent CUDA backend with an actual CUDA-driver runtime**. These are
implemented code paths, not templates or a planning-only deliverable.

The distinction between implementation and platform qualification matters:

| Plan stage | Implemented and exercised | Qualification boundary |
|---|---|---|
| 1. Mathematical profile | Complete 27-operation support manifest; separate integer/rational/BTFP contracts; independent references, ten Z3 scripts and 66 assumption-free Coq-kernel results | Model/algorithm proofs are not whole-compiler refinement proofs; transcendental/conversion approximations remain labeled |
| 2. Shared scalar pipeline | Default scalar source → checked core → LLVM; functions, loops, lazy selection, effects and failures; optimized/unoptimized differential tests | Aggregate native compatibility is explicit `--legacy-llvm`, not part of the scalar core |
| 3. CPU/runtime qualification | Linux x86-64 Debug/Release/ASan+UBSan execution; C/Python ABI tests; bounded frontend; Rust checked APIs and CI job | Rust/native-Rocq CI has not run here; additional physical CPU/toolchain qualification, longer fuzz campaigns and complete source debugging remain release work |
| 4. First GPU path | Pure loop-free int39 map → NVPTX; allocation/copies/launch/sync/errors/context cleanup; offline kernels, emulation and fake-driver tests | **Gate 4 is not closed:** no NVIDIA driver/device is available here; no device execution or GPU performance claim |
| 5. Safe optimization | Integral folding/identities, lazy constant pruning, unreachable tails, interval-based narrowing, interpreter return-flow and function lookup improvements; paired measurements | Opt-in; workload-dependent performance, including an observed native regression; no speculative reassociation, packing/vectorization or blanket speedup claim |

Thus this is a substantial implementation through Stage 5, with verified local
CPU scope, **not a declaration that every production/hardware gate has passed**.
Future native ternary/mixed-radix hardware (Stage 6) remains outside this pass.

## 2. Architecture and source files

```text
source + imports
  → bounded lexer/parser + mandatory semantic/support checking
  → ternary_ir_lowering.hpp
  → verified Ternary::IR::Program
  → optional conservative optimization + verification
      ├─ core executor (--core-run)
      ├─ default scalar LLVM → CPU execution
      └─ explicitly selected pure integral function → CUDA map LLVM/PTX
```

- `include/Ternary/IR/{Core,Program}.hpp`: typed operations, effects, scopes,
  structured control flow/functions and bounded verification.
- `include/Ternary/IR/Execute.hpp`: independent core execution. Returns propagate
  as ordinary flow rather than throwing a C++ exception on every successful call;
  function lookup uses stable indices rather than repeated linear scans.
- `include/Ternary/IR/Optimize.hpp`: verified-before/after transformations and
  conservative integral interval analysis. No floating reassociation, unsafe
  `x*0` elimination, or eager evaluation of unselected branches.
- `include/Ternary/IR/LLVM.hpp`: checked CPU scalar emission, private index-mangled
  functions, thread-local error/depth state, restricted CUDA emission and minimal
  debug metadata. Upstream LLVM parses/emits the generated IR in validation;
  this does **not** claim completed upstream MLIR integration.
- `tools/{cuda_map,run_cuda_map}.py`: LLVM NVPTX emission and real CUDA C-driver
  API calls; no CUDA toolkit/NVCC dependency. Missing hardware is explicit exit 77.
- `specs/core-profile-v0.1.json`: machine-readable operation/backend/library matrix,
  limits and qualification. `test/profile_contract.py` checks enum completeness.

Plain AST interpretation remains an independent compatibility/reference route.
`--core-opt` selects core execution for `run`; native scalar generation uses core
by default. Unsupported core aggregates produce diagnostics, never an automatic
switch to the legacy backend.

### ABI and client behavior

The shared runtime exports `bt_runtime_abi_version()` (major **1**) and thread-local
status/message functions. Python negotiates the major and checks the error channel
for configured ABI operations. Arithmetic failures now raise exceptions rather
than returning a plausible zero/NaN sentinel to the caller. Checked Rust `try_*`
APIs return owned `BtError` values; legacy convenience arithmetic fails loudly.
Rust source/tests are implemented but not locally executed.

Exported packed-word operations validate disjoint rails; positional conversion
rejects widths above 39; base-243 wrappers validate bytes/trits/nulls; array dot
products use a wide accumulator and do not silently truncate mismatched client
arrays. Python validates inputs before ctypes narrowing. Inline low-level C helpers
retain their documented valid-input preconditions. Legacy raw pointer APIs still
require valid caller-owned buffers; the runtime is not a general memory sandbox.

Generated scalar modules separately export `bt_core_abi_version()` and
`bt_core_status()`. `bt_main()` returning 1 with status 0 is a successful user
return; arithmetic failure has status 1. Thread-local isolation is regression-tested.

### Explicit limits

- Source/import expansion: 8 MiB; import depth 32; 100,000 tokens; parser recursion
  128 and 256 operators per statement segment.
- Import content identity is exact and includes the resolution directory: identical
  wrappers in different directories still resolve their respective dependencies.
- Core verification: 100,000 shared work units, nesting 512, at most 10,000
  functions and 1,000 parameters per function.
- Default core execution budget: 10 million operations; core/native call depth 512.
- CUDA host batch: at most 1 million validated integral values.

These bounds mitigate resource exhaustion; they do not prove immunity to OS stack,
memory or I/O exhaustion. Native execution is not a hostile-code sandbox. Debug
information currently identifies the main compilation unit/line, not complete
source-level variable/function debugging.

## 3. Final executable evidence

Locally tested configuration: Linux x86-64, Debian GCC **12.2.0**, C++20 standalone
build (`BT_ENABLE_MLIR=OFF`), CMake **4.4.4**, Ninja **1.13.2**, Python **3.11.2**,
llvmlite **0.50.0** / LLVM **22.1.0**, z3-solver **5.3.2.0**. The native CLI also
ran using a Zig-provided Clang **22.1.8** adapter. Tool/provenance details and source
hashes are in the evidence directory.

| Final configuration | Passed | Skipped | Failed | Wall time |
|---|---:|---:|---:|---:|
| Debug | 36 | 2 | 0 | 10.41 s |
| Release | 36 | 2 | 0 | 8.08 s |
| ASan + UBSan | 36 | 2 | 0 | 26.83 s |

The two explicit skips are actual CUDA-device qualification and native Rocq.
ASan/UBSan detected no failures in the final suite. C++ tests enable leak detection;
Python-hosted FFI tests disable host leak reporting and preload ASan plus the C++
exception runtime, while retaining shared-library sanitizer checks.

Coverage includes:

- **525 differential scalar values** plus negative/error/driver cases and **211
  generated whole-program outcomes**, both with and without core optimization.
- **10,000 seeded C++ optimizer/range cases**, checked-error preservation, symbol
  mangling, DAG/resource limits and interval containment.
- IEEE signed zero, NaN, infinities, FMA, lazy side effects/failure ordering,
  recursion and returns through nested loops/branches, native TLS concurrency,
  105 malformed-source cases and import cycle/depth/context regressions.
- Actual built C/Python runtime integration: invalid inputs/errors, all **243**
  base-243 byte roundtrips, narrowing/length rejection and multithreaded calls.
- Five program-dependent NVPTX kernels and device-free reference execution,
  including multiplication boundaries, invalid int64 lanes, inactive lanes and
  untouched failed-lane output. Fake-driver tests cover argument marshaling,
  allocation/load/launch/synchronization/free failures and context restoration.
  **Neither emulation nor a fake driver is GPU execution.**
- All **ten** root Z3 verification scripts completed successfully.
- Canonical four-file proof wrapper completed: **28 + 18 + 11 + 9 = 66** results
  with no axioms under jsCoq 0.17.1's actual Coq kernel. Only the import prefix is
  adapted for compatibility; this is not native Rocq 9 qualification.
- Standalone `--native`, conflicting CUDA CLI modes, and the checked-map example's
  offline PTX path were manually exercised. Actual execution returned unavailable
  (77), not a simulated success.

Raw logs: [evidence directory](evidence/2026-10-29-steps-1-to-5/).
Configured CI includes Debug/Release/sanitizers, all Z3 scripts, native Rocq
9.1.1/stdlib 9.1.0, and Rust 1.85.1 against the actual `libbt_runtime` build.
No remote CI result is claimed in this report. AArch64/RISC-V coverage is assembly
emission, not execution on physical machines.

## 4. Measured optimization results

[Raw JSON, compiler hash and every sample](benchmarks/2026-10-29-core-optimization.json).
100,000 iterations; seven alternating baseline/optimized samples per boundary;
LLVM target-machine optimization **0**, isolating these core passes from LLVM O3.
Every measured run checks its independently expected result. Native timing is a
warm entry call, excluding compilation/loading; core CLI timing includes process,
parsing, checking and execution. **Do not compare the two columns as a speedup.**

| Workload | Native off → on (ms) | Native ratio | Core CLI off → on (ms) | CLI ratio |
|---|---:|---:|---:|---:|
| Identities/constants | 1.343040 → 1.197559 | 1.12× | 23.645947 → 19.346743 | 1.22× |
| Lazy constant selection | 0.863201 → 0.484905 | 1.78× | 15.195900 → 9.974403 | 1.52× |
| Checked signed-map workflow | 1.457329 → 1.680550 | 0.87× | 30.163244 → 29.580734 | 1.02× |

The first two workloads reduce emitted LLVM from 5,631→4,476 and 5,626→3,905
bytes, and i128 textual mentions from 43→19 and 49→19. The checked-map workload
changes little (6,148→6,077 bytes; 37 i128 mentions unchanged), and **did not improve
native timing in this measurement**. This mixed result is why optimization remains
opt-in. Shared-host timing is variable; these samples are evidence for these
workloads, not universal throughput, statistical certification or GPU speedups.
Compilation boundaries are recorded separately as single samples in the JSON.

## 5. Reproduce and use

Install a C++20 toolchain, CMake/Ninja, Python and Clang. Pinned validation Python
dependencies are in `requirements/verification.txt`.

```sh
python3 -m pip install cmake==4.4.4 ninja==1.13.2 -r requirements/verification.txt
cmake -S . -B build/current -G Ninja -DBT_ENABLE_MLIR=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/current --target check-all -j2
for script in verify_z3*.py; do python3 "$script" || exit 1; done

# Independent core execution / default shared-core native CPU execution
build/current/bt-compiler --core-run examples/01_arithmetic_spaceship.bt
build/current/bt-compiler --native examples/01_arithmetic_spaceship.bt
build/current/bt-compiler examples/01_arithmetic_spaceship.bt --core-opt --emit-llvm -o build/current/program.ll

# Explicit legacy aggregate compatibility, not an implicit fallback
build/current/bt-compiler examples/05_sparse_matrix.bt --legacy-llvm --emit-llvm -o build/current/sparse.ll

# Program-dependent CUDA map: offline generation is explicitly labeled
build/current/bt-compiler examples/06_core_cuda_map.bt --cuda-map=transform -o build/current/map.ll
python3 tools/run_cuda_map.py build/current/map.ll --emit-ptx build/current/map.ptx
# Requires a real NVIDIA driver/device; expected successful values: [-7,-4,0,4,7]
python3 tools/run_cuda_map.py build/current/map.ll -2 -1 0 1 2
python3 test/cuda_map_test.py build/current/bt-compiler --device

python3 test/benchmark_core_optimization.py build/current/bt-compiler --output build/current/optimization.json
```

For sanitizers configure a separate Debug build with `-DBT_ENABLE_ASAN=ON` and run
CTest with `ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1`.
For the explicit compatibility proof engine:

```sh
npm install --prefix build/proof-tools jscoq@0.17.1
python3 test/check_rocq.py --jscoq build/proof-tools/node_modules/.bin/jscoq
```

The next release gates are concrete: actual CUDA hardware outputs/errors and
transfer/kernel measurements; execution of the configured Rust/native-Rocq CI;
longer fuzz/performance campaigns; additional named CPU/toolchains and fuller
debugging. None requires restarting the mathematical foundation or redesigning
the shared scalar pipeline delivered here.
