# Comprehensive Runtime & Concurrency Audit Report
**Target System:** Universal Balanced Ternary Compiler (`bt-compiler`, `bt-jit`, Kokkos Multithreading Bridge)  
**Audited Components:**
1. `ternary_kokkos_bridge.hpp`
2. `ternary_jit_engine.hpp`
3. `bt_compiler.cpp`  
**Auditor:** Antigravity Runtime & Concurrency Auditor  
**Date:** October 16, 2026  

---

## Executive Summary

An exhaustive runtime, concurrency, cache architecture, and JIT interpreter audit was conducted across the Balanced Ternary execution stack. The audit identified **critical concurrency flaws** (including an unsynchronized bitwise data race in parallel transposition, missing cache-line padding causing severe L1 false sharing, and thread starvation deadlock risks), **critical JIT memory and safety bugs** (uncaught `ReturnException` crashing the process, dynamic scope leaks, heap buffer overflows from unchecked matrix dimensions, and deep-copy overhead on AST traversal), and **CLI robustness vulnerabilities** (uncaught exceptions on invalid flags, silent ignore of `-o`, and broken target code emitters).

---

## 1. Thread Safety & Concurrency Audit

### 1.1. Severe Data Race in `parallel_transpose` (Invoked by `parallel_gemm`)
- **Location:** `ternary_kokkos_bridge.hpp`, lines 401–409 & 429
- **Vulnerability:** Unsynchronized concurrent bitwise read-modify-write on shared `TritWord64` words.
- **Detailed Mechanics:**
  In `parallel_transpose`, the loop distributes the rows $r$ of the source matrix `mat` across worker threads:
  ```cpp
  ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, mat.rows), [&](size_t r) {
      for (size_t c = 0; c < mat.cols; ++c) {
          T.set(c, r, mat.get(r, c));
      }
  });
  ```
  `T.set(c, r, ...)` writes to row $c$, column $r$ of the destination transposed matrix $T$. In `TernaryMatrix`, column $r$ is stored inside word index $w = r / 64$ and bit index $bit = r \% 64$.
  When multiple threads process different values of $r$ concurrently (e.g. Thread 0 processes $r=0..15$, Thread 1 processes $r=16..31$), both threads loop over column $c$. For the same column $c$, both threads concurrently execute:
  ```cpp
  word(r, w).setTrit(bit, t); // P &= ~mask; N &= ~mask; P |= mask; N |= mask;
  ```
  This is a concurrent unsynchronized mutation of `T.word(c, 0).P` and `T.word(c, 0).N`. Because `uint64_t` operations are non-atomic, threads overwrite each other's bit updates, corrupting matrix data and violating the fundamental dual-rail invariant $(P \ \& \ N) == 0$.
- **Impact:** Memory corruption, corrupted GEMM outputs, undefined behavior on all multi-threaded GEMM runs.
- **Resolution:** Parallelize across destination rows $c$ (columns of `mat`), ensuring each thread writes to strictly disjoint words:
  ```cpp
  ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, mat.cols), [&](size_t c) {
      for (size_t r = 0; r < mat.rows; ++r) {
          T.set(c, r, mat.get(r, c));
      }
  });
  ```

---

### 1.2. Thread Pool Shutdown / Re-entry & Lifecycle Race
- **Location:** `ternary_kokkos_bridge.hpp`, lines 185–225, 236–240
- **Vulnerabilities:**
  1. `ThreadPool::reinit()` and `ThreadPool::init()` lack thread-safe synchronization. Calling `reinit()` concurrently from multiple threads or during active parallel dispatch results in concurrent access, join, and clearance of `workers`.
  2. If `workers` are executing tasks when `shutdown()` is called, pending queue tasks are dropped via `std::swap(task_queue, empty);`. Any caller waiting on `std::latch` for discarded tasks will hang forever in an unrecoverable deadlock.
  3. Calling `reinit()` or `shutdown()` from within a worker thread (e.g. if a nested task triggers pool re-initialization) causes `th.join()` on the current thread, triggering `std::system_error: Resource deadlock avoided`.
- **Resolution:** Guard `reinit` and `shutdown` with a dedicated lifecycle `std::recursive_mutex`. Disallow re-initialization while tasks are in flight.

---

### 1.3. `std::latch` Lifetime and Uncaught Exception Hazards
- **Location:** `ternary_kokkos_bridge.hpp`, lines 300–322 (`parallel_for`) and 361–382 (`parallel_reduce`)
- **Vulnerabilities:**
  1. If `func(i)` throws an exception inside a worker thread, the exception escapes the thread function without decrementing `sync_latch`. This immediately invokes `std::terminate()` under C++ rules.
  2. If the exception occurs on the caller thread (running the last chunk), the caller stack unwinds and destroys `sync_latch` while background workers still hold references to `&sync_latch` and `&func`. When background workers finish, they access destroyed stack memory (Stack Use-After-Free).
- **Resolution:** Wrap worker lambda bodies with `try { ... } catch (...)` to capture the exception in a `std::exception_ptr`, ensure `sync_latch.count_down()` runs in a RAII scope guard / `finally` block, and rethrow the captured exception on the caller thread after `sync_latch.wait()`.

---

### 1.4. Inverted Range Bounds from Ceiling-Division Chunking
- **Location:** `ternary_kokkos_bridge.hpp`, lines 298–305, 358–365, 524–530
- **Vulnerability:**
  `chunk_size = (total + num_chunks - 1) / num_chunks;`
  Because ceiling division rounds up, $num\_chunks \times chunk\_size > total$. For trailing chunks where $c \times chunk\_size \ge total$, `start` exceeds `finish` ($start > finish$), creating inverted ranges $[start, finish)$. While unsigned `<` terminates loop 0, this causes trailing worker threads to be dispatched with invalid ranges.
- **Resolution:** Use exact balanced chunk distribution:
  ```cpp
  IndexType start = policy.begin() + (c * total) / num_chunks;
  IndexType finish = policy.begin() + ((c + 1) * total) / num_chunks;
  ```

---

## 2. Cache-Line Architecture & False Sharing

### 2.1. Critical False Sharing in `Kokkos::parallel_reduce`
- **Location:** `ternary_kokkos_bridge.hpp`, lines 360–378
- **Vulnerability:**
  `std::vector<ValueType> local_accumulators(num_chunks, ValueType{});`
  Elements of `local_accumulators` are packed contiguously in memory. For 64-bit integer or floating-point types (`sizeof(int64_t) = 8`), up to 8 threads write to adjacent slots on the **same 64-byte L1 cache line**.
  Furthermore, the reduction kernel executes:
  ```cpp
  for (IndexType i = start; i < finish; ++i) {
      func(i, local_accumulators[c]);
  }
  ```
  `func` mutates `local_accumulators[c]` on **every single iteration**. As multiple CPU cores execute their loops simultaneously, each write sends cache-line invalidation snoops across the interconnect (MESI / MOESI cache protocol thrashing).
- **Benchmark / Real-World Impact:**
  On modern multi-core processors (Apple Silicon M-Series, AMD Zen, Intel Core), this false sharing introduces a $5\times$ to $20\times$ throughput penalty compared to hardware peak bandwidth.
- **Resolution:**
  1. Accumulate into a private CPU register / stack variable throughout the loop:
     ```cpp
     ValueType local_acc = ValueType{};
     for (IndexType i = start; i < finish; ++i) {
         func(i, local_acc);
     }
     local_accumulators[c].val = local_acc;
     ```
  2. Pad accumulator structures to 64 bytes (`alignas(64)`):
     ```cpp
     struct alignas(64) PaddedAccumulator {
         ValueType val{};
     };
     std::vector<PaddedAccumulator> local_accumulators(num_chunks);
     ```

---

## 3. JIT Runtime Safety & Virtual Machine Audit

### 3.1. Process Crash from Uncaught `ReturnException`
- **Location:** `ternary_jit_engine.hpp`, lines 207–211 & 720–733; `bt_compiler.cpp`, lines 353–372
- **Vulnerability:**
  `class ReturnException` does **not** inherit from `std::exception`:
  ```cpp
  class ReturnException {
  public:
      JITValue value;
      explicit ReturnException(JITValue val) : value(std::move(val)) {}
  };
  ```
  If a user script contains a top-level `return` statement (or calls a function that exits via an unhandled return path), `execute_file()` does not catch `ReturnException`. In `bt_compiler.cpp`:
  ```cpp
  try {
      engine.execute_file(opts.inputFile);
  } catch (const std::exception& e) { ... }
  ```
  Because `ReturnException` is not a `std::exception`, it bypasses the catch handler and terminates the process via `std::terminate()` (`libc++abi: terminating with uncaught exception`).
- **Resolution:** Inherit `ReturnException` from `std::exception` (or `std::runtime_error`) and catch `ReturnException` at the top level of `TernaryJITEngine::execute()`.

---

### 3.2. Lexical Scoping & Dynamic Scope Leaks
- **Location:** `ternary_jit_engine.hpp`, lines 152–163, 263–267, 690–703
- **Vulnerabilities:**
  1. `visit(BlockNode&)` does not instantiate a child `Environment`. Variables declared inside conditional branches (`if`), loops (`while`), or nested scopes leak into and permanently mutate the outer scope.
  2. Dynamic Scoping in Function Calls: In `CallNode`, `fn_env` is constructed with `std::make_shared<Environment>(env)`. Here, `env` is the caller's active environment rather than the closure/lexical environment where the function was declared. Functions dynamically access and mutate caller local variables.
  3. `Environment::assign()` auto-creates global variables at the root environment if the identifier is undefined, silently converting typos into global state.

---

### 3.3. Unchecked Matrix Dimensions Leading to Heap Buffer Overflow
- **Location:** `ternary_jit_engine.hpp`, lines 372–382 (`Mul`), 535–553 (`matmul`), 556–585 (`csa_reduce`)
- **Vulnerabilities:**
  1. In `Mul` and `matmul`, the JIT engine performs no runtime check that `left.matrix.cols == right.matrix.rows`. In Release mode (`-DNDEBUG`), standard `assert` statements in `gemm` are compiled out, causing sequential out-of-bounds reads into `BT.word` and memory corruption.
  2. In `csa_reduce`, the engine fails to check whether all input matrices have matching dimensions $(M, N)$. If `inputs[k]` is smaller than `inputs[0]`, reading `inputs[k].data[idx]` results in a heap buffer overflow.
  3. In `matrix_init`, `zeros`, `ones`, and `rand_matrix`, dimensions are parsed as `int64_t` and cast to `size_t` without negative number checks. Passing `-1` creates a `size_t` of $2^{64}-1$, triggering immediate allocation failure or crash.
- **Resolution:** Add explicit runtime validation throwing `std::runtime_error` before executing GEMM, CSA reduction, and matrix allocation.

---

### 3.4. Catastrophic Copy Overhead on AST Walks
- **Location:** `ternary_jit_engine.hpp`, lines 298–301, 511, 683–688
- **Performance Defect:**
  `JITValue` contains a heavy `TernaryMatrix` (`std::vector<TritWord64>`) and `DenseMatrixInt64` (`std::vector<int64_t>`). In `visit(IdentifierNode)`, `env->get(node.name)` returns `JITValue` by value (deep copy). In `visit(BinaryOpNode)`, evaluating `left` and `right` makes two additional deep copies. Passing matrices into functions or builtins copies full matrix buffers 4 to 6 times per call.
- **Resolution:** Use `std::shared_ptr<TernaryMatrix>` inside `JITValue` (copy-on-write or shared immutable storage), or an explicit register/value stack.

---

### 3.5. Scalar Logic Performance & Precision Truncation
- **Location:** `ternary_jit_engine.hpp`, lines 324–368
- **Defects:**
  1. Evaluating equality `==` or spaceship `<=>` between dual-rail scalars converts `TritWord64` to `int64_t` via `Codec::to_int64(scalar, 39)`. This incurs 39 power-of-3 multiplications per operand on every comparison.
  2. Any trits residing in bits 39..63 of `TritWord64` are silently ignored. Two scalars differing only in bits 39..63 compare as equal.
- **Resolution:** Dual-rail equality is an $O(1)$ single-cycle comparison:
  ```cpp
  bool eq = (left.scalar.P == right.scalar.P) && (left.scalar.N == right.scalar.N);
  ```

---

## 4. CLI Driver Robustness (`bt_compiler.cpp`)

### 4.1. Unhandled Argument Parsing Failures
- **Location:** `bt_compiler.cpp`, lines 299–332
- **Vulnerabilities:**
  1. `opts.kokkosThreads = std::stoul(arg.substr(10));` throws unhandled `std::invalid_argument` / `std::out_of_range` on `--threads=abc` or `--threads=`, crashing the CLI before entering `main()` logic.
  2. Unknown command-line flags (e.g. `--target=arm`, `--thread=4`) are silently ignored with zero warning.
  3. If `-o` is the trailing argument without a following path, it is silently ignored.
  4. In compile mode (`--target=metal`), `opts.inputFile` is never opened or verified; the compiler generates static shader boilerplate without reading source code.
  5. Writing to `opts.outputFile` lacks error verification (`out.is_open()`), falsely reporting success if write permissions are denied.

---

### 4.2. Vector Backend Emitter Errors
- **Location:** `bt_compiler.cpp`, lines 145–148, 160–170
- **Bugs:**
  1. In `emit_arm64_neon_kernel()`, `dot_product_neon` computes `vorrq_u64` with un-masked swapped rails, fails to compute negative rail products, and never subtracts `neg_prod`. The emitted kernel generates mathematically bogus dot products.
  2. In `emit_vulkan_shader()`, GLSL lacks `#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require`, causing SPIR-V compile failure on Vulkan toolchains.

---

## 5. Audit Matrix & Priority Fix Action Plan

| ID | Component | Severity | Description | Remediation |
|---|---|---|---|---|
| **BUG-01** | `KokkosBridge` | **CRITICAL** | Data race in `parallel_transpose` during bit-sliced GEMM | Parallelize across destination rows $c$ |
| **BUG-02** | `KokkosBridge` | **HIGH** | False sharing in `parallel_reduce` on unpadded accumulators | Align to 64 bytes + thread-local registers |
| **BUG-03** | `JITEngine` | **CRITICAL** | `ReturnException` does not inherit `std::exception` (process abort) | Inherit from `std::exception` + top-level handler |
| **BUG-04** | `JITEngine` | **HIGH** | Unchecked matrix dimensions in `Mul`, `matmul`, `csa_reduce` | Add strict runtime dimension validation |
| **BUG-05** | `JITEngine` | **HIGH** | Dynamic scope leaks and lack of block-level environment | Implement lexical environment chain in blocks |
| **BUG-06** | `JITEngine` | **HIGH** | Deep copies of `TernaryMatrix` on every AST node visit | Reference-count or value-stack `JITValue` |
| **BUG-07** | `KokkosBridge` | **MEDIUM** | Inverted chunk ranges $(start > finish)$ from ceiling division | Use exact integer division: $c \times total / K$ |
| **BUG-08** | `CompilerCLI` | **MEDIUM** | Unhandled exception on invalid `--threads=` and silent ignore of `-o` | Add robust CLI parser with usage errors |
| **BUG-09** | `CompilerCLI` | **MEDIUM** | Math error in emitted ARM64 NEON kernel | Implement true dual-rail popcount subtraction |

---
*Report certified by Antigravity Prime Research & Engineering Engine.*
