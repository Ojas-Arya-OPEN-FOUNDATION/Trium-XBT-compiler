# Chapter 4: The Compiler (From Human Code to Bare-Metal Machine Instructions)

Welcome to the engine room! In [Chapter 1](01_number_systems.md) and [Chapter 2](02_core_tricks.md), we explored the mathematical elegance of balanced ternary and the bit-slicing tricks that allow binary transistors to process trits in parallel.

Now, we answer a practical question that every curious programmer asks:

> **How does a text file filled with readable code—like `var a = 84; var b = 27; var c = a + b;`—actually get transformed into microscopic bursts of electrical voltage running on your laptop's CPU at 3 billion cycles per second?**

If you have never built or studied a compiler before, fear not! Compilers are often treated as mystical, impenetrable black boxes reserved for academic computer scientists with thirty years of experience. In reality, **a compiler is simply a master translator and an orchestra conductor.** 

Just as a musical conductor reads a printed paper score of Beethoven's Fifth Symphony and translates every note, tempo, and rest into the physical movement of violin bows, brass valves, and drumsticks, the compiler reads your `.bt` source text and translates it into the exact instruction sequences your silicon processor understands.

In this chapter, we will walk side by side through every single stage of the compiler pipeline in `bt_compiler.cpp`, `ternary_frontend_ast.hpp`, and `ternary_llvm_codegen.hpp`. We will trace real code, examine assembly outputs, explore source-level debugging with DWARF 5, and recount real-world debugging stories where subtle LLVM quirks threatened to break our machine code.

---

## 1. The Complete Journey of a `.bt` Program

When you run `./bt-compiler run --native sample_program.bt`, your program goes through seven distinct phases. Each phase takes the output of the previous phase and refines it, moving from human thought down to raw silicon:

```mermaid
flowchart TD
    subgraph HumanLevel["1. Human Source Level"]
        Source[".bt Source Text<br/>(UTF-8 Character Stream)"]
    end

    subgraph Frontend["2. Compiler Frontend (ternary_frontend_ast.hpp)"]
        Lexer["Lexer (Scanner)<br/>Converts characters into Tokens"]
        Parser["Recursive-Descent Parser<br/>Builds Abstract Syntax Tree (AST)"]
        Sema["Semantic Analyzer (Sema)<br/>Validates types, scopes & symbols"]
    end

    subgraph Midend["3. LLVM CodeGen (ternary_llvm_codegen.hpp)"]
        Visitor["LLVM IR CodeGen Visitor<br/>Emits Static Single Assignment (SSA) IR"]
        Intrinsics["Branchless Ternary Intrinsics<br/>(@bt_spaceship, @bt_select, @bt_spmv)"]
        DebugMeta["DWARF-5 Metadata Generator<br/>(!DICompileUnit, !DILocation)"]
    end

    subgraph Backend["4. Optimization & Machine Codegen (Clang -O3)"]
        ClangOpt["Clang / LLVM Optimizer (-O3)<br/>Vectorization, LTO, Register Allocation"]
        NativeDylib["Native Shared Library (.dylib / .so)<br/>ARM64 / x86_64 Machine Code"]
    end

    subgraph Execution["5. Bare-Metal Execution Engine"]
        DLOpen["dlopen() & dlsym()<br/>Dynamic symbol resolution for @bt_main"]
        Silicon["Physical CPU Execution<br/>100% Native Speed (Zero Overhead)"]
    end

    Source --> Lexer
    Lexer -->|Token Stream| Parser
    Parser -->|AST Hierarchy| Sema
    Sema -->|Validated AST| Visitor
    Intrinsics -.-> Visitor
    DebugMeta -.-> Visitor
    Visitor -->|LLVM IR (.ll)| ClangOpt
    ClangOpt --> NativeDylib
    NativeDylib --> DLOpen
    DLOpen --> Silicon
```

Let us break down each of these stages using an everyday analogy and concrete code.

---

### Stage 1: The Raw Source Text (`.bt`)

Everything begins with a human sitting at a keyboard typing characters into a text file, such as `examples/01_arithmetic_spaceship.bt`:

```bt
var a = 84;
var b = 27;
var sum = a + b;
print(sum);
```

To your computer's operating system, this file is simply an array of ASCII or UTF-8 bytes stored on your disk:
`0x76 ('v'), 0x61 ('a'), 0x72 ('r'), 0x20 (' '), 0x61 ('a'), 0x20 (' '), ...`

---

### Stage 2: The Lexer (Turning Characters into Tokens)

**Analogy:** Imagine reading a book where all spaces and punctuation were removed: `THEQUICKBROWNFOX`. Your brain naturally chunks the letters into recognizable words: `THE`, `QUICK`, `BROWN`, `FOX`. 

In `ternary_frontend_ast.hpp`, the **Lexer** (or Scanner) performs this exact chunking. It scans the raw characters one by one, skips whitespace and comments (`// ...`), and groups characters into atomic units called **Tokens**:

```cpp
struct Token {
    TokenType type;
    std::string text;
    SourceLocation location; // Line and Column for error reporting
};
```

When our lexer reads `var sum = a + b;`, it emits six tokens:
1. `TokenType::KwVar` (`"var"`) at Line 3, Col 1
2. `TokenType::Ident` (`"sum"`) at Line 3, Col 5
3. `TokenType::OpAssign` (`"="`) at Line 3, Col 9
4. `TokenType::Ident` (`"a"`) at Line 3, Col 11
5. `TokenType::OpPlus` (`"+"`) at Line 3, Col 13
6. `TokenType::Ident` (`"b"`) at Line 3, Col 15
7. `TokenType::Semicolon` (`";"`) at Line 3, Col 16

Notice that the lexer records the exact `line` and `column` for every token. As we will see later, this information is essential for DWARF-5 debugging!

---

### Stage 3: The Parser (Building the Abstract Syntax Tree)

**Analogy:** Knowing individual words is not enough to understand a sentence. You need a sentence diagram: subject, verb, and direct object.

The **Parser** in `ternary_frontend_ast.hpp` uses **Recursive-Descent Parsing** to transform the flat list of tokens into a tree structure called an **Abstract Syntax Tree (AST)**. In an AST, parent nodes represent operations, and child nodes represent arguments.

For `var sum = a + b;`, the parser constructs:

```
VarDeclNode (name: "sum", type: "i64")
    └── init: BinaryOpNode (op: "+")
                ├── left:  VariableNode (name: "a")
                └── right: VariableNode (name: "b")
```

Every statement in your program becomes a node inheriting from `ASTNode`:
- `VarDeclNode`: Variable declarations (`var x = 10;`)
- `BinaryOpNode`: Math and comparisons (`+`, `-`, `*`, `<=>`, `&`, `|`, `^`)
- `WhileNode`: Repetitive loops (`while (cond) { ... }`)
- `IfNode`: Conditional branches (`if (cond) { ... } else { ... }`)
- `CallNode`: Function and intrinsic calls (`print(sum);`, `select(...)`)

---

### Stage 4: Semantic Analysis (Sema) & Type Checking

Before generating machine code, the compiler must verify that the program makes sense:
- Are all variables declared before they are used?
- Are we trying to add a 3x3 matrix to a scalar float without a defined operator?
- Are array or vector dimensions compatible?

In `ternary_frontend_ast.hpp`, the `TypeCheckerVisitor` traverses the AST. If a user writes:
```bt
var x = 10;
var y = undefined_var + 5; // Error!
```
The semantic checker halts compilation and prints a human-readable diagnostic error with line and column numbers:
`test/Sema/diagnostic_error.bt:2:9: error: use of undeclared identifier 'undefined_var'`.

---

### Stage 5: LLVM IR Code Generation (`ternary_llvm_codegen.hpp`)

Once the AST is verified, it is handed to `LLVMCodeGenVisitor` in `ternary_llvm_codegen.hpp`.

**What is LLVM IR?**
LLVM Intermediate Representation (IR) is a universal, strongly-typed assembly language for an imaginary CPU with an infinite number of registers (`%t1`, `%t2`, `%t3`, ...). Every variable is assigned once (known as **Static Single Assignment**, or **SSA Form**).

Instead of targeting Intel x86 or ARM64 directly, our compiler emits clean, portable LLVM IR text. For our addition statement `var sum = a + b;`, the visitor emits:

```llvm
  ; Load value of 'a' from stack memory into virtual register %t1
  %t1 = load i64, i64* %var_a_1, align 8
  
  ; Load value of 'b' into virtual register %t2
  %t2 = load i64, i64* %var_b_2, align 8
  
  ; Perform 64-bit integer addition
  %t3 = add i64 %t1, %t2
  
  ; Allocate space for 'sum' and store the result
  %var_sum_3 = alloca i64, align 8
  store i64 %t3, i64* %var_sum_3, align 8
```

Notice how clean and readable LLVM IR is! It strips away the complexities of specific CPU architectures while preserving full mathematical precision.

---

### Stage 6: Industrial Optimization via Clang (`-O3`)

Writing an optimizing compiler backend that knows how to vectorize loops, schedule instructions for superscalar execution units, and allocate hardware registers on modern CPUs requires millions of engineering hours.

By emitting standard LLVM IR, **our compiler immediately inherits the entire power of the world-class LLVM/Clang optimizer.**

In `NativeLLVMJIT::execute_program`, the compiler writes the IR to a temporary file (`/tmp/bt_jit_<pid>_<count>.ll`) and invokes Clang:

```bash
clang -O3 -shared -fPIC -Wno-override-module -g /tmp/bt_jit_1234_1.ll -o /tmp/bt_jit_1234_1.dylib
```

Clang applies hundreds of optimization passes:
- **Loop Vectorization**: Automatically converts iterative loops into SIMD vector instructions (ARM64 NEON or x86 AVX-512).
- **Constant Folding**: Replaces constant math like `3 + 5` with `8` at compile time.
- **Dead Code Elimination**: Removes any calculations whose results are never read.
- **Register Allocation**: Maps an infinite number of LLVM virtual registers (`%t1`, `%t2`) to the CPU's physical registers (`x0`-`x30` on ARM64, `%rax` / `%rcx` on x86).

The output is a native shared library: `.dylib` on macOS or `.so` on Linux.

---

### Stage 7: JIT Execution via `dlopen` and `dlsym`

Now comes the magic moment: how do we run this freshly compiled `.dylib` without restarting the process?

We use two standard C-runtime functions:
1. `dlopen(dylib_path, RTLD_NOW | RTLD_LOCAL)`: Asks the operating system kernel to load the compiled machine code directly into the current process's executable address space.
2. `dlsym(handle, "bt_main")`: Asks the dynamic linker to look up the 64-bit memory address of the entry point function `@bt_main`.

```cpp
// From ternary_llvm_codegen.hpp
using MainFn = int (*)();
MainFn fn = reinterpret_cast<MainFn>(dlsym(handle, "bt_main"));

// Call the function directly on bare metal!
int exit_code = fn();

// Clean up
dlclose(handle);
unlink(tmp_dylib_path.c_str());
```

When `fn()` is invoked, your CPU jumps directly to the newly generated machine instructions. **There is zero interpreter overhead, zero bytecode decoding loop, and zero virtual machine abstraction.** Your balanced ternary program runs at 100% native hardware speed!

---

## 2. The `.bt` Language Grammar and Syntax

The `.bt` language was designed with a simple philosophy: **high readability, explicit balanced ternary semantics, and zero friction.**

### Language Overview Cheat-Sheet

| Construct | Syntax Example | Meaning |
| :--- | :--- | :--- |
| **Variable Declaration** | `var x = 84;` | Declares integer variable (inferred as 64-bit balanced integer). |
| **Typed Declaration** | `var f: float = 3.5;` | Declares floating-point variable (BT-FP32 format). |
| **Spaceship Operator** | `var c = a <=> b;` | 3-way comparison: returns `+1` (greater), `0` (equal), `-1` (less). |
| **Branchless Select** | `select(c, pos, zero, neg)` | Multiplexes among three values based on ternary condition `c`. |
| **While Loop** | `while (i < 10) { ... }` | Executes block while condition evaluates to positive (`> 0`). |
| **If / Else** | `if (x > 0) { ... } else { ... }` | Standard conditional branching. |
| **Sparse Matrix** | `var S = sparse_matrix(R, C, ...);` | Declares multiplier-free 1-bit Ternary CSR matrix. |
| **Dense Vector** | `var v = vector_init(10, 20, 30);` | Declares dense integer vector for matrix multiplication. |
| **Matrix Multiply** | `var y = S * v;` | Executes multiplier-free SpMV kernel. |
| **Output** | `print(expr);` | Formatted output of integers, trits, floats, and matrices. |

### Kleene 3-Valued Logic Operators

In addition to standard arithmetic (`+`, `-`, `*`), `.bt` supports first-class 3-valued Kleene logic on trits and integers:
- `a & b`: Kleene Min / AND ($\min(a, b)$)
- `a | b`: Kleene Max / OR ($\max(a, b)$)
- `a ^ b`: Ternary Modulo-3 XOR
- `~a` or `!a`: Kleene Negation ($-a$)

---

## 3. Five Complete `.bt` Programs with Real Outputs

To truly understand `.bt`, let us explore five complete, runnable programs included in the `examples/` directory of the repository. Each program illustrates a fundamental capability of the language and compiler.

---

### Program 1: Arithmetic & The Ternary Spaceship (`<=>`) Operator

In standard binary languages like C or Java, comparing two values requires two separate checks (`if (a > b)` then `else if (a == b)`). In balanced ternary, comparison naturally produces one of three balanced states: `+1`, `0`, or `-1`. The spaceship operator (`<=>`) captures this in a single operation.

**Source Code (`examples/01_arithmetic_spaceship.bt`):**
```bt
// ==============================================================================
// Example 1: Arithmetic & Ternary Spaceship (<=>) Operator
// Universal Balanced Ternary Programming Language (.bt)
// ==============================================================================

var a = 84;
var b = 27;

// Integer Arithmetic
var sum = a + b;
var diff = a - b;
var prod = a * b;

// Ternary Spaceship Comparisons:
// Returns +1 if LHS > RHS, 0 if LHS == RHS, -1 if LHS < RHS
var cmp_gt = a <=> b;
var cmp_lt = b <=> a;
var cmp_eq = a <=> a;

print(sum);
print(diff);
print(prod);
print(cmp_gt);
print(cmp_lt);
print(cmp_eq);
```

#### Running Program 1

Execute via interpreted JIT:
```bash
./bt-compiler run examples/01_arithmetic_spaceship.bt
```

**Real Output:**
```
======================================================================
  Universal Balanced Ternary JIT Execution Engine (bt-jit)
  Input File: examples/01_arithmetic_spaceship.bt
  Kokkos Multi-Threading: DISABLED (Serial)
======================================================================

111 [trits: ++0+0]
57 [trits: +-0+0]
2268 [trits: +00+0000]
1 [trits: +]
-1 [trits: -]
0 [trits: 0]

======================================================================
  [JIT Engine] Execution successfully completed in 0.071 ms.
======================================================================
```

Execute via Native LLVM Bare-Metal JIT:
```bash
./bt-compiler run --native examples/01_arithmetic_spaceship.bt
```

**Real Output:**
```
111
57
2268
1
-1
0
```

#### Mathematical Verification
- **$84 + 27 = 111$**: In balanced ternary powers ($3^4=81, 3^3=27, 3^2=9, 3^1=3, 3^0=1$):
  $1 \cdot 81 + 1 \cdot 27 + 0 \cdot 9 + 1 \cdot 3 + 0 \cdot 1 = 111 \implies \mathbf{++0+0}$.
- **$84 - 27 = 57$**:
  $1 \cdot 81 + (-1) \cdot 27 + 0 \cdot 9 + 1 \cdot 3 + 0 \cdot 1 = 81 - 27 + 3 = 57 \implies \mathbf{+-0+0}$.
- **$84 \times 27 = 2268$**:
  $1 \cdot 3^7 + 1 \cdot 3^4 = 2187 + 81 = 2268 \implies \mathbf{+00+0000}$.
- **$84 \iff 27$**: $84 > 27 \implies \mathbf{+1}$ (`+`).
- **$27 \iff 84$**: $27 < 84 \implies \mathbf{-1}$ (`-`).
- **$84 \iff 84$**: $84 = 84 \implies \mathbf{0}$ (`0`).

---

### Program 2: 3-Way `select(cond, pos, zero, neg)` Branchless Multiplexing

CPUs hate unpredictable branches (`if / else`). A mispredicted branch flushes the instruction pipeline, wasting 15 to 20 CPU clock cycles. 

Balanced ternary solves this with **branchless selection**. The `select` intrinsic inspects a ternary condition and selects the corresponding value using native hardware conditional moves (`csel` on ARM64, `cmov` on x86_64)—with **zero branches and zero pipeline stalls!**

**Source Code (`examples/02_select_branching.bt`):**
```bt
// ==============================================================================
// Example 2: 3-Way Branchless select(cond, pos, zero, neg) Multiplexing
// Universal Balanced Ternary Programming Language (.bt)
// ==============================================================================

var threshold = 50;

// Scenario 1: Value is less than threshold (42 < 50) => status = -1
var val1 = 42;
var status1 = val1 <=> threshold;
var action1 = select(status1, 100, 0, -100);
print(action1);

// Scenario 2: Value is equal to threshold (50 == 50) => status = 0
var val2 = 50;
var status2 = val2 <=> threshold;
var action2 = select(status2, 100, 0, -100);
print(action2);

// Scenario 3: Value is greater than threshold (99 > 50) => status = +1
var val3 = 99;
var status3 = val3 <=> threshold;
var action3 = select(status3, 100, 0, -100);
print(action3);
```

#### Real Output
```bash
./bt-compiler run examples/02_select_branching.bt
```
```
-100 [trits: --+0-]
0 [trits: 0]
100 [trits: ++-0+]
```

#### Under the Hood: Generated Machine Code
Look at how Clang lowers `bt_select` to ARM64 assembly (`_bt_select`):

```assembly
_bt_select:
    cmp   x0, #1           ; Is condition == +1?
    csel  x8, x1, x2, eq   ; If equal, pick pos (x1), else pick zero (x2)
    cmn   x0, #1           ; Is condition == -1? (compare negative)
    csel  x0, x3, x8, eq   ; If equal, pick neg (x3), else keep previous choice (x8)
    ret                    ; Done! 4 instructions, 0 branches!
```

There is not a single jump instruction (`b`, `b.eq`, `jmp`). The CPU executes this at maximum superscalar throughput.

---

### Program 3: While Loops and Variables

Loops in `.bt` allow mutable state updates. In balanced ternary semantics, a condition is considered **truthy** if and only if its value is positive ($> 0$). Zero and negative values terminate the loop.

**Source Code (`examples/03_while_loops.bt`):**
```bt
// ==============================================================================
// Example 3: While Loops and Variable Mutation (Factorial Calculation)
// Universal Balanced Ternary Programming Language (.bt)
// ==============================================================================

var n = 6;
var fact = 1;
var i = 1;

while (i <= n) {
    fact = fact * i;
    i = i + 1;
}

print(n);
print(fact);
```

#### Real Output
```bash
./bt-compiler run examples/03_while_loops.bt
```
```
6 [trits: +-0]
720 [trits: +000-00]
```

#### Mathematical Verification
$6! = 6 \times 5 \times 4 \times 3 \times 2 \times 1 = 720$.
In powers of 3:
$3^6 = 729$.
$729 - 9 = 720$.
Therefore, $720 = 1 \cdot 3^6 + 0 \cdot 3^5 + 0 \cdot 3^4 + 0 \cdot 3^3 + (-1) \cdot 3^2 + 0 \cdot 3^1 + 0 \cdot 3^0 \implies \mathbf{+000-00}$.

---

### Program 4: Balanced Ternary Floating-Point (BT-FP32)

Standard binary 32-bit floats (`IEEE-754`) suffer from asymmetric rounding bias. Our compiler implements **BT-FP32**, a 32-trit floating point number composed of a 24-trit balanced mantissa ($M$) and an 8-trit balanced exponent ($E$):

$$\text{Value} = M \times 3^E$$

Because truncation in balanced ternary is naturally centered around zero, BT-FP32 exhibits zero rounding drift across billions of mathematical operations.

**Source Code (`examples/04_floating_point.bt`):**
```bt
// ==============================================================================
// Example 4: Balanced Ternary Floating-Point (BT-FP32) Calculations
// Universal Balanced Ternary Programming Language (.bt)
// ==============================================================================

var x: float = 3.5;
var y: float = 1.25;

var f_sum = x + y;
var f_diff = x - y;
var f_prod = x * y;

print(f_sum);
print(f_diff);
print(f_prod);
```

#### Running Program 4

In interpreted JIT mode, `bt-jit` prints both the decimal representation and the underlying BT-FP32 mantissa and exponent:
```bash
./bt-compiler run examples/04_floating_point.bt
```
```
4.75 [BTFP32: M=613415774, E=-17]
2.25 [BTFP32: M=871696100, E=-18]
4.375 [BTFP32: M=1694964639, E=-18]
```

In Native LLVM mode, `bt-native-jit` compiles BT-FP32 operations directly into hardware double-precision floating-point registers:
```bash
./bt-compiler run --native examples/04_floating_point.bt
```
```
4.750000
2.250000
4.375000
```

---

### Program 5: Multiplier-Free Sparse Matrix Multiplication (`sparse_matrix`)

In neural networks and physical simulations, Sparse Matrix-Vector Multiplication (SpMV) is one of the most common—and power-hungry—operations. In binary computing, every non-zero entry requires a floating-point multiplier circuit.

In balanced ternary, every non-zero entry is either $+1$ or $-1$. **Multiplying by $+1$ is just an addition; multiplying by $-1$ is just a subtraction!**

The compiler implements **1-Bit Ternary Compressed Sparse Row (T-CSR)** in `ternary_sparse.hpp` and `ternary_llvm_codegen.hpp`.

**Source Code (`examples/05_sparse_matrix.bt`):**
```bt
// ==============================================================================
// Example 5: Multiplier-Free Sparse Matrix Multiplication (sparse_matrix)
// Universal Balanced Ternary Programming Language (.bt)
// ==============================================================================

var S = sparse_matrix(3, 3,
    1,  0, -1,
    0,  1,  0,
   -1,  0,  1
);

var v = vector_init(10, 20, 30);
var res = S * v;
print(res);
```

#### Real Output
```bash
./bt-compiler run examples/05_sparse_matrix.bt
```
```
DenseMatrixInt64 [3x1]:
  [  -20 ]
  [   20 ]
  [   20 ]
```

And in native bare-metal mode:
```bash
./bt-compiler run --native examples/05_sparse_matrix.bt
```
```
-20
20
20
```

#### Hand-Worked Mathematical Proof
Let us compute $S \times v$ manually:
- **Row 0**: $[1, 0, -1] \cdot [10, 20, 30]^T = (1 \times 10) + (0 \times 20) + (-1 \times 30) = 10 - 30 = \mathbf{-20}$.
- **Row 1**: $[0, 1, 0] \cdot [10, 20, 30]^T = (0 \times 10) + (1 \times 20) + (0 \times 30) = \mathbf{20}$.
- **Row 2**: $[-1, 0, 1] \cdot [10, 20, 30]^T = (-1 \times 10) + (0 \times 20) + (1 \times 30) = -10 + 30 = \mathbf{20}$.

Look closely at the LLVM IR generated by `ternary_llvm_codegen.hpp` for `bt_spmv`:
```llvm
  %acc_add = add i64 %acc, %x_val
  %acc_sub = sub i64 %acc, %x_val
  %new_acc = select i1 %is_pos, i64 %acc_add, i64 %acc_sub
```
**Notice what is missing: There is no `mul` instruction anywhere in the loop!** The hardware executes additions, subtractions, and conditional selects. This saves massive amounts of silicon area and battery power.

---

## 4. Compiler CLI Flags in `bt_compiler.cpp`

The driver executable `bt-compiler` provides a rich set of command-line flags for compiling, optimizing, cross-targeting, and debugging.

### Complete CLI Options Reference

| Flag / Command | Description | Example Usage |
| :--- | :--- | :--- |
| `run <file.bt>` | Immediately executes program via the in-memory JIT interpreter. | `./bt-compiler run examples/01_arithmetic_spaceship.bt` |
| `run --native <file.bt>` | Compiles program to native machine code via Clang and executes at bare-metal speed via `dlopen`. | `./bt-compiler run --native examples/01_arithmetic_spaceship.bt` |
| `--target=<arch>` | Selects compilation target: `aarch64`, `x86_64`, `riscv64`, `metal`, `vulkan`. | `./bt-compiler --target=x86_64 -S prog.bt` |
| `-S, --emit-asm` | Emits human-readable target assembly code (`.s`). | `./bt-compiler --target=aarch64 -S prog.bt -o prog.s` |
| `-c, --emit-obj` | Emits standalone native object file (`.o`). | `./bt-compiler --target=aarch64 -c prog.bt -o prog.o` |
| `--emit-metal` | Emits Apple Metal Shading Language compute shader for M-series GPUs. | `./bt-compiler --emit-metal` |
| `--emit-vulkan` | Emits Vulkan SPIR-V GLSL compute shader for Android, AMD, and Intel GPUs. | `./bt-compiler --emit-vulkan` |
| `-g, --debug` | Emits DWARF 5 source-level debug metadata. | `./bt-compiler -g --emit-llvm prog.bt` |
| `--emit-llvm` | Emits human-readable LLVM Intermediate Representation (`.ll`). | `./bt-compiler --emit-llvm prog.bt` |
| `--emit-llvm-bc` | Emits binary LLVM Bitcode (`.bc`). | `./bt-compiler --emit-llvm-bc prog.bt -o prog.bc` |
| `--lto` | Enables ThinLTO Link-Time Optimization (`-flto=thin`). | `./bt-compiler --emit-llvm-bc --lto prog.bt` |
| `--test` | Runs embedded compiler invariant verification suite. | `./bt-compiler --test` |
| `--kokkos` | Enables Kokkos parallel multi-threading for large matrix operations. | `./bt-compiler run prog.bt --kokkos --threads=8` |

---

### Exploring Cross-Target Outputs

One of the most remarkable features of `bt_compiler.cpp` is its ability to compile balanced ternary algorithms to wildly different architectures.

#### Target 1: Apple Metal GPU Shaders (`--emit-metal`)
Running `./bt-compiler --emit-metal` produces Apple Metal Shading Language (MSL) code targeting Apple Silicon unified memory GPUs:

```metal
// Target: Apple Silicon M-Series GPU (Unified Memory Architecture)
#include <metal_stdlib>
using namespace metal;

kernel void bt_gemm_metal(
    device const uint64_t* A_pos      [[buffer(0)]],
    device const uint64_t* A_neg      [[buffer(1)]],
    device const uint64_t* BT_pos     [[buffer(2)]],
    device const uint64_t* BT_neg     [[buffer(3)]],
    device int64_t*        C_out      [[buffer(4)]],
    constant uint&         M          [[buffer(5)]],
    constant uint&         N          [[buffer(6)]],
    uint2                  gid        [[thread_position_in_grid]])
{
    // Dual-rail bitwise multiplication & SIMD hardware popcount
    uint64_t p_prod = (ap & bp) | (an & bn);
    uint64_t n_prod = (ap & bn) | (an & bp);
    acc += static_cast<int64_t>(popcount(p_prod)) - static_cast<int64_t>(popcount(n_prod));
}
```

#### Target 2: Vulkan SPIR-V GLSL Shaders (`--emit-vulkan`)
Running `./bt-compiler --emit-vulkan` produces cross-platform Vulkan compute shaders for Qualcomm Adreno, ARM Mali, AMD Radeon, and Intel Arc graphics:

```glsl
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 16, local_size_y = 16) in;
// Dual-rail 64-bit bitplane parallel reductions
```

#### Target 3: Cross-Target Assembly (`-S`)
Compile `examples/01_arithmetic_spaceship.bt` directly to ARM64 assembly:
```bash
./bt-compiler --target=aarch64 -S examples/01_arithmetic_spaceship.bt -o /tmp/arm64.s
```

Or cross-compile to Intel x86_64 assembly:
```bash
./bt-compiler --target=x86_64 -S examples/01_arithmetic_spaceship.bt -o /tmp/x86.s
```

Or cross-compile to RISC-V 64 assembly:
```bash
./bt-compiler --target=riscv64 -S examples/01_arithmetic_spaceship.bt -o /tmp/riscv.s
```

---

## 5. Source-Level Debugging: Emitting DWARF-5 Metadata

Have you ever wondered how debuggers like `gdb` or `lldb` let you step line by line through your code?

When a CPU executes machine code, it only sees memory addresses (e.g., `0x100003f40: add x0, x1, x2`). The CPU has no idea what a variable named `sum` or a file called `examples/01_arithmetic_spaceship.bt` is!

### What is DWARF?

**Analogy:** Imagine a surveyor mapping out a brand new city. The surveyor writes a detailed directory: *"The front door of the bakery at 14 Elm Street is located at GPS coordinates 37.7749° N, 122.4194° W."*

**DWARF** is that exact directory for your computer code. It is an industry-standard binary debugging format that maps machine code memory addresses back to original source file names, line numbers, and column offsets.

---

### Emitting DWARF-5 in `ternary_llvm_codegen.hpp`

When the `-g` flag is passed, `LLVMCodeGenVisitor` emits DWARF 5 metadata structures into the LLVM IR module:

```llvm
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "bt-compiler", emissionKind: LineTablesOnly)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "examples/01_arithmetic_spaceship.bt", directory: ".")
!4 = distinct !DISubprogram(name: "bt_main", scope: !3, file: !3, line: 1, unit: !0)
```

Next, as the visitor walks the AST, every single emitted instruction is tagged with its source location using the `!dbg` tag:

```llvm
  %t12 = call i64 @bt_spaceship(i64 %t10, i64 %t11), !dbg !24
  %var_cmp1_6 = alloca i64, align 8, !dbg !25
  store i64 %t12, i64* %var_cmp1_6, align 8, !dbg !25
```

Where metadata node `!24` points directly to Line 14, Column 14 of `examples/01_arithmetic_spaceship.bt`:

```llvm
!24 = !DILocation(line: 14, column: 14, scope: !4)
```

### The Debugger Experience (Stepping with LLDB)

Because our compiler emits compliant DWARF-5 metadata, you can inspect your `.bt` programs with standard developer tools:

```bash
lldb -- bt-compiler run --native -g examples/01_arithmetic_spaceship.bt
```

Inside LLDB:
```
(lldb) breakpoint set --file 01_arithmetic_spaceship.bt --line 14
Breakpoint 1: where = bt_main + 48 at 01_arithmetic_spaceship.bt:14:14
(lldb) run
Process stopped
* thread #1, stop reason = breakpoint 1.1
    frame #0: bt_main at 01_arithmetic_spaceship.bt:14:14
   11   var diff = a - b;
   12   var prod = a * b;
   13   
-> 14   var cmp_gt = a <=> b;
   15   var cmp_lt = b <=> a;
   16   var cmp_eq = a <=> a;
(lldb) step
```

You can set breakpoints, single-step line by line, and inspect hardware registers directly inside your `.bt` files!

---

## 6. The Engineering Detective Stories: Hard Bugs in Code Generation

Building a compiler is one of the most rewarding challenges in computer engineering, but it is also one of the most unforgiving. Compilers have zero tolerance for ambiguity. 

During the development of our LLVM backend, we encountered subtle bugs that produced mysterious compilation failures. Let us look at how two of the most critical bugs were diagnosed and solved in `.prime-research/deep_audit_compiler_codegen.md` and `ternary_llvm_codegen.hpp`.

---

### Bug Story 1: Variable Shadowing & Duplicate SSA Alloca (`CODEGEN-01`)

#### The Symptom
A test program declared a temporary variable `i` inside two consecutive blocks:

```bt
{
    var i = 0;
}
{
    var i = 10;
}
```

When compiled with `run --native`, Clang suddenly aborted with a fatal error:
```
error: multiple definition of local value named 'var_i'
```

#### The Root Cause
In an early prototype of `ternary_llvm_codegen.hpp`, the code generator created stack allocations using the variable's source name:

```cpp
// ❌ FLAWED IMPLEMENTATION
std::string ptr_reg = "%var_" + node.name;
fn_body_ << "  " << ptr_reg << " = alloca i64, align 8\n";
```

In LLVM IR, **every local identifier within a function must be globally unique.** When the second block declared `var i`, the visitor emitted a second `%var_i = alloca i64`, which violated LLVM's fundamental single-assignment rule!

Furthermore, the compiler used a single flat symbol table (`std::unordered_map<std::string, std::string> symbols_`). If an inner scope declared a variable with the same name as an outer variable (variable shadowing), the inner variable permanently overwrote the outer variable's pointer. When the inner scope exited, the outer variable was lost forever!

#### The Fix
We introduced two robust mechanisms in `ternary_llvm_codegen.hpp`:
1. **Scoped Symbol Stacks**: We replaced the flat map with a stack of symbol tables:
   ```cpp
   std::vector<std::unordered_map<std::string, std::string>> symbol_stack_;
   
   void enter_scope() { symbol_stack_.emplace_back(); }
   void exit_scope()  { if (!symbol_stack_.empty()) symbol_stack_.pop_back(); }
   ```
2. **Globally Unique SSA Allocation Counters**: Every variable allocation receives an atomic monotonic suffix:
   ```cpp
   std::string ptr_reg = "%var_" + node.name + "_" + std::to_string(++alloca_counter_);
   ```

Now, declaring `var i` twice emits `%var_i_1` and `%var_i_2`. Both variables have distinct, valid SSA registers, and variable shadowing works flawlessly across arbitrary nesting levels!

---

### Bug Story 2: Basic Block Terminators & Dead Instructions (`CODEGEN-03`)

#### The Symptom
When compiling code with early returns inside conditional blocks:

```bt
if (x > 0) {
    return 10;
}
return 0;
```

Clang rejected the generated LLVM IR with:
```
error: instruction expected, found 'br' after terminator
```

#### The Root Cause
In LLVM intermediate representation, a basic block is a straight-line sequence of instructions with **exactly one entry point and exactly one exit point.** 

Instructions that transfer control flow—such as `ret` (return) and `br` (branch)—are called **Terminators**. Under LLVM rules, **a terminator must be the absolute final instruction in a basic block.** Nothing may follow it.

In our early visitor implementation for `IfNode`:

```cpp
// ❌ FLAWED IMPLEMENTATION
fn_body_ << then_lbl << ":\n";
if (node.then_block) node.then_block->accept(*this);
fn_body_ << "  br label %" << exit_lbl << "\n"; // Unconditional branch!
```

If `then_block` contained a `return` statement, `visit(ReturnNode)` emitted `ret i32 10`. Then, immediately afterward, the visitor blindly emitted `br label %if_exit`! This produced:

```llvm
if_then_1:
  ret i32 10
  br label %if_exit_3   ; ❌ FATAL ERROR: Instruction placed after terminator!
```

#### The Fix
We added explicit basic block termination tracking via a boolean flag `block_terminated_`:

```cpp
// In ternary_llvm_codegen.hpp
void visit(frontend::ReturnNode& node) override {
    if (block_terminated_) return; // Skip if already terminated
    
    // Emit return instruction...
    fn_body_ << "  ret i32 " << ret_val << "\n";
    block_terminated_ = true; // Mark current block as finished!
}

void visit(frontend::IfNode& node) override {
    // ...
    fn_body_ << then_lbl << ":\n";
    block_terminated_ = false;
    if (node.then_block) node.then_block->accept(*this);
    
    // Only emit branch to exit if the block has NOT terminated with a return!
    if (!block_terminated_) {
        fn_body_ << "  br label %" << exit_lbl << "\n";
    }
}
```

With this change, the compiler intelligently prunes dead branch instructions and avoids generating unreachable empty blocks, ensuring 100% LLVM SSA conformance.

---

### Summary of Fixed Compiler Anomalies

| Bug ID | Component | Root Cause | Engineering Solution |
| :--- | :--- | :--- | :--- |
| `CODEGEN-01` | `ternary_llvm_codegen.hpp` | Duplicate `%var_name` allocas in nested scopes. | Implemented `symbol_stack_` with monotonic `alloca_counter_`. |
| `CODEGEN-02` | `ternary_llvm_codegen.hpp` | `@bt_main` returned `i32` but user return emitted `ret i64`. | Added automatic `trunc i64 to i32` and `fptosi double to i32`. |
| `CODEGEN-03` | `ternary_llvm_codegen.hpp` | Emitted unconditional `br` immediately following `ret`. | Added `block_terminated_` tracking to prune post-terminator dead code. |
| `CODEGEN-04` | `ternary_llvm_codegen.hpp` | Mixed float/int operations emitted `fadd` without cast. | Added automatic `sitofp` promotion for integer operands. |

---

## 7. Check Yourself!

Test your understanding of compiler architecture, LLVM IR, and balanced ternary code generation with these questions:

1. **The Compiler Pipeline**: Why does our compiler emit LLVM Intermediate Representation (IR) instead of emitting raw x86_64 or ARM64 machine bytes directly from the AST? What major engineering advantages does this provide?
2. **Branchless Execution**: In Program 2, why does the 3-way `select(cond, pos, zero, neg)` intrinsic execute faster on a modern CPU than an equivalent chain of `if ... else if ... else` statements?
3. **LLVM SSA Rules**: What does Static Single Assignment (SSA) mean, and why did our original code generator fail when compiling duplicate variable names across distinct curly brace blocks?
4. **Multiplier-Free SpMV**: Look at the inner loop of `bt_spmv` in `ternary_llvm_codegen.hpp`. How does the code calculate the product of a ternary matrix entry and a vector element without using a CPU multiplication instruction?
5. **DWARF Debugging**: What is the purpose of the `!DILocation` metadata attached to LLVM IR instructions? How does a debugger like `lldb` use it when you hit a breakpoint?

---

### Answers & Explanations

1. Emitting LLVM IR decouples the compiler frontend from physical hardware. Instead of writing custom instruction selectors, register allocators, and loop optimizers for ARM64, x86_64, RISC-V, and GPU architectures, our frontend targets a single, clean intermediate language. LLVM's industrial backend handles loop vectorization, constant propagation, and native machine code generation for every CPU on Earth.
2. An `if ... else` chain requires conditional branch instructions (`b.eq`, `jmp`). If the CPU's branch predictor guesses incorrectly, the entire instruction pipeline must be flushed, wasting 15–20 clock cycles. The `select` intrinsic maps directly to conditional select/move instructions (`csel` on ARM64, `cmov` on x86), which execute unconditionally in a single cycle without branches.
3. In SSA form, every register is assigned a value exactly once. When two scopes both declared `var i`, the early compiler emitted `%var_i = alloca ...` twice within the same function. LLVM rejected this as a duplicate local symbol definition. Appending an atomic counter (`%var_i_1`, `%var_i_2`) ensures global uniqueness within the function.
4. In balanced ternary, non-zero entries are restricted to $+1$ or $-1$. In `bt_spmv`, a sign bit indicates whether the entry is positive (bit 1) or negative (bit 0). The kernel computes both `acc + x_val` and `acc - x_val`, and uses a branchless `select` based on the sign bit to pick the correct sum. No hardware multipliers are ever triggered!
5. `!DILocation` records the exact source file, line number, and column offset for an instruction. When Clang compiles the IR, it translates these tags into DWARF line table sections (`.debug_line`). When `lldb` stops at a machine address, it queries the line table, identifies the corresponding line in the `.bt` source file, and displays the source code context to the developer.
