# Chapter 2: The Core Bit-Slicing Tricks (How Binary Chips Run Ternary)

Welcome back! In [Chapter 1](01_number_systems.md), we discovered the pure mathematical beauty of balanced ternary: zero rounding bias, zero-cost sign negation, and the highest radix economy of any integer base.

Now comes the million-dollar engineering challenge: **Every modern silicon chip in the world—from your laptop's Intel, AMD, or Apple Silicon CPU to NVIDIA graphics cards—is built entirely out of binary transistors.**

If you tried to simulate balanced ternary naively in C++ using an `enum` or integer:
```cpp
// ❌ NAIVE APPROACH: Slow, branching, pipeline-stalling code
if (trit == 1) {
    result += value;
} else if (trit == -1) {
    result -= value;
}
```
your CPU would spend half its time stalled, waiting for branch predictors to guess whether the next trit is positive, negative, or zero. It would run **10x to 50x slower** than native binary code!

In this chapter, you will learn the secret weapons that allow this compiler to run balanced ternary math **faster than standard 32-bit binary floats on standard binary processors**. We will break down every trick step by step with everyday analogies, ASCII diagrams, Mermaid data-flow graphs, and hand-worked math!

---

## 1. Dual-Rail Bit-Slicing: Two Sheets of Paper (P and N Registers)

How do you represent three possible values ($-1, 0, +1$) using standard binary bits that can only be `0` or `1`?

### The Two Sheets of Paper Analogy
Imagine you are a teacher grading a 64-question true/false/abstain test:
- A student can answer correctly ($+1$).
- A student can answer incorrectly ($-1$).
- A student can leave the answer blank ($0$).

Instead of trying to squeeze a three-way score into a single column, you take **two clear sheets of tracing paper**:
1. **Sheet P (Positive)**: You put a mark in box $i$ if question $i$ was $+1$.
2. **Sheet N (Negative)**: You put a mark in box $i$ if question $i$ was $-1$.

```
Question Slot:      #0     #1     #2     #3     #4
---------------------------------------------------
Sheet P (Positive): [ 1 ]  [ 0 ]  [ 0 ]  [ 1 ]  [ 0 ]
Sheet N (Negative): [ 0 ]  [ 1 ]  [ 0 ]  [ 0 ]  [ 1 ]
---------------------------------------------------
Interpreted Trit:    +1     -1      0     +1     -1
```

Look at what happens:
- If box $i$ has a mark on **Sheet P only** $\to$ the trit is $\mathbf{+1}$.
- If box $i$ has a mark on **Sheet N only** $\to$ the trit is $\mathbf{-1}$.
- If box $i$ has **no mark on either sheet** $\to$ the trit is $\mathbf{0}$.
- What if both sheets have a mark? **That is physically impossible!** A question cannot be both correct and incorrect simultaneously. In computer terms, $P$ and $N$ must always be **disjoint** (`(P & N) == 0`).

### The C++ Implementation in `ternary_core.hpp`
In `ternary_core.hpp`, this is implemented as `DualRailTrit` for single trits, and `TritWord64` for 64 trits in parallel:

```cpp
struct TritWord64 {
    uint64_t P{0}; // Positive rail: bit i is 1 if trit i == +1
    uint64_t N{0}; // Negative rail: bit i is 1 if trit i == -1

    constexpr bool isValid() const noexcept {
        return (P & N) == 0; // Both rails must never be active at the same time
    }
};
```

In just **128 bits of memory** (two 64-bit CPU registers), we store **64 complete balanced trits**.

```mermaid
flowchart TD
    subgraph TritWord64["TritWord64 Register Pair (128 bits)"]
        P["uint64_t P (Positive Rail Bitmask)"]
        N["uint64_t N (Negative Rail Bitmask)"]
    end
    
    P --> |Bitwise AND| V["Validation Invariant: (P & N) == 0"]
    N --> |Bitwise AND| V
    
    P --> Dec["Parallel ALU Decoding"]
    N --> Dec
    Dec --> T0["Trit 0: (P[0] - N[0])"]
    Dec --> T1["Trit 1: (P[1] - N[1])"]
    Dec --> TDots["..."]
    Dec --> T63["Trit 63: (P[63] - N[63])"]
```

### The 0-Cycle Negation Superpower
In binary computers, negating a number requires Two's Complement: invert all bits (`NOT`) and add 1 (`ADD`), which requires electric charge to ripple through the full adder circuit.

How do you negate a 64-trit number in dual-rail balanced ternary?
To flip $+1 \leftrightarrow -1$, you simply **swap the two sheets of paper**!
- Marks on Sheet P become Sheet N.
- Marks on Sheet N become Sheet P.
- Blanks ($0$) stay blanks.

Look at `ternary_core.hpp`:
```cpp
// 0-Cycle Negation: compile-time register swap!
constexpr TritWord64 operator-() const noexcept {
    return {N, P};
}
```

In compiled assembly code, swapping two registers costs **literally zero clock cycles** (the compiler simply renames the registers in its Static Single Assignment table, or emits a single `mov` instruction without any arithmetic latency).

---

## 2. Multiplying Trits with AND and OR Logic (Zero Multipliers!)

In standard binary math, multiplying two numbers requires dedicated hardware multipliers: large grids of transistors called Wallace trees or Booth encoders that consume enormous amounts of silicon and battery power.

In balanced ternary, multiplying two trits ($A \times B$) follows the elementary school signs rule:
- $(+1) \times (+1) = \mathbf{+1}$
- $(-1) \times (-1) = \mathbf{+1}$
- $(+1) \times (-1) = \mathbf{-1}$
- $(-1) \times (+1) = \mathbf{-1}$
- Anything $\times 0 = \mathbf{0}$

Let's look at the complete truth table:

| Input $A$ | Input $B$ | Product ($A \times B$) |
| :---: | :---: | :---: |
| $+1$ | $+1$ | **$+1$** |
| $+1$ | $0$ | **$0$** |
| $+1$ | $-1$ | **$-1$** |
| $0$ | $+1$ | **$0$** |
| $0$ | $0$ | **$0$** |
| $0$ | $-1$ | **$0$** |
| $-1$ | $+1$ | **$-1$** |
| $-1$ | $0$ | **$0$** |
| $-1$ | $-1$ | **$+1$** |

### Deriving the Dual-Rail Logic
Now let's ask: under what conditions is the product positive, and under what conditions is it negative?

1. **When is the product positive ($+1$)?**
   The product is positive if and only if:
   - Both inputs are positive: $(P_A \land P_B)$
   - **OR** both inputs are negative: $(N_A \land N_B)$
   
   Therefore, the positive rail of the product is:
   $$\mathbf{P_{prod} = (P_A \ \& \ P_B) \ | \ (N_A \ \& \ N_B)}$$

2. **When is the product negative ($-1$)?**
   The product is negative if and only if:
   - $A$ is positive and $B$ is negative: $(P_A \land N_B)$
   - **OR** $A$ is negative and $B$ is positive: $(N_A \land P_B)$
   
   Therefore, the negative rail of the product is:
   $$\mathbf{N_{prod} = (P_A \ \& \ N_B) \ | \ (N_A \ \& \ P_B)}$$

3. **When is the product zero ($0$)?**
   If neither condition holds, both $P_{prod}$ and $N_{prod}$ will be `0`, meaning the trit is automatically $0$!

```mermaid
flowchart TD
    subgraph Inputs["Dual-Rail Inputs (64 trits each)"]
        PA["A.P"]
        NA["A.N"]
        PB["B.P"]
        NB["B.N"]
    end
    
    PA & PB --> |Bitwise AND| AND_PP["P_A & P_B"]
    NA & NB --> |Bitwise AND| AND_NN["N_A & N_B"]
    AND_PP & AND_NN --> |Bitwise OR| P_prod["P_prod (Positive Terms)"]
    
    PA & NB --> |Bitwise AND| AND_PN["P_A & N_B"]
    NA & PB --> |Bitwise AND| AND_NP["N_A & P_B"]
    AND_PN & AND_NP --> |Bitwise OR| N_prod["N_prod (Negative Terms)"]
```

Look at what we just accomplished:
**64 multiplications happen simultaneously across 64 pairs of trits using just 4 bitwise instructions (`&`, `&`, `|`, `|`).**
There are **zero hardware multipliers**, zero floating-point units, and zero conditional branches involved!

---

## 3. The Dot Product with POPCNT: 4 Bitwise Ops + 2 Popcounts

The dot product (multiplying two vectors element-by-element and adding up the results) is the absolute foundation of modern computing: it powers Artificial Intelligence (Large Language Models, neural networks), computer graphics, physics engines, and matrix algebra.

In standard binary 32-bit floating point, computing the dot product of two 64-element vectors requires:
- 64 floating-point multiplies
- 63 floating-point additions

In balanced ternary, we do the entire 64-element dot product in **just 7 CPU instructions**!

### The POPCNT Magic
Recall what we got from our bitwise multiplication:
- `P_prod` is a 64-bit integer with a `1` at every position where the product was $+1$.
- `N_prod` is a 64-bit integer with a `1` at every position where the product was $-1$.

To find the total sum of all 64 products, what do we need to do?
$$\text{Dot Product} = (\text{Count of all }+1\text{s}) - (\text{Count of all }-1\text{s})$$

Every modern CPU (x86_64, ARM64 Apple Silicon) has a built-in silicon instruction that counts how many `1` bits are in a register in a single clock cycle: **Population Count** (`POPCNT` on x86, `CNT` on ARM).

Look at `dotProduct` in `ternary_core.hpp`:
```cpp
inline int64_t dotProduct(const TritWord64& a, const TritWord64& b) noexcept {
    // 4 bitwise instructions to calculate all 64 products simultaneously:
    uint64_t P_prod = (a.P & b.P) | (a.N & b.N);
    uint64_t N_prod = (a.P & b.N) | (a.N & b.P);

    // 2 hardware popcounts + 1 integer subtraction:
    return static_cast<int64_t>(TritWord64::popcount64(P_prod)) - 
           static_cast<int64_t>(TritWord64::popcount64(N_prod));
}
```

That is the entire dot product:
$$\mathbf{4 \text{ bitwise ops} + 2 \text{ popcounts} + 1 \text{ subtraction} = 7 \text{ instructions for } 64 \text{ elements}!}$$

---

### Worked 4-Trit Example by Hand

Let's work through a complete example by hand, tracing every single bit.

Suppose we have two 4-trit vectors:
$$\vec{A} = (+1, \ -1, \ 0, \ +1)$$
$$\vec{B} = (+1, \ +1, \ -1, \ -1)$$

#### Step 1: Write down the dual-rail binary bitmasks
Bit 0 is the leftmost slot, Bit 3 is the rightmost slot:

```
Index:               0   1   2   3
-----------------------------------
Vector A:           +1  -1   0  +1
A.P (positive):      1   0   0   1   (binary: 1001_2)
A.N (negative):      0   1   0   0   (binary: 0010_2)

Vector B:           +1  +1  -1  -1
B.P (positive):      1   1   0   0   (binary: 0011_2)
B.N (negative):      0   0   1   1   (binary: 1100_2)
```

#### Step 2: Calculate Positive Product ($P_{prod}$)
$$P_A \ \& \ P_B = 1001_2 \ \& \ 0011_2 = \mathbf{0001_2} \quad (\text{only bit } 0 \text{ is } 1)$$
$$N_A \ \& \ N_B = 0010_2 \ \& \ 1100_2 = \mathbf{0000_2}$$
$$P_{prod} = 0001_2 \ | \ 0000_2 = \mathbf{0001_2}$$

Interpretation: Bit 0 is the only position where the product is $+1$.

#### Step 3: Calculate Negative Product ($N_{prod}$)
$$P_A \ \& \ N_B = 1001_2 \ \& \ 1100_2 = \mathbf{1000_2} \quad (\text{bit } 3 \text{ is } 1)$$
$$N_A \ \& \ P_B = 0010_2 \ \& \ 0011_2 = \mathbf{0010_2} \quad (\text{bit } 1 \text{ is } 1)$$
$$N_{prod} = 1000_2 \ | \ 0010_2 = \mathbf{1010_2}$$

Interpretation: Bits 1 and 3 are positions where the product is $-1$.

#### Step 4: Hardware Population Counts
$$\text{popcount}(P_{prod}) = \text{popcount}(0001_2) = \mathbf{1}$$
$$\text{popcount}(N_{prod}) = \text{popcount}(1010_2) = \mathbf{2}$$

#### Step 5: Final Subtraction
$$\text{Result} = 1 - 2 = \mathbf{-1}$$

#### Verification by Traditional School Arithmetic:
$$(+1 \times +1) + (-1 \times +1) + (0 \times -1) + (+1 \times -1)$$
$$= (+1) + (-1) + (0) + (-1) = 0 + 0 - 1 = \mathbf{-1}$$

It matches identically! In silicon, this math runs at the maximum physical speed of the CPU execution pipelines.

---

## 4. BTHA: Balanced Ternary Half-Adder & Carry-Save Adders (CSA)

What about addition?
When you add two trits $A + B$, the sum can range from $-2$ to $+2$:
$$A + B = 3 \times \text{Carry} + \text{Sum}$$

Let's look at all possible sums:
- $(+1) + (+1) = +2 = 3(+1) + (-1) \implies \mathbf{\text{Carry} = +1, \ \text{Sum} = -1}$
- $(+1) + (0)  = +1 = 3(0)  + (+1) \implies \mathbf{\text{Carry} = 0,  \ \text{Sum} = +1}$
- $(+1) + (-1) =  0 = 3(0)  + (0)  \implies \mathbf{\text{Carry} = 0,  \ \text{Sum} = 0}$
- $(0)  + (0)  =  0 = 3(0)  + (0)  \implies \mathbf{\text{Carry} = 0,  \ \text{Sum} = 0}$
- $(-1) + (0)  = -1 = 3(0)  + (-1) \implies \mathbf{\text{Carry} = 0,  \ \text{Sum} = -1}$
- $(-1) + (-1) = -2 = 3(-1) + (+1) \implies \mathbf{\text{Carry} = -1, \ \text{Sum} = +1}$

Notice the incredible symmetry:
1. A **Carry of $+1$** occurs *only* when both inputs are $+1$:
   $$\mathbf{PC = A.P \ \& \ B.P}$$
2. A **Carry of $-1$** occurs *only* when both inputs are $-1$:
   $$\mathbf{NC = A.N \ \& \ B.N}$$
3. Look at the Sum when a carry is generated:
   - When Carry is $+1$, the Sum is $-1$!
   - When Carry is $-1$, the Sum is $+1$!
   - The carry and the sum naturally oppose each other!

### The 6-Instruction BTHA in `ternary_core.hpp`
Here is how `BTHA` computes both Sum and Carry for 64 trits simultaneously:

```cpp
inline AdderOutput BTHA(const TritWord64& A, const TritWord64& B) noexcept {
    // 1. Carries (2 instructions)
    uint64_t PC = A.P & B.P;
    uint64_t NC = A.N & B.N;

    // 2. Sums (4 instructions)
    uint64_t PS = ((A.P ^ B.P) & ~(A.N | B.N)) | NC;
    uint64_t NS = ((A.N ^ B.N) & ~(A.P | B.P)) | PC;

    return { TritWord64(PS, NS), TritWord64(PC, NC) };
}
```

Six single-cycle bitwise instructions (`&`, `&`, `^`, `|`, `~&`, `|`) add 64 trits in parallel without any ripple stalls!

---

### Carry-Save Adders (CSA): O(1) Constant-Time Accumulation

In ordinary binary math, when you add numbers repeatedly in a loop (like accumulating 10,000 matrix rows), carries must ripple from right to left across all 64 bits. This ripple delay creates CPU pipeline stalls.

**The Solution: Carry-Save Addition (CSA).**
Instead of forcing carries to ripple immediately, a Carry-Save Adder maintains two running vectors:
- Vector **$S$** (Running Sums)
- Vector **$C$** (Running Carries)

When a new 64-trit vector $X$ arrives, we do **not** do a slow ripple add. We pass $(S, C \ll 1, X)$ into a **Balanced Ternary Full-Adder (BTFA / 3-to-2 Compressor)**:

```mermaid
flowchart TD
    subgraph Accumulator["BT_CSA Register State"]
        S["Sum Vector (S)"]
        C["Carry Vector (C)"]
    end
    
    C --> |Shift Left by 1 Trit| C_sh["C << 1 (Multiplied by 3)"]
    X["New Input Vector (X)"] --> Compressor["BTFA (3-to-2 Compressor)"]
    S --> Compressor
    C_sh --> Compressor
    
    Compressor --> S_new["New Sum Vector (S')"]
    Compressor --> C_new["New Carry Vector (C')"]
```

Because 3 inputs are compressed into 2 outputs in constant time, **accumulation takes $O(1)$ time regardless of whether you are adding 10 numbers or 10,000,000 numbers!**

### Canonicalization at the Exit Gate
When the loop finishes, we must finally resolve $(S, C)$ into a single final vector. In binary, resolving carries requires a full 64-bit ripple.

In balanced ternary, because carries can be positive ($+1$) or negative ($-1$), **positive and negative carries collide and cancel each other out!** In mathematical tests and Z3 SMT proofs (`verify_z3_ternary.py`), the canonicalization loop:
```cpp
while (curr_C.P != 0 || curr_C.N != 0) {
    uint64_t sh_P = curr_C.P << 1;
    uint64_t sh_N = curr_C.N << 1;
    AdderOutput ha = BTHA(curr_S, TritWord64(sh_P, sh_N));
    curr_S = ha.Sum;
    curr_C = ha.Carry;
}
```
terminates in just **2 to 3 iterations**!

---

## 5. Base-243 Packing & Division-Free Magic Multipliers

When storing millions of trits in RAM or on disk, how should they be packed?
- If we store 1 trit in a 1-byte variable, we waste 75% of our RAM.
- If we use dual-rail registers, we use 2 bits per trit. But mathematically, one trit represents $\log_2(3) \approx 1.585$ bits of information.

### The Base-243 Discovery: 5 Trits in 1 Byte
Look at the powers of 3:
$$3^1 = 3, \quad 3^2 = 9, \quad 3^3 = 27, \quad 3^4 = 81, \quad \mathbf{3^5 = 243}$$

A standard computer byte (8 bits) holds values from $0$ to $255$.
Since **$243 \le 255$**, we can pack **5 entire trits into a single 8-bit byte**!
- Storage rate: $\frac{8 \text{ bits}}{5 \text{ trits}} = \mathbf{1.6 \text{ bits per trit}}$.
- Storage efficiency: $\frac{1.585}{1.600} = \mathbf{99.06\%}$ of theoretical Shannon limit!

To pack 5 trits into byte $B \in [0, 242]$:
Map trits $\{-1, 0, +1\}$ to $\{0, 1, 2\}$ by adding 1:
$$B = d_0 + 3d_1 + 9d_2 + 27d_3 + 81d_4$$

---

### The Unpacking Bottleneck
Packing is easy: simple multiplies and adds. But how do you **unpack** byte $B$ back into its 5 trits?
The traditional mathematical way is repeated division by 3:
- $q_1 = \lfloor B / 3 \rfloor, \quad d_0 = B - 3q_1$
- $q_2 = \lfloor q_1 / 3 \rfloor, \quad d_1 = q_1 - 3q_2$
- $q_3 = \lfloor q_2 / 3 \rfloor, \quad d_2 = q_2 - 3q_3$
- $q_4 = \lfloor q_3 / 3 \rfloor, \quad d_3 = q_3 - 3q_4, \quad d_4 = q_4$

**The Disaster:** On modern processors, integer division (`IDIV`) is the slowest instruction in the entire CPU. It cannot be pipelined easily and takes **15 to 40 clock cycles per division**. Unpacking 5 trits would take over **100 clock cycles**!

---

### The Cure: Division-Free Magic Multipliers `(171, 57, 19, 203)`

Can we divide by 3 without using a division instruction?
**Yes! By multiplying by a fixed-point reciprocal and shifting right!**

Let's see why:
Suppose you want to divide an integer $B$ by 3.
In binary fractions, $\frac{1}{3} \approx 0.01010101\dots_2$.
If we multiply $B$ by a carefully chosen integer $M$ and shift right by $S$ bits:
$$\lfloor B / D \rfloor = (B \times M) \gg S$$

Let's inspect the 4 magic multipliers used in `Base243Codec::unpack5_fast` in `ternary_core.hpp`:

| Divisor $D$ | Magic Multiplier $M$ | Shift $S$ | Reciprocal Ratio $\frac{M}{2^S}$ | Exact Fraction $\frac{1}{D}$ | Error for $B \le 242$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$3$** | **$171$** | **$9$** ($2^9 = 512$) | $\frac{171}{512} \approx 0.333984$ | $\frac{1}{3} \approx 0.333333$ | Zero integer error! |
| **$9$** | **$57$** | **$9$** ($2^9 = 512$) | $\frac{57}{512} \approx 0.111328$ | $\frac{1}{9} \approx 0.111111$ | Zero integer error! |
| **$27$** | **$19$** | **$9$** ($2^9 = 512$) | $\frac{19}{512} \approx 0.037109$ | $\frac{1}{27} \approx 0.037037$ | Zero integer error! |
| **$81$** | **$203$** | **$14$** ($2^{14} = 16384$) | $\frac{203}{16384} \approx 0.012390$ | $\frac{1}{81} \approx 0.012345$ | Zero integer error! |

#### Mathematical Proof for $M=171, S=9$:
Notice that $171 \times 3 = 513 = 512 + 1 = 2^9 + 1$.
Therefore:
$$\frac{B \times 171}{512} = \frac{B \times \frac{512 + 1}{3}}{512} = \frac{B}{3} + \frac{B}{3 \times 512} = \frac{B}{3} + \frac{B}{1536}$$

Since $B$ is a packed byte, its maximum value is $242$.
What is the largest possible error term?
$$\frac{B}{1536} \le \frac{242}{1536} \approx \mathbf{0.1575}$$

Because the error $0.1575$ is strictly less than $\frac{1}{3} \approx 0.3333$, **it can never push the result over the next integer threshold!**
Therefore, taking the floor via bit-shift:
$$\mathbf{\lfloor (B \times 171) \gg 9 \rfloor = \lfloor B / 3 \rfloor \quad \text{for all } B \in [0, 242]!}$$

---

### Worked Numerical Unpack Example by Hand

Let's unpack the byte **$B = 200$** using only these magic multipliers.

1. **Calculate quotients:**
   - $q_1 = (200 \times 171) \gg 9 = 34200 \gg 9 = \lfloor 34200 / 512 \rfloor = \mathbf{66}$
     *(Check: $200 / 3 = 66.666\dots \implies 66$. Perfect!)*
   - $q_2 = (200 \times 57) \gg 9 = 11400 \gg 9 = \lfloor 11400 / 512 \rfloor = \mathbf{22}$
     *(Check: $200 / 9 = 22.222\dots \implies 22$. Perfect!)*
   - $q_3 = (200 \times 19) \gg 9 = 3800 \gg 9 = \lfloor 3800 / 512 \rfloor = \mathbf{7}$
     *(Check: $200 / 27 = 7.407\dots \implies 7$. Perfect!)*
   - $q_4 = (200 \times 203) \gg 14 = 40600 \gg 14 = \lfloor 40600 / 16384 \rfloor = \mathbf{2}$
     *(Check: $200 / 81 = 2.469\dots \implies 2$. Perfect!)*

2. **Extract trits by subtracting shifted multiples and shifting by $-1$:**
   - $t_0 = (B - 3 \times q_1) - 1 = (200 - 3 \times 66) - 1 = (200 - 198) - 1 = 2 - 1 = \mathbf{+1}$
   - $t_1 = (q_1 - 3 \times q_2) - 1 = (66 - 3 \times 22) - 1 = (66 - 66) - 1 = 0 - 1 = \mathbf{-1}$
   - $t_2 = (q_2 - 3 \times q_3) - 1 = (22 - 3 \times 7) - 1 = (22 - 21) - 1 = 1 - 1 = \mathbf{0}$
   - $t_3 = (q_3 - 3 \times q_4) - 1 = (7 - 3 \times 2) - 1 = (7 - 6) - 1 = 1 - 1 = \mathbf{0}$
   - $t_4 = q_4 - 1 = 2 - 1 = \mathbf{+1}$

The unpacked trits are: **$(+1, -1, 0, 0, +1)$**.

#### Verify Packing:
$$\text{Digits: } d_0 = 2, \ d_1 = 0, \ d_2 = 1, \ d_3 = 1, \ d_4 = 2$$
$$B = 2 \times 1 + 0 \times 3 + 1 \times 9 + 1 \times 27 + 2 \times 81$$
$$B = 2 + 0 + 9 + 27 + 162 = \mathbf{200}$$

Exact match! On modern CPUs, multiplication and shift instructions execute in **1 single clock cycle**. Unpacking 5 trits takes under **4 nanoseconds**!

---

## 6. Douglas W. Jones SWAR Tree Reduction

Suppose you have 16 trits stored in a 32-bit register and you want to convert them into a standard binary integer.
If you use a loop:
```cpp
// ❌ Slow serial loop: 16 iterations, 16 multiplications by 3
for (int i = 0; i < 16; ++i) {
    result = result * 3 + trit[i];
}
```
the CPU must execute 16 sequential operations, where each step must wait for the previous step to finish.

In the 1990s, computer scientist Douglas W. Jones invented an ingenious algorithm using **SWAR (SIMD Within A Register)** to reduce the entire 16-trit vector into binary in **just 4 parallel stages**!

### The SWAR Tournament Analogy
Think of a sports tournament with 16 teams. Instead of having Team 1 play Team 2, then the winner play Team 3, and so on (15 serial rounds), you pair them up and run games simultaneously:
- **Round 1**: 8 simultaneous games (Pairs)
- **Round 2**: 4 simultaneous games (Quads)
- **Round 3**: 2 simultaneous games (Octets)
- **Round 4**: 1 final game (The Championship)

In just $\log_2(16) = 4$ rounds, the tournament is finished!

```mermaid
flowchart TD
    subgraph Stage0["Input: 16 Packed 2-bit Trits (Base 3)"]
        T["T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11 T12 T13 T14 T15"]
    end
    
    T --> Stage1["Stage 1: Pairs -> Base 9<br/>(8 parallel 4-bit fields: V1 = 3 * odd + even)"]
    Stage1 --> Stage2["Stage 2: Quads -> Base 81<br/>(4 parallel 8-bit fields: V2 = 9 * odd + even)"]
    Stage2 --> Stage3["Stage 3: Octets -> Base 6561<br/>(2 parallel 16-bit fields: V3 = 81 * odd + even)"]
    Stage3 --> Stage4["Stage 4: Hexadecs -> Binary (Base 3^16 = 43,046,721)<br/>(1 32-bit binary field: V4 = 6561 * odd + even)"]
```

### The C++ Implementation: `bct_to_bin_swar32`
Look at `bct_to_bin_swar32` in `ternary_core.hpp`:

```cpp
constexpr inline uint32_t bct_to_bin_swar32(uint32_t bct) noexcept {
    // Stage 1: Pairs -> Base 9 (8 parallel 4-bit fields)
    uint32_t v1 = (bct & 0x33333333U) + 3U * ((bct >> 2) & 0x33333333U);

    // Stage 2: Quads -> Base 81 (4 parallel 8-bit fields)
    uint32_t v2 = (v1 & 0x0F0F0F0FU) + 9U * ((v1 >> 4) & 0x0F0F0F0FU);

    // Stage 3: Octets -> Base 6561 (2 parallel 16-bit fields)
    uint32_t v3 = (v2 & 0x00FF00FFU) + 81U * ((v2 >> 8) & 0x00FF00FFU);

    // Stage 4: Hexadecs -> Binary (Base 3^16 = 43046721)
    uint32_t v4 = (v3 & 0x0000FFFFU) + 6561U * (v3 >> 16);

    return v4;
}
```

Notice the constant multipliers at each stage:
- Stage 1: multiplies by **$3$** ($3^1$)
- Stage 2: multiplies by **$9$** ($3^2$)
- Stage 3: multiplies by **$81$** ($3^4$)
- Stage 4: multiplies by **$6561$** ($3^8$)

### Converting Full 64-Trit Words: `swar_to_int64`
In `ternary_core.hpp`, `swar_to_int64` takes a 64-trit `TritWord64`:
1. It uses `spread_16_to_32` to expand the positive rail ($P$) and negative rail ($N$) into 2-bit fields.
2. It runs `bct_to_bin_swar32` on the $P$ bits and $N$ bits independently.
3. It subtracts: $\text{Result} = \text{Binary}(P) - \text{Binary}(N)$.
4. For larger words, it scales Chunk 1 (trits 16..31) by $3^{16} = 43,046,721$ and Chunk 2 (trits 32..38) by $3^{32} = 1,853,020,188,851,841$.

In fewer than 30 assembly instructions, a full 39-trit word is converted to an `int64_t` without a single loop or branch!

---

## 7. Summary of Microarchitectural Cost Model

To prove why balanced ternary outperforms binary on silicon, look at `CycleCostModel` in `ternary_core.hpp`:

| Operation | Standard Binary Approach | Balanced Ternary Bit-Sliced | Speedup / Advantage |
| :--- | :--- | :--- | :--- |
| **Negation ($-X$)** | `NEG` (inversion + carry ripple) | Register rail swap (`{N, P}`) | **0 cycles (Instant)** |
| **Dot Product (64-wide)** | 64 FMAs (Floating-Point Multiply-Adds) | 4 bitwise ops + 2 `POPCNT` + 1 `SUB` | **~4 CPU cycles total** |
| **Accumulation** | Ripple-carry adder stall | Constant-time BT-CSA (3-to-2 compressor) | **$O(1)$ constant time** |
| **Trit Multiplication** | Silicon hardware multiplier gates | 2 `AND` + 1 `OR` gates | **Single-cycle, low power** |
| **Byte Unpacking** | 4 slow hardware `IDIV` instructions | 4 multiply-shifts `(171, 57, 19, 203)` | **~20x faster than `IDIV`** |

---

### Check Yourself!

1. **Dual-Rail Representation**: If a `TritWord64` has `P = 0x05` and `N = 0x0A`, what are the trits at positions 0, 1, 2, and 3? Is this word valid?
   *(Hint: In binary, 0x05 is `0101_2` and 0x0A is `1010_2`).*

2. **0-Cycle Negation**: If `w` has `P = 0x05` and `N = 0x0A`, what are `P` and `N` for `-w`?

3. **Bitwise Multiplication**: Suppose $A = -1$ and $B = -1$. Using the dual-rail equations $P_{prod} = (P_A \ \& \ P_B) \ | \ (N_A \ \& \ N_B)$ and $N_{prod} = (P_A \ \& \ N_B) \ | \ (N_A \ \& \ P_B)$, prove that the product is $+1$.

4. **Magic Division**: Using the magic multiplier $M = 57$ and shift $S = 9$, calculate $\lfloor 180 / 9 \rfloor$. Does it match $180 / 9 = 20$?

5. **SWAR Reduction**: Why does `bct_to_bin_swar32` run in 4 stages instead of 16 iterations? What are the place value multipliers in each stage?

---

### Answers & Explanations

1. `P = 0101_2` (bits 0 and 2 are 1), `N = 1010_2` (bits 1 and 3 are 1).
   - Position 0: $P=1, N=0 \implies \mathbf{+1}$.
   - Position 1: $P=0, N=1 \implies \mathbf{-1}$.
   - Position 2: $P=1, N=0 \implies \mathbf{+1}$.
   - Position 3: $P=0, N=1 \implies \mathbf{-1}$.
   - Validity check: `P & N = 0101 & 1010 = 0000 = 0`. Yes, the word is 100% valid!
2. Negating simply swaps `P` and `N`. For `-w`, `P = 0x0A` and `N = 0x05`. Position 0 becomes $-1$, position 1 becomes $+1$, etc.
3. For $A = -1$: $P_A = 0, N_A = 1$. For $B = -1$: $P_B = 0, N_B = 1$.
   - $P_{prod} = (0 \ \& \ 0) \ | \ (1 \ \& \ 1) = 0 \ | \ 1 = \mathbf{1}$.
   - $N_{prod} = (0 \ \& \ 1) \ | \ (1 \ \& \ 0) = 0 \ | \ 0 = \mathbf{0}$.
   - Since $P_{prod} = 1$ and $N_{prod} = 0$, the resulting trit is $\mathbf{+1}$.
4. $180 \times 57 = 10260$. Now shift right by 9: $\lfloor 10260 / 512 \rfloor = \mathbf{20}$. Remainder is $10260 - 20 \times 512 = 10260 - 10240 = 20 < 512$. Exact match!
5. Because SWAR uses a parallel divide-and-conquer binary tree. In each stage $k$, pairs of fields are merged simultaneously across the entire 32-bit register. The multipliers are powers of 3 corresponding to field widths: $3^1 = 3$, $3^2 = 9$, $3^4 = 81$, and $3^8 = 6561$.
