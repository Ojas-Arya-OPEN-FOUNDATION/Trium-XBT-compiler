# Chapter 1: Number Systems From Zero

Have you ever wondered why we count the way we do? Why do we have numbers like 10, 20, 100? Why do computers use 0s and 1s? And what makes "balanced ternary" so mathematically special that computer science pioneer Donald Knuth called it *"perhaps the prettiest number system of all"*?

In this chapter, we will build number systems from scratch. By the time you finish, you will be able to convert any decimal number into balanced ternary in your head or on paper with a pen!

---

## 1. How Counting Works: Positional Base Systems

When you were a child, you learned to count on your ten fingers. Because humans have 10 fingers, human civilization settled on **Decimal** (Base 10).

When you write the number **$352$** in decimal, what does it actually mean?
It means:
$$352 = (3 \times 10^2) + (5 \times 10^1) + (2 \times 10^0)$$
$$352 = (3 \times 100) + (5 \times 10) + (2 \times 1)$$

Every time you move one column to the left, the place value multiplies by **$10$**:
$$\dots, 1000, 100, 10, 1$$

---

## 2. Why Computers Use Binary (Base 2)

A copper wire inside a silicon chip doesn't have ten fingers. It is an electrical circuit. It is easy to measure whether a wire has electricity running through it or not:
- **Voltage High ($\approx 1.2\text{V}$)**: Represents `1`
- **Voltage Low ($0\text{V}$)**: Represents `0`

Because there are only two digits (`0` and `1`), binary place values multiply by **$2$** every time you move one position to the left:
$$\dots, 64, 32, 16, 8, 4, 2, 1$$

To write the number **$13$** in binary:
$$13 = 8 + 4 + 0 + 1 = (1 \times 2^3) + (1 \times 2^2) + (0 \times 2^1) + (1 \times 2^0) = \mathbf{1101_2}$$

### The Big Flaw of Binary: How Do You Write $-13$?
Binary digits can only be `0` or `1`. There is no minus switch!
To represent $-13$, binary computers must invent complex conventions:
1. They reserve the leftmost bit as a **sign bit** (`0` for positive, `1` for negative).
2. Or they use a mathematical trick called **Two's Complement**, where negative numbers wrap around backwards.
3. This creates weird edge cases: you can have numbers where $-128$ exists but $+128$ doesn't fit in an 8-bit byte!

---

## 3. What is Balanced Ternary?

Now let's look at **Ternary (Base 3)**.
In base 3, every position to the left multiplies by **$3$**:
$$\dots, 81, 27, 9, 3, 1$$

Standard "unbalanced" ternary uses the digits `0`, `1`, and `2`.
For example: $5 = (1 \times 3) + (2 \times 1) = 12_3$. That still has the same problem as binary—it cannot represent negative numbers naturally.

### The Breakthrough: Balance Around Zero!
Instead of using `0, 1, 2`, **Balanced Ternary** shifts the digits to center around zero:
$$\mathbf{-1, \quad 0, \quad +1}$$

In computer text, we often write these digits using compact symbols:
- `+` or `1` means $+1$
- `0` means $0$
- `-` or `T` (short for *Trit negative*, or $\bar{1}$) means $-1$

Look at the place values of balanced ternary:
| Place Value | Calculation | Value in Decimal |
| :---: | :---: | :---: |
| $3^4$ | $3 \times 3 \times 3 \times 3$ | **$81$** |
| $3^3$ | $3 \times 3 \times 3$ | **$27$** |
| $3^2$ | $3 \times 3$ | **$9$** |
| $3^1$ | $3$ | **$3$** |
| $3^0$ | $1$ | **$1$** |

Because each digit can be $+1$, $0$, or $-1$, every digit can **add** its place value, **ignore** its place value, or **subtract** its place value!

---

## 4. The Scale Analogy: Measuring Weights with 3-Pan Balance

Imagine an old-fashioned balance scale with two pans:

```
              ┌─────────▲─────────┐
              │                   │
         [Left Pan]          [Right Pan]
         (Subtract)           (Add)
```

Suppose you have weights of sizes **$1\text{ lb}, 3\text{ lb}, 9\text{ lb}, 27\text{ lb}$**.
You put an unknown object on the left pan. How do you measure its weight?
- If you place a weight in the **Right Pan**, it **adds** to the measurement ($+1$).
- If you leave the weight on the **Table**, it **does nothing** ($0$).
- If you place a weight in the **Left Pan** alongside the object, it **subtracts** from the measurement ($-1$).

With just four weights ($1, 3, 9, 27$), you can weigh **every single integer from $-40$ to $+40$**!

---

## 5. Hand-Worked Conversion Examples

Let's convert decimal numbers into balanced ternary step by step.

### Example 1: Convert Decimal $+5$ to Balanced Ternary
Look at our powers of 3: $1, 3, 9, 27\dots$
- Can we make 5 using 9?
  $$9 - 3 - 1 = 5!$$
- In balanced ternary place values:
  - In the **$9$** place: $+1$ (we add 9)
  - In the **$3$** place: $-1$ (we subtract 3)
  - In the **$1$** place: $-1$ (we subtract 1)
- Written in balanced ternary:
  $$\mathbf{+ - -} \quad \text{or} \quad \mathbf{1 \bar{1} \bar{1}} \quad \text{or} \quad \mathbf{1 T T}$$

Let's check the math:
$$(+1 \times 9) + (-1 \times 3) + (-1 \times 1) = 9 - 3 - 1 = \mathbf{5}$$
It works perfectly!

---

### Example 2: Convert Decimal $+14$ to Balanced Ternary
Look at our powers of 3: $1, 3, 9, 27\dots$
- 14 is close to 27:
  $$27 - 9 - 3 - 1 = 14!$$
- Place values:
  - In the **$27$** place: $+1$
  - In the **$9$** place: $-1$
  - In the **$3$** place: $-1$
  - In the **$1$** place: $-1$
- Result:
  $$\mathbf{+ - - -} = (27 - 9 - 3 - 1) = \mathbf{14}$$

---

### Example 3: Convert Decimal $+8$ to Balanced Ternary
- 8 is $9 - 1$:
  - $9$ place: $+1$
  - $3$ place: $0$
  - $1$ place: $-1$
- Result:
  $$\mathbf{+ 0 -} = (9 + 0 - 1) = \mathbf{8}$$

---

### Example 4: The Magic of Negative Numbers (Zero-Cycle Negation!)
Now, what is **$-5$** in balanced ternary?
In decimal or binary, you have to do math to negate a number.
In balanced ternary, to negate a number, **you simply flip every $+$ to $-$ and every $-$ to $+$**:

$$+5 = \mathbf{+ - -}$$
$$-5 = \mathbf{- + +}$$

Let's check the math for $-5$:
$$(-1 \times 9) + (+1 \times 3) + (+1 \times 1) = -9 + 3 + 1 = \mathbf{-5}$$

**Negation is 100% free!**
There is no sign bit, no carry propagation, no two's complement inversion, and no negative-zero anomaly. A computer chip can negate a balanced ternary number by just swapping two wires!

---

## 6. The Mystery of Fractions: Why $\frac{1}{2} = 0.1111\dots_3$

Here is one of the most beautiful mathematical properties of balanced ternary, and why our compiler can do floating-point math with **zero rounding bias**.

What is $\frac{1}{2}$ in balanced ternary?
Let's look at the negative powers of 3 (the fractional places after the ternary point):
$$3^{-1} = \frac{1}{3}, \quad 3^{-2} = \frac{1}{9}, \quad 3^{-3} = \frac{1}{27}, \quad 3^{-4} = \frac{1}{81}\dots$$

If you add up all the positive fractional trits ($+1$ at every fractional place):
$$\sum_{k=1}^\infty \left(\frac{1}{3}\right)^k = \frac{\frac{1}{3}}{1 - \frac{1}{3}} = \frac{\frac{1}{3}}{\frac{2}{3}} = \mathbf{\frac{1}{2}}!$$

So in balanced ternary:
$$\mathbf{+\frac{1}{2} = 0.111111\dots_3 = 0.++++++\dots}$$
$$\mathbf{-\frac{1}{2} = 0.\bar{1}\bar{1}\bar{1}\bar{1}\dots_3 = 0.------\dots}$$

### Why This Eliminates Rounding Bias
In binary, truncation rounds towards zero or towards infinity, causing simulations to accumulate drift.
In balanced ternary:
- The maximum fraction you can discard by truncating trits is $+\frac{1}{2}$ ($0.++++\dots$).
- The minimum fraction you can discard is $-\frac{1}{2}$ ($0.----\dots$).
- The error interval is strictly symmetric around zero: **$[-0.5, +0.5]$**.

When you discard digits (truncation rounding), the average error over random numbers is:
$$\mathbb{E}[\text{error}] = \int_{-0.5}^{+0.5} x \, dx = \left[ \frac{x^2}{2} \right]_{-0.5}^{+0.5} = \frac{0.25}{2} - \frac{0.25}{2} = \mathbf{0.000000}$$

**Balanced ternary floating-point numbers have ZERO round-off bias!** In simulations running for millions of steps, numbers never systematically drift up or down.

---

## 7. Radix Economy: Why Base 3 is the Most Efficient Base

In information theory, how "efficient" is a numbering system?
Suppose you want to represent numbers up to $N$. You use base $R$ (radix).
The number of digits needed is roughly $\log_R N$.
The hardware cost to store each digit is proportional to the base $R$.
So the total hardware cost (Radix Economy) is proportional to:
$$E(R) = R \times \log_R N = R \frac{\ln N}{\ln R}$$

To find the most efficient base $R$ in the universe, calculus tells us to take the derivative of $\frac{R}{\ln R}$ and set it to zero:
$$\frac{d}{dR}\left(\frac{R}{\ln R}\right) = \frac{\ln R - 1}{(\ln R)^2} = 0 \implies \ln R = 1 \implies \mathbf{R = e \approx 2.71828}$$

The mathematically ideal base for a computer is Euler's number **$e \approx 2.718$**.
Since computers must use whole numbers:
- Is 2 closer to $e$? $|2 - 2.718| = 0.718$
- Is 3 closer to $e$? $|3 - 2.718| = \mathbf{0.282}$!

**Base 3 is closer to mathematical perfection than Base 2.** Base 3 has the highest radix economy of any integer base in mathematics!

---

### Check Yourself!
1. What decimal number does the balanced ternary word `+ - +` represent?
   *(Hint: place values are 9, 3, 1).*
2. How do you negate the balanced ternary number `+ + - 0`?
3. What is the decimal value of the negated number?
4. Why does discarding fractional trits produce an average error of exactly zero?

*(Answers: 1. $(+1 \times 9) + (-1 \times 3) + (+1 \times 1) = 9 - 3 + 1 = 7$. 2. Flip all signs: `- - + 0`. 3. $(-1 \times 27) + (-1 \times 9) + (+1 \times 3) + (0 \times 1) = -27 - 9 + 3 = -33$. The original was $+33$. 4. Because the fractional truncation range is perfectly symmetric between $-0.5$ and $+0.5$, meaning positive and negative rounding errors cancel each other out completely).*
