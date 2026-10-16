-- Formal Verification of Balanced Ternary Foundations in Lean 4
-- Author: Antigravity Prime Research Engine

inductive Trit : Type where
  | Neg  : Trit  -- -1 (denoted as - or T)
  | Zero : Trit  --  0 (denoted as 0)
  | Pos  : Trit  -- +1 (denoted as + or 1)
  deriving Repr, DecidableEq

namespace Trit

/-- The integer valuation of a trit. -/
def val : Trit → Int
  | Neg  => -1
  | Zero =>  0
  | Pos  =>  1

/-- Symmetric negation: + becomes -, - becomes +, 0 stays 0. -/
def neg : Trit → Trit
  | Neg  => Pos
  | Zero => Zero
  | Pos  => Neg

theorem val_neg (t : Trit) : val (neg t) = - val t := by
  cases t <;> rfl

theorem neg_neg (t : Trit) : neg (neg t) = t := by
  cases t <;> rfl

/-- Half-adder sum trit -/
def ha_sum : Trit → Trit → Trit
  | Pos,  Pos  => Neg   -- 1 + 1 = 2 = 3(1) - 1
  | Pos,  Zero => Pos   -- 1 + 0 = 1
  | Pos,  Neg  => Zero  -- 1 - 1 = 0
  | Zero, Pos  => Pos   -- 0 + 1 = 1
  | Zero, Zero => Zero  -- 0 + 0 = 0
  | Zero, Neg  => Neg   -- 0 - 1 = -1
  | Neg,  Pos  => Zero  -- -1 + 1 = 0
  | Neg,  Zero => Neg   -- -1 + 0 = -1
  | Neg,  Neg  => Pos   -- -1 - 1 = -2 = 3(-1) + 1

/-- Half-adder carry trit -/
def ha_carry : Trit → Trit → Trit
  | Pos,  Pos  => Pos   -- 1 + 1 = 2 = 3(1) - 1
  | Pos,  Zero => Zero
  | Pos,  Neg  => Zero
  | Zero, Pos  => Zero
  | Zero, Zero => Zero
  | Zero, Neg  => Zero
  | Neg,  Pos  => Zero
  | Neg,  Zero => Zero
  | Neg,  Neg  => Neg   -- -1 - 1 = -2 = 3(-1) + 1

/-- Theorem: Half-adder equation holds: a + b = 3 * carry + sum -/
theorem ha_correct (a b : Trit) :
    val a + val b = 3 * val (ha_carry a b) + val (ha_sum a b) := by
  cases a <;> cases b <;> rfl

/-- Full-adder sum trit -/
def fa_sum (a b c : Trit) : Trit :=
  match a, b, c with
  -- sum = 3 -> 3(1) + 0
  | Pos, Pos, Pos => Zero
  -- sum = 2 -> 3(1) - 1
  | Pos, Pos, Zero => Neg
  | Pos, Zero, Pos => Neg
  | Zero, Pos, Pos => Neg
  -- sum = 1 -> 3(0) + 1
  | Pos, Pos, Neg => Pos
  | Pos, Zero, Zero => Pos
  | Pos, Neg, Pos => Pos
  | Zero, Pos, Zero => Pos
  | Zero, Zero, Pos => Pos
  | Neg, Pos, Pos => Pos
  -- sum = 0 -> 3(0) + 0
  | Pos, Zero, Neg => Zero
  | Pos, Neg, Zero => Zero
  | Zero, Pos, Neg => Zero
  | Zero, Zero, Zero => Zero
  | Zero, Neg, Pos => Zero
  | Neg, Pos, Zero => Zero
  | Neg, Zero, Pos => Zero
  -- sum = -1 -> 3(0) - 1
  | Pos, Neg, Neg => Neg
  | Zero, Zero, Neg => Neg
  | Zero, Neg, Zero => Neg
  | Neg, Pos, Neg => Neg
  | Neg, Zero, Zero => Neg
  | Neg, Neg, Pos => Neg
  -- sum = -2 -> 3(-1) + 1
  | Zero, Neg, Neg => Pos
  | Neg, Zero, Neg => Pos
  | Neg, Neg, Zero => Pos
  -- sum = -3 -> 3(-1) + 0
  | Neg, Neg, Neg => Zero

/-- Full-adder carry trit -/
def fa_carry (a b c : Trit) : Trit :=
  match a, b, c with
  | Pos, Pos, Pos => Pos
  | Pos, Pos, Zero => Pos
  | Pos, Zero, Pos => Pos
  | Zero, Pos, Pos => Pos
  | Pos, Pos, Neg => Zero
  | Pos, Zero, Zero => Zero
  | Pos, Neg, Pos => Zero
  | Zero, Pos, Zero => Zero
  | Zero, Zero, Pos => Zero
  | Neg, Pos, Pos => Zero
  | Pos, Zero, Neg => Zero
  | Pos, Neg, Zero => Zero
  | Zero, Pos, Neg => Zero
  | Zero, Zero, Zero => Zero
  | Zero, Neg, Pos => Zero
  | Neg, Pos, Zero => Zero
  | Neg, Zero, Pos => Zero
  | Pos, Neg, Neg => Zero
  | Zero, Zero, Neg => Zero
  | Zero, Neg, Zero => Zero
  | Neg, Pos, Neg => Zero
  | Neg, Zero, Zero => Zero
  | Neg, Neg, Pos => Zero
  | Zero, Neg, Neg => Neg
  | Neg, Zero, Neg => Neg
  | Neg, Neg, Zero => Neg
  | Neg, Neg, Neg => Neg

/-- Theorem: Full-adder equation holds: a + b + c = 3 * carry + sum -/
theorem fa_correct (a b c : Trit) :
    val a + val b + val c = 3 * val (fa_carry a b c) + val (fa_sum a b c) := by
  cases a <;> cases b <;> cases c <;> rfl

end Trit

/-!
### Dual-Bitplane Representation (Binary Emulation on Modern Hardware)
On standard 64-bit CPU/GPU architectures, a trit is emulated via two bits:
  P (Positive bitmask)
  N (Negative bitmask)
Valid condition: P and N cannot be simultaneously 1 (i.e. P && N == 0).
-/

structure BitplaneTrit where
  p : Bool
  n : Bool
  disjoint : (p && n) = false
  deriving Repr

namespace BitplaneTrit

def val (b : BitplaneTrit) : Int :=
  (if b.p then 1 else 0) - (if b.n then 1 else 0)

/-- Isomorphism between Trit and valid BitplaneTrit -/
def toTrit (b : BitplaneTrit) : Trit :=
  match b.p, b.n with
  | true, false => Trit.Pos
  | false, true => Trit.Neg
  | false, false => Trit.Zero
  | true, true => Trit.Zero -- unreachable by disjointness

def fromTrit : Trit → BitplaneTrit
  | Trit.Pos  => ⟨true, false, rfl⟩
  | Trit.Zero => ⟨false, false, rfl⟩
  | Trit.Neg  => ⟨false, true, rfl⟩

theorem val_fromTrit (t : Trit) : val (fromTrit t) = Trit.val t := by
  cases t <;> rfl

/-- Bit-sliced multiplication:
    P_z = (P_x && P_y) || (N_x && N_y)
    N_z = (P_x && N_y) || (N_x && P_y)
-/
def mul (x y : BitplaneTrit) : BitplaneTrit :=
  let pz := (x.p && y.p) || (x.n && y.n)
  let nz := (x.p && y.n) || (x.n && y.p)
  have h_disjoint : (pz && nz) = false := by
    rcases x with ⟨xp, xn, hx⟩
    rcases y with ⟨yp, yn, hy⟩
    cases xp <;> cases xn <;> cases yp <;> cases yn <;> revert hx hy <;> decide
  ⟨pz, nz, h_disjoint⟩

/-- Theorem: Bit-sliced multiplication exactly matches algebraic integer multiplication -/
theorem mul_val_correct (x y : BitplaneTrit) :
    val (mul x y) = val x * val y := by
  rcases x with ⟨xp, xn, hx⟩
  rcases y with ⟨yp, yn, hy⟩
  cases xp <;> cases xn <;> cases yp <;> cases yn <;> revert hx hy <;> decide

end BitplaneTrit

/-!
### Balanced Ternary Radix-3 Expansions
Evaluating arbitrary-length trit sequences into Integers.
-/

def evalTritList : List Trit → Int
  | [] => 0
  | t :: ts => Trit.val t + 3 * evalTritList ts

theorem evalTritList_cons (t : Trit) (ts : List Trit) :
    evalTritList (t :: ts) = Trit.val t + 3 * evalTritList ts := rfl

/-- Negating a trit sequence negates the evaluated integer value -/
def negTritList : List Trit → List Trit
  | [] => []
  | t :: ts => Trit.neg t :: negTritList ts

theorem eval_negTritList (ts : List Trit) :
    evalTritList (negTritList ts) = - evalTritList ts := by
  induction ts with
  | nil => rfl
  | cons t ts ih =>
    simp [evalTritList, negTritList, Trit.val_neg]
    rw [ih]
    omega

/-!
### Balanced Ternary Euclidean Step (divMod3)
Decomposing any integer z into a quotient q and a unique remainder trit t ∈ {-1, 0, 1}
such that z = 3 * q + t.
-/

def divMod3 (z : Int) : Int × Trit :=
  let r := z % 3
  if r = 0 then
    (z / 3, Trit.Zero)
  else if r = 1 || r = -2 then
    ((z - 1) / 3, Trit.Pos)
  else
    ((z + 1) / 3, Trit.Neg)

theorem divMod3_correct (z : Int) :
    z = 3 * (divMod3 z).1 + Trit.val (divMod3 z).2 := by
  have h_rem : z % 3 = -2 ∨ z % 3 = -1 ∨ z % 3 = 0 ∨ z % 3 = 1 ∨ z % 3 = 2 := by
    omega
  dsimp [divMod3]
  rcases h_rem with h | h | h | h | h
  · -- r = -2
    have h1 : ¬(z % 3 = 0) := by omega
    have h2 : (z % 3 = 1 || z % 3 = -2) = true := by
      rw [h]; decide
    simp [h1, h2, Trit.val]
    omega
  · -- r = -1
    have h1 : ¬(z % 3 = 0) := by omega
    have h2 : (z % 3 = 1 || z % 3 = -2) = false := by
      rw [h]; decide
    simp [h1, h2, Trit.val]
    omega
  · -- r = 0
    simp [h, Trit.val]
    omega
  · -- r = 1
    have h1 : ¬(z % 3 = 0) := by omega
    have h2 : (z % 3 = 1 || z % 3 = -2) = true := by
      rw [h]; decide
    simp [h1, h2, Trit.val]
    omega
  · -- r = 2
    have h1 : ¬(z % 3 = 0) := by omega
    have h2 : (z % 3 = 1 || z % 3 = -2) = false := by
      rw [h]; decide
    simp [h1, h2, Trit.val]
    omega

/-!
### Three-Valued Logic (Kleene / Łukasiewicz)
AND = min, OR = max, NOT = neg.
-/

namespace Logic3

def kleeneAnd (a b : Trit) : Trit :=
  match a, b with
  | Trit.Neg, _ => Trit.Neg
  | _, Trit.Neg => Trit.Neg
  | Trit.Zero, _ => Trit.Zero
  | _, Trit.Zero => Trit.Zero
  | Trit.Pos, Trit.Pos => Trit.Pos

def kleeneOr (a b : Trit) : Trit :=
  match a, b with
  | Trit.Pos, _ => Trit.Pos
  | _, Trit.Pos => Trit.Pos
  | Trit.Zero, _ => Trit.Zero
  | _, Trit.Zero => Trit.Zero
  | Trit.Neg, Trit.Neg => Trit.Neg

/-- De Morgan's Law 1: neg (and a b) = or (neg a) (neg b) -/
theorem deMorgan1 (a b : Trit) :
    Trit.neg (kleeneAnd a b) = kleeneOr (Trit.neg a) (Trit.neg b) := by
  cases a <;> cases b <;> rfl

/-- De Morgan's Law 2: neg (or a b) = and (neg a) (neg b) -/
theorem deMorgan2 (a b : Trit) :
    Trit.neg (kleeneOr a b) = kleeneAnd (Trit.neg a) (Trit.neg b) := by
  cases a <;> cases b <;> rfl

end Logic3

/-!
### Vector Dot Product & Bit-Sliced Popcount Identity
Proving that the algebraic inner product of two ternary vectors
is equal to popcount(P_dot) - popcount(N_dot).
-/

def popcount : List Bool → Int
  | [] => 0
  | true :: bs => 1 + popcount bs
  | false :: bs => popcount bs

def bitplaneDot : List BitplaneTrit → List BitplaneTrit → Int
  | [], _ => 0
  | _, [] => 0
  | x :: xs, y :: ys => BitplaneTrit.val (BitplaneTrit.mul x y) + bitplaneDot xs ys

def positiveBits : List BitplaneTrit → List BitplaneTrit → List Bool
  | [], _ => []
  | _, [] => []
  | x :: xs, y :: ys => (BitplaneTrit.mul x y).p :: positiveBits xs ys

def negativeBits : List BitplaneTrit → List BitplaneTrit → List Bool
  | [], _ => []
  | _, [] => []
  | x :: xs, y :: ys => (BitplaneTrit.mul x y).n :: negativeBits xs ys

/-- Theorem: Vector dot product on modern hardware is EXACTLY the difference between two popcounts!
    dot(u, v) = popcount(P_dot) - popcount(N_dot) -/
theorem dot_eq_popcount_diff (xs ys : List BitplaneTrit) :
    bitplaneDot xs ys = popcount (positiveBits xs ys) - popcount (negativeBits xs ys) := by
  induction xs generalizing ys with
  | nil =>
    cases ys <;> rfl
  | cons x xs ih =>
    cases ys with
    | nil => rfl
    | cons y ys =>
      have ih_spec := ih ys
      have hz_val : BitplaneTrit.val (BitplaneTrit.mul x y) =
        (if (BitplaneTrit.mul x y).p then 1 else 0) - (if (BitplaneTrit.mul x y).n then 1 else 0) := rfl
      have hp : (BitplaneTrit.mul x y).p = true ∨ (BitplaneTrit.mul x y).p = false := by cases (BitplaneTrit.mul x y).p <;> decide
      have hn : (BitplaneTrit.mul x y).n = true ∨ (BitplaneTrit.mul x y).n = false := by cases (BitplaneTrit.mul x y).n <;> decide
      rcases hp with hp | hp <;> rcases hn with hn | hn
      · have hdis := (BitplaneTrit.mul x y).disjoint
        revert hdis; simp [hp, hn]
      · simp [bitplaneDot, positiveBits, negativeBits, hp, hn, popcount, hz_val, ih_spec]
        omega
      · simp [bitplaneDot, positiveBits, negativeBits, hp, hn, popcount, hz_val, ih_spec]
        omega
      · simp [bitplaneDot, positiveBits, negativeBits, hp, hn, popcount, hz_val, ih_spec]
