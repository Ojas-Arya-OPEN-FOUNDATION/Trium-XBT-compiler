/-
  TernaryAlgebraProofs.lean
  Formal Verification Suite for Balanced Ternary Dense Algebra, Logic, and Dot Products
  Author: Antigravity Prime Verification Engine
  Verified via Lean 4 (v4.34.0, zero external dependencies)
-/

-- ==============================================================================
-- 1. Primitive Trit Definition and Kleene 3-Valued Logic
-- ==============================================================================

inductive Trit : Type where
  | Neg  : Trit  -- -1
  | Zero : Trit  --  0
  | Pos  : Trit  -- +1
  deriving DecidableEq, Repr

namespace Trit

def val : Trit → Int
  | Neg  => -1
  | Zero =>  0
  | Pos  =>  1

theorem val_bounds (t : Trit) : -1 ≤ val t ∧ val t ≤ 1 := by
  cases t <;> decide

-- Kleene NOT (involution)
def kleeneNot : Trit → Trit
  | Neg  => Pos
  | Zero => Zero
  | Pos  => Neg

theorem val_kleeneNot (t : Trit) : val (kleeneNot t) = - val t := by
  cases t <;> rfl

theorem kleeneNot_involutive (t : Trit) : kleeneNot (kleeneNot t) = t := by
  cases t <;> rfl

-- Kleene AND (min)
def kleeneAnd : Trit → Trit → Trit
  | Neg, _ => Neg
  | _, Neg => Neg
  | Zero, _ => Zero
  | _, Zero => Zero
  | Pos, Pos => Pos

theorem val_kleeneAnd (a b : Trit) : val (kleeneAnd a b) = min (val a) (val b) := by
  cases a <;> cases b <;> rfl

theorem kleeneAnd_comm (a b : Trit) : kleeneAnd a b = kleeneAnd b a := by
  cases a <;> cases b <;> rfl

theorem kleeneAnd_assoc (a b c : Trit) :
    kleeneAnd (kleeneAnd a b) c = kleeneAnd a (kleeneAnd b c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem kleeneAnd_idem (a : Trit) : kleeneAnd a a = a := by
  cases a <;> rfl

-- Kleene OR (max)
def kleeneOr : Trit → Trit → Trit
  | Pos, _ => Pos
  | _, Pos => Pos
  | Zero, _ => Zero
  | _, Zero => Zero
  | Neg, Neg => Neg

theorem val_kleeneOr (a b : Trit) : val (kleeneOr a b) = max (val a) (val b) := by
  cases a <;> cases b <;> rfl

theorem kleeneOr_comm (a b : Trit) : kleeneOr a b = kleeneOr b a := by
  cases a <;> cases b <;> rfl

theorem kleeneOr_assoc (a b c : Trit) :
    kleeneOr (kleeneOr a b) c = kleeneOr a (kleeneOr b c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem kleeneOr_idem (a : Trit) : kleeneOr a a = a := by
  cases a <;> rfl

-- De Morgan's Laws
theorem de_morgan_and (a b : Trit) :
    kleeneNot (kleeneAnd a b) = kleeneOr (kleeneNot a) (kleeneNot b) := by
  cases a <;> cases b <;> rfl

theorem de_morgan_or (a b : Trit) :
    kleeneNot (kleeneOr a b) = kleeneAnd (kleeneNot a) (kleeneNot b) := by
  cases a <;> cases b <;> rfl

-- Kleene XOR (multiplication of signs)
def kleeneXor (a b : Trit) : Trit :=
  match a, b with
  | Pos, Pos => Pos
  | Neg, Neg => Pos
  | Pos, Neg => Neg
  | Neg, Pos => Neg
  | _, _     => Zero

theorem val_kleeneXor (a b : Trit) : val (kleeneXor a b) = val a * val b := by
  cases a <;> cases b <;> rfl

-- Spaceship Comparison (<=>)
def ternaryCmp (a b : Trit) : Trit :=
  match a, b with
  | Pos, Zero => Pos
  | Pos, Neg  => Pos
  | Zero, Neg => Pos
  | Neg, Pos  => Neg
  | Neg, Zero => Neg
  | Zero, Pos => Neg
  | _, _      => Zero

theorem ternaryCmp_sound (a b : Trit) :
    val (ternaryCmp a b) = (if val a > val b then 1 else (if val a < val b then -1 else 0)) := by
  cases a <;> cases b <;> rfl

end Trit

-- ==============================================================================
-- 2. Branchless 3-Way Conditional Selection
-- ==============================================================================

-- High-level 3-way conditional selection
def ternarySelect {α : Type} (c : Trit) (pos_val zero_val neg_val : α) : α :=
  match c with
  | Trit.Pos  => pos_val
  | Trit.Zero => zero_val
  | Trit.Neg  => neg_val

theorem ternarySelect_sound {α : Type} (c : Trit) (x y z : α) :
    ternarySelect c x y z = (if Trit.val c = 1 then x else (if Trit.val c = 0 then y else z)) := by
  cases c <;> rfl

-- Hardware Dual-Rail Bitplane Representation
structure DualRailTrit where
  p : Bool
  n : Bool
  disjoint : (p && n) = false
  deriving Repr, DecidableEq

namespace DualRailTrit

def val (d : DualRailTrit) : Int :=
  (if d.p then 1 else 0) - (if d.n then 1 else 0)

-- Branchless bit-level multiplexer equations
def branchlessSelect (cond : DualRailTrit) (x y z : DualRailTrit) : DualRailTrit :=
  let out_p := (cond.p && x.p) || (!cond.p && !cond.n && y.p) || (cond.n && z.p)
  let out_n := (cond.p && x.n) || (!cond.p && !cond.n && y.n) || (cond.n && z.n)
  have h_disj : (out_p && out_n) = false := by
    rcases cond with ⟨cp, cn, hc⟩
    rcases x with ⟨xp, xn, hx⟩
    rcases y with ⟨yp, yn, hy⟩
    rcases z with ⟨zp, zn, hz⟩
    cases cp <;> cases cn <;> cases xp <;> cases xn <;>
    cases yp <;> cases yn <;> cases zp <;> cases zn <;>
    revert hc hx hy hz <;> decide
  ⟨out_p, out_n, h_disj⟩

theorem branchlessSelect_sound (cond : DualRailTrit) (x y z : DualRailTrit) :
    val (branchlessSelect cond x y z) =
      (if val cond = 1 then val x else (if val cond = 0 then val y else val z)) := by
  rcases cond with ⟨cp, cn, hc⟩
  rcases x with ⟨xp, xn, hx⟩
  rcases y with ⟨yp, yn, hy⟩
  rcases z with ⟨zp, zn, hz⟩
  cases cp <;> cases cn <;> cases xp <;> cases xn <;>
  cases yp <;> cases yn <;> cases zp <;> cases zn <;>
  revert hc hx hy hz <;> decide

end DualRailTrit

-- ==============================================================================
-- 3. Carry-Save Accumulator (BT-CSA) & 3-to-2 Compressor Soundness
-- ==============================================================================

namespace BT_CSA

-- Balanced Ternary Half-Adder
def btha (a b : Trit) : Trit × Trit :=
  match a, b with
  | Trit.Pos, Trit.Pos => (Trit.Pos, Trit.Neg)   -- 1 + 1 = 2 = 3*(1) - 1
  | Trit.Neg, Trit.Neg => (Trit.Neg, Trit.Pos)   -- -1 - 1 = -2 = 3*(-1) + 1
  | Trit.Pos, Trit.Zero => (Trit.Zero, Trit.Pos)
  | Trit.Zero, Trit.Pos => (Trit.Zero, Trit.Pos)
  | Trit.Neg, Trit.Zero => (Trit.Zero, Trit.Neg)
  | Trit.Zero, Trit.Neg => (Trit.Zero, Trit.Neg)
  | _, _                => (Trit.Zero, Trit.Zero)

theorem btha_sound (a b : Trit) :
    3 * Trit.val (btha a b).1 + Trit.val (btha a b).2 = Trit.val a + Trit.val b := by
  cases a <;> cases b <;> rfl

-- Balanced Ternary Full-Adder (BTFA / 3-to-2 Compressor)
def btfa (a b d : Trit) : Trit × Trit :=
  let (c1, s1) := btha a b
  let (c2, s)  := btha s1 d
  -- Merge carries (No overlap theorem: c1 and c2 never have same sign)
  let carry := match c1, c2 with
    | Trit.Pos, Trit.Neg => Trit.Zero
    | Trit.Neg, Trit.Pos => Trit.Zero
    | Trit.Pos, _        => Trit.Pos
    | _, Trit.Pos        => Trit.Pos
    | Trit.Neg, _        => Trit.Neg
    | _, Trit.Neg        => Trit.Neg
    | _, _               => Trit.Zero
  (carry, s)

theorem btfa_sound (a b d : Trit) :
    3 * Trit.val (btfa a b d).1 + Trit.val (btfa a b d).2 =
      Trit.val a + Trit.val b + Trit.val d := by
  cases a <;> cases b <;> cases d <;> rfl

theorem btfa_carry_bound (a b d : Trit) :
    -1 ≤ Trit.val (btfa a b d).1 ∧ Trit.val (btfa a b d).1 ≤ 1 := by
  cases a <;> cases b <;> cases d <;> decide

theorem btfa_sum_bound (a b d : Trit) :
    -1 ≤ Trit.val (btfa a b d).2 ∧ Trit.val (btfa a b d).2 ≤ 1 := by
  cases a <;> cases b <;> cases d <;> decide

/-- No-Overlap Carry Theorem: The two internal half-adder carries c1 and c2 in BTFA
    never have the same non-zero sign (c1 * c2 ≠ 1), so their sum c1 + c2 is always in {-1, 0, +1}. -/
theorem btfa_no_overlap_carry (a b d : Trit) :
    -1 ≤ Trit.val (btha a b).1 + Trit.val (btha (btha a b).2 d).1 ∧
    Trit.val (btha a b).1 + Trit.val (btha (btha a b).2 d).1 ≤ 1 := by
  cases a <;> cases b <;> cases d <;> decide

/-- 4-to-2 Balanced Ternary Carry-Save Compressor:
    Compresses 4 input trits (a, b, c, d) into a sum trit s2 and two carry trits (c1, c2). -/
def bt_compress_4to2 (a b c d : Trit) : Trit × Trit × Trit :=
  let (c1, s1) := btfa a b c
  let (c2, s2) := btha s1 d
  (c1, c2, s2)

theorem bt_compress_4to2_sound (a b c d : Trit) :
    Trit.val (bt_compress_4to2 a b c d).2.2 +
    3 * (Trit.val (bt_compress_4to2 a b c d).1 + Trit.val (bt_compress_4to2 a b c d).2.1) =
      Trit.val a + Trit.val b + Trit.val c + Trit.val d := by
  cases a <;> cases b <;> cases c <;> cases d <;> rfl

end BT_CSA

-- ==============================================================================
-- 4. Planar POPCNT Dot-Product Identity Theorem
-- ==============================================================================

def popcount : List Bool → Int
  | [] => 0
  | true :: bs => 1 + popcount bs
  | false :: bs => popcount bs

def bitwiseAnd : List Bool → List Bool → List Bool
  | [], _ => []
  | _, [] => []
  | x :: xs, y :: ys => (x && y) :: bitwiseAnd xs ys

def bitwiseOr : List Bool → List Bool → List Bool
  | [], _ => []
  | _, [] => []
  | x :: xs, y :: ys => (x || y) :: bitwiseOr xs ys

def tritVal (p n : Bool) : Int :=
  (if p then 1 else 0) - (if n then 1 else 0)

def planarDot : List Bool → List Bool → List Bool → List Bool → Int
  | pu :: pus, nu :: nus, pv :: pvs, nv :: nvs =>
    (tritVal pu nu * tritVal pv nv) + planarDot pus nus pvs nvs
  | _, _, _, _ => 0

inductive ValidBitplane : List Bool → List Bool → Prop where
  | nil : ValidBitplane [] []
  | cons (p n : Bool) (ps ns : List Bool)
      (hdis : (p && n) = false) (hrest : ValidBitplane ps ns) :
      ValidBitplane (p :: ps) (n :: ns)

/--
  Master Planar POPCNT Dot-Product Theorem:
  For any balanced ternary vectors u and v represented as disjoint dual bitplanes
  (pus, nus) and (pvs, nvs):
    u · v = popcount((pus ∧ pvs) ∨ (nus ∧ nvs)) - popcount((pus ∧ nvs) ∨ (nus ∧ pvs))
  Proved by structural induction on bitplane lists with zero linear arithmetic gaps.
-/
theorem planar_popcount_dot_identity
    (pus nus pvs nvs : List Bool)
    (hu : ValidBitplane pus nus)
    (hv : ValidBitplane pvs nvs)
    (hlen : pus.length = pvs.length) :
    planarDot pus nus pvs nvs =
      popcount (bitwiseOr (bitwiseAnd pus pvs) (bitwiseAnd nus nvs)) -
      popcount (bitwiseOr (bitwiseAnd pus nvs) (bitwiseAnd nus pvs)) := by
  induction hu generalizing pvs nvs with
  | nil =>
    cases hv with
    | nil => rfl
    | cons pv nv pvs nvs hdv hrv => contradiction
  | cons pu nu pus nus hdu hru ih =>
    cases hv with
    | nil => contradiction
    | cons pv nv pvs nvs hdv hrv =>
      simp only [List.length_cons, Nat.succ.injEq] at hlen
      have ih_app := ih pvs nvs hrv hlen
      dsimp [planarDot, bitwiseAnd, bitwiseOr]
      rw [ih_app]
      have h_cases : tritVal pu nu * tritVal pv nv =
                     (if (pu && pv || nu && nv) then 1 else 0) -
                     (if (pu && nv || nu && pv) then 1 else 0) := by
        dsimp [tritVal]
        cases pu <;> cases nu <;> cases pv <;> cases nv <;>
        revert hdu hdv <;> decide
      rw [h_cases]
      cases h1 : (pu && pv || nu && nv) <;> cases h2 : (pu && nv || nu && pv)
      · dsimp [popcount]
        omega
      · dsimp [popcount]
        omega
      · dsimp [popcount]
        omega
      · have hdisj : ((pu && pv || nu && nv) && (pu && nv || nu && pv)) = false := by
          cases pu <;> cases nu <;> cases pv <;> cases nv <;>
          revert hdu hdv <;> decide
        rw [h1, h2] at hdisj
        contradiction
