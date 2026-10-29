(** * BalancedTernaryFoundations.v
    Formal Verification of Balanced Ternary Hardware Foundations & Symplectic Mechanics
    Target: Rocq / standard library ZArith + Bool + Lia.
    See test/check_rocq.py for actual engine-specific validation; source presence
    alone is not a claim of native Rocq 9 execution.
*)

From Stdlib Require Import Bool.Bool.
From Stdlib Require Import ZArith.ZArith.
From Stdlib Require Import micromega.Lia.

Open Scope Z_scope.

(** =========================================================================
    1. Algebraic Trit Type & Dual-Rail Binary Pair Isomorphism
    ========================================================================= *)

Inductive Trit : Type :=
  | TNeg  : Trit  (* -1 *)
  | TZero : Trit  (*  0 *)
  | TPos  : Trit. (* +1 *)

Definition trit_val (t : Trit) : Z :=
  match t with
  | TNeg  => -1
  | TZero =>  0
  | TPos  =>  1
  end.

(** Dual-Rail binary valuation: v(p, n) = (if p then 1 else 0) - (if n then 1 else 0) *)
Definition rail_val (p n : bool) : Z :=
  (if p then 1 else 0) - (if n then 1 else 0).

(** Disjointness invariant: p && n = false *)
Definition rail_valid (p n : bool) : Prop :=
  andb p n = false.

Definition rail_of_trit (t : Trit) : bool * bool :=
  match t with
  | TNeg  => (false, true)
  | TZero => (false, false)
  | TPos  => (true, false)
  end.

Definition trit_to_rail := rail_of_trit.

Definition trit_of_rail (p n : bool) : Trit :=
  match p, n with
  | true, false  => TPos
  | false, true  => TNeg
  | false, false => TZero
  | true, true   => TZero (* unreachable under rail_valid *)
  end.

Definition rail_to_trit := trit_of_rail.

(** Theorem 1a: rail_of_trit always produces a valid disjoint pair *)
Theorem rail_of_trit_wf : forall (t : Trit),
  let '(p, n) := rail_of_trit t in rail_valid p n.
Proof.
  intros []; reflexivity.
Qed.

Theorem trit_to_rail_valid : forall (t : Trit),
  let '(p, n) := trit_to_rail t in rail_valid p n.
Proof.
  exact rail_of_trit_wf.
Qed.

(** Theorem 1b: Isomorphism round-trip 1: trit_of_rail (rail_of_trit t) = t *)
Theorem trit_of_rail_of_trit : forall (t : Trit),
  let '(p, n) := rail_of_trit t in trit_of_rail p n = t.
Proof.
  intros []; reflexivity.
Qed.

Theorem trit_rail_roundtrip : forall (t : Trit),
  let '(p, n) := trit_to_rail t in rail_to_trit p n = t.
Proof.
  exact trit_of_rail_of_trit.
Qed.

(** Theorem 1c: Isomorphism round-trip 2: valid (p, n) round-trips through Trit *)
Theorem rail_of_trit_of_rail : forall (p n : bool),
  rail_valid p n ->
  rail_of_trit (trit_of_rail p n) = (p, n).
Proof.
  intros [] [] Hvalid; try reflexivity.
  discriminate Hvalid.
Qed.

Theorem rail_trit_roundtrip : forall (p n : bool),
  rail_valid p n ->
  trit_to_rail (rail_to_trit p n) = (p, n).
Proof.
  exact rail_of_trit_of_rail.
Qed.

(** Theorem 1d: Trit valuation bounds and dual-rail valuation preservation *)
Theorem trit_val_bounds : forall (t : Trit),
  -1 <= trit_val t <= 1.
Proof.
  intros []; simpl; lia.
Qed.

Theorem rail_val_isomorphic : forall (p n : bool),
  rail_valid p n ->
  rail_val p n = trit_val (trit_of_rail p n) /\
  (rail_val p n = -1 \/ rail_val p n = 0 \/ rail_val p n = 1).
Proof.
  intros [] [] Hvalid; try discriminate Hvalid; unfold rail_val, trit_of_rail, trit_val; simpl; lia.
Qed.

(** =========================================================================
    2. Zero-Gate Negation (Wire Swap)
    ========================================================================= *)

Definition rail_neg (r : bool * bool) : bool * bool :=
  let '(p, n) := r in (n, p).

Theorem rail_neg_wf : forall (p n : bool),
  rail_valid p n -> rail_valid n p.
Proof.
  intros [] [] Hvalid; try discriminate Hvalid; reflexivity.
Qed.

Theorem rail_neg_val : forall (p n : bool),
  rail_val n p = - rail_val p n.
Proof.
  intros [] []; unfold rail_val; simpl; lia.
Qed.

Theorem rail_neg_involutive : forall (r : bool * bool),
  rail_neg (rail_neg r) = r.
Proof.
  intros [p n]; reflexivity.
Qed.

Theorem zero_gate_negation : forall (p n : bool),
  rail_valid p n ->
  rail_valid n p /\ rail_val n p = - rail_val p n.
Proof.
  intros p n Hvalid.
  split; [apply rail_neg_wf; exact Hvalid | apply rail_neg_val].
Qed.

(** =========================================================================
    3. Dual-Rail Bitwise Multiplication
    ========================================================================= *)

Definition rail_mul_p (pa na pb nb : bool) : bool :=
  orb (andb pa pb) (andb na nb).

Definition rail_mul_n (pa na pb nb : bool) : bool :=
  orb (andb pa nb) (andb na pb).

Theorem rail_mul_wf : forall (pa na pb nb : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_valid (rail_mul_p pa na pb nb) (rail_mul_n pa na pb nb).
Proof.
  intros [] [] [] [] Ha Hb;
    try discriminate Ha; try discriminate Hb;
    reflexivity.
Qed.

Theorem rail_mul_correct : forall (pa na pb nb : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_val (rail_mul_p pa na pb nb) (rail_mul_n pa na pb nb) =
    rail_val pa na * rail_val pb nb.
Proof.
  intros [] [] [] [] Ha Hb;
    try discriminate Ha; try discriminate Hb;
    reflexivity.
Qed.

Theorem rail_mul_comm : forall (pa na pb nb : bool),
  (rail_mul_p pa na pb nb, rail_mul_n pa na pb nb) =
  (rail_mul_p pb nb pa na, rail_mul_n pb nb pa na).
Proof.
  intros [] [] [] []; reflexivity.
Qed.

Theorem rail_mul_assoc : forall (pa na pb nb pc nc : bool),
  let p_ab := rail_mul_p pa na pb nb in
  let n_ab := rail_mul_n pa na pb nb in
  let p_bc := rail_mul_p pb nb pc nc in
  let n_bc := rail_mul_n pb nb pc nc in
  (rail_mul_p p_ab n_ab pc nc, rail_mul_n p_ab n_ab pc nc) =
  (rail_mul_p pa na p_bc n_bc, rail_mul_n pa na p_bc n_bc).
Proof.
  intros [] [] [] [] [] []; reflexivity.
Qed.

Theorem dual_rail_mul_correct : forall (pa na pb nb : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_valid (rail_mul_p pa na pb nb) (rail_mul_n pa na pb nb) /\
  rail_val (rail_mul_p pa na pb nb) (rail_mul_n pa na pb nb) =
    rail_val pa na * rail_val pb nb.
Proof.
  intros pa na pb nb Ha Hb.
  split; [apply rail_mul_wf | apply rail_mul_correct]; assumption.
Qed.

(** =========================================================================
    4. 6-Gate Balanced Ternary Half-Adder (BTHA)
    ========================================================================= *)

Definition btha_cp (pa pb : bool) : bool := andb pa pb.
Definition btha_cn (na nb : bool) : bool := andb na nb.

Definition btha_sp (pa na pb nb : bool) : bool :=
  orb (andb (xorb pa pb) (negb (orb na nb))) (btha_cn na nb).

Definition btha_sn (pa na pb nb : bool) : bool :=
  orb (andb (xorb na nb) (negb (orb pa pb))) (btha_cp pa pb).

Theorem btha_sum_wf : forall (pa na pb nb : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_valid (btha_sp pa na pb nb) (btha_sn pa na pb nb).
Proof.
  intros [] [] [] [] Ha Hb;
    try discriminate Ha; try discriminate Hb;
    reflexivity.
Qed.

Theorem btha_carry_wf : forall (pa na pb nb : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_valid (btha_cp pa pb) (btha_cn na nb).
Proof.
  intros [] [] [] [] Ha Hb;
    try discriminate Ha; try discriminate Hb;
    reflexivity.
Qed.

(** Theorem 4: 6-gate BTHA preserves output disjointness and satisfies s + 3*c = a + b *)
Theorem btha_correct : forall (pa na pb nb : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  let sp := btha_sp pa na pb nb in
  let sn := btha_sn pa na pb nb in
  let cp := btha_cp pa pb in
  let cn := btha_cn na nb in
  rail_valid sp sn /\
  rail_valid cp cn /\
  rail_val sp sn + 3 * rail_val cp cn = rail_val pa na + rail_val pb nb.
Proof.
  intros [] [] [] [] Ha Hb;
    try discriminate Ha; try discriminate Hb;
    simpl; repeat split; reflexivity.
Qed.

(** =========================================================================
    5. BTFA (3-to-2 Compressor) & No-Overlap Carry Theorem
    ========================================================================= *)

Theorem btfa_no_overlap_carry : forall (pa na pb nb pd nd : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_valid pd nd ->
  let c1p := btha_cp pa pb in
  let c1n := btha_cn na nb in
  let s1p := btha_sp pa na pb nb in
  let s1n := btha_sn pa na pb nb in
  let c2p := btha_cp s1p pd in
  let c2n := btha_cn s1n nd in
  andb c1p c2p = false /\ andb c1n c2n = false.
Proof.
  intros [] [] [] [] [] [] Ha Hb Hd;
    try discriminate Ha; try discriminate Hd; try discriminate Hb;
    simpl; split; reflexivity.
Qed.

Theorem btfa_correct : forall (pa na pb nb pd nd : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_valid pd nd ->
  let c1p := btha_cp pa pb in
  let c1n := btha_cn na nb in
  let s1p := btha_sp pa na pb nb in
  let s1n := btha_sn pa na pb nb in
  let c2p := btha_cp s1p pd in
  let c2n := btha_cn s1n nd in
  let sp  := btha_sp s1p s1n pd nd in
  let sn  := btha_sn s1p s1n pd nd in
  let cp  := andb (orb c1p c2p) (negb (orb c1n c2n)) in
  let cn  := andb (orb c1n c2n) (negb (orb c1p c2p)) in
  rail_valid sp sn /\
  rail_valid cp cn /\
  rail_val sp sn + 3 * rail_val cp cn =
    rail_val pa na + rail_val pb nb + rail_val pd nd.
Proof.
  intros [] [] [] [] [] [] Ha Hb Hd;
    try discriminate Ha; try discriminate Hb; try discriminate Hd;
    simpl; repeat split; reflexivity.
Qed.

Theorem btfa_and_no_overlap_carry : forall (pa na pb nb pd nd : bool),
  rail_valid pa na ->
  rail_valid pb nb ->
  rail_valid pd nd ->
  let c1p := btha_cp pa pb in
  let c1n := btha_cn na nb in
  let s1p := btha_sp pa na pb nb in
  let s1n := btha_sn pa na pb nb in
  let c2p := btha_cp s1p pd in
  let c2n := btha_cn s1n nd in
  let sp  := btha_sp s1p s1n pd nd in
  let sn  := btha_sn s1p s1n pd nd in
  let cp  := andb (orb c1p c2p) (negb (orb c1n c2n)) in
  let cn  := andb (orb c1n c2n) (negb (orb c1p c2p)) in
  andb c1p c2p = false /\
  andb c1n c2n = false /\
  rail_valid sp sn /\
  rail_valid cp cn /\
  rail_val sp sn + 3 * rail_val cp cn =
    rail_val pa na + rail_val pb nb + rail_val pd nd.
Proof.
  intros [] [] [] [] [] [] Ha Hb Hd;
    try discriminate Ha; try discriminate Hb; try discriminate Hd;
    simpl; repeat split; reflexivity.
Qed.

(** =========================================================================
    6. Base-243 Packing Range & Symplectic Leapfrog / Verlet Reversibility
    ========================================================================= *)

Definition base243_pack5 (t0 t1 t2 t3 t4 : Trit) : Z :=
  (trit_val t0 + 1) +
  3 * (trit_val t1 + 1) +
  9 * (trit_val t2 + 1) +
  27 * (trit_val t3 + 1) +
  81 * (trit_val t4 + 1).

Theorem base243_pack5_bounds : forall (t0 t1 t2 t3 t4 : Trit),
  0 <= base243_pack5 t0 t1 t2 t3 t4 <= 242.
Proof.
  intros [] [] [] [] []; unfold base243_pack5; simpl; lia.
Qed.

Section SymplecticMechanics.

  (** Generalized 2-stage shear step (Kick-Drift) with arbitrary potentials f, g : Z -> Z *)
  Definition step_fwd_2stage (f g : Z -> Z) (st : Z * Z) : Z * Z :=
    let '(q, p) := st in
    let p' := p + f q in
    let q' := q + g p' in
    (q', p').

  Definition step_back_2stage (f g : Z -> Z) (st : Z * Z) : Z * Z :=
    let '(q', p') := st in
    let q := q' - g p' in
    let p := p' - f q in
    (q, p).

  (** Theorem 6a: 2-stage Symplectic Leapfrog is 100% bit-exactly time-reversible for ANY f, g *)
  Theorem symplectic_2stage_reversible : forall (f g : Z -> Z) (q p : Z),
    step_back_2stage f g (step_fwd_2stage f g (q, p)) = (q, p).
  Proof.
    intros f g q p.
    unfold step_fwd_2stage, step_back_2stage.
    f_equal; [lia |].
    replace (q + g (p + f q) - g (p + f q)) with q by lia.
    lia.
  Qed.

  Theorem symplectic_2stage_invertible : forall (f g : Z -> Z) (q p : Z),
    step_fwd_2stage f g (step_back_2stage f g (q, p)) = (q, p).
  Proof.
    intros f g q p.
    unfold step_fwd_2stage, step_back_2stage.
    f_equal; [| lia].
    replace (p - f (q - g p) + f (q - g p)) with p by lia.
    lia.
  Qed.

  (** 3-Stage Kick-Drift-Kick (Verlet) matching MultiWordSymplecticEngine in symplectic_ternary_engine.hpp *)
  Definition verlet_kdk_fwd (F : Z -> Z) (st : Z * Z) : Z * Z :=
    let '(q, p) := st in
    let p_half := p + F q in
    let q_next := q + p_half in
    let p_next := p_half + F q_next in
    (q_next, p_next).

  Definition verlet_kdk_back (F : Z -> Z) (st : Z * Z) : Z * Z :=
    let '(q_next, p_next) := st in
    let p_half := p_next - F q_next in
    let q_prev := q_next - p_half in
    let p_prev := p_half - F q_prev in
    (q_prev, p_prev).

  (** Theorem 6b: 3-stage Verlet KDK is 100% bit-exactly time-reversible for ANY force law F : Z -> Z *)
  Theorem verlet_kdk_reversible : forall (F : Z -> Z) (q p : Z),
    verlet_kdk_back F (verlet_kdk_fwd F (q, p)) = (q, p).
  Proof.
    intros F q p.
    unfold verlet_kdk_fwd, verlet_kdk_back.
    set (p_half := p + F q).
    set (q_next := q + p_half).
    replace (p_half + F q_next - F q_next) with p_half by lia.
    replace (q_next - p_half) with q by (unfold q_next; lia).
    f_equal; unfold p_half; lia.
  Qed.

  Theorem verlet_kdk_invertible : forall (F : Z -> Z) (q p : Z),
    verlet_kdk_fwd F (verlet_kdk_back F (q, p)) = (q, p).
  Proof.
    intros F q p.
    unfold verlet_kdk_fwd, verlet_kdk_back.
    set (p_half := p - F q).
    set (q_prev := q - p_half).
    replace (p_half - F q_prev + F q_prev) with p_half by lia.
    replace (q_prev + p_half) with q by (unfold q_prev; lia).
    f_equal; unfold p_half; lia.
  Qed.

End SymplecticMechanics.
