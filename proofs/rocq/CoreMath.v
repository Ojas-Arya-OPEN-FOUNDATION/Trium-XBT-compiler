(** Hardware-independent integer mathematics. No floating-point surrogate model.
    The C++ implementation is tested against these contracts; this file does not
    prove extraction or correctness of the entire compiler. *)
From Stdlib Require Import ZArith.ZArith Lists.List Bool.Bool micromega.Lia.
Import ListNotations.
Open Scope Z_scope.
Module TriumCore.

Inductive trit := Neg | Zero | Pos.
Definition value (t : trit) : Z :=
  match t with Neg => -1 | Zero => 0 | Pos => 1 end.
Definition negate (t : trit) : trit :=
  match t with Neg => Pos | Zero => Zero | Pos => Neg end.
Fixpoint decode (digits : list trit) : Z :=
  match digits with [] => 0 | d :: rest => value d + 3 * decode rest end.
Fixpoint capacity (width : nat) : Z :=
  match width with O => 0 | S rest => 3 * capacity rest + 1 end.
Fixpoint place (width : nat) : Z :=
  match width with O => 1 | S rest => 3 * place rest end.

Theorem capacity_formula : forall width,
  2 * capacity width + 1 = place width.
Proof.
  induction width; [reflexivity|].
  change (2 * (3 * capacity width + 1) + 1 = 3 * place width). nia.
Qed.
Theorem decode_bounds : forall digits,
  - capacity (length digits) <= decode digits <= capacity (length digits).
Proof.
  induction digits as [|d rest IH]; [cbn; lia|].
  change (- (3 * capacity (length rest) + 1) <= value d + 3 * decode rest <=
          3 * capacity (length rest) + 1).
  destruct d; unfold value; lia.
Qed.
Theorem negate_value : forall t, value (negate t) = - value t.
Proof. intros []; reflexivity. Qed.
Theorem negate_decode : forall digits,
  decode (map negate digits) = - decode digits.
Proof.
  induction digits as [|d rest IH]; [reflexivity|].
  change (value (negate d) + 3 * decode (map negate rest) = -(value d + 3 * decode rest)).
  rewrite negate_value, IH; lia.
Qed.
Example scalar_capacity : capacity 39 = 2026277576509488133.
Proof. reflexivity. Qed.

Definition checked (bound result : Z) : option Z :=
  if ((-bound <=? result) && (result <=? bound))%bool
  then Some result else None.
Definition add (h a b : Z) := checked h (a+b).
Definition sub (h a b : Z) := checked h (a-b).
Definition mul (h a b : Z) := checked h (a*b).

Theorem checked_exact : forall h x y,
  checked h x = Some y <-> y = x /\ -h <= x <= h.
Proof.
  intros h x y. unfold checked.
  destruct ((-h <=? x) && (x <=? h))%bool eqn:E.
  - apply Bool.andb_true_iff in E. destruct E as [L R].
    apply Z.leb_le in L. apply Z.leb_le in R.
    split; intro H; [inversion H; lia|destruct H; subst; reflexivity].
  - split; intro H; [discriminate|].
    destruct H as [_ [L R]].
    assert (T : ((-h <=? x) && (x <=? h))%bool = true).
    { apply Bool.andb_true_iff; split; apply Z.leb_le; assumption. }
    congruence.
Qed.
Theorem checked_rejects : forall h x,
  checked h x = None <-> ~ (-h <= x <= h).
Proof.
  intros h x. unfold checked.
  destruct ((-h <=? x) && (x <=? h))%bool eqn:E.
  - apply Bool.andb_true_iff in E. destruct E as [L R].
    apply Z.leb_le in L. apply Z.leb_le in R. split; [discriminate|tauto].
  - split; intro H; [|reflexivity]. intro B.
    assert (T : ((-h <=? x) && (x <=? h))%bool = true).
    { apply Bool.andb_true_iff; split; apply Z.leb_le; lia. }
    congruence.
Qed.
Theorem add_exact : forall h a b r,
  add h a b = Some r <-> r = a+b /\ -h <= a+b <= h.
Proof. intros; apply checked_exact. Qed.
Theorem sub_exact : forall h a b r,
  sub h a b = Some r <-> r = a-b /\ -h <= a-b <= h.
Proof. intros; apply checked_exact. Qed.
Theorem mul_exact : forall h a b r,
  mul h a b = Some r <-> r = a*b /\ -h <= a*b <= h.
Proof. intros; apply checked_exact. Qed.

(** Euclidean remainder is nonnegative for BOTH divisor signs. *)
Definition divmod (a b : Z) : option (Z * Z) :=
  if b =? 0 then None else
  let d := Z.abs b in
  Some ((if b <? 0 then -(a / d) else a / d), a mod d).
Theorem divmod_zero : forall a, divmod a 0 = None.
Proof. reflexivity. Qed.
Theorem divmod_correct : forall a b,
  b <> 0 -> exists q r,
  divmod a b = Some (q,r) /\ a = b*q+r /\ 0 <= r < Z.abs b.
Proof.
  intros a b Hb.
  assert (Hd : 0 < Z.abs b) by (apply Z.abs_pos; exact Hb).
  pose proof (Z.div_mod a (Z.abs b) ltac:(lia)) as Eq.
  pose proof (Z.mod_pos_bound a (Z.abs b) Hd) as Bound.
  unfold divmod. assert (E : (b =? 0) = false) by (apply Z.eqb_neq; exact Hb).
  rewrite E.
  destruct (b <? 0) eqn:Sign.
  - apply Z.ltb_lt in Sign. exists (-(a / Z.abs b)), (a mod Z.abs b).
    split; [reflexivity|]. split; [assert (Ab : Z.abs b = -b) by (apply Z.abs_neq; lia); nia|exact Bound].
  - apply Z.ltb_ge in Sign. exists (a / Z.abs b), (a mod Z.abs b).
    split; [reflexivity|]. split; [assert (Ab : Z.abs b = b) by (apply Z.abs_eq; lia); nia|exact Bound].
Qed.
Theorem divmod_unique : forall a b q r q' r',
  b <> 0 -> a=b*q+r -> a=b*q'+r' ->
  0 <= r < Z.abs b -> 0 <= r' < Z.abs b -> q=q' /\ r=r'.
Proof.
  intros a b q r q' r' Hb H H' B B'.
  destruct (Z_lt_ge_dec b 0).
  - rewrite Z.abs_neq in * by lia.
    assert (Q : -q = -q' /\ r = r').
    { apply (Z.div_mod_unique (-b) (-q) (-q') r r'); [left; exact B|left; exact B'|nia]. }
    lia.
  - rewrite Z.abs_eq in * by lia.
    apply (Z.div_mod_unique b q q' r r'); [left; exact B|left; exact B'|lia].
Qed.

Definition select {A : Type} (condition : Z) (p z n : unit -> A) : A :=
  if 0 <? condition then p tt else if condition <? 0 then n tt else z tt.
Theorem select_positive : forall A c (p z n : unit -> A),
  0<c -> select c p z n = p tt.
Proof. intros. unfold select. destruct (Z.ltb_spec 0 c); [reflexivity|lia]. Qed.
Theorem select_negative : forall A c (p z n : unit -> A),
  c<0 -> select c p z n = n tt.
Proof.
  intros. unfold select. destruct (Z.ltb_spec 0 c); [lia|].
  destruct (Z.ltb_spec c 0); [reflexivity|lia].
Qed.
Theorem select_zero : forall A (p z n : unit -> A), select 0 p z n = z tt.
Proof. reflexivity. Qed.

Definition bind (x : option Z) (f : Z -> option Z) :=
  match x with Some v => f v | None => None end.
(** Optimization warning: checked addition is NOT freely reassociative. *)
Example no_unconditional_reassociation :
  bind (add 13 13 1) (fun x => sub 13 x 1) <>
  bind (sub 13 1 1) (fun x => add 13 13 x).
Proof. vm_compute. discriminate. Qed.

Theorem core_math_complete : forall h x y,
  checked h x = Some y -> y=x /\ -h <= y <= h.
Proof. intros h x y H. apply checked_exact in H. destruct H; subst; auto. Qed.
Print Assumptions core_math_complete.
End TriumCore.
