(** Arithmetic obligations for core lowering/optimization. These lemmas justify
    the stated algorithms; they are not a proof of the C++/LLVM translator. *)
From Stdlib Require Import ZArith.ZArith micromega.Lia.
Open Scope Z_scope.
Module CoreLowering.
Theorem gpu_mul_precheck : forall a b h, 0<a -> 0<=b -> 0<=h ->
  (b<=h/a <-> a*b<=h).
Proof.
  intros a b h A B H.
  pose proof (Z.div_mod h a ltac:(lia)) as E.
  pose proof (Z.mod_pos_bound h a A) as R.
  split; intros; nia.
Qed.
Theorem add_interval : forall a b c d x y,
  a<=x<=b -> c<=y<=d -> a+c<=x+y<=b+d.
Proof. intros; lia. Qed.
Theorem sub_interval : forall a b c d x y,
  a<=x<=b -> c<=y<=d -> a-d<=x-y<=b-c.
Proof. intros; lia. Qed.
Theorem scale_interval : forall a b x k, a<=x<=b ->
  Z.min (a*k) (b*k)<=x*k<=Z.max (a*k) (b*k).
Proof.
  intros a b x k X.
  destruct (Z_le_gt_dec 0 k).
  - assert (0<=(x-a)*k) by (apply Z.mul_nonneg_nonneg; lia).
    assert (0<=(b-x)*k) by (apply Z.mul_nonneg_nonneg; lia).
    pose proof (Z.le_min_l (a*k) (b*k)). pose proof (Z.le_max_r (a*k) (b*k)). nia.
  - assert (0<=(x-a)*(-k)) by (apply Z.mul_nonneg_nonneg; lia).
    assert (0<=(b-x)*(-k)) by (apply Z.mul_nonneg_nonneg; lia).
    pose proof (Z.le_min_r (a*k) (b*k)). pose proof (Z.le_max_l (a*k) (b*k)). nia.
Qed.
Theorem mul_interval : forall a b c d x y,
  a<=x<=b -> c<=y<=d ->
  Z.min (Z.min (a*c) (a*d)) (Z.min (b*c) (b*d)) <= x*y <=
  Z.max (Z.max (a*c) (a*d)) (Z.max (b*c) (b*d)).
Proof.
  intros a b c d x y X Y.
  pose proof (scale_interval a b x y X) as XY.
  pose proof (scale_interval c d y a Y) as AY.
  pose proof (scale_interval c d y b Y) as BY.
  replace (c*a) with (a*c) in AY by ring.
  replace (d*a) with (a*d) in AY by ring.
  replace (y*a) with (a*y) in AY by ring.
  replace (c*b) with (b*c) in BY by ring.
  replace (d*b) with (b*d) in BY by ring.
  replace (y*b) with (b*y) in BY by ring.
  pose proof (Z.le_min_l (Z.min (a*c) (a*d)) (Z.min (b*c) (b*d))).
  pose proof (Z.le_min_r (Z.min (a*c) (a*d)) (Z.min (b*c) (b*d))).
  pose proof (Z.le_max_l (Z.max (a*c) (a*d)) (Z.max (b*c) (b*d))).
  pose proof (Z.le_max_r (Z.max (a*c) (a*d)) (Z.max (b*c) (b*d))).
  pose proof (Z.min_spec (a*y) (b*y)).
  pose proof (Z.max_spec (a*y) (b*y)).
  destruct H3 as [[? ?]|[? ?]]; destruct H4 as [[? ?]|[? ?]]; lia.
Qed.
Example int39_add_fits_i64 : 2*2026277576509488133 < 9223372036854775808.
Proof. reflexivity. Qed.
Example int39_mul_fits_i128 :
  2026277576509488133*2026277576509488133 < 170141183460469231731687303715884105728.
Proof. reflexivity. Qed.
Inductive outcome := Value : Z -> outcome | Failure : outcome.
Definition multiply (a b:outcome) : outcome :=
  match a,b with Value x, Value y => Value (x*y) | _,_ => Failure end.
Theorem zero_does_not_erase_failure : multiply (Value 0) Failure=Failure.
Proof. reflexivity. Qed.
Theorem core_lowering_complete : True. Proof. exact I. Qed.
End CoreLowering.
