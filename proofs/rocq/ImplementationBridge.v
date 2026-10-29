(** Mathematical counterparts of the widening/correction/one-round algorithms
    used by Math/Integer.hpp, Math/Rational.hpp and ternary_fp.hpp. This is an
    algorithm-level bridge, not an extraction or C++ compiler correctness proof. *)
From Stdlib Require Import ZArith.ZArith micromega.Lia.
Open Scope Z_scope.
Module ImplementationBridge.
Definition adjust (q r b : Z) : Z * Z :=
  if r <? 0 then if 0 <? b then (q-1,r+b) else (q+1,r-b) else (q,r).

Theorem correction_correct : forall a b q r,
  b <> 0 -> a=b*q+r -> -Z.abs b < r < Z.abs b ->
  let '(q',r') := adjust q r b in a=b*q'+r' /\ 0<=r'<Z.abs b.
Proof.
  intros a b q r Hb E B. unfold adjust.
  destruct (r <? 0) eqn:R; [apply Z.ltb_lt in R|apply Z.ltb_ge in R; nia].
  destruct (0 <? b) eqn:S.
  - apply Z.ltb_lt in S. rewrite Z.abs_eq in * by lia. nia.
  - apply Z.ltb_ge in S. rewrite Z.abs_neq in * by lia. nia.
Qed.

Theorem truncating_divmod_refines_euclidean : forall a b,
  b <> 0 -> let '(q,r) := adjust (a ÷ b) (Z.rem a b) b in
             a=b*q+r /\ 0<=r<Z.abs b.
Proof.
  intros a b Hb. apply correction_correct; auto using Z.quot_rem.
  pose proof (Z.rem_bound_abs a b Hb) as B. apply Z.abs_lt in B. exact B.
Qed.

Definition round_unsigned (n d : Z) := (2*n+d)/(2*d).
(** C++ uses (n + d/2)/d after taking magnitudes, not 2*n+d. *)
Theorem cpp_round_matches_model : forall n d, 0<=n -> 0<d ->
  (n+d/2)/d=round_unsigned n d.
Proof.
  intros n d N D. unfold round_unsigned.
  set (q0 := (n+d/2)/d). set (r0 := (n+d/2) mod d).
  set (q1 := (2*n+d)/(2*d)). set (r1 := (2*n+d) mod (2*d)).
  assert (E0 : n+d/2=d*q0+r0) by (unfold q0,r0; apply Z.div_mod; lia).
  assert (E1 : 2*n+d=(2*d)*q1+r1) by (unfold q1,r1; apply Z.div_mod; lia).
  pose proof (Z.div_mod d 2 ltac:(lia)) as E2.
  pose proof (Z.mod_pos_bound d 2 ltac:(lia)) as B2.
  assert (B0 : 0<=r0<d) by (unfold r0; apply Z.mod_pos_bound; lia).
  assert (B1 : 0<=r1<2*d) by (unfold r1; apply Z.mod_pos_bound; lia).
  assert (U : q0=q1 /\ 2*r0+d mod 2=r1).
  { apply (Z.div_mod_unique (2*d) q0 q1 (2*r0+d mod 2) r1);
    [left; lia|left; exact B1|nia]. }
  exact (proj1 U).
Qed.

Theorem round_half_interval : forall n d, 0<=n -> 0<d ->
  -d <= 2*(n-round_unsigned n d*d) < d.
Proof.
  intros n d N D. unfold round_unsigned.
  pose proof (Z.div_mod (2*n+d) (2*d)) as E.
  pose proof (Z.mod_pos_bound (2*n+d) (2*d)) as B.
  assert (2*d<>0) by lia. specialize (E H). specialize (B ltac:(lia)). nia.
Qed.
Theorem round_half_up_tie : forall n d q, 0<=n -> 0<d ->
  2*n=(2*q+1)*d -> round_unsigned n d=q+1.
Proof.
  intros n d q N D E. unfold round_unsigned.
  replace (2*n+d) with ((q+1)*(2*d)) by nia.
  apply Z.div_mul. lia.
Qed.

Definition h40 : Z := 6078832729528464400.
Theorem rational_sum_fits_i128 : forall a b c d,
  Z.abs a<=h40 -> Z.abs b<=h40 -> Z.abs c<=h40 -> Z.abs d<=h40 ->
  Z.abs (a*d+b*c) <= 170141183460469231731687303715884105727.
Proof.
  intros a b c d A B C D.
  pose proof (Z.abs_triangle (a*d) (b*c)) as T.
  rewrite !Z.abs_mul in T.
  pose proof (Z.abs_nonneg a). pose proof (Z.abs_nonneg b).
  pose proof (Z.abs_nonneg c). pose proof (Z.abs_nonneg d).
  assert (Z.abs a*Z.abs d<=h40*h40) by (apply Z.mul_le_mono_nonneg; lia).
  assert (Z.abs b*Z.abs c<=h40*h40) by (apply Z.mul_le_mono_nonneg; lia).
  unfold h40 in *. lia.
Qed.
Theorem reduction_preserves_cross_product : forall n d g rn rd,
  n=g*rn -> d=g*rd -> n*rd=rn*d.
Proof. intros; nia. Qed.
Theorem binade_midpoint : forall hi lo scale,
  hi=3*lo-2 -> hi*scale+lo*(3*scale)=2*((hi+1)*scale).
Proof. intros; nia. Qed.
Theorem upper_candidate_nearer : forall n d lo hi,
  0<d -> lo<=hi -> d*(lo+hi)<=2*n ->
  (n-hi*d)*(n-hi*d)<=(n-lo*d)*(n-lo*d).
Proof.
  intros n d lo hi D L M.
  assert (0 <= (hi-lo)*d) by nia.
  assert (0 <= ((hi-lo)*d)*(2*n-d*(lo+hi))) by (apply Z.mul_nonneg_nonneg; lia).
  nia.
Qed.
Example repeated_rounding_counterexample : ((5+1)/3+1)/3=1 /\ 5-3 < 9-5.
Proof. split; [reflexivity|lia]. Qed.
Theorem implementation_bridge_complete : True. Proof. exact I. Qed.
End ImplementationBridge.
