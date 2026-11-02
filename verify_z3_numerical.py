#!/usr/bin/env python3
"""Numerical-v1 guard/capacity/root models, not full C++ or libm proofs."""
import sys
try:
 from z3 import *
except ImportError:
 print('SKIP install z3-solver');sys.exit(77)
checks=0
def prove(assumptions,bad):
 global checks
 s=Solver();s.set(timeout=30000);s.add(*assumptions,bad);result=s.check();assert result==unsat,result;checks+=1
# Positive powered-midpoint order is equivalent to nearest-root selection.
a,b,t=Reals('lower upper root')
for degree in (2,3):
 prove([0<=a,a<b,0<=t,t<(a+b)/2],t**degree>=((a+b)/2)**degree)
 prove([0<=a,a<b,t>=(a+b)/2],t**degree<((a+b)/2)**degree)
# Dominant hypot tail: y^2 < a*u => sqrt(a^2+y^2) < a+u/2.
a,y,u=Reals('a y unit')
prove([a>0,u>0,y>=0,y*y<a*u],a*a+y*y>=(a+u/2)**2)
# The implementation's delta>W guard makes that strict inequality valid.
for w in range(1,41):
 L,H=(3**(w-1)+1)//2,(3**w-1)//2
 prove([],IntVal(H*H)>=IntVal(L*3**(2*(w+1))))
# Worst-case coefficient/alignment/work bounds remain far inside 2048-bit scratch.
H=(3**40-1)//2
for bound in (4096*H**2*3**256,17*H**17*3**256,H**16):
 # Factor allows normalization and midpoint comparison headroom.
 prove([],IntVal(bound*16*(H+2)**3)>=IntVal(2**2048))
# Per-limb division constructs a valid u128 dividend for every nonzero u64 divisor.
r,d,word=Ints('remainder divisor word')
prove([1<=d,d<2**64,0<=r,r<d,0<=word,word<2**64],r*2**64+word>=2**128)
# Bounded C array addressing and checked window subtraction in signed i64.
i,n,low,high=Ints('index count low high')
prove([0<=i,i<n,n<=4096],24*i+24>24*4096)
prove([-2**36<=low,low<=high,high<=2**36],Or(high-low<0,high-low>=2**63))
# Obligation A: Chunked power-of-3 decomposition for 0 <= delta <= 256.
delta_sym=Int('delta_sym')
c0=If(delta_sym>40,40,delta_sym)
r1=delta_sym-c0
c1=If(r1>40,40,r1)
r2=r1-c1
c2=If(r2>40,40,r2)
r3=r2-c2
c3=If(r3>40,40,r3)
r4=r3-c3
c4=If(r4>40,40,r4)
r5=r4-c4
c5=If(r5>40,40,r5)
r6=r5-c5
c6=If(r6>40,40,r6)
r7=r6-c6
prove([0<=delta_sym,delta_sym<=256],Or(r7!=0,c0+c1+c2+c3+c4+c5+c6!=delta_sym))
for delta in range(257):
 rem=delta
 chunks=[]
 while rem>0:
  step=40 if rem>40 else rem
  chunks.append(step)
  rem-=step
 sum_chunks=sum(chunks)
 prod_chunks=1
 for c in chunks:
  prod_chunks*=(3**c)
 prove([],Or(IntVal(sum_chunks)!=IntVal(delta),IntVal(prod_chunks)!=IntVal(3**delta)))
# Obligation B: Bounded polynomial term exponent c_e + i * e.
poly_i,in_e,c_e=Ints('poly_i in_e c_e')
term_exp=c_e+poly_i*in_e
prove([0<=poly_i,poly_i<=16,-2**31<=in_e,in_e<=2**31-1,-2**31<=c_e,c_e<=2**31-1],Or(term_exp>=2**63-1,term_exp<=-(2**63-1)))
# Obligation C: Word and bit shift equivalence ((v * 2^b) * 2^(64*w) == v * 2^(64*w + b)).
M=2048
v_bv=BitVec('v_bv',M)
w_bv=BitVec('w_bv',M)
b_bv=BitVec('b_bv',M)
N_limbs=32
prove([ULE(w_bv,BitVecVal(N_limbs-1,M)),ULE(b_bv,BitVecVal(63,M))],((v_bv<<b_bv)<<(64*w_bv))!=(v_bv<<(64*w_bv+b_bv)))
v_int,p_b,p_w=Ints('v_int p_b p_w')
prove([],(v_int*p_b)*p_w!=v_int*(p_b*p_w))
print(f'PASS {checks} numerical root/tail, fixed scratch, guarded address and arithmetic-capacity model obligations')
