#!/usr/bin/env python3
"""Finite BTFP profile-1 arithmetic model lemmas, NOT a C++/RTL refinement proof."""
from z3 import *

def prove(assumptions,violation):
    s=Solver();s.set(timeout=30000);s.add(*assumptions,violation)
    result=s.check();assert result==unsat,(result,s.reason_unknown())

# Exact limb multiply-add identity and carry bounds for Wide::times.
B=2**64
x,y,c=Ints('limb_x limb_y carry')
p=x*y+c
prove([0<=x,x<B,0<=y,y<B,0<=c,c<B],Or(p<0,p>=B*B))
prove([0<=x,x<B,0<=y,y<B,0<=c,c<B],p!=(p%B)+B*(p/B))
# Signed-magnitude subtraction's low limb and borrow, including rhs==2^64.
x,y,c=Ints('sx sy borrow')
prove([0<=x,x<B,0<=y,y<B,Or(c==0,c==1)],
      x-y-c != (x-y-c)%B-B*If(x<y+c,1,0))
# Exact signed fast path: after checked alignment, same-sign addition needs
# the explicit headroom check; opposite signs cannot overflow signed i128.
a,b=Ints('aligned_magnitude tail_magnitude');limit=2**127-1
prove([0<=a,a<=limit,0<=b,b<=limit,a<=limit-b],a+b>limit)
prove([0<=a,a<=limit,0<=b,b<=limit],Or(a-b < -limit,a-b>limit))
# sqrt scaling: after the last multiply-by-9, N<9L^2; 4N and
# (2H+2)^2 fit signed i128 for all supported widths.
for w in range(1,41):
    L=(3**(w-1)+1)//2;H=(3**w-1)//2
    assert H==3*L-2 and 36*L*L<2**127 and (2*H+2)**2<2**127
    n=Int('radicand')
    prove([n>=0,n<L*L],Or(9*n<0,9*n>=9*L*L))
# Exact sqrt midpoint test is equivalent to the nearest-neighbor distance test.
r,l,u=Reals('root lower upper')
prove([0<=l,l<u,l<=r,r<=u],
      Xor(4*r*r<(l+u)*(l+u),(r-l)*(r-l)<(u-r)*(u-r)))
# A remote coefficient is <3^80, delta>=161. The nearest nonzero distance
# from an integer-scaled dominant value to a rounding midpoint is >=
# 1/(2*3^40) at W<=40, so the tail can affect ONLY exact ties.
assert 2*3**40*3**80 < 3**161
# Close alignment stays below 512 bits, including a carry and midpoint tests.
assert 8*3**241 < 2**512
# Exact binary64 import/export scratch bounds. 2^1074 denominators and
# their ternary mantissa scaling fit 1280 bits. Export early exits are safe.
H=(3**40-1)//2
assert (2**1074)*3*(H+2)*2 < 2**1280
assert 3**800 < 2**1270 and H*3**646 < 2**1090
assert 3**647 > 2**1024
assert H*2**1075 < 3**801
print('PASS limb/carry/borrow, sqrt bounds/midpoints, remote-tail separation and binary64 capacity models')
