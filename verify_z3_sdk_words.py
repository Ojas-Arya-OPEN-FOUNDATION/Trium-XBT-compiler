#!/usr/bin/env python3
"""Int39 word/native64 model obligations; not a whole-shader/compiler proof."""
from z3 import *
B=2**32;H=(3**39-1)//2
x0,x1,y0,y1=Ints('x0 x1 y0 y1')
s=Solver();s.set(timeout=30000)
s.add(*[And(v>=0,v<B) for v in (x0,x1,y0,y1)])
wide=(x0+B*x1)+(y0+B*y1)
pair=(x0+y0)%B+B*((x1+y1+(x0+y0)/B)%B)
s.add(pair!=wide%(B*B));assert s.check()==unsat
# Exact unsigned schoolbook product decomposition underlying the 4-word helper.
s=Solver();s.set(timeout=30000)
s.add((x0+B*x1)*(y0+B*y1)!=x0*y0+B*(x0*y1+x1*y0)+B*B*x1*y1)
assert s.check()==unsat
# Every product of two legal magnitudes fits four uint32 limbs. It must still
# be checked against H BEFORE signed conversion, not only against int64.
a,b=Ints('a b');s=Solver();s.set(timeout=30000);s.add(0<=a,a<=H,0<=b,b<=H,Or(a*b<0,a*b>=2**128));assert s.check()==unsat
# Checked addition/subtraction fit the 64-bit representation before int39 guard.
a,b=Ints('ca cb');s=Solver();s.add(-H<=a,a<=H,-H<=b,b<=H,Or(a+b<-(2**63),a+b>=2**63,a-b<-(2**63),a-b>=2**63));assert s.check()==unsat
# Explicit witness: checking a wrapped signed product is insufficient.
wrapped=(H*9)%(2**64);signed=wrapped-2**64 if wrapped>=2**63 else wrapped
assert -H<=signed<=H and H*9>H
# Native64 guarded multiplication: Euclidean quotient is exactly the admissible
# multiplier limit. a==0 uses divisor 1 and product 0; no zero division occurs.
a,b,q,r=Ints('na nb nq nr');s=Solver();s.set(timeout=30000)
s.add(a>=1,a<=H,b>=0,b<=H,q>=0,r>=0,r<a,H==q*a+r,
      Or(And(b<=q,a*b>H),And(b>q,a*b<=H)))
assert s.check()==unsat
assert H//1==H and 0*H==0
print('PASS uint32 addition, schoolbook product, capacity and pre-conversion overflow and native64 division-guard obligations')
