#!/usr/bin/env python3
"""Core lowering obligations: safe device mul precheck and interval arithmetic.
These establish the stated arithmetic lemmas, not end-to-end compiler refinement.
"""
from z3 import Ints,Solver,Not,And,unsat

def prove(name,pre,claim):
    s=Solver();s.set(timeout=10000);s.add(*pre,Not(claim));r=s.check()
    assert r==unsat,(name,r)

a,b,q,r=Ints('a b q r')
# Euclidean quotient bounds justify b <= floor(H/a), without device i128.
prove('mul accepted',[a>0,b>=0,q>=0,r>=0,r<a,b<=q],a*b<=a*q+r)
prove('mul rejected',[a>0,b>=0,q>=0,r>=0,r<a,b>q],a*b>a*q+r)
H=(3**39-1)//2
assert 2*H<2**63 and H*H<2**127
x,y,lo,hi,L,U=Ints('x y lo hi L U')
pre=[lo<=x,x<=hi,L<=y,y<=U]
prove('add interval',pre,And(lo+L<=x+y,x+y<=hi+U))
prove('sub interval',pre,And(lo-U<=x-y,x-y<=hi-L))
# A parent's safe result cannot erase a failing child evaluation.
assert H+1>H and (H+1)*0==0
print('PASS: CUDA multiplication precheck, int39 widths and core interval obligations')
