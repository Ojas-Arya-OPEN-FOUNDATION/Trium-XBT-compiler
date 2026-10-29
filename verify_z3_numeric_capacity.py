#!/usr/bin/env python3
"""Machine-width obligations for the exact rational/BTFP accumulator paths.
This is not a proof of all BTFP transcendental functions or compiler refinement.
"""
from z3 import Int, Solver, Not, unsat

def prove(name,assumptions,claim):
    s=Solver();s.set(timeout=10000);s.add(*assumptions,Not(claim))
    answer=s.check()
    assert answer==unsat,(name,answer)

signed_max=2**127-1
for w in range(1,41):
    h=(3**w-1)//2
    # Reduced canonical components are bounded; a rational sum uses two products.
    assert 2*h*h<=signed_max
    n,d=Int(f'n{w}'),Int(f'd{w}')
    top=h+1
    # Full int64 ratio operands, including |INT64_MIN|. A normalization scale-up
    # remains below this threshold; its next condition also fits signed i128.
    dmax=max(2**63,3*2**63//top)
    prove(f'ratio right shift W={w}',[n<=2**63,d>0,n>=top*d],3*d<=dmax)
    prove(f'ratio scale W={w}',[n>=0,d>0,d<=dmax,n<top*d],3*n<=signed_max)
    assert 3*top*dmax<=signed_max
    # Retained explicit FMA alignment capacities bound BOTH aligned addends.
    k=max(0,78-2*w)
    assert h*h*3**k+h<=signed_max
    k=78-w
    assert h*3**k+h*h<=signed_max
# Unsigned normalization can consume even the magnitude of INT128_MIN without
# ever computing 3**81 (which would overflow unsigned i128).
assert 2**127//3**80<=1 and 3**80<2**128
print('PASS: rational, ratio-normalization and exact FMA capacities for widths 1..40')
