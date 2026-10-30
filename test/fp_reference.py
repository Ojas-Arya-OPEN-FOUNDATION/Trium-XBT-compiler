"""Independent arbitrary-precision oracle. Compare exact distances among
normalized neighbors; no binary floating arithmetic or C++ algorithm calls.
Square root uses squared rational midpoints, not rounded approximate roots.
"""
from fractions import Fraction as F
from math import isqrt

def power(e):
    return F(3**e) if e >= 0 else F(1,3**-e)


def nearest(x,w):
    if not x: return (0,0)
    sign = -1 if x < 0 else 1
    x = abs(x)
    lo,hi = (3**(w-1)+1)//2,(3**w-1)//2
    e = 0
    while x > hi*power(e): e += 1
    while x < lo*power(e): e -= 1
    candidates = []
    for k in range(e-2,e+3):
        y = x/power(k)
        m = min(hi,max(lo,(2*y.numerator+y.denominator)//(2*y.denominator)))
        v = m*power(k)
        candidates.append((abs(v-x),-v,sign*m,k))
    _,_,m,k = min(candidates)
    return m,k


def nearest_sqrt(x,w):
    if x < 0: raise ValueError('domain')
    if not x: return (0,0)
    lo,hi = (3**(w-1)+1)//2,(3**w-1)//2
    e = 0
    while x > (hi*power(e))**2: e += 1
    while x < (lo*power(e))**2: e -= 1
    candidates = {}
    for k in range(e-2,e+3):
        y = x/power(2*k)
        q = isqrt(y.numerator//y.denominator)
        for m in (q,q+1,lo,hi):
            m = max(lo,min(hi,m)); candidates[m*power(k)] = (m,k)
    lower = max(v for v in candidates if v*v <= x)
    upper = min(v for v in candidates if v*v >= x)
    return candidates[upper if x >= ((lower+upper)/2)**2 else lower]


def value(pair): return pair[0]*power(pair[1])

