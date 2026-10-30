#!/usr/bin/env python3
"""Independent exact Fraction oracle: normalized ternary rounding and rationals.
Candidates are compared by exact distance, not by imitating C++ normalization.
"""
from fractions import Fraction as F
from math import isqrt
import random
import subprocess
import sys


from fp_reference import power,nearest,nearest_sqrt,value

rng = random.Random(0x46504d415448)
commands,expected,labels = [],[],[]
def add(command,result):
    commands.append(command+'\n'); expected.append(result); labels.append(command)

for w in (1,2,3,10,20,40):
    h=(3**w-1)//2
    for m in list(range(-100,101)) + [h-1,h,h+1,-h-1,-2**63,2**63-1] + [rng.randint(-h,h) for _ in range(100)]:
        e=rng.randint(-5,5)
        add(f'fp norm {w} {m} {e} 0 0 0 0',nearest(m*power(e),w))
    ratios=[(-2**63,-2**63),(2**63-1,-2**63),(-2**63,1),(0,0),(1,0)]
    ratios += [(rng.randint(-2**63,2**63-1),rng.randint(-2**63,2**63-1)) for _ in range(500)]
    for n,d in ratios:
        try: result=nearest(F(n,d),w)
        except ZeroDivisionError: result='domain'
        add(f'fp ratio {w} {n} 0 {d} 0 0 0',result)
    for _ in range(150):
        m,n,c=[rng.randint(-h,h) for _ in range(3)]
        e,f,g=[rng.randint(-5,5) for _ in range(3)]
        a,b,z=[value(nearest(k*power(j),w)) for k,j in ((m,e),(n,f),(c,g))]
        for op in ('add','mul','div','fma','sqrt'):
            try:
                result = nearest_sqrt(a,w) if op=='sqrt' else nearest(
                    a+b if op=='add' else a*b if op=='mul' else a/b if op=='div' else a*b+z,w)
            except (ZeroDivisionError,ValueError): result='domain'
            add(f'fp {op} {w} {m} {e} {n} {f} {c} {g}',result)

H=(3**39-1)//2
for _ in range(1500):
    a,b,c,d=[rng.randint(-100000,100000) for _ in range(4)]
    for op in ('add','sub','mul','div'):
        try:
            x,y=F(a,b),F(c,d)
            z=x+y if op=='add' else x-y if op=='sub' else x*y if op=='mul' else x/y
            result = (z.numerator,z.denominator) if abs(z.numerator)<=H and z.denominator<=H else 'overflow'
        except ZeroDivisionError: result='domain'
        add(f'rational {op} {a} {b} {c} {d}',result)

process=subprocess.run([sys.argv[1]],input=''.join(commands),capture_output=True,text=True,timeout=60)
assert process.returncode==0, process.stderr
actual=process.stdout.splitlines()
assert len(actual)==len(expected),(len(actual),len(expected))
failures=[]
for label,line,wanted in zip(labels,actual,expected):
    got=tuple(map(int,line.split())) if line not in ('domain','overflow') else line
    if isinstance(got,tuple) and label.startswith('fp norm '):
        _,_,w,m,e,*_=label.split()
        original=int(m)*power(int(e));rounded=value(got)
        if original: assert abs(rounded-original)*3**(int(w)-1)<=abs(original)
    if got!=wanted: failures.append((label,got,wanted))
if failures:
    for failure in failures[:20]: print('MISMATCH',failure)
    raise AssertionError(f'{len(failures)} exact numeric oracle mismatches')
print(f'PASS: {len(expected)} exact rational/BTFP cases, including binade boundaries')
