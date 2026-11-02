#!/usr/bin/env python3
"""Check/rebuild four binary64 CORDIC tables using exact rational intervals.
This certifies stored constants, NOT the full CORDIC algorithms/range reduction.
No mpmath/libm/Decimal dependency; --rewrite explicitly updates the header.
"""
from fractions import Fraction as F
from math import isqrt
from pathlib import Path
import argparse,re,struct

def atan_interval(x):
 s=F(0)
 for n in range(80):s+=(-1 if n%2 else 1)*x**(2*n+1)/(2*n+1)
 tail=x**161/161
 return s,s+tail # 80 terms end with a negative term

def atanh_interval(x):
 s=sum((x**(2*n+1)/(2*n+1) for n in range(80)),F(0))
 return s,s+x**161/(161*(1-x*x))

def rounded_interval(interval):
 lo,hi=interval;assert lo<=hi
 a,b=float(lo),float(hi)
 assert a==b,'interval must select a unique binary64 value'
 return a

def rounded_sqrt(x):
 # Values are in [1/sqrt(2), 3/sqrt(8)]; choose the local binary binade.
 shift=53 if x<1 else 52
 scaled=x*2**(2*shift);q=isqrt(scaled.numerator//scaled.denominator)
 comparison=4*scaled.numerator-scaled.denominator*(2*q+1)**2
 if comparison>0 or (comparison==0 and q%2):q+=1
 return float(F(q,2**shift))

def tables():
 a,b=atan_interval(F(1,5)),atan_interval(F(1,239))
 angles=[rounded_interval((4*a[0]-b[1],4*a[1]-b[0]))]
 angles += [rounded_interval(atan_interval(F(1,3**i))) for i in range(1,21)]
 sequence=[1,1]+list(range(2,21))
 return {'CORDIC_CIRC_THETA':angles,
  'CORDIC_CIRC_K':[rounded_sqrt(F(1)/(1+F(1,3**(2*i)))) for i in range(21)],
  'CORDIC_HYP_THETA':[rounded_interval(atanh_interval(F(1,3**i))) for i in sequence],
  'CORDIC_HYP_K':[rounded_sqrt(F(1)/(1-F(1,3**(2*i)))) for i in sequence]}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--rewrite',action='store_true');args=parser.parse_args()
 path=Path(__file__).resolve().parents[1]/'ternary_fp.hpp';text=path.read_text();changes=[]
 for name,values in tables().items():
  pattern=r'(constexpr std::array<double, 21> '+name+r' = \{)(.*?)(\n\};)'
  match=re.search(pattern,text,re.S);assert match,name
  actual=[float(line.split('//')[0].strip().rstrip(',')) for line in match[2].splitlines() if line.split('//')[0].strip()]
  assert len(actual)==21
  for i,(a,b) in enumerate(zip(actual,values)):
   if struct.pack('>d',a)!=struct.pack('>d',b):changes.append((name,i,a,b))
  if args.rewrite:
   body='\n'+'\n'.join(f'    {value:.18e}'+(',' if i<20 else '') for i,value in enumerate(values))
   text=text[:match.start(2)]+body+text[match.end(2):]
 if args.rewrite:path.write_text(text)
 elif changes:raise AssertionError(changes)
 print(f'PASS 84 interval-certified binary64 CORDIC constants; {len(changes)} corrected' if args.rewrite else 'PASS 84 interval-certified binary64 CORDIC constants')
 if args.rewrite:
  for name,i,a,b in changes:print(f'{name}[{i}]: {a:.18e} -> {b:.18e}')
if __name__=='__main__':main()
