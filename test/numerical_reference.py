#!/usr/bin/env python3
"""Exact represented-input oracle. Roots compare powered rational midpoints;
no libm tolerances or C++ normalizer are used to certify new finite operations.
"""
from fractions import Fraction as Q
from pathlib import Path
import ctypes as C
import itertools,random,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ffi/python'))
from fp_reference import nearest,nearest_sqrt,power,value
import trium
rng=random.Random(0x4e554d32);checks=0

def root_floor(n,r):
 lo,hi=0,1<<((n.bit_length()+r-1)//r)
 while lo<hi:
  m=(lo+hi+1)//2
  if m**r<=n:lo=m
  else:hi=m-1
 return lo

def cbrt(x,w):
 if not x:return (0,0)
 sign=-1 if x<0 else 1;x=abs(x);lo,hi=(3**(w-1)+1)//2,(3**w-1)//2;e=0
 while x>(hi*power(e))**3:e+=1
 while x<(lo*power(e))**3:e-=1
 candidates={}
 for k in range(e-2,e+3):
  y=x/power(3*k);q=root_floor(y.numerator//y.denominator,3)
  for m in (q,q+1,lo,hi):
   m=min(hi,max(lo,m));candidates[m*power(k)]=(sign*m,k)
 lower=max(v for v in candidates if v**3<=x);upper=min(v for v in candidates if v**3>=x)
 return candidates[upper if x>=((lower+upper)/2)**3 else lower]

def pair(a):return a.mantissa,a.exponent
def match(a,expected):
 global checks
 assert pair(a)==expected,(a,expected);checks+=1

def fail(fn,code):
 global checks
 try:fn()
 except trium.Error as e:assert e.code==code,(e.code,str(e),code)
 else:raise AssertionError('accepted invalid/over-limit input')
 checks+=1

with trium.Context(library=str(Path(sys.argv[1]).resolve())) as ctx:
 assert ctx.info['api_minor']>=2 and ctx.info['numerical_max_terms']==4096
 for w in range(1,41):
  L,H=(3**(w-1)+1)//2,(3**w-1)//2
  make=lambda n,e=0:trium.TernaryFloat(w,*nearest(Q(n)*power(e),w))
  samples=[make(m,rng.randint(-6,6)) for m in [0,L,H,-L,-H]+[rng.randint(-H,H) for _ in range(15)]]
  for a in samples:
   x=value(pair(a));match(ctx.float_apply('cbrt',a),cbrt(x,w))
   if x:match(ctx.float_apply('reciprocal',a),nearest(1/x,w))
   for p in (-16,-3,-1,0,1,2,3,16):
    if x or p>=0:match(ctx.float_powi(a,p),nearest(x**p,w))
   b=rng.choice(samples);y=value(pair(b));match(ctx.float_apply('hypot',a,b),nearest_sqrt(x*x+y*y,w))
  for _ in range(15):
   a=[rng.choice(samples) for _ in range(rng.randint(0,9))];b=[rng.choice(samples) for _ in a]
   xs=[value(pair(t)) for t in a];ys=[value(pair(t)) for t in b]
   match(ctx.float_reduce('sum',a,width=w),nearest(sum(xs),w))
   match(ctx.float_reduce('dot',a,b,width=w),nearest(sum(x*y for x,y in zip(xs,ys)),w))
   match(ctx.float_reduce('norm2',a,width=w),nearest_sqrt(sum(x*x for x in xs),w))
   # Unit argument strips powers of three before alignment; degree 16 works at W=40.
   one=make(1);coefficients=[make(rng.randint(-20,20)) for _ in range(17)]
   match(ctx.float_polynomial(one,coefficients),nearest(sum(value(pair(c)) for c in coefficients),w))
   # General argument, exact full polynomial (not multiply-rounded Horner).
   argument=trium.TernaryFloat(w,rng.choice([L,H,-L,-H]),0);z=value(pair(argument))
   coefficients=[trium.TernaryFloat(w,rng.choice([L,H,-L,-H]),rng.randint(-2,2)) for _ in range(6)]
   match(ctx.float_polynomial(argument,coefficients),nearest(sum(value(pair(c))*z**i for i,c in enumerate(coefficients)),w))
  # Permutation-independent cancellation and products, final range only.
  a=make(100);b=make(-100);c=make(1)
  expected=nearest(value(pair(a))+value(pair(b))+value(pair(c)),w)
  for terms in itertools.permutations([a,b,c]):match(ctx.float_reduce('sum',terms,width=w),expected)
  for e in (-2**31,-2**31+1,2**31-2,2**31-1):
   a=trium.TernaryFloat(w,H,e)
   r=cbrt(Q(H)*power(e%3),w);match(ctx.float_apply('cbrt',a),(r[0],r[1]+e//3))
   # Far-tail hypot returns the dominant magnitude, with no huge power allocation.
   small=trium.TernaryFloat(w,L,-2**31)
   if e>-(2**31)+w:match(ctx.float_apply('hypot',a,small),(H,e))
   match(ctx.float_reduce('sum',[a,trium.TernaryFloat(w,-H,e)],width=w),(0,0))
   for p in (-1,0,1,2):
    r=nearest(Q(H)**p,w);target=(r[0],r[1]+p*e)
    if -(2**31)<=target[1]<2**31:match(ctx.float_powi(a,p),target)
    else:fail(lambda:ctx.float_powi(a,p),2)
  # Equal enormous exponents: squaring must not fail prematurely in hypot/norm.
  for e in (-2**31,2**31-1):
   a=trium.TernaryFloat(w,L,e);r=nearest_sqrt(Q(2*L*L),w);target=(r[0],r[1]+e)
   for fn in (lambda:ctx.float_apply('hypot',a,a),lambda:ctx.float_reduce('norm2',[a,a],width=w)):
    if -(2**31)<=target[1]<2**31:match(fn(),target)
    else:fail(fn,2)
  fail(lambda:ctx.float_powi(make(0),-1),1)
  fail(lambda:ctx.float_apply('reciprocal',make(0)),1)
  fail(lambda:ctx.float_powi(make(1),17),6)
 # Exhaustive small normalized formats and binade-gap cubic midpoint sides.
 for w in (1,2):
  L,H=(3**(w-1)+1)//2,(3**w-1)//2
  for m in range(-H,H+1):
   for e in range(-4,5):
    a=ctx.float_from_ratio(m,width=w,exponent=e);match(ctx.float_apply('cbrt',a),cbrt(value(pair(a)),w))
  for m in range(L,H+1):
   following=m+1 if m<H else H+2
   midpoint=Q(m+following,2)**3
   for delta in (-Q(1,1000),Q(0),Q(1,1000)):
    # Input conversion may round; oracle uses the actual represented input.
    q=midpoint+delta;a=ctx.float_from_ratio(q.numerator,q.denominator,width=w)
    match(ctx.float_apply('cbrt',a),cbrt(value(pair(a)),w))
 w=40;one=ctx.float_from_ratio(1,width=w);huge=ctx.float_from_ratio(1,width=w,exponent=1000)
 fail(lambda:ctx.float_reduce('sum',[one,huge]),6)
 fail(lambda:ctx.float_reduce('dot',[one]*4097,[one]*4097),6)
 fail(lambda:ctx.float_polynomial(one,[one]*18),6)
 fail(lambda:ctx.float_reduce('sum',[trium.TernaryFloat(w,0,1)]),1)
 fail(lambda:ctx.float_apply('hypot',one,trium.TernaryFloat(1,1,0)),1)
 # Active count and full-degree capacity boundaries, not just tiny arrays.
 match(ctx.float_reduce('sum',[one]*4096),nearest(Q(4096),40))
 match(ctx.float_reduce('dot',[one]*4096,[one]*4096),nearest(Q(4096),40))
 match(ctx.float_reduce('norm2',[one]*4096),nearest(Q(64),40))
 H=(3**40-1)//2;large=trium.TernaryFloat(40,H,0)
 match(ctx.float_polynomial(large,[large]*17),nearest(sum(Q(H)**(i+1) for i in range(17)),40))
 extreme=trium.TernaryFloat(40,H,2**31-1);negative=trium.TernaryFloat(40,-H,2**31-1)
 match(ctx.float_reduce('dot',[extreme,negative],[extreme,extreme]),(0,0))
 # Concrete useful difference: a rounded sequential sum loses the small term.
 big=trium.TernaryFloat(3,13,4);neg=trium.TernaryFloat(3,-13,4);unit=ctx.float_from_ratio(1,width=3)
 match(ctx.float_reduce('sum',[big,unit,neg]),pair(unit))
 match(ctx.float_apply('add',ctx.float_apply('add',big,unit),neg),(0,0))
 # Exact root midpoint ties at binade gaps (not approximated binary ties).
 u=ctx.float_from_ratio(1,width=1)
 match(ctx.float_reduce('norm2',[u]*4),(1,1)) # sqrt(4)=2, equidistant from 1 and 3 -> 3
 three=ctx.float_from_ratio(3,width=2);four=ctx.float_from_ratio(4,width=2)
 match(ctx.float_apply('hypot',three,four),(2,1)) # 5 halfway between 4 and 6 -> 6
 match(ctx.float_reduce('norm2',[three,four]),(2,1))
 match(ctx.float_polynomial(one,[]),(0,0))
 # Raw ABI: output transactions and aliasing, including late invalid records.
 lib=ctx.lib;err=C.create_string_buffer(1024);a=one._record();out=trium._Float(40,0,12345,77,0)
 def raw_fail(fn,code):
  global checks
  before=bytes(out);assert fn()==code and bytes(out)==before;checks+=1
 raw_fail(lambda:lib.bt_sdk_float_polynomial(C.byref(a),None,18,C.byref(out),err,len(err)),1)
 raw_fail(lambda:lib.bt_sdk_float_polynomial(C.byref(a),C.byref(a),18,C.byref(out),err,len(err)),6)
 raw_fail(lambda:lib.bt_sdk_float_powi(C.byref(a),17,C.byref(out),err,len(err)),6)
 records=(trium._Float*2)(a,trium._Float(40,1,a.mantissa,a.exponent,0))
 raw_fail(lambda:lib.bt_sdk_float_reduce(40,1,records,None,2,C.byref(out),err,len(err)),1)
 raw_fail(lambda:lib.bt_sdk_float_reduce(40,1,None,None,4097,C.byref(out),err,len(err)),6)
 raw_fail(lambda:lib.bt_sdk_float_reduce(40,99,None,None,0,C.byref(out),err,len(err)),1)
 assert lib.bt_sdk_float_powi(C.byref(a),3,C.byref(a),err,len(err))==0;match(trium.TernaryFloat._from_record(a),pair(one))
 records=(trium._Float*2)(one._record(),one._record())
 assert lib.bt_sdk_float_reduce(40,1,records,None,2,C.byref(records[1]),err,len(err))==0
 match(trium.TernaryFloat._from_record(records[1]),nearest(Q(2),40))
 # Strict window boundary after removing exact radix factors.
 a=trium.TernaryFloat(1,1,0)
 match(ctx.float_reduce('sum',[a,trium.TernaryFloat(1,1,256)]),nearest(Q(1)+power(256),1))
 fail(lambda:ctx.float_reduce('sum',[a,trium.TernaryFloat(1,1,257)]),6)
print(f'PASS {checks} exact advanced FP/numerical checks: all widths, powered roots, reductions, polynomial, extreme exponents, limits and ABI transactions')
