#!/usr/bin/env python3
"""Deterministic independent exact-reference coverage of finite BTFP profile 1.
No tolerance-based comparisons: mantissa/exponent and binary64 bits must agree.
"""
from fractions import Fraction as F
from pathlib import Path
import ctypes as C
import json
import itertools
import math
import random
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ffi/python'))
import trium
from fp_reference import nearest, nearest_sqrt, value, power
rng=random.Random(0x54465031)
contract=json.loads((Path(__file__).resolve().parents[1]/'specs/ternary-float-v1.json').read_text())
assert contract['version']==1 and contract['width_min']==1 and contract['width_max']==40
assert contract['primitive_operations']==['add','sub','mul','div','sqrt','fma']
assert C.sizeof(trium._Float)==24
checks=0

def pair(v):return v.mantissa,v.exponent

def bits(x):return struct.pack('>d',x)

def binary(x):
    try:return float(x)
    except OverflowError:return -math.inf if x<0 else math.inf

def matches(actual,expected):
    global checks
    assert pair(actual)==expected,(actual,expected)
    checks+=1

def invalid(fn,code=1):
    global checks
    try:fn()
    except trium.Error as e:assert e.code==code,(e.code,e)
    else:raise AssertionError('invalid operation accepted')
    checks+=1

with trium.Context(library=str(Path(sys.argv[1]).resolve())) as ctx:
    for w in range(1,41):
        lo,hi=(3**(w-1)+1)//2,(3**w-1)//2
        make=lambda m,e=0:trium.TernaryFloat(w,*nearest(m*power(e),w))
        for n,d in [(0,1),(1,2),(-1,2),(-2**63,-2**63),(2**63-1,-2**63)]+[(rng.randint(-2**63,2**63-1),rng.choice([-1,1])*rng.randint(1,2**63-1)) for _ in range(24)]:
            matches(ctx.float_from_ratio(n,d,width=w),nearest(F(n,d),w))
        samples=[0,lo,hi,-lo,-hi]+[rng.randint(-hi,hi) for _ in range(24)]
        for m in samples:
            a=make(m,rng.randint(-12,12));b=make(rng.choice(samples),rng.randint(-12,12));c=make(rng.choice(samples),rng.randint(-12,12))
            x,y,z=map(lambda t:value(pair(t)),(a,b,c))
            for op,reference in [('add',lambda:x+y),('sub',lambda:x-y),('mul',lambda:x*y),('fma',lambda:x*y+z)]:
                matches(ctx.float_apply(op,a,b,c),nearest(reference(),w))
            if y:matches(ctx.float_apply('div',a,b),nearest(x/y,w))
            if x>=0:matches(ctx.float_apply('sqrt',a),nearest_sqrt(x,w))
        # Exact binary imports: include subnormals, maxima, signed zero and bitwise random inputs.
        inputs=[0.0,-0.0,0.5,-0.5,1.0,math.nextafter(1.0,0.0),math.nextafter(1.0,math.inf),float.fromhex('0x0.0000000000001p-1022'),float.fromhex('0x1.fffffffffffffp1023')]
        inputs += [struct.unpack('>d',rng.getrandbits(64).to_bytes(8,'big'))[0] for _ in range(24)]
        for x in inputs:
            if not math.isfinite(x):continue
            a=ctx.float_from_double(x,width=w);matches(a,nearest(F.from_float(x),w))
            expected=binary(value(pair(a)))
            assert bits(ctx.float_to_double(a))==bits(expected),(w,x,a,ctx.float_to_double(a),expected)
            checks+=1
        # Export covers rounding, subnormal/normal and finite/infinite boundaries.
        for e in [-801,-800,-720,-700,-680,-679,-678,-650,-647,-646,-645,-1,0,1,640,645,646,647,800]:
            for m in (lo,hi,-lo,-hi):
                a=trium.TernaryFloat(w,m,e);expected=binary(value(pair(a)))
                assert bits(ctx.float_to_double(a))==bits(expected),(a,ctx.float_to_double(a),expected)
                checks+=1
        # Sqrt parity/extreme exponents without allocating 3**INT_MAX in the oracle.
        for e in (-2**31,-2**31+1,2**31-2,2**31-1):
            for m in (lo,hi):
                r=nearest_sqrt(F(m)*power(e%2),w)
                matches(ctx.float_apply('sqrt',trium.TernaryFloat(w,m,e)),(r[0],r[1]+e//2))
        for op in ('add','sub','mul','div','fma','sqrt'):
            a=make(-1 if op=='sqrt' else 1);zero=make(0)
            if op=='sqrt':invalid(lambda:ctx.float_apply(op,a))
            elif op=='div':invalid(lambda:ctx.float_apply(op,a,zero))
        invalid(lambda:ctx.float_from_ratio(1,0,width=w))
        invalid(lambda:ctx.float_from_double(math.inf,width=w))
        invalid(lambda:ctx.float_from_double(math.nan,width=w))
        invalid(lambda:ctx.float_apply('add',trium.TernaryFloat(w,0,1),make(1)))
        invalid(lambda:ctx.float_apply('mul',trium.TernaryFloat(w,hi,2**31-1),make(3)),2)
        invalid(lambda:ctx.float_apply('div',trium.TernaryFloat(w,lo,-2**31),make(3)),2)
    # Exhaustive small-format neighborhoods for binary operations and square root.
    for w in (1,2,3):
        lo,hi=(3**(w-1)+1)//2,(3**w-1)//2
        values=[trium.TernaryFloat(w,0,0)]+[trium.TernaryFloat(w,s*m,e) for s in (-1,1) for m in range(lo,hi+1) for e in range(-2,3)]
        for a,b in itertools.product(values,repeat=2):
            x,y=value(pair(a)),value(pair(b))
            for op,r in [('add',x+y),('sub',x-y),('mul',x*y)]:matches(ctx.float_apply(op,a,b),nearest(r,w))
            if y:matches(ctx.float_apply('div',a,b),nearest(x/y,w))
        if w==1:
            for a,b,c in itertools.product(values,repeat=3):matches(ctx.float_apply('fma',a,b,c),nearest(value(pair(a))*value(pair(b))+value(pair(c)),w))
    # Remote FMA terms must break BOTH signs of a binade midpoint, even across extreme gaps.
    for e in (-161,-200,-2**31):
        for s,t in itertools.product((-1,1),repeat=2):
            a=trium.TernaryFloat(3,s*6,0);b=trium.TernaryFloat(3,7,0);c=trium.TernaryFloat(3,t*5,e)
            expected=(s*5,2) if s==t else (s*13,1)
            matches(ctx.float_apply('fma',a,b,c),expected)
    # Test close and threshold-aligned FMA products at every width, including W40.
    for w in range(1,41):
        h=(3**w-1)//2;l=(3**(w-1)+1)//2
        for e in (-161,-160,-81,-40,-1,0,1,40,81,160,161):
            a=trium.TernaryFloat(w,h,0);b=trium.TernaryFloat(w,-l,0);c=trium.TernaryFloat(w,h,e)
            matches(ctx.float_apply('fma',a,b,c),nearest(F(-h*l)+h*power(e),w))
    # Construct exact product/binade midpoint ties at every odd width >=3.
    for w in range(3,41,2):
        h=(3**w-1)//2;l=(3**(w-1)+1)//2
        for sign,tail in itertools.product((-1,1),repeat=2):
            a=trium.TernaryFloat(w,sign*2*3**(w-2),0)
            b=trium.TernaryFloat(w,(3**w+1)//4,0)
            c=trium.TernaryFloat(w,tail*l,-2**31)
            matches(ctx.float_apply('fma',a,b,c),(sign*l,w-1) if sign==tail else (sign*h,w-2))
    # Correct binary ties-to-even; negative zero is preserved on underflow export.
    for n in (2**53+1,2**53+3,-2**53-1,-2**53-3):
        a=ctx.float_from_ratio(n,width=40);assert bits(ctx.float_to_double(a))==bits(float(n));checks+=1
    for _ in range(1000):
        x=struct.unpack('>d',rng.getrandbits(64).to_bytes(8,'big'))[0]
        if not math.isfinite(x) or not x:continue
        a=ctx.float_from_double(x,width=40);assert bits(ctx.float_to_double(a))==bits(x);checks+=1
    # Malformed raw ABI inputs: reserved fields, widths, nulls and failed output transactions.
    lib=ctx.lib;err=C.create_string_buffer(256);sentinel=trium._Float(40,0,123,4,0)
    before=bytes(sentinel)
    for record in (trium._Float(40,1,1,0,0),trium._Float(41,0,1,0,0),trium._Float(40,0,0,1,0),trium._Float(40,0,1,0,0),trium._Float(40,0,1,-2**31,0),trium._Float(40,0,-2**63,2**31-1,0)):
        code=lib.bt_sdk_float_apply(5,C.byref(record),None,None,C.byref(sentinel),err,len(err));assert code==1
        assert bytes(sentinel)==before;checks+=1
    invalid(lambda:ctx.float_apply('add',trium.TernaryFloat(3,6,0),trium.TernaryFloat(2,2,0)))
    # Result aliasing is safe: all operands are consumed before commit.
    record=ctx.float_from_ratio(9,width=40)._record()
    assert lib.bt_sdk_float_apply(1,C.byref(record),C.byref(record),None,C.byref(record),err,len(err))==0
    assert ctx.float_to_double(trium.TernaryFloat._from_record(record))==18.0;checks+=1
    assert lib.bt_sdk_float_from_ratio(40,1,0,0,C.byref(sentinel),err,len(err))==1
    assert bytes(sentinel)==before;checks+=1
    assert lib.bt_sdk_float_from_double(40,math.inf,C.byref(sentinel),err,len(err))==1
    assert bytes(sentinel)==before;checks+=1

print(f'PASS {checks} exact floating-foundation checks: all 40 widths, small exhaustive domains, fused/sticky ties, binary64 bits and ABI transactions')
