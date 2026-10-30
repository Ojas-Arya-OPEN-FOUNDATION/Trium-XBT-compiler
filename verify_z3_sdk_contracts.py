#!/usr/bin/env python3
"""SDK transition/admission/address model obligations, not a C++/driver proof.
Concrete concurrency, TLS, cache and failure behavior also require runtime tests.
"""
import sys
try:
 from z3 import *
except ImportError:
 print('SKIP install z3-solver to check SDK models');sys.exit(77)
def prove(assumptions,bad):
 s=Solver();s.set(timeout=30000);s.add(*assumptions,bad);assert s.check()==unsat
# Generations never wrap: clear rejects MAX and increments under its mutex.
e=Int('epoch');prove([1<=e,e<2**64-1],Or(e+1<=e,e+1>=2**64))
# Native static admission uses a nonzero divisor and bounds aggregate work.
n,w,b=Ints('count work budget')
prove([0<=n,1<=w,0<=b,n<=b/w],n*w>b)
# After eviction establishes its loop exit, inserting one bounded source fits.
entries,used,length=Ints('entries bytes source_length')
prove([0<=entries,entries<8,0<=used,0<length,length<=8*1024**2,used+length<=16*1024**2],Or(entries+1>8,used+length>16*1024**2))
# An invalidated cache reference cannot free a shared object with accepted calls.
refs=Int('accepted_references');cached=Bool('cached')
freed=And(Not(cached),refs==0)
prove([refs>0],freed)
# Failed calls preserve output and metrics under the commit-only-on-success rule.
status,old,new,old_metrics,new_metrics=Ints('status old new old_metrics new_metrics')
prove([status!=0],Or(If(status==0,new,old)!=old,If(status==0,new_metrics,old_metrics)!=old_metrics))
# Reused GPU storage is safe only under the push-count lane guard.
i,n,capacity=Ints('lane n capacity')
prove([0<=i,i<n,n<=capacity],Or(8*i+8>8*capacity,4*i+4>4*capacity))
# Uniform native argument words stay inside their caller-owned slot array.
i,count=Ints('argument count')
prove([0<=i,i<count,count<=1000],8*i+8>8*count)
print('PASS SDK generation, admission, cache ownership, transactional output and guarded buffer-address models')
