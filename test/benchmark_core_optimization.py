#!/usr/bin/env python3
"""Alternating, correctness-checked opt-off/opt-on measurements.
Target-machine opt=0 isolates this compiler's passes from an LLVM O3 pipeline.
These CPU workloads are not a universal benchmark or GPU speedup claim.
"""
from compiler_test_support import Harness
from contextlib import ExitStack
from pathlib import Path
import argparse
import ctypes as C
import hashlib
import json
import platform
import statistics
import tempfile
import time

p=argparse.ArgumentParser()
p.add_argument('compiler');p.add_argument('--output',type=Path)
p.add_argument('--repeats',type=int,default=7)
a=p.parse_args();assert a.repeats>=3
n=100000;expected=n*(n-1)//2+20*n
map_expected=sum(3*x+(1 if x>0 else -1 if x<0 else 0) for x in ((i%101)-50 for i in range(n)))
sources={
 'identities_and_constants':f'func step(x:int)->int{{return (x*1+0)+(2+3)*4;}}var i=0;var total=0;while(i<{n}){{total=total+step(i);i=i+1;}}return total <=> {expected};',
 'lazy_constant_selection':f'var i=0;var total=0;while(i<{n}){{total=total+select(1,(i*1+0)+(2+3)*4,2026277576509488133+1,0);i=i+1;}}return total <=> {expected};',
 'checked_signed_map':f'func transform(x:int)->int{{var scaled=x*3;return select(x,scaled+1,0,scaled-1);}}var i=0;var x=-50;var total=0;while(i<{n}){{total=total+transform(x);x=x+1;if(x>50){{x=-50;}}i=i+1;}}return total <=> {map_expected};'
}
result={'platform':platform.platform(),'iterations':n,'samples':a.repeats,
 'native_llvm_opt_level':0,'sample_order':'alternating baseline/optimized each repeat',
 'compiler_sha256':hashlib.sha256(Path(a.compiler).read_bytes()).hexdigest(),'workloads':{}}
for name,source in sources.items():
 data={};functions={};libraries=[]
 with ExitStack() as stack:
  for optimize in (False,True):
   mode='optimized' if optimize else 'baseline'
   work=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='bt-core-opt-bench-')))
   h=Harness(a.compiler,flags=['--core-opt'] if optimize else [])
   src,lib=h.emit(source,work)
   check=h.native(lib);assert check.returncode==0,check.stderr
   library=C.CDLL(str(lib));libraries.append(library)
   entry=library.bt_main;entry.restype=C.c_int;entry.argtypes=[]
   def core(h=h,src=src):
    check=h.run([h.compiler,'--core-run',str(src),*h.flags])
    assert check.returncode==0,check.stderr
    return check.returncode
   functions[mode]={'native_warm_kernel':entry,'core_cli':core}
   ir=(work/'program.ll').read_text()
   data[mode]={'compilation_single_sample_ms':h.metrics.copy(),
      'llvm_bytes':len(ir.encode()),'i128_mentions':ir.count('i128')}
  for boundary in ('native_warm_kernel','core_cli'):
   samples={mode:[] for mode in functions}
   for fns in functions.values():assert fns[boundary]()==0
   for repeat in range(a.repeats):
    order=('baseline','optimized') if repeat%2==0 else ('optimized','baseline')
    for mode in order:
     start=time.perf_counter_ns();status=functions[mode][boundary]()
     elapsed=(time.perf_counter_ns()-start)/1e6
     assert status==0
     samples[mode].append(elapsed)
   for mode,values in samples.items():
    data[mode][boundary]={'median_ms':statistics.median(values),'samples_ms':values}
 data['native_ratio']=data['baseline']['native_warm_kernel']['median_ms']/data['optimized']['native_warm_kernel']['median_ms']
 data['core_cli_ratio']=data['baseline']['core_cli']['median_ms']/data['optimized']['core_cli']['median_ms']
 result['workloads'][name]=data;print(name,json.dumps(data),flush=True)
if a.output:
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,indent=2)+'\n')
