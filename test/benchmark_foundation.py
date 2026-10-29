#!/usr/bin/env python3
"""Repeatable scalar benchmark; no GPU, MLIR, or universal speedup claims.
CLI columns include process/parse/typechecking. Warm native excludes compilation
and loading, and is deliberately NOT used for an apples-to-apples speedup.
"""
from compiler_test_support import Harness
import argparse
import ctypes
import json
import platform
from pathlib import Path
import statistics
import tempfile
import time

parser=argparse.ArgumentParser()
parser.add_argument('compiler');parser.add_argument('--iterations',type=int,default=100000)
parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--output',type=Path)
a=parser.parse_args()
assert 1<=a.iterations<=100000 and a.repeats>=3
h=Harness(a.compiler)
n=a.iterations
checksum=0;x=1
for _ in range(n):
    x=x*3+17
    for j in range(3):
        if x>333331: x-=333331
    checksum+=x
step='''func step(x:int)->int { var y=x*3+17;
if(y>333331){y=y-333331;} if(y>333331){y=y-333331;} if(y>333331){y=y-333331;} return y; }'''
kernels={
    'checked_sum':f'var i=0; var sum=0; while(i < {n}) {{ sum=sum+i; i=i+1; }} return sum <=> {n*(n-1)//2};',
    'calls_recurrence':step+f'var i=0;var x=1;var sum=0;while(i < {n}){{ x=step(x);sum=sum+x;i=i+1;}} return sum <=> {checksum};'
}
result={'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),
        'iterations':n,'repeats':a.repeats,'native_llvm_opt_level':0,'kernels':{}}
try:
    result['cpu_model']=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
except (OSError,StopIteration): pass

def measure(fn):
    assert fn()==0
    samples=[]
    for _ in range(a.repeats):
        start=time.perf_counter_ns();status=fn();elapsed=(time.perf_counter_ns()-start)/1e6
        assert status==0,status
        samples.append(elapsed)
    return {'median_ms':statistics.median(samples),'min_ms':min(samples),'max_ms':max(samples),'samples_ms':samples}

for name,source in kernels.items():
    with tempfile.TemporaryDirectory(prefix='bt-bench-') as d:
        path,shared=h.emit(source,Path(d))
        data={'compilation':h.metrics.copy()}
        for label,args in [('ast_cli',[h.compiler,'run',str(path)]),('core_cli',[h.compiler,'--core-run',str(path)])]:
            def execute(args=args):
                p=h.run(args)
                assert p.returncode==0,(label,p.stderr)
                return p.returncode
            data[label]=measure(execute)
        isolated=h.native(shared)
        assert isolated.returncode==0,isolated.stderr
        library=ctypes.CDLL(str(shared));entry=library.bt_main;entry.restype=ctypes.c_int;entry.argtypes=[]
        data['native_warm_kernel']=measure(entry)
        data['ast_to_core_cli_ratio']=data['ast_cli']['median_ms']/data['core_cli']['median_ms']
        result['kernels'][name]=data
        print(name,json.dumps(data),flush=True)
if a.output:
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
