#!/usr/bin/env python3
"""Effect/error/IEEE/ABI and bounded malformed-source regression tests."""
from compiler_test_support import Harness
from pathlib import Path
import ctypes as C
import json
import math
import random
import tempfile
import threading
import sys
h=Harness(sys.argv[1]);H=(3**39-1)//2
cases=[
 ('func side(x:int)->int{print(x);return x;} print(side(1)+side(2));',0,[1,2,3]),
 (f'print(1);print(({H}+1)*0);print(99);',1,[1]),
 (f'print(1);print(({H}+1)-1);print(99);',1,[1]),
 (f'func fail()->int{{return {H}+1;}} print(2);print(fail());print(99);',1,[2]),
 (f'print(select(0,{H}+1,7,-{H}-1));',0,[7]),
 ('var nan=sqrt(-1.0);print(nan!=nan);print(nan==nan);print(nan<=0.0);print(select(nan,1,2,3));',0,[1,-1,-1,2]),
 ('print(-0.0);print(sqrt(1e-300));print(fma(1.0,2.0,0.25));',0,[-0.0,1e-150,2.25]),
 ('func f(n:int)->int{if(n>0){return f(n-1);}else{return 0;}}return f(513);',1,[]),
 ('return 1;',1,[]),
 ('var big=1e308;var infinity=big*big;print(infinity);print(-infinity);print(select(infinity,1,2,3));var nan=infinity-infinity;print(nan!=nan);print(select(nan,4,5,6));',0,[float('inf'),float('-inf'),1,1,5]),
 ('func first(n:int)->int{var i=0;while(i<n){if(i==3){return i;}i=i+1;}return -1;}print(first(7));print(first(2));',0,[3,-1]),
]
with tempfile.TemporaryDirectory(prefix='bt-core-gates-') as d:
 work=Path(d)
 for index,(source,status,expected) in enumerate(cases):
  for flags in ([],['--core-opt']):
   tool=Harness(h.compiler,flags=flags);src,shared=tool.emit(source,work)
   native=tool.native(shared);assert native.returncode==status,(index,flags,native.stderr)
   actual=[float(v) for v in native.stdout.splitlines()]
   assert len(actual)==len(expected),(index,native.stdout,expected)
   for a,b in zip(actual,expected):assert math.isclose(a,b,rel_tol=1e-14,abs_tol=0),(index,a,b)
   if index==6:assert native.stdout.splitlines()[0].startswith('-'),native.stdout
   core=tool.run([h.compiler,'--core-run',str(src),*flags]);assert core.returncode==status,(index,core.stderr)
   # Direct ABI inspection lives in an isolated process so a machine-code
   # regression cannot take down the test controller. Main return=1 is NOT
   # necessarily an arithmetic failure; the status channel distinguishes it.
   code=f'''import ctypes as C,threading
l=C.CDLL({str(shared)!r});assert l.bt_core_abi_version()==1
assert l.bt_core_status()==0
codes=[]
def run():
 assert l.bt_core_status()==0
 codes.append((l.bt_main(),l.bt_core_status()))
threads=[threading.Thread(target=run) for _ in range(4)]
for t in threads:t.start()
for t in threads:t.join()
assert len(codes)==4 and all(a=={status} and b=={1 if status and index!=8 else 0} for a,b in codes),codes
assert l.bt_core_status()==0
'''
   p=h.run([sys.executable,'-c',code]);assert p.returncode==0,(index,p.stderr)
 # Structured negative fuzz cases must reject, never crash/hang.
 rng=random.Random(0x46555a5a)
 fragments=['var','func','(','}',')','return','select','+','*','1','x',':',';']
 bad=['var = '+' '.join(rng.choices(fragments,k=20)) for _ in range(100)]
 bad += ['print('+('('*300)+'1'+(')'*300)+');',
         'print('+('~'*300)+'1);','return '+('1 + '*300)+'1;',
         ('if(1) '*300)+'return 0;', '//'+('x'*(8*1024*1024))]
 for source in bad:
  src=work/'bad.bt';src.write_text(source)
  p=h.run([h.compiler,'--core-run',str(src)])
  assert p.returncode==1 and ('Error' in p.stderr or 'Failure' in p.stderr),p.stderr
 # Cycles resolve once by exact content identity; deep chains fail deliberately.
 a=work/'a.bt';b=work/'b.bt';a.write_text('import "b.bt"; var a=1;print(a+b);');b.write_text('import "a.bt";var b=2;')
 p=h.run([h.compiler,'--core-run',str(a)]);assert p.returncode==0 and p.stdout.startswith('3 '),p.stderr
 for folder,name in [('left','left_value'),('right','right_value')]:
  (work/folder).mkdir();(work/folder/'wrapper.bt').write_text('import "leaf.bt";')
  (work/folder/'leaf.bt').write_text(f'var {name}=1;')
 a.write_text('import "left/wrapper.bt";import "right/wrapper.bt";print(left_value+right_value);')
 p=h.run([h.compiler,'--core-run',str(a)]);assert p.returncode==0 and p.stdout.startswith('2 '),p.stderr
 for i in range(40):(work/f'd{i}.bt').write_text(f'import "d{i+1}.bt";' if i<39 else 'return 0;')
 p=h.run([h.compiler,'--core-run',str(work/'d0.bt')]);assert p.returncode==1 and 'budget' in p.stderr,p.stderr
print('PASS: native/core optimized+unoptimized effects, IEEE edges, TLS ABI, 105 malformed-source cases and imports')
