#!/usr/bin/env python3
"""Seeded whole-program generation; exact independent Python outcomes."""
from compiler_test_support import Harness
from pathlib import Path
import random
import os
import re
import sys
import tempfile
h=Harness(sys.argv[1],flags=['--core-opt'] if os.environ.get('BT_TEST_CORE_OPT')=='1' else []);rng=random.Random(0x50524f47)
source=['func bump(x:int)->int { return x+1; }']
expected=[]
for i in range(200):
    n=rng.randrange(16);x=rng.randint(-100,100);threshold=rng.randint(-1000,1000);selector=rng.randint(-2,2)
    source.append(f'''func case_{i}(x:int)->int {{
      var count=0; var total=0;
      while(count < {n}) {{ total=total + (x+count)*(x-count); count=count+1; }}
      if(total > ({threshold})) {{ var x=7; total=total+x; }} else {{ total=total-3; }}
      return select({selector}, total+1, total-2, bump(total));
    }}
    print(case_{i}({x}));''')
    total=sum((x+j)*(x-j) for j in range(n));total+=7 if total>threshold else -3
    expected.append(total+1 if selector>0 else total-2 if selector==0 else total+1)
source.extend(['func fact(n:int)->int { if(n > 0) { return n*fact(n-1); } else { return 1; } }',
               'print(fact(8));','func promoted(x:float)->float{return x*0.125;} print(promoted(5));',
               'var shadow=3; { var shadow=9; print(shadow); } print(shadow);'])
expected.extend([40320,0.625,9,3])
# Source function names must not interpose libc or generated helper/entry symbols.
for name in ('step','puts','malloc','sin'):
    source.append(f'func {name}(x:int)->int{{return x+7;}} print({name}(9));')
    expected.append(16)
source.append('func branch(x:int)->int { select(x) { pos: {return 11;} zero: {return 22;} neg: {return 33;} } }')
source.extend(['print(branch(1));','print(branch(0));','print(branch(-1));'])
expected.extend([11,22,33])
with tempfile.TemporaryDirectory(prefix='bt-programs-') as d:
    src,shared=h.emit('\n'.join(source),Path(d))
    outputs=[h.native(shared),h.run([h.compiler,'run',str(src)]),h.run([h.compiler,'--core-run',str(src),*h.flags])]
    for name,p in zip(('native','AST','typed core'),outputs):
        assert p.returncode==0,(name,p.stderr)
        tokens=p.stdout.splitlines() if name=='native' else re.findall(r'^(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) \[',p.stdout,re.M)
        actual=[float(t) if any(c in t for c in '.eE') else int(t) for t in tokens]
        assert actual==expected,(name,actual,expected)
print(f'PASS: {len(expected)} generated whole-program outcomes across native/AST/core/Python')
