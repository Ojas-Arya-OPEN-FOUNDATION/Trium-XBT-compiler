#!/usr/bin/env python3
"""Actual native shared-object compilation/execution; no simulated compiler."""
import ctypes as C
from pathlib import Path
import os,sys,random,math,struct
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ffi/python'))
import trium
library=str(Path(sys.argv[1]).resolve());H=(3**39-1)//2
rng=random.Random(0x53444b32)
def fails(fn,code):
    try:fn()
    except trium.Error as e:assert e.code==code,(e.code,str(e));return
    raise AssertionError('expected failure')
def equivalent(a,b):
    return isinstance(a,float) and math.isnan(a) and isinstance(b,float) and math.isnan(b) or (struct.pack('d',a)==struct.pack('d',b) if isinstance(a,float) else a==b)
with trium.Context(library=library) as ctx:
    if not ctx.cache_info['native_compiler_captured']:
        print('SKIP no candidate native compiler');sys.exit(77)
    values=[-H,-H+1,-100,-1,0,1,100,H-1,H]+[rng.randint(-10000,10000) for _ in range(100)]
    cases=[
        'return x;', 'return select(x,1,2,3);',
        'return x*x;', 'return x*9;',
        'if(x>0){return x-1;}else{return x+1;}',
        'return select(x,7,0,'+str(H)+'*'+str(H)+');',
        'return abs(x);',
        'return btha_sum_p(x,0,3,0);',
        'return btha_sum_n(x,3,1,0);',
        'return btha_carry_p(x,0,3,0);',
        'return btha_carry_n(0,x,0,-7);',
        'return x & -42;', 'return x | 13;', 'return x ^ -17;',
        'return planar_dot64(x,0,511,0);',
        'var a:int=x+1;return a*3;'
    ]
    compared=0
    for body in cases:
        with ctx.compile('func f(x:int)->int{'+body+'}') as m:
            for x in values:
                try:expected=m.invoke('f',x)
                except trium.Error as e:fails(lambda:m.invoke('f',x,backend='native'),e.code)
                else:assert m.invoke('f',x,backend='native')==expected
                compared+=1
            good=[]
            for x in values:
                try:m.invoke('f',x)
                except trium.Error:continue
                good.append(x)
            reference=m.map('f',good)[0];actual,metrics=m.map('f',good,backend='native')
            assert actual==reference and metrics['backend']==2 and metrics['compile_ms']==0
    with ctx.compile('func g(x:float,y:float)->float{return fma(x,y,0.5);}func s(x:float)->float{return sqrt(x);}func mix(x:int,y:float)->float{return x+y;}func sign(x:float)->int{return select(x,1,0,-1);}func v()->void{return;}') as m:
        cases=[('g',(1.25,2.0)),('g',(float('inf'),0.0)),('s',(-1.0,)),('s',(-0.0,)),('s',(2.0,)),('mix',(H,0.125)),('sign',(float('nan'),)),('sign',(-0.0,)),('v',())]
        for name,args in cases:assert equivalent(m.invoke(name,*args),m.invoke(name,*args,backend='native')),(name,args)
        fails(lambda:m.invoke('mix',1.0,2.0,backend='native'),1)
        fails(lambda:m.invoke('s',1,backend='native'),1)
    with ctx.compile('func tr(x:trit)->trit{return x;}') as m:
        assert m.info['functions'][0]['parameters'][0]['type']=='trit'
        for x in (-1,0,1):assert m.invoke('tr',x,backend='native')==x
        fails(lambda:m.invoke('tr',2,backend='native'),1)
    # Exercise bounded native-image eviction without changing surviving handles.
    with ctx.compile(''.join(f'func f{i}(x:int)->int{{return x+{i};}}' for i in range(18))) as m:
        before=ctx.cache_info['native_builds']
        for i in range(18):assert m.invoke(f'f{i}',1,backend='native')==i+1
        assert ctx.cache_info['native_builds']==before+18 and m.invoke('f0',2,backend='native')==2
    # Single-flight build and independent thread-local error states.
    with ctx.compile('func f(x:int)->int{return x*9;}') as m:
        before=ctx.cache_info['native_builds']
        def worker(i):
            if i%3==0:fails(lambda:m.invoke('f',H,backend='native'),2)
            else:assert m.invoke('f',i,backend='native')==i*9
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(80)))
        assert ctx.cache_info['native_builds']==before+1
        assert m.invoke('f',2,backend='native')==18
        # A failing lane must not partially publish outputs OR metrics.
        data=(C.c_int64*2)(1,H);out=(C.c_int64*2)(123,456);metrics=trium.Metrics();metrics.struct_size=C.sizeof(metrics);metrics.total_ms=123
        snapshot=bytes(metrics);error=C.create_string_buffer(1024)
        assert ctx.lib.bt_sdk_map(m.handle,b'f',2,data,2,out,C.byref(metrics),error,len(error))==2
        assert list(out)==[123,456] and bytes(metrics)==snapshot
        builds=ctx.cache_info['native_builds'];ctx.clear_cache()
        assert m.invoke('f',2,backend='native')==18 and ctx.cache_info['native_builds']==builds+1
        # Recompiling live identical source shares an immutable module/cache but not its handle.
        with ctx.compile('func f(x:int)->int{return x*9;}') as same:
            same.invoke('f',3,backend='native')
            builds=ctx.cache_info['native_builds']
            with ctx.compile('func f(x:int)->int{return x*9;}') as duplicate:
                assert duplicate.handle.value!=same.handle.value
                assert duplicate.invoke('f',4,backend='native')==36
                assert ctx.cache_info['native_builds']==builds
    # Invalidating cached dlopen objects while other accepted calls run is safe.
    with ctx.compile('func race(x:int)->int{return x+1;}') as m:
        m.invoke('race',1,backend='native')
        def racing(i):
            if i%7==0:ctx.clear_cache()
            else:assert m.invoke('race',i,backend='native')==i+1
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(racing,range(48)))
        assert m.invoke('race',2,backend='native')==3
    # Compiler discovery/configuration is explicit; reference still works.
    old=os.environ.get('BT_SDK_CLANG');os.environ['BT_SDK_CLANG']='/nonexistent/trium-clang'
    try:
        with trium.Context(library=library) as missing:
            with missing.compile('func f(x:int)->int{return x+1;}') as m:
                assert m.invoke('f',1)==2
                fails(lambda:m.invoke('f',1,backend='native'),4)
    finally:
        if old is None:os.environ.pop('BT_SDK_CLANG',None)
        else:os.environ['BT_SDK_CLANG']=old
print(f'PASS {compared} native/reference cases, actual mixed typed ABI, TLS concurrency, cache reuse/invalidation and failure transactions')
