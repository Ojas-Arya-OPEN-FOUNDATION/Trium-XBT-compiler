#!/usr/bin/env python3
"""Mandatory API-1.1 contracts: no compiler or GPU required."""
import ctypes as C
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ffi/python'))
import trium
library=str(Path(sys.argv[1]).resolve())
def fails(fn,code):
    try:fn()
    except trium.Error as e:assert e.code==code,(e.code,str(e));return str(e)
    raise AssertionError('expected failure')
with trium.Context(library=library) as ctx:
    assert ctx.info['api_minor']>=1 and ctx.info['sizes']['metrics']==C.sizeof(trium.Metrics)
    source='func f(x:int)->int{return x*3;}'
    first=ctx.compile(source);second=ctx.compile(source)
    assert first.handle.value!=second.handle.value
    assert ctx.cache_info['source_misses']==1 and ctx.cache_info['source_hits']==1
    assert first.info['functions'][0]['parameters']==[{'index':0,'type':'int39'}]
    assert first.info['functions'][0]['native'] and first.info['functions'][0]['map']
    first.close();assert second.invoke('f',7)==21
    second.close()
    third=ctx.compile(source);assert ctx.cache_info['source_hits']==2;third.close()
    with ThreadPoolExecutor(max_workers=8) as pool:
        def compile_once(_):
            with ctx.compile(source) as module:return module.invoke('f',2)
        assert list(pool.map(compile_once,range(32)))==[6]*32
    assert ctx.cache_info['source_misses']==1
    for i in range(12):
        with ctx.compile(f'func f(x:int)->int{{return x+{i};}}'):pass
    assert ctx.cache_info['source_entries']<=8 and ctx.cache_info['source_bytes']<=16777216
    generation=ctx.cache_info['generation'];ctx.clear_cache()
    assert ctx.cache_info['generation']==generation+1 and ctx.cache_info['source_entries']==0
    with ctx.compile('func loop(x:int)->int{while(x>0){x=x-1;}return x;}func recursive(x:int)->int{return recursive(x);}func io(x:int)->int{print(x);return x;}') as m:
        info={f['name']:f for f in m.info['functions']}
        assert info['loop']['reference'] and not info['loop']['native']
        assert info['recursive']['reference'] and not info['recursive']['native']
        assert not info['io']['reference'] and not info['io']['native']
        assert m.invoke('loop',5)==0
        fails(lambda:m.invoke('loop',1,backend='native'),4)
        fails(lambda:m.invoke('io',1),1)
    for text in ['func f( ->int{}','func f(x:int)->int{return unknown;}']:
        message=fails(lambda:ctx.compile(text),1);assert 'Error' in message and ':' in message
    # Transitive IO propagation across a long call chain, without executing it.
    chain=''.join(f'func f{i}(x:int)->int{{return f{i+1}(x);}}' for i in range(128))+'func f128(x:int)->int{print(x);return x;}'
    with ctx.compile(chain) as m:
        assert len(m.info['functions'])==129 and all(not f['reference'] for f in m.info['functions'])
        fails(lambda:m.invoke('f0',1),1)
    # Publish only fully initialized metadata when a large source is compiled concurrently.
    ctx.clear_cache()
    def compile_chain(_):
        with ctx.compile(chain) as m:return m.handle.value,m.info
    with ThreadPoolExecutor(max_workers=8) as pool:metadata=list(pool.map(compile_chain,range(24)))
    assert all(len(info['functions'])==129 and all(not f['reference'] for f in info['functions']) for _,info in metadata)
    # Length query/copy failure leaves text untouched and returns required size.
    error=C.create_string_buffer(1024);size=C.c_size_t(2);out=C.create_string_buffer(b'ZZ',3)
    status=ctx.lib.bt_sdk_info(out,C.byref(size),error,len(error));assert status==1 and out.raw==b'ZZ\0' and size.value>3
    assert ctx.lib.bt_sdk_info(None,None,error,len(error))==1
    tiny=C.create_string_buffer(1);size=C.c_size_t(0)
    assert ctx.lib.bt_sdk_info(None,None,tiny,1)==1 and tiny.raw==b'\0'
    fails(lambda:trium._query(ctx.lib,'module_info',ctx.handle),1)
    with ctx.compile('func f(x:int)->int{return x+1;}') as m:
        fails(lambda:trium._query(ctx.lib,'cache_info',m.handle),1)
        snapshot=m.info
        ctx.clear_cache();assert m.invoke('f',2)==3 and m.info==snapshot
        ctx.close();fails(lambda:m.invoke('f',2),5)
with trium.Context(library=library,max_elements=1,operation_budget=2) as c:
    with c.compile('func f(x:int)->int{return x*3;}func loop(x:int)->int{while(x>0){x=x-1;}return x;}') as m:
        fails(lambda:m.map('f',[1,2]),6)
        fails(lambda:m.invoke('f',1,backend='native'),6)
        fails(lambda:m.invoke('loop',10),6)
        raw=(C.c_int64*2)(1,2);output=(C.c_int64*2)(91,92);error=C.create_string_buffer(128)
        assert c.lib.bt_sdk_map(m.handle,b'f',0,raw,2,output,None,error,len(error))==6 and list(output)==[91,92]
print('PASS API layouts, typed metadata, diagnostics, bounded source cache, concurrent compilation, invalidation and resource statuses')
