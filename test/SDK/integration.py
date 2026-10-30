#!/usr/bin/env python3
from pathlib import Path
import os,sys,threading,ctypes as C
os.environ['BT_SDK_LIBRARY']=str(Path(sys.argv[1]).resolve())
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'ffi/python'))
import trium
H=(3**39-1)//2

def fails(fn,code=None):
 try:fn()
 except trium.Error as e:
  assert code is None or code==e.code,(code,e.code,str(e));return
 raise AssertionError('invalid call silently accepted')
with trium.Context() as ctx:
 for bad in ['import "not-allowed.bt";','print(7);','func x()->int{return missing;}']:
  fails(lambda:ctx.compile(bad))
 with ctx.compile('func transform(x:int)->int{return select(x,x*3+1,0,x*3-1);}') as mod:
  assert mod.exports==['transform']
  assert mod.map('transform',[-2,-1,0,1,2])[0]==[-7,-4,0,4,7]
  assert mod.map('transform',[])[0]==[]
  fails(lambda:mod.map('transform',[H]),2)
  fails(lambda:mod.map('transform',[H+1]),2)
  fails(lambda:mod.map('wrong',[1]))
  data=(C.c_int64*2)(1,H);out=(C.c_int64*2)(123,456)
  fails(lambda:trium._check(ctx.lib,'map',mod.handle,b'transform',0,data,2,out,None),2)
  assert list(out)==[123,456]
  errors=[]
  def worker(i):
   try:assert mod.map('transform',[i])[0]==[i*3+1]
   except BaseException as e:errors.append(e)
  threads=[threading.Thread(target=worker,args=(i,)) for i in range(1,20)]
  for t in threads:t.start()
  for t in threads:t.join()
  assert not errors,errors
  try:
   import numpy as np
  except ImportError:pass
  else:
   array=np.arange(10,dtype=np.int64);result,metrics=mod.map_array('transform',array);assert list(result)==[0]+[3*x+1 for x in range(1,10)]
   try:mod.map_array('transform',array[::-1])
   except ValueError:pass
   else:raise AssertionError('negative stride accepted')
   unaligned=np.ndarray(shape=(10,),dtype=np.int64,buffer=np.zeros(81,dtype=np.uint8),offset=1)
   assert not unaligned.flags.aligned
   try:mod.map_array('transform',unaligned)
   except ValueError:pass
   else:raise AssertionError('unaligned int64 input accepted')
   array.flags.writeable=False
   assert list(mod.map_array('transform',array)[0])==[0]+[3*x+1 for x in range(1,10)]
  from itertools import repeat
  try:mod.map('transform',repeat(1))
  except ValueError:pass
  else:raise AssertionError('unbounded iterable accepted')

 with ctx.compile('func f(x:int)->int{if(x>0){return x+f(x-1);}return 0;}func g(x:float)->float{return fma(x,2.0,0.5);}') as m:
  assert m.invoke('f',10)==55 and m.invoke('g',1.25)==3.0
  fails(lambda:m.invoke('g',1))
 with ctx.compile('func noisy(x:int)->int{print(x);return x;}') as m:fails(lambda:m.invoke('noisy',1))
 for n in [-(2**63),-H,-1,0,1,H,2**63-1]:assert ctx.decode(*ctx.encode(n))==n
 fails(lambda:ctx.decode(1,1),1)
 assert ctx.matmul([1,2,3,4],[5,6,7,8],2,2,2)==[19,22,43,50]
 mod=ctx.compile('func transform(x:int)->int{return x+1;}')
 ctx.close();fails(lambda:mod.map('transform',[1]),5);mod.close()
print('PASS in-process CPU compiler SDK, transactions, exports, scalar calls, math, ownership and concurrency')
