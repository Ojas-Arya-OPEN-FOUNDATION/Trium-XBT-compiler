#!/usr/bin/env python3
"""Real NVPTX compilation + explicitly labeled device-free execution checks.
With --device: actual CUDA execution or an explicit dependency skip, never fallback.
"""
from pathlib import Path
import ctypes as C
import random
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cuda_map import Driver,DeviceUnavailable,DeviceExecutionError,ptx_from_llvm,H
from compiler_test_support import Harness
h=Harness(sys.argv[1]);device='--device' in sys.argv
if device:
    try: driver=Driver()
    except DeviceUnavailable as error:
        print('SKIP: real GPU qualification unavailable:',error);raise SystemExit(77)
rng=random.Random(0x475055)
values=[-H,-H+1,-100,-1,0,1,100,H-1,H]+[rng.randint(-100000,100000) for _ in range(200)]
cases=[
 ('return x*x+3;',lambda x:x*x+3),
 ('return select(x, 2026277576509488133+1, 7, -2026277576509488133-1);',lambda x:7 if x==0 else H+1),
 ('var y=x; if(y>0){y=y-1;}else{y=y+1;} return y;',lambda x:x-1 if x>0 else x+1),
 ('return (x & 13) ^ (x | -7);',None),
 ('return btha_sum_p(x,0,1152921504606846975,0);',None),
]
for index,(body,reference) in enumerate(cases):
  with tempfile.TemporaryDirectory(prefix='bt-cuda-') as d:
    work=Path(d);source=work/'map.bt';module=work/'map.ll'
    source.write_text('func map(x:int)->int{'+body+'}')
    p=h.run([h.compiler,str(source),'--cuda-map=map','--core-opt','-o',str(module)])
    assert p.returncode==0,p.stderr
    text=module.read_text();ptx=ptx_from_llvm(text,driver.architecture() if device else 'sm_50')
    assert '.entry bt_kernel' in ptx and 'st.global' in ptx
    if reference is None:
      # Independent source interpreter supplies the digit/intrinsic reference;
      # CPU-emulated device IR is still never described as physical GPU evidence.
      expected=[]
      for x in values:
        source.write_text('func map(x:int)->int{'+body+'} print(map('+str(x)+'));')
        p=h.run([h.compiler,'--core-run',str(source)])
        expected.append(int(p.stdout.split()[0]) if p.returncode==0 else None)
    else:
      expected=[]
      for x in values:
        n=reference(x);expected.append(n if -H<=n<=H else None)
    if device:
      safe=[x for x,v in zip(values,expected) if v is not None]
      want=[v for v in expected if v is not None]
      assert driver.execute(ptx,safe)==want
      bad=[x for x,v in zip(values,expected) if v is None]
      if bad:
        try: driver.execute(ptx,bad[:16])
        except DeviceExecutionError: pass
        else: raise AssertionError('GPU silently accepted overflowing batch')
      continue
    # Host emulation replaces ONLY hardware indexing intrinsics. Arithmetic,
    # checks, memory writes and control flow are exactly the device module.
    for name in ('tid','ctaid','ntid'):
      intrinsic=f'llvm.nvvm.read.ptx.sreg.{name}.x'
      text=text.replace('@'+intrinsic,'@emu.'+name)
      text=text.replace(f'declare i32 @emu.{name}()', '')
    text+='''
@emu.lane = internal global i32 0
define i32 @emu.tid(){ %v = load i32, ptr @emu.lane
 ret i32 %v }
define i32 @emu.ctaid(){ ret i32 0 }
define i32 @emu.ntid(){ ret i32 1 }
define void @set_lane(i32 %v){ store i32 %v, ptr @emu.lane
 ret void }
'''
    mod=h.llvm.parse_assembly(text);mod.verify();mod.triple=h.llvm.get_default_triple();mod.data_layout=str(h.tm.target_data)
    obj=work/'map.o';obj.write_bytes(h.tm.emit_object(mod));shared=work/'map.so'
    p=h.run([h.linker,'-shared',str(obj),'-o',str(shared)]);assert p.returncode==0,p.stderr
    lib=C.CDLL(str(shared));lib.set_lane.argtypes=[C.c_int];lib.set_lane.restype=None
    lib.bt_kernel.argtypes=[C.POINTER(C.c_int64),C.POINTER(C.c_int64),C.POINTER(C.c_int32),C.c_uint64];lib.bt_kernel.restype=None
    n=len(values);src=(C.c_int64*n)(*values);dest=(C.c_int64*n)(*([-123456]*n));errors=(C.c_int32*n)(*([-1]*n))
    for lane in range(n):lib.set_lane(lane);lib.bt_kernel(src,dest,errors,n)
    for i,want in enumerate(expected):
      assert errors[i]==(1 if want is None else 0),(index,i,errors[i],want)
      assert dest[i]==(-123456 if want is None else want),(index,i,dest[i],want)
    for invalid in (-2**63,2**63-1):
      src[0]=invalid;dest[0]=-123456;errors[0]=-1
      lib.set_lane(0);lib.bt_kernel(src,dest,errors,n)
      assert errors[0]==1 and dest[0]==-123456
    # Inactive lanes, including an empty dispatch, cannot touch output buffers.
    before=list(dest),list(errors)
    for lane,count in [(n,n),(0,0)]:
      lib.set_lane(lane);lib.bt_kernel(src,dest,errors,count)
      assert (list(dest),list(errors))==before
if not device:
  for body in ['while(x>0){x=x-1;}return x;','print(x);return x;','return sqrt(x);']:
    with tempfile.TemporaryDirectory() as d:
      src=Path(d)/'bad.bt';src.write_text('func map(x:int)->int{'+body+'}')
      p=h.run([h.compiler,str(src),'--cuda-map=map']);assert p.returncode!=0,p.stdout
print('PASS: '+('real CUDA device execution' if device else 'NVPTX compilation and device-free emulation (NOT GPU qualification)'))
