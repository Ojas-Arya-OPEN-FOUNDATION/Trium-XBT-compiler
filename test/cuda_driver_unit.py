#!/usr/bin/env python3
"""Device-free tests of CUDA argument marshaling, ownership and failure cleanup.
The fake C API is explicitly NOT device qualification.
"""
from pathlib import Path
import ctypes as C
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cuda_map import Driver,DeviceExecutionError,H,ptx_from_llvm

def ptrwrite(pointer,value,kind=C.c_void_p):C.cast(pointer,C.POINTER(kind)).contents.value=value
class FakeDriver(Driver):
    def __init__(self,fail=''):
        self.device=C.c_int(0);self.fail=fail;self.memory={};self.events=[];self.current=123;self.allocations=0
    def check(self,status,operation):
        if status:raise DeviceExecutionError(operation)
    def get_current(self,p):ptrwrite(p,self.current);return 0
    def set_current(self,p):self.current=p.value;self.events.append('restore');return 0
    def create(self,p,flags,device):ptrwrite(p,456);self.current=456;self.events.append('create');return 0
    def destroy(self,p):self.current=0;self.memory.clear();self.events.append('destroy');return 0
    def load(self,p,ptx):
        if self.fail=='load':return 1
        ptrwrite(p,789);self.events.append('load');return 0
    def unload(self,p):self.events.append('unload');return 0
    def get_function(self,p,module,name):assert name==b'bt_kernel';ptrwrite(p,999);return 0
    def allocate(self,p,size):
        self.allocations+=1
        if self.fail=='allocation' and self.allocations==2:return 1
        buf=C.create_string_buffer(size);addr=C.addressof(buf);self.memory[addr]=buf;ptrwrite(p,addr,C.c_uint64);return 0
    def free(self,p):self.memory.pop(p.value);self.events.append('free');return 1 if self.fail=='free' else 0
    def htod(self,dest,src,size):C.memmove(dest.value,src,size);return 0
    def dtoh(self,dest,src,size):C.memmove(dest,src.value,size);self.events.append(('copy',size));return 0
    def launch(self,k,gx,gy,gz,bx,by,bz,shared,stream,args,extra):
        assert (gy,gz,bx,by,bz,shared,stream,extra)==(1,1,256,1,1,0,None,None)
        if self.fail=='launch':return 1
        src,out,err,n=[C.cast(v,C.POINTER(C.c_uint64)).contents.value for v in args]
        assert gx==(n+255)//256
        source=C.cast(src,C.POINTER(C.c_int64));dest=C.cast(out,C.POINTER(C.c_int64));errors=C.cast(err,C.POINTER(C.c_int32))
        for i in range(n):
            value=source[i]*source[i]+1;errors[i]=0 if value<=H else 1
            if errors[i]==0:dest[i]=value
        return 0
    def sync(self):return 1 if self.fail=='sync' else 0

fake=FakeDriver();assert fake.execute('fake PTX',[0,1,-2])==[1,2,5]
assert not fake.memory and fake.current==123
for fail in ('load','allocation','launch','sync','free'):
    fake=FakeDriver(fail)
    try:fake.execute('fake PTX',[1,2])
    except DeviceExecutionError:pass
    else:raise AssertionError(fail)
    assert fake.current==123 and not fake.memory,(fail,fake.events)
fake=FakeDriver()
try:fake.execute('fake PTX',[H,1])
except DeviceExecutionError:pass
else:raise AssertionError('arithmetic failure was not propagated')
assert ('copy',8) in fake.events and ('copy',16) not in fake.events,fake.events
for values in ([H+1],[True],[1.5]):
    fake=FakeDriver()
    try:fake.execute('fake PTX',values)
    except ValueError:pass
    else:raise AssertionError(values)
    assert not fake.events
assert FakeDriver().execute('fake PTX',[])==[]
for arch in ['sm_999','sm_00','sm_50;injected',None]:
    try:ptx_from_llvm('',arch)
    except ValueError:pass
    else:raise AssertionError(arch)
print('PASS: CUDA C-argument marshaling, transactional results, cleanup and context restoration (fake driver)')
