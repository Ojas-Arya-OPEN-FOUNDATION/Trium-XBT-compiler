#!/usr/bin/env python3
"""Real CUDA-driver execution for the verified core int39 map profile.
Offline PTX emission and CPU emulation are not device qualification.
No CUDA toolkit dependency: LLVM emits PTX; a real NVIDIA driver/device is needed
for execute(). Driver failures propagate, and failed batches return no partial data.
"""
import ctypes as C
import ctypes.util

H=(3**39-1)//2
class DeviceUnavailable(RuntimeError): pass
class DeviceExecutionError(RuntimeError): pass

# Base architectures recognized by the pinned LLVM 22 NVPTX backend.
# This is code-emission support, not physical-device qualification.
CUDA_ARCHITECTURES={f'sm_{n}' for n in (50,52,53,60,61,62,70,72,75,80,86,87,88,89,90,100,101,103,110,120,121)}
def ptx_from_llvm(text,arch='sm_50'):
    if not isinstance(arch,str) or arch not in CUDA_ARCHITECTURES:
        raise ValueError('unsupported CUDA base architecture for the pinned LLVM backend')
    from llvmlite import binding as llvm
    llvm.initialize_all_targets();llvm.initialize_all_asmprinters()
    module=llvm.parse_assembly(text);module.verify()
    if module.triple!='nvptx64-nvidia-cuda': raise ValueError('expected CUDA LLVM module')
    tm=llvm.Target.from_triple(module.triple).create_target_machine(cpu=arch,opt=3)
    module.data_layout=str(tm.target_data)
    return tm.emit_assembly(module)

class Driver:
    def __init__(self,device=0):
        try: self.lib=C.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        except OSError as error: raise DeviceUnavailable('CUDA driver library is not available') from error
        self.init=self.bind('cuInit',[C.c_uint])
        status=self.init(0)
        if status in (35,100): raise DeviceUnavailable(f'CUDA device/driver unavailable ({status})')
        self.check(status,'cuInit')
        count=C.c_int();self.check(self.bind('cuDeviceGetCount',[C.POINTER(C.c_int)])(C.byref(count)),'device count')
        if count.value==0: raise DeviceUnavailable('no CUDA devices')
        if not isinstance(device,int) or not 0<=device<count.value: raise ValueError('invalid CUDA device ordinal')
        self.device=C.c_int();self.check(self.bind('cuDeviceGet',[C.POINTER(C.c_int),C.c_int])(C.byref(self.device),device),'device selection')
        self.get_current=self.bind('cuCtxGetCurrent',[C.POINTER(C.c_void_p)])
        self.set_current=self.bind('cuCtxSetCurrent',[C.c_void_p])
        self.create=self.bind('cuCtxCreate_v2',[C.POINTER(C.c_void_p),C.c_uint,C.c_int])
        self.destroy=self.bind('cuCtxDestroy_v2',[C.c_void_p])
        self.allocate=self.bind('cuMemAlloc_v2',[C.POINTER(C.c_uint64),C.c_size_t])
        self.free=self.bind('cuMemFree_v2',[C.c_uint64])
        self.htod=self.bind('cuMemcpyHtoD_v2',[C.c_uint64,C.c_void_p,C.c_size_t])
        self.dtoh=self.bind('cuMemcpyDtoH_v2',[C.c_void_p,C.c_uint64,C.c_size_t])
        self.load=self.bind('cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
        self.unload=self.bind('cuModuleUnload',[C.c_void_p])
        self.get_function=self.bind('cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self.launch=self.bind('cuLaunchKernel',[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)])
        self.sync=self.bind('cuCtxSynchronize',[])
    def bind(self,name,args):
        fn=getattr(self.lib,name);fn.restype=C.c_int;fn.argtypes=args;return fn
    def check(self,status,operation):
        if status:
            text=C.c_char_p()
            try:
                self.bind('cuGetErrorString',[C.c_int,C.POINTER(C.c_char_p)])(status,C.byref(text))
                detail=text.value.decode('utf8','replace') if text.value else str(status)
            except AttributeError: detail=str(status)
            raise DeviceExecutionError(f'{operation}: CUDA error {status}: {detail}')
    def architecture(self):
        attr=self.bind('cuDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int]);major=C.c_int();minor=C.c_int()
        self.check(attr(C.byref(major),75,self.device),'compute capability major')
        self.check(attr(C.byref(minor),76,self.device),'compute capability minor')
        return f'sm_{major.value}{minor.value}'
    def execute(self,ptx,values):
        checked=[]
        for value in values:
            if len(checked)>=1000000: raise ValueError('CUDA map batch exceeds 1,000,000 elements; split into chunks')
            if type(value) is not int or not -H<=value<=H: raise ValueError('CUDA map input outside int39 domain')
            checked.append(value)
        values=checked
        if not values: return []
        count=len(values);context=C.c_void_p();previous=C.c_void_p();module=C.c_void_p();buffers=[]
        self.check(self.get_current(C.byref(previous)),'save context')
        self.check(self.create(C.byref(context),0,self.device),'create context')
        # Preserve the original failure during cleanup; report cleanup failures on
        # an otherwise successful batch. Do not leak the caller's current context.
        failure=None;cleanup_errors=[]
        try:
            ptx_buffer=C.create_string_buffer(ptx.encode())
            self.check(self.load(C.byref(module),ptx_buffer),'load generated PTX')
            kernel=C.c_void_p();self.check(self.get_function(C.byref(kernel),module,b'bt_kernel'),'resolve kernel')
            for size in (8*count,8*count,4*count):
                ptr=C.c_uint64();self.check(self.allocate(C.byref(ptr),size),'allocate device buffer');buffers.append(ptr)
            src=(C.c_int64*count)(*values);errors=(C.c_int32*count)(*([-1]*count));dest=(C.c_int64*count)()
            self.check(self.htod(buffers[0],src,C.sizeof(src)),'copy input')
            self.check(self.htod(buffers[2],errors,C.sizeof(errors)),'initialize error buffer')
            n=C.c_uint64(count);arguments=(C.c_void_p*4)(*[C.cast(C.byref(v),C.c_void_p) for v in (*buffers,n)])
            self.check(self.launch(kernel,(count+255)//256,1,1,256,1,1,0,None,arguments,None),'launch')
            self.check(self.sync(),'synchronize')
            self.check(self.dtoh(errors,buffers[2],C.sizeof(errors)),'copy lane errors')
            bad=[i for i,status in enumerate(errors) if status!=0]
            if bad: raise DeviceExecutionError(f'checked arithmetic failed at lanes {bad[:16]} (total {len(bad)})')
            self.check(self.dtoh(dest,buffers[1],C.sizeof(dest)),'copy results')
            result=list(dest)
        except BaseException as error:
            failure=error
        finally:
            for ptr in reversed(buffers):
                status=self.free(ptr)
                if status: cleanup_errors.append(('free buffer',status))
            if module.value:
                status=self.unload(module)
                if status: cleanup_errors.append(('unload module',status))
            status=self.destroy(context)
            if status: cleanup_errors.append(('destroy context',status))
            status=self.set_current(previous)
            if status: cleanup_errors.append(('restore context',status))
        if failure is not None: raise failure
        if cleanup_errors: raise DeviceExecutionError(f'CUDA cleanup failed: {cleanup_errors}')
        return result
