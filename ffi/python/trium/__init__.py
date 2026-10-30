"""Synchronous Trium SDK v1. Explicit backend, no hidden GPU/CPU fallback.
Set BT_SDK_LIBRARY to an absolute trusted libtrium_sdk path for source builds.
"""
import ctypes as C
import os
from pathlib import Path
import json
import numbers
from itertools import islice
from dataclasses import dataclass

class Error(RuntimeError):
    def __init__(self,code,message): self.code=code;super().__init__(message)
class _Options(C.Structure):
    _fields_=[('struct_size',C.c_uint32),('abi_major',C.c_uint32),('max_elements',C.c_uint64),('operation_budget',C.c_uint64),('device_index',C.c_uint32),('workgroup',C.c_uint32),('representation',C.c_uint32),('reserved',C.c_uint32)]
class _Value(C.Structure):
    _fields_=[('kind',C.c_uint32),('reserved',C.c_uint32),('integer',C.c_int64),('floating',C.c_double)]
class _Float(C.Structure):
    _fields_=[('width',C.c_uint32),('reserved',C.c_uint32),('mantissa',C.c_int64),('exponent',C.c_int32),('reserved2',C.c_uint32)]
@dataclass(frozen=True)
class TernaryFloat:
    """Finite normalized value mantissa * 3**exponent. Not a binary Python float."""
    width:int
    mantissa:int
    exponent:int
    def _record(self):
        if type(self.width)is not int or not 1<=self.width<=40 or type(self.exponent)is not int or not -(2**31)<=self.exponent<2**31:raise ValueError('invalid BTFP profile')
        return _Float(self.width,0,_integer(self.mantissa),self.exponent,0)
    @classmethod
    def _from_record(cls,r):return cls(r.width,r.mantissa,r.exponent)
class Metrics(C.Structure):
    _fields_=[('struct_size',C.c_uint32),('backend',C.c_uint32)]+[(n,C.c_double) for n in ('compile_ms','upload_ms','execute_ms','download_ms','total_ms','kernel_ms','gpu_upload_ms','gpu_download_ms')]+[('workgroup',C.c_uint32),('representation',C.c_uint32),('device_uuid',C.c_uint8*16)]
    def as_dict(self):return {name:(bytes(getattr(self,name)).hex() if name=='device_uuid' else getattr(self,name)) for name,_ in self._fields_}

def _library(path=None):
    path=path or os.environ.get('BT_SDK_LIBRARY') or str(Path(__file__).parent/'libtrium_sdk.so')
    path=Path(path)
    if not path.is_absolute():raise ValueError('SDK native library must have an absolute trusted path')
    lib=C.CDLL(str(path));lib.bt_sdk_abi_version.restype=C.c_uint32
    if lib.bt_sdk_abi_version()!=1:raise RuntimeError('unsupported SDK ABI')
    if hasattr(lib,'bt_sdk_api_version'):
        lib.bt_sdk_api_version.restype=C.c_uint32;lib.bt_sdk_api_version.argtypes=[]
        lib._api_version=lib.bt_sdk_api_version()
    else:lib._api_version=0x10000
    if lib._api_version>>16!=1:raise RuntimeError('unsupported SDK API major')
    h=C.c_uint64;err=[C.c_void_p,C.c_size_t]
    signatures={
     'float_from_ratio':[C.c_uint32,C.c_int64,C.c_int64,C.c_int64,C.POINTER(_Float)],
     'float_from_double':[C.c_uint32,C.c_double,C.POINTER(_Float)],
     'float_to_double':[C.POINTER(_Float),C.POINTER(C.c_double)],
     'float_apply':[C.c_uint32,C.POINTER(_Float),C.POINTER(_Float),C.POINTER(_Float),C.POINTER(_Float)],
     'context_create':[C.POINTER(_Options),C.POINTER(h)],'release':[h],
     'compile':[h,C.c_char_p,C.c_size_t,C.POINTER(h)],'exports':[h,C.c_void_p,C.c_size_t],
     'invoke':[h,C.c_char_p,C.POINTER(_Value),C.c_uint32,C.POINTER(_Value)],
     'map':[h,C.c_char_p,C.c_uint32,C.POINTER(C.c_int64),h,C.POINTER(C.c_int64),C.POINTER(Metrics)],
     'shader':[h,C.c_char_p,C.c_uint32,C.c_void_p,C.POINTER(C.c_size_t)],
     'devices':[C.c_void_p,C.c_size_t],
     'dual_encode':[C.c_int64,C.c_uint32,C.POINTER(h),C.POINTER(h)],
     'dual_decode':[h,h,C.c_uint32,C.POINTER(C.c_int64)],
     'matmul':[C.POINTER(C.c_int64),h,C.POINTER(C.c_int64),h,C.POINTER(C.c_int64),h,h,h,h],
    }
    if lib._api_version>=0x10001:
        signatures.update({'info':[C.c_void_p,C.POINTER(C.c_size_t)],'module_info':[h,C.c_void_p,C.POINTER(C.c_size_t)],'cache_info':[h,C.c_void_p,C.POINTER(C.c_size_t)],'cache_clear':[h],
          'invoke_ex':[h,C.c_char_p,C.c_uint32,C.POINTER(_Value),C.c_uint32,C.POINTER(_Value),C.POINTER(Metrics)]})
    for name,args in signatures.items():
        fn=getattr(lib,'bt_sdk_'+name);fn.argtypes=args+err;fn.restype=C.c_int32
    return lib

def _check(lib,name,*args):
    error=C.create_string_buffer(4096);code=getattr(lib,'bt_sdk_'+name)(*args,error,len(error))
    if code:raise Error(code,error.value.decode('utf-8','replace'))

def _query(lib,name,*args):
    if lib._api_version<0x10001:raise Error(4,'this feature requires SDK API 1.1')
    size=C.c_size_t();_check(lib,name,*args,None,C.byref(size))
    for _ in range(4):
        if not 0<size.value<=16*1024*1024:raise Error(6,'SDK metadata capacity exceeded')
        out=C.create_string_buffer(size.value);capacity=size.value
        try:_check(lib,name,*args,out,C.byref(size))
        except Error as e:
            if e.code==1 and size.value>capacity:continue
            raise
        return json.loads(out.value)
    raise Error(6,'metadata changed repeatedly; retry query')

def _backend(name):
    if name not in ('cpu','vulkan','native'):raise ValueError('backend must be cpu, vulkan or native; no implicit fallback')
    return {'cpu':0,'vulkan':1,'native':2}[name]

def _integer(n):
    if not isinstance(n,numbers.Integral) or isinstance(n,bool):raise ValueError('expected exact signed int64 input')
    value=int(n)
    if not -(2**63)<=value<2**63:raise ValueError('expected exact signed int64 input')
    return value
def _name(name):
    data=name.encode('utf-8')
    if b'\0' in data:raise ValueError('NUL in export name')
    return data

class Context:
    def __init__(self,library=None,*,device=0,workgroup=64,representation='portable32',max_elements=1000000,operation_budget=10000000):
        if not 0<=device<2**32 or workgroup not in (32,64,128) or not 0<max_elements<=1000000 or not 0<operation_budget<=100000000:raise ValueError('invalid context options')
        if representation not in ('portable32','native64','auto'):raise ValueError('invalid representation')
        self.lib=_library(library);self.handle=C.c_uint64()
        options=_Options(C.sizeof(_Options),1,max_elements,operation_budget,device,workgroup,{'portable32':0,'native64':1,'auto':2}[representation],0)
        if self.lib._api_version>=0x10001:
            info=_query(self.lib,'info')
            if info['abi_major']!=1 or info['sizes']!={'options':C.sizeof(_Options),'metrics':C.sizeof(Metrics),'value':C.sizeof(_Value),'float_value':C.sizeof(_Float)}:raise RuntimeError('SDK ABI layout mismatch')
        _check(self.lib,'context_create',C.byref(options),C.byref(self.handle))
    @property
    def info(self):return _query(self.lib,'info')
    @property
    def cache_info(self):return _query(self.lib,'cache_info',self.handle)
    def clear_cache(self):
        if self.lib._api_version<0x10001:raise Error(4,'cache control requires API 1.1')
        _check(self.lib,'cache_clear',self.handle)
    def compile(self,source):
        data=source.encode('utf-8');handle=C.c_uint64()
        _check(self.lib,'compile',self.handle,data,len(data),C.byref(handle));return Module(self,handle)
    def devices(self):
        out=C.create_string_buffer(65536);_check(self.lib,'devices',out,len(out));return json.loads(out.value)
    def encode(self,value,width=64):
        if not 1<=width<=64:raise ValueError('width must be 1..64')
        p=C.c_uint64();n=C.c_uint64();_check(self.lib,'dual_encode',_integer(value),width,C.byref(p),C.byref(n));return p.value,n.value
    def decode(self,p,n,width=64):
        if not all(type(x)is int and 0<=x<2**64 for x in (p,n)) or not 1<=width<=64:raise ValueError('invalid rails/width')
        value=C.c_int64();_check(self.lib,'dual_decode',p,n,width,C.byref(value));return value.value
    def float_from_ratio(self,n,d=1,*,width=20,exponent=0):
        if type(width)is not int or not 1<=width<=40:raise ValueError('width must be 1..40')
        out=_Float();_check(self.lib,'float_from_ratio',width,_integer(n),_integer(d),_integer(exponent),C.byref(out));return TernaryFloat._from_record(out)
    def float_from_double(self,value,*,width=20):
        if type(value)is not float or type(width)is not int or not 1<=width<=40:raise ValueError('expected binary64 float and width 1..40')
        out=_Float();_check(self.lib,'float_from_double',width,value,C.byref(out));return TernaryFloat._from_record(out)
    def float_to_double(self,value):
        record=value._record();out=C.c_double();_check(self.lib,'float_to_double',C.byref(record),C.byref(out));return out.value
    def float_apply(self,operation,a,b=None,c=None):
        operations={'add':1,'sub':2,'mul':3,'div':4,'sqrt':5,'fma':6}
        if operation not in operations:raise ValueError('unknown BTFP operation')
        records=[v._record() if v is not None else None for v in (a,b,c)];out=_Float()
        _check(self.lib,'float_apply',operations[operation],*[C.byref(v) if v is not None else None for v in records],C.byref(out));return TernaryFloat._from_record(out)
    def matmul(self,a,b,m,k,n):
        if any(type(x)is not int or x<0 or x>1000000 for x in (m,k,n)) or m*k!=len(a) or k*n!=len(b) or m*n>1000000:raise ValueError('matrix shapes/storage mismatch')
        av=(C.c_int64*len(a))(*map(_integer,a));bv=(C.c_int64*len(b))(*map(_integer,b));out=(C.c_int64*(m*n))()
        _check(self.lib,'matmul',av,len(a),bv,len(b),out,m*n,m,k,n);return list(out)
    def close(self):
        if getattr(self,'handle',None) and self.handle.value:_check(self.lib,'release',self.handle);self.handle.value=0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass

class Module:
    def __init__(self,context,handle):self.context=context;self.lib=context.lib;self.handle=handle
    @property
    def exports(self):
        if self.lib._api_version>=0x10001:return [f['name'] for f in self.info['functions']]
        out=C.create_string_buffer(1024*1024);_check(self.lib,'exports',self.handle,out,len(out));return out.value.decode().splitlines()
    @property
    def info(self):return _query(self.lib,'module_info',self.handle)
    def invoke(self,name,*args,backend='cpu',with_metrics=False):
        values=(_Value*len(args))()
        for i,x in enumerate(args):
            if type(x)is float:values[i]=_Value(2,0,0,x)
            else:values[i]=_Value(1,0,_integer(x),0)
        output=_Value();metrics=Metrics();metrics.struct_size=C.sizeof(metrics);selected=_backend(backend)
        if self.lib._api_version>=0x10001:_check(self.lib,'invoke_ex',self.handle,_name(name),selected,values,len(args),C.byref(output),C.byref(metrics))
        else:
            if selected or with_metrics:raise Error(4,'native execution and metrics require API 1.1')
            _check(self.lib,'invoke',self.handle,_name(name),values,len(args),C.byref(output))
        result=output.integer if output.kind==1 else output.floating if output.kind==2 else None
        return (result,metrics.as_dict()) if with_metrics else result
    def map(self,name,values,*,backend='cpu'):
        values=list(islice(values,1000001))
        if len(values)>1000000:raise ValueError('batch limit exceeded')
        data=(C.c_int64*len(values))(*map(_integer,values));out=(C.c_int64*len(values))()
        metrics=self._map(name,data,len(values),out,backend);return list(out),metrics
    def map_array(self,name,values,*,backend='cpu'):
        import numpy as np
        if not isinstance(values,np.ndarray) or values.dtype!=np.dtype('int64') or values.ndim!=1 or not values.flags.c_contiguous or not values.flags.aligned:raise ValueError('expected aligned native-endian contiguous rank-1 numpy int64 array')
        if len(values)>1000000:raise ValueError('batch limit exceeded')
        output=np.empty_like(values)
        metrics=self._map(name,values.ctypes.data_as(C.POINTER(C.c_int64)),len(values),output.ctypes.data_as(C.POINTER(C.c_int64)),backend)
        return output,metrics
    def _map(self,name,data,count,out,backend):
        selected=_backend(backend)
        if selected==2 and self.lib._api_version<0x10001:raise Error(4,'native execution requires API 1.1')
        metrics=Metrics();metrics.struct_size=C.sizeof(metrics)
        _check(self.lib,'map',self.handle,_name(name),selected,data,count,out,C.byref(metrics));return metrics.as_dict()
    def shader(self,name,workgroup=64):
        size=C.c_size_t();_check(self.lib,'shader',self.handle,_name(name),workgroup,None,C.byref(size));out=C.create_string_buffer(size.value)
        _check(self.lib,'shader',self.handle,_name(name),workgroup,out,C.byref(size));return out.value.decode()
    def close(self):
        if self.handle.value:_check(self.lib,'release',self.handle);self.handle.value=0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
