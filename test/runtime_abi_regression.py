#!/usr/bin/env python3
"""Load the actual built C ABI through checked Python wrappers."""
from pathlib import Path
import ctypes as C
import os
import math
import sys
import threading
os.environ['BT_RUNTIME_LIBRARY']=str(Path(sys.argv[1]).resolve())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ffi/python'))
import bt_ffi as b

def rejects(fn,kind=ValueError):
    try:fn()
    except kind:return
    raise AssertionError('invalid ABI/client input was silently accepted')
assert b._LIB.bt_runtime_abi_version()==1
valid=b.make_word64(1,2);bad=b.BtWord64(p=1,n=1)
rejects(lambda:b.make_word64(1,1));rejects(lambda:b.make_word64(-1,0))
rejects(lambda:b.dot_product(bad,valid));rejects(lambda:b.swar_to_int64(valid,40))
rejects(lambda:b.dot_product_array([valid],[]));rejects(lambda:b.dot_product_array([bad],[valid]))
rejects(lambda:b.kogge_stone_add64(valid,valid,256));rejects(lambda:b.kogge_stone_add64(valid,valid,2))
rejects(lambda:b.base243_pack([0,0,0,0,256]));rejects(lambda:b.base243_pack([0]))
for byte in (-1,243,256):rejects(lambda:b.base243_unpack(byte))
for value in (float('nan'),float('inf')):rejects(lambda:b.fp32_from_float(value))
zero=b.fp32_from_float(0);one=b.fp32_from_float(1)
rejects(lambda:b.fp32_div(one,zero));assert b.fp32_to_float(zero)==0
assert b.dot_product_array([],[])==0
assert not b.verify_disjointness(bad)
# Exhaustive legal base-243 byte roundtrip.
for value in range(243):assert b.base243_pack(b.base243_unpack(value))==value
# Raw null/invalid calls use status, not crashes; wrappers surface the status.
rejects(lambda:b._LIB.bt_base243_pack_val(None))
rejects(lambda:b._LIB.bt_base243_unpack_val(255,None))
assert math.isclose(b.fp32_to_float(one),1.0,rel_tol=1e-12)
failures=[]
def worker(index):
    try:
        if index%2:rejects(lambda:b.fp32_div(one,zero))
        else:assert math.isclose(b.fp32_to_float(one),1.0,rel_tol=1e-12)
    except BaseException as error:failures.append(error)
threads=[threading.Thread(target=worker,args=(i,)) for i in range(20)]
for t in threads:t.start()
for t in threads:t.join()
assert not failures,failures
assert b._LIB.bt_last_error_code()==0
print('PASS: versioned C/Python ABI, propagated errors, no narrowing/truncation, 243 roundtrips and TLS')
