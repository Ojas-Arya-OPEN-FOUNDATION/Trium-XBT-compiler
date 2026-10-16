#!/usr/bin/env python3
"""
Balanced Ternary System - High-Performance Python ctypes FFI Module
Compatible with Python 3.8+ and libbt_runtime.dylib (or .so / .dll).
"""

import os
import sys
import math
import ctypes
from typing import List, Tuple, Optional

# ============================================================================
# 1. Dynamic Shared Runtime Library Loader
# ============================================================================

def _find_and_load_library() -> ctypes.CDLL:
    lib_names = ["libbt_runtime.dylib", "libbt_runtime.so", "bt_runtime.dll"]
    search_paths = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
        "/usr/local/lib",
        "/usr/lib",
    ]

    for path in search_paths:
        for name in lib_names:
            full_path = os.path.abspath(os.path.join(path, name))
            if os.path.exists(full_path):
                try:
                    return ctypes.CDLL(full_path)
                except OSError as e:
                    pass

    # Fallback to system linker search
    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

    raise FileNotFoundError(
        "Could not load libbt_runtime.dylib. Please compile it first using: "
        "clang++ -std=c++20 -O3 -shared -fPIC lib/Ternary/ternary_c_abi.cpp -I. -o libbt_runtime.dylib"
    )

_LIB = _find_and_load_library()

# ============================================================================
# 2. ctypes Structure Definitions (Standard C ABI Layout)
# ============================================================================

class BtWord64(ctypes.Structure):
    """64-trit dual-rail word representation. Invariant: (p & n) == 0."""
    _fields_ = [
        ("p", ctypes.c_uint64),
        ("n", ctypes.c_uint64),
    ]

    def __repr__(self) -> str:
        return f"BtWord64(p=0x{self.p:016X}, n=0x{self.n:016X})"

class BtFp32(ctypes.Structure):
    """Balanced Ternary Floating-Point 32-bit equivalent (mantissa * 3^exponent)."""
    _fields_ = [
        ("mantissa", ctypes.c_int64),
        ("exponent", ctypes.c_int32),
    ]

    def __repr__(self) -> str:
        val = fp32_to_float(self)
        return f"BtFp32(mantissa={self.mantissa}, exponent={self.exponent}, approx={val:.7g})"

class BtCsr(ctypes.Structure):
    """T-CSR 1-Bit Compressed Sparse Row Matrix representation."""
    _fields_ = [
        ("rows", ctypes.c_uint64),
        ("cols", ctypes.c_uint64),
        ("nnz", ctypes.c_uint64),
        ("row_ptr", ctypes.POINTER(ctypes.c_uint32)),
        ("col_indices", ctypes.POINTER(ctypes.c_uint32)),
        ("signs", ctypes.POINTER(ctypes.c_uint64)),
    ]

class BtBthaResult(ctypes.Structure):
    """Balanced Ternary Half-Adder (BTHA) output."""
    _fields_ = [
        ("sum", BtWord64),
        ("carry", BtWord64),
    ]

# ============================================================================
# 3. Function Signatures Configuration
# ============================================================================

_LIB.bt_swar_word64_to_int64.argtypes = [BtWord64, ctypes.c_size_t]
_LIB.bt_swar_word64_to_int64.restype = ctypes.c_int64

_LIB.bt_dot_product_array.argtypes = [
    ctypes.POINTER(BtWord64),
    ctypes.POINTER(BtWord64),
    ctypes.c_size_t,
]
_LIB.bt_dot_product_array.restype = ctypes.c_int64

_LIB.bt_fp32_from_double.argtypes = [ctypes.c_double]
_LIB.bt_fp32_from_double.restype = BtFp32

_LIB.bt_fp32_to_double.argtypes = [BtFp32]
_LIB.bt_fp32_to_double.restype = ctypes.c_double

_LIB.bt_fp32_add.argtypes = [BtFp32, BtFp32]
_LIB.bt_fp32_add.restype = BtFp32

_LIB.bt_fp32_sub.argtypes = [BtFp32, BtFp32]
_LIB.bt_fp32_sub.restype = BtFp32

_LIB.bt_fp32_mul.argtypes = [BtFp32, BtFp32]
_LIB.bt_fp32_mul.restype = BtFp32

_LIB.bt_fp32_div.argtypes = [BtFp32, BtFp32]
_LIB.bt_fp32_div.restype = BtFp32

_LIB.bt_csr_spmv_exec.argtypes = [
    ctypes.POINTER(BtCsr),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
]
_LIB.bt_csr_spmv_exec.restype = None

_LIB.bt_radix3_fft_forward.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
_LIB.bt_radix3_fft_forward.restype = ctypes.c_int

_LIB.bt_radix3_fft_inverse.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
]
_LIB.bt_radix3_fft_inverse.restype = ctypes.c_int

_LIB.bt_dot_product_64_val.argtypes = [BtWord64, BtWord64]
_LIB.bt_dot_product_64_val.restype = ctypes.c_int64

_LIB.bt_btha_step_val.argtypes = [BtWord64, BtWord64]
_LIB.bt_btha_step_val.restype = BtBthaResult

_LIB.bt_base243_pack_val.argtypes = [ctypes.POINTER(ctypes.c_int8)]
_LIB.bt_base243_pack_val.restype = ctypes.c_uint8

_LIB.bt_base243_unpack_val.argtypes = [ctypes.c_uint8, ctypes.POINTER(ctypes.c_int8)]
_LIB.bt_base243_unpack_val.restype = None

_LIB.bt_hamming_distance_64_val.argtypes = [BtWord64, BtWord64]
_LIB.bt_hamming_distance_64_val.restype = ctypes.c_uint32

_LIB.bt_verify_disjointness_val.argtypes = [BtWord64]
_LIB.bt_verify_disjointness_val.restype = ctypes.c_bool

# ============================================================================
# 4. Safe Pythonic Wrapper Functions
# ============================================================================

def make_word64(p: int, n: int) -> BtWord64:
    """Creates a 64-trit dual-rail word."""
    return BtWord64(p=p, n=n)

def verify_disjointness(w: BtWord64) -> bool:
    """Verifies dual-rail disjointness: (p & n) == 0."""
    return bool(_LIB.bt_verify_disjointness_val(w))

def swar_to_int64(w: BtWord64, num_trits: int = 39) -> int:
    """Converts up to 39 packed trits to a 64-bit signed integer via Douglas W. Jones' SWAR reduction."""
    return int(_LIB.bt_swar_word64_to_int64(w, num_trits))

def dot_product(a: BtWord64, b: BtWord64) -> int:
    """Computes 64-trit bit-sliced dot product using dual-rail popcount contraction."""
    return int(_LIB.bt_dot_product_64_val(a, b))

def dot_product_array(a_list: List[BtWord64], b_list: List[BtWord64]) -> int:
    """Computes dot product across arrays of 64-trit dual-rail words."""
    length = min(len(a_list), len(b_list))
    if length == 0:
        return 0
    arr_a = (BtWord64 * length)(*a_list[:length])
    arr_b = (BtWord64 * length)(*b_list[:length])
    return int(_LIB.bt_dot_product_array(arr_a, arr_b, length))

def btha_step(a: BtWord64, b: BtWord64) -> Tuple[BtWord64, BtWord64]:
    """Balanced Ternary Half-Adder step returning (sum, carry)."""
    res = _LIB.bt_btha_step_val(a, b)
    return res.sum, res.carry

def hamming_distance(a: BtWord64, b: BtWord64) -> int:
    """Counts differing trits between two 64-trit words."""
    return int(_LIB.bt_hamming_distance_64_val(a, b))

def fp32_from_float(val: float) -> BtFp32:
    """Constructs a BT-FP32 from a Python float."""
    return _LIB.bt_fp32_from_double(ctypes.c_double(val))

def fp32_to_float(a: BtFp32) -> float:
    """Converts a BT-FP32 to a Python float."""
    return float(_LIB.bt_fp32_to_double(a))

def fp32_add(a: BtFp32, b: BtFp32) -> BtFp32:
    """Balanced ternary float addition with 0-bias truncation rounding."""
    return _LIB.bt_fp32_add(a, b)

def fp32_sub(a: BtFp32, b: BtFp32) -> BtFp32:
    """Balanced ternary float subtraction."""
    return _LIB.bt_fp32_sub(a, b)

def fp32_mul(a: BtFp32, b: BtFp32) -> BtFp32:
    """Balanced ternary float multiplication."""
    return _LIB.bt_fp32_mul(a, b)

def fp32_div(a: BtFp32, b: BtFp32) -> BtFp32:
    """Balanced ternary float division."""
    return _LIB.bt_fp32_div(a, b)

def csr_spmv(
    rows: int,
    cols: int,
    row_ptr: List[int],
    col_indices: List[int],
    signs: List[int],
    x: List[int],
) -> List[int]:
    """Executes multiplier-free SpMV: y = mat * x using T-CSR bit-packed signs."""
    assert len(x) >= cols, "Vector x length must be >= cols"
    assert len(row_ptr) == rows + 1, "row_ptr length must be rows + 1"
    nnz = len(col_indices)

    c_row_ptr = (ctypes.c_uint32 * len(row_ptr))(*row_ptr)
    c_col_indices = (ctypes.c_uint32 * len(col_indices))(*col_indices) if nnz > 0 else None
    c_signs = (ctypes.c_uint64 * len(signs))(*signs) if len(signs) > 0 else None

    mat = BtCsr(
        rows=rows,
        cols=cols,
        nnz=nnz,
        row_ptr=c_row_ptr,
        col_indices=c_col_indices,
        signs=c_signs,
    )

    c_x = (ctypes.c_int64 * len(x))(*x)
    c_y = (ctypes.c_int64 * rows)()

    _LIB.bt_csr_spmv_exec(ctypes.byref(mat), c_x, c_y)
    return [c_y[i] for i in range(rows)]

def radix3_fft(
    real: List[float],
    imag: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """Forward Radix-3 Cooley-Tukey FFT for sequence length N = 3^k."""
    n = len(real)
    if n == 0:
        return [], []
    c_real_in = (ctypes.c_double * n)(*real)
    c_imag_in = (ctypes.c_double * n)(*(imag if imag is not None else [0.0] * n))
    c_real_out = (ctypes.c_double * n)()
    c_imag_out = (ctypes.c_double * n)()

    err = _LIB.bt_radix3_fft_forward(c_real_in, c_imag_in, c_real_out, c_imag_out, n)
    if err != 0:
        raise ValueError(f"Radix-3 FFT forward error code: {err} (length {n} must be power of 3)")

    return [c_real_out[i] for i in range(n)], [c_imag_out[i] for i in range(n)]

def radix3_ifft(
    real: List[float],
    imag: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """Inverse Radix-3 Cooley-Tukey FFT with 1/N normalization for N = 3^k."""
    n = len(real)
    if n == 0:
        return [], []
    c_real_in = (ctypes.c_double * n)(*real)
    c_imag_in = (ctypes.c_double * n)(*(imag if imag is not None else [0.0] * n))
    c_real_out = (ctypes.c_double * n)()
    c_imag_out = (ctypes.c_double * n)()

    err = _LIB.bt_radix3_fft_inverse(c_real_in, c_imag_in, c_real_out, c_imag_out, n)
    if err != 0:
        raise ValueError(f"Radix-3 FFT inverse error code: {err} (length {n} must be power of 3)")

    return [c_real_out[i] for i in range(n)], [c_imag_out[i] for i in range(n)]

def base243_pack(trits: List[int]) -> int:
    """Packs 5 balanced trits into 1 byte [0..242]."""
    assert len(trits) == 5
    c_trits = (ctypes.c_int8 * 5)(*trits)
    return int(_LIB.bt_base243_pack_val(c_trits))

def base243_unpack(byte_val: int) -> List[int]:
    """Unpacks 1 byte [0..242] into 5 balanced trits."""
    c_trits = (ctypes.c_int8 * 5)()
    _LIB.bt_base243_unpack_val(ctypes.c_uint8(byte_val), c_trits)
    return [c_trits[i] for i in range(5)]

# ============================================================================
# 5. Verification Suite & CLI Runner
# ============================================================================

def run_self_test():
    print("=" * 70)
    print("  [Python ctypes FFI] Balanced Ternary Shared Runtime Self-Test")
    print("=" * 70)

    # 1. Dual-Rail Word and Disjointness
    w1 = make_word64(0x5555555555555555, 0xAAAAAAAAAAAAAAAA)
    assert verify_disjointness(w1), "w1 disjointness failed"
    w_bad = make_word64(0x3, 0x1)
    assert not verify_disjointness(w_bad), "w_bad disjointness check failed"
    print("  [FFI 1/7] Dual-Rail Word & Disjointness Invariant ... PASSED")

    # 2. SWAR Tree Reduction
    assert swar_to_int64(make_word64(0, 0), 10) == 0
    assert swar_to_int64(make_word64(1, 0), 1) == 1
    assert swar_to_int64(make_word64(0, 1), 1) == -1
    assert swar_to_int64(make_word64(2, 0), 2) == 3   # trit 0=0, trit 1=+1 -> 3
    assert swar_to_int64(make_word64(3, 0), 2) == 4   # trit 0=+1, trit 1=+1 -> 4
    assert swar_to_int64(make_word64(0, 3), 2) == -4  # trit 0=-1, trit 1=-1 -> -4
    print("  [FFI 2/7] Douglas W. Jones SWAR Tree Reduction ....... PASSED")

    # 3. Bit-Sliced Dot Product
    # a: trits [1, 1, -1, -1]
    # b: trits [1, -1, 1, -1]
    # products: 1*1 + 1*(-1) + (-1)*1 + (-1)*(-1) = 1 - 1 - 1 + 1 = 0
    a = make_word64(0b0011, 0b1100)
    b = make_word64(0b0101, 0b1010)
    dot = dot_product(a, b)
    assert dot == 0, f"Expected dot product 0, got {dot}"

    # Array dot product
    arr_dot = dot_product_array([a, make_word64(1, 0)], [b, make_word64(1, 0)])
    assert arr_dot == 1, f"Expected array dot product 1, got {arr_dot}"
    print("  [FFI 3/7] Bit-Sliced Popcount Dot Product ........... PASSED")

    # 4. Balanced Ternary Half-Adder (BTHA)
    s, c = btha_step(make_word64(1, 0), make_word64(1, 0)) # 1 + 1 = 2 = 3*(+1) + (-1)
    assert s.p == 0 and s.n == 1, f"Sum trit expected -1, got p={s.p}, n={s.n}"
    assert c.p == 1 and c.n == 0, f"Carry trit expected +1, got p={c.p}, n={c.n}"
    print("  [FFI 4/7] Branchless 6-Op BTHA Half-Adder ........... PASSED")

    # 5. BT-FP32 Floating-Point Arithmetic
    fa = fp32_from_float(13.5)
    fb = fp32_from_float(4.5)
    f_sum = fp32_add(fa, fb)
    f_sub = fp32_sub(fa, fb)
    f_mul = fp32_mul(fa, fb)
    f_div = fp32_div(fa, fb)

    assert abs(fp32_to_float(f_sum) - 18.0) < 1e-4, f"Sum error: {fp32_to_float(f_sum)}"
    assert abs(fp32_to_float(f_sub) - 9.0) < 1e-4, f"Sub error: {fp32_to_float(f_sub)}"
    assert abs(fp32_to_float(f_mul) - 60.75) < 1e-3, f"Mul error: {fp32_to_float(f_mul)}"
    assert abs(fp32_to_float(f_div) - 3.0) < 1e-4, f"Div error: {fp32_to_float(f_div)}"
    print("  [FFI 5/7] BT-FP32 Arithmetic (Add/Sub/Mul/Div) ...... PASSED")

    # 6. Multiplier-Free T-CSR SpMV
    # 3x3 matrix:
    # Row 0: (+1 at 0), (-1 at 2)
    # Row 1: (+1 at 1)
    # Row 2: (-1 at 0), (+1 at 1), (+1 at 2)
    # x = [10, 20, 30]
    # y0 = 10 - 30 = -20
    # y1 = 20
    # y2 = -10 + 20 + 30 = 40
    row_ptr = [0, 2, 3, 6]
    col_indices = [0, 2, 1, 0, 1, 2]
    # signs bit-packed: idx 0=+1(1), 1=-1(0), 2=+1(1), 3=-1(0), 4=+1(1), 5=+1(1) -> 0b110101 = 0x35
    signs = [0x35]
    x_vec = [10, 20, 30]
    y_res = csr_spmv(3, 3, row_ptr, col_indices, signs, x_vec)
    assert y_res == [-20, 20, 40], f"SpMV result mismatch: {y_res}"
    print("  [FFI 6/7] Multiplier-Free T-CSR 1-Bit SpMV .......... PASSED")

    # 7. Radix-3 Cooley-Tukey FFT & IFFT Roundtrip (N = 9 = 3^2)
    signal = [1.0, 2.0, 3.0, -1.0, 0.0, 1.5, -2.5, 4.0, 0.5]
    f_re, f_im = radix3_fft(signal)
    inv_re, inv_im = radix3_ifft(f_re, f_im)

    for i in range(len(signal)):
        diff = abs(inv_re[i] - signal[i])
        assert diff < 1e-10, f"FFT roundtrip error at {i}: {diff}"
        assert abs(inv_im[i]) < 1e-10, f"FFT imaginary residual at {i}: {inv_im[i]}"
    print("  [FFI 7/7] Radix-3 FFT/IFFT Reversible Transform ..... PASSED")

    # 8. Base-243 Codec
    test_trits = [1, -1, 0, 1, -1]
    byte_val = base243_pack(test_trits)
    unpacked = base243_unpack(byte_val)
    assert unpacked == test_trits, f"Base-243 codec mismatch: {unpacked} vs {test_trits}"
    print("  [Bonus]   Base-243 5-Trit/Byte Codec .............. PASSED")

    print("=" * 70)
    print("  -> ALL PYTHON FFI VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_self_test()
