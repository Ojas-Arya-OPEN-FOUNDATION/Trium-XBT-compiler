# C ABI Specification, Thread-Local Error Propagation & Multi-Language FFI
## Production C Interface, Zero-Overhead Rust Wrappers, and Python Bindings

> **AI Context Brief:**
> This document details the external interoperability layer of Trium-XBT:
> 1. **C ABI Interface (`include/Ternary/ternary_c_abi.h` & `lib/Ternary/ternary_c_abi.cpp`)**: Strict standard-layout `repr(C)` structs and thread-local error state (`bt_last_error_code()`).
> 2. **Rust Safe Wrappers (`ffi/rust/src/lib.rs`)**: Idiomatic, memory-safe, zero-overhead Rust bindings implementing `Add`, `Sub`, `Mul`, `Div`, `PartialEq`, and error-checked builders.
> 3. **Dynamic Shared Library (`libbt_runtime.so` / `.dylib`)**: Binary runtime loadable by Python `ctypes`, Julia `ccall`, or C/C++.

---

## 1. C ABI Header (`ternary_c_abi.h`)

```c
#ifndef TERNARY_C_ABI_H
#define TERNARY_C_ABI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status Codes
typedef enum {
    BT_STATUS_OK = 0,
    BT_STATUS_INVALID_ARGUMENT = 1,
    BT_STATUS_OVERFLOW = 2,
    BT_STATUS_INTERNAL_ERROR = 3
} bt_status_t;

// ABI Metadata & Thread-Local Diagnostics
uint32_t bt_runtime_abi_version(void);
int bt_last_error_code(void);
const char* bt_last_error_message(void);
void bt_clear_error(void);

// -----------------------------------------------------------------------------
// Standard Layout Structs (repr(C))
// -----------------------------------------------------------------------------

// 64-Trit Dual-Rail Word. Invariant: (p & n) == 0
typedef struct {
    uint64_t p;
    uint64_t n;
} bt_word64_t;

// BT-FP32 Floating-Point (Value = mantissa * 3^exponent)
typedef struct {
    int64_t mantissa;
    int32_t exponent;
} bt_fp32_t;

// Ternary Compressed Sparse Row (T-CSR) 1-Bit Matrix
typedef struct {
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    const uint32_t* row_ptr;     // size: rows + 1
    const uint32_t* col_indices; // size: nnz
    const uint64_t* signs;       // bit-packed: 1 bit per non-zero (0 = -1, 1 = +1)
} bt_csr_t;

// Output of Balanced Ternary Half-Adder
typedef struct {
    bt_word64_t sum;
    bt_word64_t carry;
} bt_btha_result_t;

// -----------------------------------------------------------------------------
// Core Runtime API Functions
// -----------------------------------------------------------------------------
int64_t bt_swar_word64_to_int64(bt_word64_t w, size_t num_trits);
int64_t bt_dot_product_array(const bt_word64_t* a, const bt_word64_t* b, size_t num_words);

bt_fp32_t bt_fp32_from_double(double val);
double    bt_fp32_to_double(bt_fp32_t a);
bt_fp32_t bt_fp32_add(bt_fp32_t a, bt_fp32_t b);
bt_fp32_t bt_fp32_sub(bt_fp32_t a, bt_fp32_t b);
bt_fp32_t bt_fp32_mul(bt_fp32_t a, bt_fp32_t b);
bt_fp32_t bt_fp32_div(bt_fp32_t a, bt_fp32_t b);
bt_fp32_t bt_fp32_fma(bt_fp32_t a, bt_fp32_t b, bt_fp32_t c);

void bt_csr_spmv_exec(const bt_csr_t* mat, const int64_t* x, int64_t* y);

uint8_t bt_base243_pack_val(const int8_t trits[5]);
void    bt_base243_unpack_val(uint8_t byte, int8_t out_trits[5]);

#ifdef __cplusplus
}
#endif

#endif // TERNARY_C_ABI_H
```

---

## 2. Memory-Safe Rust Bindings (`ffi/rust/src/lib.rs`)

```rust
//! Multi-Language Rust FFI Bindings for the Balanced Ternary Engine
//! Zero-overhead, memory-safe Rust abstractions over `libbt_runtime`.

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BtWord64 {
    pub p: u64,
    pub n: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BtError {
    pub code: i32,
    pub message: String,
}

impl fmt::Display for BtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BtError({}): {}", self.code, self.message)
    }
}
impl std::error::Error for BtError {}

impl BtWord64 {
    pub fn try_new(p: u64, n: u64) -> Result<Self, BtError> {
        let w = Self { p, n };
        if w.is_valid() {
            Ok(w)
        } else {
            Err(BtError {
                code: 1,
                message: "overlapping word rails (p & n != 0)".into(),
            })
        }
    }

    pub fn new(p: u64, n: u64) -> Self {
        Self::try_new(p, n).expect("invalid ternary word")
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        (self.p & self.n) == 0
    }

    #[inline]
    pub fn negate(&self) -> Self {
        Self { p: self.n, n: self.p } // 0-cycle rail swap
    }

    pub fn try_dot(&self, other: Self) -> Result<i64, BtError> {
        if !self.is_valid() || !other.is_valid() {
            return Err(BtError {
                code: 1,
                message: "invalid operand: overlapping rails".into(),
            });
        }
        let arr_a = [*self];
        let arr_b = [other];
        let res = unsafe { bt_dot_product_array(arr_a.as_ptr(), arr_b.as_ptr(), 1) };
        Ok(res)
    }

    pub fn dot(&self, other: &Self) -> i64 {
        self.try_dot(*other).expect("dot product failed")
    }
}

// -----------------------------------------------------------------------------
// BT-FP32 Floating-Point Rust Type
// -----------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BtFp32 {
    pub mantissa: i64,
    pub exponent: i32,
}

impl BtFp32 {
    pub fn try_from_f64(val: f64) -> Result<Self, BtError> {
        if val.is_nan() || val.is_infinite() {
            return Err(BtError {
                code: 1,
                message: "NaN and Infinity are not representable in BT-FP32".into(),
            });
        }
        let out = unsafe { bt_fp32_from_double(val) };
        Ok(out)
    }

    pub fn try_to_f64(&self) -> Result<f64, BtError> {
        Ok(unsafe { bt_fp32_to_double(*self) })
    }

    pub fn try_add(&self, other: Self) -> Result<Self, BtError> {
        Ok(unsafe { bt_fp32_add(*self, other) })
    }

    pub fn try_div(&self, other: Self) -> Result<Self, BtError> {
        if other.mantissa == 0 {
            return Err(BtError {
                code: 1,
                message: "division by zero".into(),
            });
        }
        Ok(unsafe { bt_fp32_div(*self, other) })
    }
}

// Rust Operator Overloading
impl Add for BtFp32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { self.try_add(rhs).unwrap() }
}
impl Sub for BtFp32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { unsafe { bt_fp32_sub(self, rhs) } }
}
impl Mul for BtFp32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { unsafe { bt_fp32_mul(self, rhs) } }
}
impl Div for BtFp32 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self { self.try_div(rhs).unwrap() }
}

// Raw extern "C" linking
extern "C" {
    pub fn bt_runtime_abi_version() -> u32;
    pub fn bt_last_error_code() -> i32;
    pub fn bt_last_error_message() -> *const std::os::raw::c_char;
    pub fn bt_dot_product_array(a: *const BtWord64, b: *const BtWord64, num_words: usize) -> i64;
    pub fn bt_fp32_from_double(val: f64) -> BtFp32;
    pub fn bt_fp32_to_double(a: BtFp32) -> f64;
    pub fn bt_fp32_add(a: BtFp32, b: BtFp32) -> BtFp32;
    pub fn bt_fp32_sub(a: BtFp32, b: BtFp32) -> BtFp32;
    pub fn bt_fp32_mul(a: BtFp32, b: BtFp32) -> BtFp32;
    pub fn bt_fp32_div(a: BtFp32, b: BtFp32) -> BtFp32;
}
```
