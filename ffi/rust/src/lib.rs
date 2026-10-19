//! Balanced Ternary System - High-Performance Rust FFI Bindings
//!
//! Provides zero-overhead, memory-safe Rust abstractions over the
//! C ABI shared runtime library (`libbt_runtime.dylib` / `so` / `dll`).

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// ============================================================================
// 1. #[repr(C)] Layout-Compatible Structures
// ============================================================================

/// 64-trit dual-rail word representation.
/// Invariant: (p & n) == 0.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BtWord64 {
    pub p: u64,
    pub n: u64,
}

/// Balanced Ternary Floating-Point (BT-FP32) equivalent representation.
/// Value = mantissa * 3^exponent. Canonical zero has mantissa = 0, exponent = 0.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BtFp32 {
    pub mantissa: i64,
    pub exponent: i32,
}

/// Ternary Compressed Sparse Row (T-CSR) 1-Bit Matrix representation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BtCsr {
    pub rows: u64,
    pub cols: u64,
    pub nnz: u64,
    pub row_ptr: *const u32,
    pub col_indices: *const u32,
    pub signs: *const u64,
}

/// Output of a Balanced Ternary Half-Adder (BTHA) step.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BtBthaResult {
    pub sum: BtWord64,
    pub carry: BtWord64,
}

// ============================================================================
// 2. Raw extern "C" ABI Declarations
// ============================================================================

extern "C" {
    pub fn bt_swar_word64_to_int64(w: BtWord64, num_trits: usize) -> i64;
    pub fn bt_dot_product_array(a: *const BtWord64, b: *const BtWord64, num_words: usize) -> i64;
    pub fn bt_fp32_from_double(val: f64) -> BtFp32;
    pub fn bt_fp32_to_double(a: BtFp32) -> f64;
    pub fn bt_fp32_add(a: BtFp32, b: BtFp32) -> BtFp32;
    pub fn bt_fp32_sub(a: BtFp32, b: BtFp32) -> BtFp32;
    pub fn bt_fp32_mul(a: BtFp32, b: BtFp32) -> BtFp32;
    pub fn bt_fp32_div(a: BtFp32, b: BtFp32) -> BtFp32;
    pub fn bt_fp32_fma(a: BtFp32, b: BtFp32, c: BtFp32) -> BtFp32;
    pub fn bt_fp32_sqrt(a: BtFp32) -> BtFp32;
    pub fn bt_fp32_abs(a: BtFp32) -> BtFp32;
    pub fn bt_fp32_neg(a: BtFp32) -> BtFp32;
    pub fn bt_fp32_pow_int(a: BtFp32, exp: i64) -> BtFp32;
    pub fn bt_csr_spmv_exec(mat: *const BtCsr, x: *const i64, y: *mut i64);
    pub fn bt_radix3_fft_forward(
        real_in: *const f64,
        imag_in: *const f64,
        real_out: *mut f64,
        imag_out: *mut f64,
        n: usize,
    ) -> i32;
    pub fn bt_radix3_fft_inverse(
        real_in: *const f64,
        imag_in: *const f64,
        real_out: *mut f64,
        imag_out: *mut f64,
        n: usize,
    ) -> i32;
    pub fn bt_radix3_convolve(
        a: *const f64,
        na: usize,
        b: *const f64,
        nb: usize,
        out: *mut f64,
    ) -> i32;
    pub fn bt_kogge_stone_add64_val(a: BtWord64, b: BtWord64, cin: i8, cout_msb: *mut i8) -> BtWord64;
    pub fn bt_dot_product_64_val(a: BtWord64, b: BtWord64) -> i64;
    pub fn bt_btha_step_val(a: BtWord64, b: BtWord64) -> BtBthaResult;
    pub fn bt_base243_pack_val(trits: *const i8) -> u8;
    pub fn bt_base243_unpack_val(byte: u8, out_trits: *mut i8);
    pub fn bt_hamming_distance_64_val(a: BtWord64, b: BtWord64) -> u32;
    pub fn bt_verify_disjointness_val(word: BtWord64) -> bool;
}

// ============================================================================
// 3. Safe Rust High-Level Wrappers
// ============================================================================

impl BtWord64 {
    /// Creates a new dual-rail word.
    #[inline]
    pub fn new(p: u64, n: u64) -> Self {
        Self { p, n }
    }

    /// Verifies dual-rail disjointness invariant: (p & n) == 0.
    #[inline]
    pub fn is_valid(&self) -> bool {
        unsafe { bt_verify_disjointness_val(*self) }
    }

    /// Converts up to 39 packed trits to an i64 using Douglas W. Jones' SWAR parallel tree reduction.
    #[inline]
    pub fn to_i64(&self, num_trits: usize) -> i64 {
        unsafe { bt_swar_word64_to_int64(*self, num_trits) }
    }

    /// Computes 64-trit bit-sliced dot product using dual-rail popcount contraction.
    #[inline]
    pub fn dot(&self, other: &Self) -> i64 {
        unsafe { bt_dot_product_64_val(*self, *other) }
    }

    /// Branchless Balanced Ternary Half-Adder (BTHA) step.
    #[inline]
    pub fn btha_step(&self, other: &Self) -> BtBthaResult {
        unsafe { bt_btha_step_val(*self, *other) }
    }

    /// Computes branchless dual-rail Hamming distance.
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        unsafe { bt_hamming_distance_64_val(*self, *other) }
    }
}

impl BtFp32 {
    /// Constructs a BT-FP32 from an IEEE-754 f64 double.
    #[inline]
    pub fn from_f64(val: f64) -> Self {
        unsafe { bt_fp32_from_double(val) }
    }

    /// Converts this BT-FP32 to standard IEEE-754 f64 double.
    #[inline]
    pub fn to_f64(&self) -> f64 {
        unsafe { bt_fp32_to_double(*self) }
    }

    /// Returns true if this value is the canonical zero (mantissa == 0).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    /// Fused multiply-add with single unbiased rounding: (self * b) + c.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        unsafe { bt_fp32_fma(self, b, c) }
    }

    /// Balanced ternary square root with exact midpoint rounding: sqrt(self).
    #[inline]
    pub fn sqrt(self) -> Self {
        unsafe { bt_fp32_sqrt(self) }
    }

    /// Absolute value of a balanced ternary float.
    #[inline]
    pub fn abs(self) -> Self {
        unsafe { bt_fp32_abs(self) }
    }

    /// Zero-cost negation of a balanced ternary float.
    #[inline]
    pub fn neg(self) -> Self {
        unsafe { bt_fp32_neg(self) }
    }

    /// Integer power of a balanced ternary float: self^exp.
    #[inline]
    pub fn pow_int(self, exp: i64) -> Self {
        unsafe { bt_fp32_pow_int(self, exp) }
    }
}

impl fmt::Display for BtFp32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [M={}, E={}]",
            self.to_f64(),
            self.mantissa,
            self.exponent
        )
    }
}

impl Add for BtFp32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        unsafe { bt_fp32_add(self, rhs) }
    }
}

impl Sub for BtFp32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        unsafe { bt_fp32_sub(self, rhs) }
    }
}

impl Mul for BtFp32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        unsafe { bt_fp32_mul(self, rhs) }
    }
}

impl Div for BtFp32 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        unsafe { bt_fp32_div(self, rhs) }
    }
}

/// Computes bit-sliced dot product across slices of dual-rail words.
pub fn dot_product_slices(a: &[BtWord64], b: &[BtWord64]) -> i64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    unsafe { bt_dot_product_array(a.as_ptr(), b.as_ptr(), len) }
}

/// Executes multiplier-free SpMV: y = mat * x.
pub fn csr_spmv(mat: &BtCsr, x: &[i64]) -> Vec<i64> {
    assert!(
        x.len() >= mat.cols as usize,
        "x slice length must be at least mat.cols"
    );
    let mut y = vec![0i64; mat.rows as usize];
    unsafe {
        bt_csr_spmv_exec(mat as *const BtCsr, x.as_ptr(), y.as_mut_ptr());
    }
    y
}

/// Forward Radix-3 Cooley-Tukey FFT for sequence length N = 3^k.
pub fn radix3_fft(real: &[f64], imag: Option<&[f64]>) -> Result<(Vec<f64>, Vec<f64>), i32> {
    let n = real.len();
    let mut real_out = vec![0.0; n];
    let mut imag_out = vec![0.0; n];
    let imag_ptr = match imag {
        Some(im) => {
            assert_eq!(im.len(), n, "imaginary slice length must match real");
            im.as_ptr()
        }
        None => std::ptr::null(),
    };

    let status = unsafe {
        bt_radix3_fft_forward(
            real.as_ptr(),
            imag_ptr,
            real_out.as_mut_ptr(),
            imag_out.as_mut_ptr(),
            n,
        )
    };

    if status == 0 {
        Ok((real_out, imag_out))
    } else {
        Err(status)
    }
}

/// Inverse Radix-3 Cooley-Tukey FFT with 1/N normalization for sequence length N = 3^k.
pub fn radix3_ifft(real: &[f64], imag: Option<&[f64]>) -> Result<(Vec<f64>, Vec<f64>), i32> {
    let n = real.len();
    let mut real_out = vec![0.0; n];
    let mut imag_out = vec![0.0; n];
    let imag_ptr = match imag {
        Some(im) => {
            assert_eq!(im.len(), n, "imaginary slice length must match real");
            im.as_ptr()
        }
        None => std::ptr::null(),
    };

    let status = unsafe {
        bt_radix3_fft_inverse(
            real.as_ptr(),
            imag_ptr,
            real_out.as_mut_ptr(),
            imag_out.as_mut_ptr(),
            n,
        )
    };

    if status == 0 {
        Ok((real_out, imag_out))
    } else {
        Err(status)
    }
}

/// O(N log3 N) discrete spectral convolution via Radix-3 FFT.
pub fn radix3_convolve(a: &[f64], b: &[f64]) -> Result<Vec<f64>, i32> {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return Ok(Vec::new());
    }
    let out_len = na + nb - 1;
    let mut out = vec![0.0; out_len];
    let status = unsafe {
        bt_radix3_convolve(a.as_ptr(), na, b.as_ptr(), nb, out.as_mut_ptr())
    };
    if status == 0 {
        Ok(out)
    } else {
        Err(status)
    }
}

/// Packs 5 balanced trits into 1 byte [0..242].
pub fn base243_pack(trits: &[i8; 5]) -> u8 {
    unsafe { bt_base243_pack_val(trits.as_ptr()) }
}

/// Unpacks 1 byte [0..242] into 5 balanced trits using magic reciprocal multipliers.
pub fn base243_unpack(byte: u8) -> [i8; 5] {
    let mut trits = [0i8; 5];
    unsafe {
        bt_base243_unpack_val(byte, trits.as_mut_ptr());
    }
    trits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disjointness_and_dot() {
        let a = BtWord64::new(0b0011, 0b1100);
        let b = BtWord64::new(0b0101, 0b1010);
        assert!(a.is_valid());
        assert!(b.is_valid());
        let bad = BtWord64::new(1, 1);
        assert!(!bad.is_valid());
        assert_eq!(a.dot(&b), 0);
    }
}
