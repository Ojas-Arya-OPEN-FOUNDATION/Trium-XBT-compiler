/**
 * @file ternary_c_abi.h
 * @brief Production C ABI / FFI Interface for the Balanced Ternary Engine
 * Compatible with C99, C++20, Rust FFI (#[repr(C)]), and Python ctypes.
 * @author Antigravity Prime Research & Engineering Engine
 */

#ifndef TERNARY_C_ABI_H
#define TERNARY_C_ABI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Thread-local error state. Message remains valid until the next ABI call on
 * this thread. Legacy value APIs return a neutral/NaN result on failure; callers
 * must query this state. New checked APIs return bt_status_t directly. */
typedef enum {
    BT_STATUS_OK = 0,
    BT_STATUS_INVALID_ARGUMENT = 1,
    BT_STATUS_OVERFLOW = 2,
    BT_STATUS_INTERNAL_ERROR = 3
} bt_status_t;
/* ABI major 1: structure layouts follow the host C ABI; incompatible changes
 * require a new major. Capability additions do not imply new hardware support. */
uint32_t bt_runtime_abi_version(void);
int bt_last_error_code(void);
const char* bt_last_error_message(void);
void bt_clear_error(void);


/* ============================================================================
 * 1. Data Structures (repr(C) / Standard Layout)
 * ============================================================================ */

/**
 * @brief 64-trit dual-rail word representation.
 * Disjointness invariant: (p & n) == 0.
 * Trit i in {-1, 0, +1} is given by ((p >> i) & 1) - ((n >> i) & 1).
 */
typedef struct {
    uint64_t p;
    uint64_t n;
} bt_word64_t;

/**
 * @brief Balanced Ternary Floating-Point 32-bit equivalent (BT-FP32).
 * Exact normalized value = mantissa * 3^exponent.
 * Unique canonical zero: mantissa = 0, exponent = 0.
 */
typedef struct {
    int64_t mantissa;
    int32_t exponent;
} bt_fp32_t;

/**
 * @brief Ternary Compressed Sparse Row (T-CSR) 1-Bit Matrix representation.
 * Memory savings: 32x to 64x over IEEE FP32/FP64 CSR.
 */
typedef struct {
    uint64_t rows;
    uint64_t cols;
    uint64_t nnz;
    const uint32_t* row_ptr;     /* size: rows + 1 */
    const uint32_t* col_indices; /* size: nnz */
    const uint64_t* signs;       /* bit-packed: 1 bit per nnz (0 = -1, 1 = +1) */
} bt_csr_t;

bt_status_t bt_csr_spmv_checked(const bt_csr_t* mat, const int64_t* x,
    size_t x_len, int64_t* y, size_t y_len,
    size_t row_ptr_len, size_t col_indices_len, size_t signs_len);

/**
 * @brief Output of a Balanced Ternary Half-Adder (BTHA) step.
 */
typedef struct {
    bt_word64_t sum;
    bt_word64_t carry;
} bt_btha_result_t;

/**
 * @brief 5 balanced trits container.
 */
typedef struct {
    int8_t trits[5];
} bt_trits5_t;

/* ============================================================================
 * 2. Inline / C-Callable Arithmetic & Bitwise Functions
 * ============================================================================ */

#if defined(__GNUC__) || defined(__clang__)
#define BT_POPCOUNT64(x) ((uint32_t)__builtin_popcountll((unsigned long long)(x)))
#else
static inline uint32_t bt_popcount64_fallback(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (uint32_t)((x * 0x0101010101010101ULL) >> 56);
}
#define BT_POPCOUNT64(x) bt_popcount64_fallback(x)
#endif

/**
 * @brief Creates a 64-trit dual-rail word.
 */
static inline bt_word64_t bt_make_word64(uint64_t p, uint64_t n) {
    bt_word64_t w;
    w.p = p;
    w.n = n;
    return w;
}

/**
 * @brief Computes 64-trit bit-sliced dot product using dual-rail popcount contraction.
 * dot = popcount((ap & bp) | (an & bn)) - popcount((ap & bn) | (an & bp))
 */
static inline int64_t bt_dot_product_64(bt_word64_t a, bt_word64_t b) {
    uint64_t pos = (a.p & b.p) | (a.n & b.n);
    uint64_t neg = (a.p & b.n) | (a.n & b.p);
    return (int64_t)BT_POPCOUNT64(pos) - (int64_t)BT_POPCOUNT64(neg);
}

/**
 * @brief Raw 4-operand variant of bt_dot_product_64.
 */
static inline int64_t bt_dot_product_64_raw(uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn) {
    bt_word64_t a = { ap, an };
    bt_word64_t b = { bp, bn };
    return bt_dot_product_64(a, b);
}

/**
 * @brief Branchless 6-instruction Balanced Ternary Half-Adder (BTHA) step.
 * Computes Sum and Carry planes in parallel without carry ripples.
 * Invariant: A + B = 3 * Carry + Sum.
 */
static inline bt_btha_result_t bt_btha_step(bt_word64_t a, bt_word64_t b) {
    uint64_t pc = a.p & b.p;
    uint64_t nc = a.n & b.n;

    uint64_t xor_p = a.p ^ b.p;
    uint64_t not_or_n = ~(a.n | b.n);
    uint64_t ps = (xor_p & not_or_n) | nc;

    uint64_t xor_n = a.n ^ b.n;
    uint64_t not_or_p = ~(a.p | b.p);
    uint64_t ns = (xor_n & not_or_p) | pc;

    bt_btha_result_t res;
    res.sum.p = ps;
    res.sum.n = ns;
    res.carry.p = pc;
    res.carry.n = nc;
    return res;
}

/**
 * @brief Pointer-based variant of bt_btha_step for C ABI callers.
 */
static inline void bt_btha_step_ptrs(bt_word64_t a, bt_word64_t b, bt_word64_t* out_sum, bt_word64_t* out_carry) {
    bt_btha_result_t res = bt_btha_step(a, b);
    if (out_sum) *out_sum = res.sum;
    if (out_carry) *out_carry = res.carry;
}

/**
 * @brief Packs 5 balanced trits (each in {-1, 0, +1}) into 1 byte [0..242].
 * Encoding: sum_{k=0..4} (trit[k] + 1) * 3^k.
 */
static inline uint8_t bt_base243_pack(const int8_t trits[5]) {
    int32_t b = (trits[0] + 1) +
                (trits[1] + 1) * 3 +
                (trits[2] + 1) * 9 +
                (trits[3] + 1) * 27 +
                (trits[4] + 1) * 81;
    return (uint8_t)b;
}

/**
 * @brief Decompresses 1 byte [0..242] into 5 balanced trits using magic reciprocal multipliers.
 * Zero runtime integer divisions (only multiplication and bit shifts).
 */
static inline void bt_base243_unpack(uint8_t byte, int8_t out_trits[5]) {
    uint32_t b = byte;
    uint32_t q1 = (b * 171U) >> 9;
    uint32_t q2 = (b * 57U) >> 9;
    uint32_t q3 = (b * 19U) >> 9;
    uint32_t q4 = (b * 203U) >> 14;

    out_trits[0] = (int8_t)((int32_t)(b - 3U * q1) - 1);
    out_trits[1] = (int8_t)((int32_t)(q1 - 3U * q2) - 1);
    out_trits[2] = (int8_t)((int32_t)(q2 - 3U * q3) - 1);
    out_trits[3] = (int8_t)((int32_t)(q3 - 3U * q4) - 1);
    out_trits[4] = (int8_t)((int32_t)(q4) - 1);
}

/**
 * @brief Struct-returning helper for base243 unpack.
 */
static inline bt_trits5_t bt_base243_unpack5(uint8_t byte) {
    bt_trits5_t res;
    bt_base243_unpack(byte, res.trits);
    return res;
}

/**
 * @brief Douglas W. Jones SWAR Parallel BCT-to-Binary Tree Reduction.
 * Converts 32 packed trits in dual-rail (p, n) to a 64-bit binary integer in log2(32)=5 tree stages.
 * Eliminates 32 sequential Horner loop data-hazard stalls.
 */
static inline int64_t bt_swar_bct_to_bin32(uint32_t p, uint32_t n) {
    /* Level 1: 16 pairs -> Base 9 in [-4..+4] */
    int32_t v1[16];
    for (int i = 0; i < 16; ++i) {
        int d_even = (int)((p >> (2 * i)) & 1U) - (int)((n >> (2 * i)) & 1U);
        int d_odd  = (int)((p >> (2 * i + 1)) & 1U) - (int)((n >> (2 * i + 1)) & 1U);
        v1[i] = 3 * d_odd + d_even;
    }
    /* Level 2: 8 quads -> Base 81 in [-40..+40] */
    int32_t v2[8];
    for (int i = 0; i < 8; ++i) {
        v2[i] = 9 * v1[2 * i + 1] + v1[2 * i];
    }
    /* Level 3: 4 octets -> Base 6561 in [-3280..+3280] */
    int32_t v3[4];
    for (int i = 0; i < 4; ++i) {
        v3[i] = 81 * v2[2 * i + 1] + v2[2 * i];
    }
    /* Level 4: 2 hexadecs -> Base 43046721 in [-21520080..+21520080] */
    int64_t v4[2];
    for (int i = 0; i < 2; ++i) {
        v4[i] = (int64_t)6561 * v3[2 * i + 1] + v3[2 * i];
    }
    /* Level 5: Full 32 trits to 64-bit binary integer */
    return (int64_t)43046721 * v4[1] + v4[0];
}

/**
 * @brief Fast Branchless Dual-Rail Ternary Hamming Distance across 64 trits.
 * Counts differing trits between words a and b in 1-2 CPU clock cycles.
 * Two dual-rail trits differ iff (a.p ^ b.p) | (a.n ^ b.n) != 0.
 */
static inline uint32_t bt_hamming_distance_64(bt_word64_t a, bt_word64_t b) {
    uint64_t diff = (a.p ^ b.p) | (a.n ^ b.n);
    return BT_POPCOUNT64(diff);
}

/**
 * @brief Verifies dual-rail disjointness invariant: (p & n) == 0.
 */
static inline bool bt_verify_disjointness(bt_word64_t word) {
    return (word.p & word.n) == 0;
}

/* ============================================================================
 * 3. Exported Dynamic Shared Library Runtime API (libbt_runtime)
 * ============================================================================ */

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef BT_BUILD_SHARED
    #define BT_API __declspec(dllexport)
  #else
    #define BT_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define BT_API __attribute__((visibility("default")))
  #else
    #define BT_API
  #endif
#endif

/**
 * @brief Converts up to 39 packed trits in a dual-rail word to an int64_t binary integer
 * using Douglas W. Jones' SWAR tree reduction.
 */
BT_API int64_t bt_swar_word64_to_int64(bt_word64_t w, size_t num_trits);

/**
 * @brief Computes the bit-sliced dot product over arrays of 64-trit dual-rail words.
 */
BT_API int64_t bt_dot_product_array(const bt_word64_t* a, const bt_word64_t* b, size_t num_words);

/**
 * @brief Constructs a balanced ternary float from IEEE-754 double precision float.
 */
BT_API bt_fp32_t bt_fp32_from_double(double val);

/**
 * @brief Converts balanced ternary float to standard IEEE-754 double precision float.
 */
BT_API double bt_fp32_to_double(bt_fp32_t a);

/**
 * @brief Adds two balanced ternary floats with 0-bias truncation rounding.
 */
BT_API bt_fp32_t bt_fp32_add(bt_fp32_t a, bt_fp32_t b);

/**
 * @brief Subtracts two balanced ternary floats.
 */
BT_API bt_fp32_t bt_fp32_sub(bt_fp32_t a, bt_fp32_t b);

/**
 * @brief Multiplies two balanced ternary floats.
 */
BT_API bt_fp32_t bt_fp32_mul(bt_fp32_t a, bt_fp32_t b);

/**
 * @brief Divides two balanced ternary floats with symmetric nearest-integer rounding.
 */
BT_API bt_fp32_t bt_fp32_div(bt_fp32_t a, bt_fp32_t b);

/**
 * @brief Fused multiply-add with single unbiased rounding: (a * b) + c.
 */
BT_API bt_fp32_t bt_fp32_fma(bt_fp32_t a, bt_fp32_t b, bt_fp32_t c);

/**
 * @brief Balanced ternary square root with exact midpoint rounding: sqrt(a).
 */
BT_API bt_fp32_t bt_fp32_sqrt(bt_fp32_t a);

/**
 * @brief Absolute value of a balanced ternary float.
 */
BT_API bt_fp32_t bt_fp32_abs(bt_fp32_t a);

/**
 * @brief Zero-cost negation of a balanced ternary float.
 */
BT_API bt_fp32_t bt_fp32_neg(bt_fp32_t a);

/**
 * @brief Integer power of a balanced ternary float: a^exp.
 */
BT_API bt_fp32_t bt_fp32_pow_int(bt_fp32_t a, int64_t exp);

/**
 * @brief Multiplier-Free Sparse Matrix-Vector Multiply (SpMV): y = mat * x.
 */
BT_API void bt_csr_spmv_exec(const bt_csr_t* mat, const int64_t* x, int64_t* y);

/**
 * @brief Forward Radix-3 Cooley-Tukey FFT for sequence length N = 3^k.
 * @return 0 on success, -1 on invalid args, -2 on non-power-of-3 size.
 */
BT_API int bt_radix3_fft_forward(const double* real_in, const double* imag_in, double* real_out, double* imag_out, size_t n);

/**
 * @brief Inverse Radix-3 Cooley-Tukey FFT with 1/N normalization for N = 3^k.
 * @return 0 on success, -1 on invalid args, -2 on non-power-of-3 size.
 */
BT_API int bt_radix3_fft_inverse(const double* real_in, const double* imag_in, double* real_out, double* imag_out, size_t n);

/**
 * @brief O(N log3 N) discrete spectral convolution via Radix-3 FFT.
 * Output buffer must have size >= na + nb - 1.
 */
BT_API int bt_radix3_convolve(const double* a, size_t na, const double* b, size_t nb, double* out);

/**
 * @brief Exported 6-stage Kogge-Stone adder with 35% reduced monotone transfer lattice.
 */
BT_API bt_word64_t bt_kogge_stone_add64_val(bt_word64_t a, bt_word64_t b, int8_t cin, int8_t* cout_msb);

/**
 * @brief Exported single-word 64-trit dot product for dynamic FFI bindings.
 */
BT_API int64_t bt_dot_product_64_val(bt_word64_t a, bt_word64_t b);

/**
 * @brief Exported branchless BTHA step for dynamic FFI bindings.
 */
BT_API bt_btha_result_t bt_btha_step_val(bt_word64_t a, bt_word64_t b);

/**
 * @brief Exported Base-243 pack for dynamic FFI bindings.
 */
BT_API uint8_t bt_base243_pack_val(const int8_t trits[5]);

/**
 * @brief Exported Base-243 unpack for dynamic FFI bindings.
 */
BT_API void bt_base243_unpack_val(uint8_t byte, int8_t out_trits[5]);

/**
 * @brief Exported Hamming distance across 64 trits for dynamic FFI bindings.
 */
BT_API uint32_t bt_hamming_distance_64_val(bt_word64_t a, bt_word64_t b);

/**
 * @brief Exported disjointness verifier for dynamic FFI bindings.
 */
BT_API bool bt_verify_disjointness_val(bt_word64_t word);

/**
 * @brief Exported Kleene 3-valued AND for 64-bit integer scalars.
 */
BT_API int64_t bt_kleene_and(int64_t a, int64_t b);

/**
 * @brief Exported Kleene 3-valued OR for 64-bit integer scalars.
 */
BT_API int64_t bt_kleene_or(int64_t a, int64_t b);

/**
 * @brief Exported Ternary XOR (GF(3) product) for 64-bit integer scalars.
 */
BT_API int64_t bt_kleene_xor(int64_t a, int64_t b);

/**
 * @brief Universal XBT GEMM for unmodified FP32 matrices: C[M x N] = A[M x K] * B[K x N].
 * Uses 8-plane Balanced Ternary decomposition with active-plane bitmask skipping.
 */
BT_API void xbt_sgemm(const float* a, const float* b, float* c, size_t m, size_t k, size_t n, int min_plane_sum_cutoff);

/**
 * @brief Vectorized RMSNorm layer: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i].
 * Uses 3rd-order Cubic Householder inverse square root.
 */
BT_API void xbt_rmsnorm(const float* x, const float* weight, float* out, size_t n, float eps);

/**
 * @brief Numerically stable Radix-3 vector Softmax: out[i] = exp(x[i] - max) / sum.
 */
BT_API void xbt_softmax(const float* x, float* out, size_t n);

/**
 * @brief Radix-3 range-reduced exponential e^x.
 */
BT_API double xbt_exp(double x);

/**
 * @brief Odd-symmetric Balanced Ternary tanh(x).
 */
BT_API double xbt_tanh(double x);

/**
 * @brief Centered Balanced Ternary sigmoid(x).
 */
BT_API double xbt_sigmoid(double x);

/**
 * @brief Fast SiLU / Swish activation: x * sigmoid(x).
 */
BT_API double xbt_silu(double x);

/**
 * @brief Fast GELU activation.
 */
BT_API double xbt_gelu(double x);

/**
 * @brief 3rd-order Cubic Householder 1 / sqrt(x).
 */
BT_API double xbt_rsqrt(double x);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_C_ABI_H */

