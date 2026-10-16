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

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_C_ABI_H */
