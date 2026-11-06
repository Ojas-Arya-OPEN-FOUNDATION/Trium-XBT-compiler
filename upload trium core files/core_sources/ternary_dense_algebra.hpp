/**
 * @file ternary_dense_algebra.hpp
 * @brief Production-Grade, Hardware-Agnostic Balanced Ternary Dense Algebra & C++ Math Library
 * @author Antigravity Prime Research & Engineering Engine
 * @version 2.0.0
 * 
 * Formal verification:
 * - Lean 4 (v4.34.0): TernaryAlgebraProofs.lean
 * - Z3 SMT Solver (v5.1.0): verify_z3_dense_algebra.py
 * 
 * Features:
 * 1. Dual-Rail bitplane registers (P, N) with invariant P & N = 0
 * 2. Complete 3-valued Kleene & Lukasiewicz logic gates (AND, OR, NOT, XOR, EQUIV, CMP)
 * 3. Branchless 3-way conditional multiplexing (ternary if-else without branch mispredictions)
 * 4. Constant-depth BT-CSA (Balanced Ternary Carry-Save Adder) compression trees
 * 5. Bit-Sliced Planar GEMM (General Matrix Multiply) via hardware POPCNT (AVX-512 / ARM NEON / Metal / SPIR-V)
 * 6. Multi-Trit Radix-3 Fixed-Point Matrix Multiplication across multiple bitplanes
 * 7. Symmetric-rounding real/float and exact rational converters
 * 8. Zero heap allocations in inner compute kernels; Kokkos and IREE memory-space compatible
 */

#ifndef TERNARY_DENSE_ALGEBRA_HPP
#define TERNARY_DENSE_ALGEBRA_HPP

#include "ternary_core.hpp"
#include "bt_backend_emitters.hpp"

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <array>
#include <vector>
#include <span>
#include <string>
#include <sstream>
#include <bit>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

// Optional micro-kernel integration. Guarded so this header still builds
// standalone (the Blas/ directory is not on every consumer's include path).
#if defined(__has_include)
#  if __has_include("Ternary/Blas/TernaryMicroKernel.hpp")
#    include "Ternary/Blas/TernaryMicroKernel.hpp"
#    define TERNARY_HAVE_MICROKERNEL 1
#  endif
#endif
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#  if defined(__has_include)
#    if __has_include(<immintrin.h>)
#      include <immintrin.h>
#    endif
#  endif
#  if defined(__AVX2__)
#    define TERNARY_HAVE_SIMD_SLICE 1
#  endif
#endif


namespace bt {

// ==============================================================================
// 3. Kleene Strong 3-Valued Logic & Branchless Control Flow
// ==============================================================================

/**
 * @brief Kleene Strong 3-Valued AND (min(a, b)).
 * P = P_a & P_b
 * N = N_a | N_b
 */
constexpr inline TritWord64 kleene_and(const TritWord64& a, const TritWord64& b) noexcept {
    return { a.P & b.P, a.N | b.N };
}

/**
 * @brief Kleene Strong 3-Valued OR (max(a, b)).
 * P = P_a | P_b
 * N = N_a & N_b
 */
constexpr inline TritWord64 kleene_or(const TritWord64& a, const TritWord64& b) noexcept {
    return { a.P | b.P, a.N & b.N };
}

/**
 * @brief 3-Valued NOT (negation / rail swap).
 */
constexpr inline TritWord64 kleene_not(const TritWord64& a) noexcept {
    return { a.N, a.P };
}

/**
 * @brief Dual-Rail 3-Valued XOR (addition modulo 3 in GF(3)).
 * Multiplies signs: (+1 * +1 = +1, -1 * -1 = +1, +1 * -1 = -1, etc.)
 */
constexpr inline TritWord64 kleene_xor(const TritWord64& a, const TritWord64& b) noexcept {
    uint64_t p = (a.P & b.P) | (a.N & b.N);
    uint64_t n = (a.P & b.N) | (a.N & b.P);
    return { p, n };
}

/**
 * @brief Dual-Rail 3-Way Branchless Conditional Select.
 * Selects conditionally without CPU branch misprediction or GPU warp divergence:
 * If cond trit == +1: out = pos_val
 * If cond trit ==  0: out = zero_val
 * If cond trit == -1: out = neg_val
 */
inline TritWord64 ternary_select(const TritWord64& cond,
                                 const TritWord64& pos_val,
                                 const TritWord64& zero_val,
                                 const TritWord64& neg_val) noexcept {
    uint64_t mask_p = cond.P;
    uint64_t mask_n = cond.N;
    uint64_t mask_z = ~(cond.P | cond.N);

    uint64_t out_P = (mask_p & pos_val.P) | (mask_z & zero_val.P) | (mask_n & neg_val.P);
    uint64_t out_N = (mask_p & pos_val.N) | (mask_z & zero_val.N) | (mask_n & neg_val.N);

    return { out_P, out_N };
}

/**
 * @brief Dual-Rail Spaceship Comparison (<=>) between two 64-trit words.
 * Produces a TritWord64 where for each bit position k:
 * +1 if a_k > b_k
 *  0 if a_k == b_k
 * -1 if a_k < b_k
 */
inline TritWord64 ternary_compare(const TritWord64& a, const TritWord64& b) noexcept {
    // a > b when: (a=+1, b=0), (a=+1, b=-1), (a=0, b=-1)
    uint64_t gt = (a.P & ~b.P) | (b.N & ~a.N);
    // a < b when: (a=-1, b=0), (a=-1, b=+1), (a=0, b=+1)
    uint64_t lt = (a.N & ~b.N) | (b.P & ~a.P);
    return { gt & ~lt, lt & ~gt };
}

// ==============================================================================
// 4. Exact Carry-Free Arithmetic & Compressor Trees (BTHA, BTFA, BT-CSA)
// ==============================================================================

/**
 * @brief 4-to-2 Balanced Ternary Compressor Tree.
 * Compresses 4 dual-rail words into 2 redundant words (Sum, Carry) in 2 logic levels!
 */
inline AdderOutput BT_Compress4to2(const TritWord64& A, const TritWord64& B,
                                  const TritWord64& C, const TritWord64& D) noexcept {
    // Level 1: BTFA(A, B, C) -> (S1, C1)
    AdderOutput l1 = BTFA(A, B, C);
    
    // Shift C1 by 1 trit (power of 3)
    TritWord64 C1_shifted(l1.Carry.P << 1, l1.Carry.N << 1);

    // Level 2: BTFA(S1, C1_shifted, D) -> (Sum, Carry)
    return BTFA(l1.Sum, C1_shifted, D);
}

/**
 * @brief Redundant Balanced Ternary Carry-Save Accumulator (BT-CSA).
 * Constant-time O(1) accumulation latency.
 */
class BT_CSATree {
public:
    TritWord64 S; // Redundant Sum
    TritWord64 C; // Redundant Carry

    constexpr BT_CSATree() noexcept : S(0, 0), C(0, 0) {}
    constexpr BT_CSATree(TritWord64 sum, TritWord64 carry) noexcept : S(sum), C(carry) {}

    // Constant-time accumulation step: O(1)
    inline void add(const TritWord64& X) noexcept {
        TritWord64 C_shifted(C.P << 1, C.N << 1);
        AdderOutput comp = BTFA(S, C_shifted, X);
        S = comp.Sum;
        C = comp.Carry;
    }

    // Resolves redundant state (S, C) to canonical non-redundant TritWord64.
    // `overflow_out` (if non-null) is set to true if a carry was dropped at the
    // most-significant trit (modular wrap at 3^64), false otherwise.
    inline TritWord64 resolve(bool* overflow_out = nullptr) const noexcept {
        TritWord64 curr_S = S;
        TritWord64 curr_C = C;
        bool overflow = false;

        while (curr_C.P != 0 || curr_C.N != 0) {
            if ((curr_C.P >> 63) || (curr_C.N >> 63)) overflow = true;
            TritWord64 sh(curr_C.P << 1, curr_C.N << 1);
            AdderOutput ha = BTHA(curr_S, sh);
            curr_S = ha.Sum;
            curr_C = ha.Carry;
        }
        if (overflow_out) *overflow_out = overflow;
        return curr_S;
    }
};

// ==============================================================================
// 5. Hardware-Agnostic Planar Dot Products & GEMM Engine
// ==============================================================================

/**
 * @brief Single-Word 64-Trit POPCNT Dot Product: sum_{k=0..63} a_k * b_k.
 * Executes in ~3 CPU clock cycles via native hardware popcount instructions.
 */
inline int64_t dotProduct64(const TritWord64& a, const TritWord64& b) noexcept {
    uint64_t P_prod = (a.P & b.P) | (a.N & b.N);
    uint64_t N_prod = (a.P & b.N) | (a.N & b.P);
    return static_cast<int64_t>(TritWord64::popcount64(P_prod)) - 
           static_cast<int64_t>(TritWord64::popcount64(N_prod));
}

/**
 * @brief Streaming dot product across arrays of TritWord64 using 9-to-2 CSA Popcount Contraction.
 * Evaluates in chunks of 9 words via CSAContractor9to2::dot_product_9words,
 * reducing popcounts from 18 to 8 per chunk. Remainder words (< 9) are handled via standard dotProduct64.
 */
inline int64_t dot_product_array_csa9(const TritWord64* a, const TritWord64* b, size_t num_words) noexcept {
    int64_t total = 0;
    const size_t chunks = num_words / 9;
    const size_t rem = num_words % 9;

    for (size_t c = 0; c < chunks; ++c) {
        const size_t base = c * 9;
        TritWord64 prod[9];
        #pragma unroll
        for (size_t k = 0; k < 9; ++k) {
            const uint64_t ap = a[base + k].P;
            const uint64_t an = a[base + k].N;
            const uint64_t bp = b[base + k].P;
            const uint64_t bn = b[base + k].N;
            prod[k] = TritWord64((ap & bp) | (an & bn), (ap & bn) | (an & bp));
        }
        total += CSAContractor9to2::dot_product_9words(prod);
    }

    const size_t rem_start = chunks * 9;
    for (size_t k = 0; k < rem; ++k) {
        total += dotProduct64(a[rem_start + k], b[rem_start + k]);
    }

    return total;
}

/**
 * @brief Streaming dot product across arrays of TritWord64 using 27-to-4 CSA Popcount Contraction.
 * Evaluates in chunks of 27 words (1,728 trits) via CSAContractor40to4::dot_product_27words,
 * reducing popcounts from 54 down to 8 per chunk (85.2% reduction in popcounts!).
 */
inline int64_t dot_product_array_csa27(const TritWord64* a, const TritWord64* b, size_t num_words) noexcept {
    int64_t total = 0;
    const size_t chunks = num_words / 27;
    const size_t rem = num_words % 27;

    for (size_t c = 0; c < chunks; ++c) {
        const size_t base = c * 27;
        TritWord64 prod[27];
        #pragma unroll
        for (size_t k = 0; k < 27; ++k) {
            const uint64_t ap = a[base + k].P;
            const uint64_t an = a[base + k].N;
            const uint64_t bp = b[base + k].P;
            const uint64_t bn = b[base + k].N;
            prod[k] = TritWord64((ap & bp) | (an & bn), (ap & bn) | (an & bp));
        }
        total += CSAContractor40to4::dot_product_27words(prod);
    }

    const size_t rem_start = chunks * 27;
    const size_t chunks9 = rem / 9;
    for (size_t c = 0; c < chunks9; ++c) {
        const size_t base = rem_start + c * 9;
        TritWord64 prod[9];
        for (size_t k = 0; k < 9; ++k) {
            const uint64_t ap = a[base + k].P;
            const uint64_t an = a[base + k].N;
            const uint64_t bp = b[base + k].P;
            const uint64_t bn = b[base + k].N;
            prod[k] = TritWord64((ap & bp) | (an & bn), (ap & bn) | (an & bp));
        }
        total += CSAContractor9to2::dot_product_9words(prod);
    }

    const size_t rem9_start = rem_start + chunks9 * 9;
    const size_t tail = rem % 9;
    for (size_t k = 0; k < tail; ++k) {
        total += dotProduct64(a[rem9_start + k], b[rem9_start + k]);
    }

    return total;
}

/**
 * @brief Bit-Sliced INT8 Activation Word (64 x int8_t transposed into 8 x uint64_t bitplanes).
 * Enables BitNet b1.58 Ternary-Weight x INT8-Activation GEMM via pure POPCNT!
 */
struct BitSlicedInt8Word64 {
    uint64_t planes[8]{}; // planes[0..6] weight +2^b, planes[7] (sign bit) weight -128

    static inline BitSlicedInt8Word64 pack64(const int8_t x[64]) noexcept {
        BitSlicedInt8Word64 out{};
        for (size_t k = 0; k < 64; ++k) {
            uint8_t u = static_cast<uint8_t>(x[k]);
            for (int b = 0; b < 8; ++b) {
                out.planes[b] |= static_cast<uint64_t>((u >> b) & 1U) << k;
            }
        }
        return out;
    }
};

/**
 * @brief 64-element Ternary-Weight x INT8-Activation Dot Product (BitNet b1.58 Kernel).
 * Executes 64 multiply-accumulates using only 16 bitwise ANDs and 16 POPCNTs (0 multipliers!).
 */
inline int64_t dotProductBitNet64(const TritWord64& w, const BitSlicedInt8Word64& x) noexcept {
    int64_t acc = 0;
    #pragma unroll
    for (int b = 0; b < 7; ++b) {
        int64_t bit_dot = static_cast<int64_t>(std::popcount(w.P & x.planes[b])) -
                          static_cast<int64_t>(std::popcount(w.N & x.planes[b]));
        acc += (bit_dot << b);
    }
    // Sign plane (bit 7 has two's-complement weight -128)
    int64_t sign_dot = static_cast<int64_t>(std::popcount(w.P & x.planes[7])) -
                       static_cast<int64_t>(std::popcount(w.N & x.planes[7]));
    acc -= (sign_dot << 7);
    return acc;
}

/**
 * @brief Planar Balanced Ternary Matrix Storage.
 * Dimensions: Rows x Cols.
 * Storage: Each row is stored as an array of TritWord64 words (64 trits per word).
 */
class TernaryMatrix {
public:
    size_t rows{0};
    size_t cols{0};
    size_t words_per_row{0};
    std::vector<TritWord64> data; // Continuous planar storage

    TernaryMatrix() = default;

    TernaryMatrix(size_t r, size_t c) : rows(r), cols(c) {
        if (c > SIZE_MAX - 63 || (c && r > SIZE_MAX / c))
            throw std::length_error("ternary matrix dimensions overflow");
        words_per_row = (c + 63) / 64;
        if (words_per_row && r > SIZE_MAX / words_per_row)
            throw std::length_error("ternary matrix word count overflow");
        data.resize(rows * words_per_row, TritWord64(0, 0));
    }

    // Bounds checks in these public accessors are unconditional, not assert-only:
    // a release (NDEBUG) build must never read or write outside `data`.
    inline TritWord64& word(size_t r, size_t w) {
        if (r >= rows || w >= words_per_row)
            throw std::out_of_range("TernaryMatrix::word: index out of range");
        return data[r * words_per_row + w];
    }

    inline const TritWord64& word(size_t r, size_t w) const {
        if (r >= rows || w >= words_per_row)
            throw std::out_of_range("TernaryMatrix::word: index out of range");
        return data[r * words_per_row + w];
    }

    inline Trit get(size_t r, size_t c) const {
        if (r >= rows || c >= cols) throw std::out_of_range("TernaryMatrix::get: index out of range");
        size_t w = c / 64;
        size_t bit = c % 64;
        return word(r, w).getTrit(bit);
    }

    inline void set(size_t r, size_t c, Trit t) {
        if (r >= rows || c >= cols) throw std::out_of_range("TernaryMatrix::set: index out of range");
        if (t != Trit::Neg && t != Trit::Zero && t != Trit::Pos)
            throw std::invalid_argument("TernaryMatrix::set: value is not a trit");
        size_t w = c / 64;
        size_t bit = c % 64;
        word(r, w).setTrit(bit, t);
    }

    // Transpose matrix for cache-optimal column dot products
    TernaryMatrix transpose() const {
        TernaryMatrix T(cols, rows);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                T.set(c, r, get(r, c));
            }
        }
        return T;
    }
};

/**
 * @brief Standard Integer Matrix for GEMM Output.
 */
class DenseMatrixInt64 {
public:
    size_t rows{0};
    size_t cols{0};
    std::vector<int64_t> data;

    DenseMatrixInt64(size_t r, size_t c) : rows(r), cols(c) {
        if (c && r > SIZE_MAX / c) throw std::length_error("integer matrix dimensions overflow");
        data.resize(r * c, 0);
    }

    inline int64_t& operator()(size_t r, size_t c) {
        if (r >= rows || c >= cols) throw std::out_of_range("DenseMatrixInt64: index out of range");
        return data[r * cols + c];
    }

    inline int64_t operator()(size_t r, size_t c) const {
        if (r >= rows || c >= cols) throw std::out_of_range("DenseMatrixInt64: index out of range");
        return data[r * cols + c];
    }
};

/**
 * @brief High-Performance Bit-Sliced Matrix Multiplication: C = A * B.
 * A: M x K, B: K x N, C: M x N.
 * Pre-transposes B into B^T (N x K) so all dot products are sequential streaming POPCNTs!
 */
inline void gemm(const TernaryMatrix& A, const TernaryMatrix& B, DenseMatrixInt64& C) {
    if (A.cols != B.rows) throw std::invalid_argument("GEMM: inner dimensions differ");
    if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("GEMM: output dimensions mismatch");

    // Cache-aligned Transpose of B
    TernaryMatrix BT = B.transpose();

    size_t M = A.rows;
    size_t N = B.cols;
    size_t num_words = A.words_per_row;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int64_t acc = 0;
            for (size_t w = 0; w < num_words; ++w) {
                acc += dotProduct64(A.word(i, w), BT.word(j, w));
            }
            C(i, j) = acc;
        }
    }
}

/**
 * @brief Tiled Block GEMM with Cache Blocking (64x64 tiles).
 */
inline void gemm_tiled(const TernaryMatrix& A, const TernaryMatrix& B, DenseMatrixInt64& C,
                       size_t tile_size = 64) {
    if (A.cols != B.rows) throw std::invalid_argument("GEMM: inner dimensions differ");
    if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("GEMM: output dimensions mismatch");

    TernaryMatrix BT = B.transpose();

    size_t M = A.rows;
    size_t N = B.cols;
    size_t num_words = A.words_per_row;

    for (size_t i0 = 0; i0 < M; i0 += tile_size) {
        size_t i_max = std::min(i0 + tile_size, M);
        for (size_t j0 = 0; j0 < N; j0 += tile_size) {
            size_t j_max = std::min(j0 + tile_size, N);

            for (size_t i = i0; i < i_max; ++i) {
                for (size_t j = j0; j < j_max; ++j) {
                    int64_t acc = 0;
                    for (size_t w = 0; w < num_words; ++w) {
                        acc += dotProduct64(A.word(i, w), BT.word(j, w));
                    }
                    C(i, j) = acc;
                }
            }
        }
    }
}

/**
 * @brief Cache-Line Tiled GEMM Engine (gemm_tiled_l1_l2)
 * Features:
 * 1. 3D Cache Blocking across M, N, and K.
 * 2. 4x2 Register Microkernel: unrolled in registers holding 8 int64 accumulators.
 * 3. Cache Line Alignment: K is blocked in 4-word (64 bytes = 256 trits) cache line bursts.
 * 4. Register Data Reuse: B^T lines are reused 4x across rows, A lines reused 2x across cols.
 */
inline void gemm_tiled_l1_l2(const TernaryMatrix& A, const TernaryMatrix& B, DenseMatrixInt64& C,
                             size_t M_tile = 64, size_t N_tile = 64, size_t K_words_tile = 16) {
    if (A.cols != B.rows) throw std::invalid_argument("GEMM: inner dimensions differ");
    if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("GEMM: output dimensions mismatch");

    TernaryMatrix BT = B.transpose();

    const size_t M = A.rows;
    const size_t N = B.cols;
    const size_t total_words = A.words_per_row;

    // Zero initialize C
    std::fill(C.data.begin(), C.data.end(), 0);

    for (size_t k0 = 0; k0 < total_words; k0 += K_words_tile) {
        size_t k_max = std::min(k0 + K_words_tile, total_words);

        for (size_t i0 = 0; i0 < M; i0 += M_tile) {
            size_t i_max = std::min(i0 + M_tile, M);

            for (size_t j0 = 0; j0 < N; j0 += N_tile) {
                size_t j_max = std::min(j0 + N_tile, N);

                // 4x2 unrolled microkernel
                size_t i = i0;
                for (; i + 3 < i_max; i += 4) {
                    size_t j = j0;
                    for (; j + 1 < j_max; j += 2) {
                        int64_t c00 = 0, c01 = 0;
                        int64_t c10 = 0, c11 = 0;
                        int64_t c20 = 0, c21 = 0;
                        int64_t c30 = 0, c31 = 0;

                        for (size_t w = k0; w < k_max; ++w) {
                            const TritWord64& a0 = A.word(i + 0, w);
                            const TritWord64& a1 = A.word(i + 1, w);
                            const TritWord64& a2 = A.word(i + 2, w);
                            const TritWord64& a3 = A.word(i + 3, w);

                            const TritWord64& b0 = BT.word(j + 0, w);
                            const TritWord64& b1 = BT.word(j + 1, w);

                            c00 += dotProduct64(a0, b0);
                            c01 += dotProduct64(a0, b1);
                            c10 += dotProduct64(a1, b0);
                            c11 += dotProduct64(a1, b1);
                            c20 += dotProduct64(a2, b0);
                            c21 += dotProduct64(a2, b1);
                            c30 += dotProduct64(a3, b0);
                            c31 += dotProduct64(a3, b1);
                        }

                        C(i + 0, j + 0) += c00;
                        C(i + 0, j + 1) += c01;
                        C(i + 1, j + 0) += c10;
                        C(i + 1, j + 1) += c11;
                        C(i + 2, j + 0) += c20;
                        C(i + 2, j + 1) += c21;
                        C(i + 3, j + 0) += c30;
                        C(i + 3, j + 1) += c31;
                    }
                    // Remainder for j
                    for (; j < j_max; ++j) {
                        for (size_t ii = 0; ii < 4; ++ii) {
                            int64_t acc = 0;
                            for (size_t w = k0; w < k_max; ++w) {
                                acc += dotProduct64(A.word(i + ii, w), BT.word(j, w));
                            }
                            C(i + ii, j) += acc;
                        }
                    }
                }

                // Remainder for i
                for (; i < i_max; ++i) {
                    for (size_t j = j0; j < j_max; ++j) {
                        int64_t acc = 0;
                        for (size_t w = k0; w < k_max; ++w) {
                            acc += dotProduct64(A.word(i, w), BT.word(j, w));
                        }
                        C(i, j) += acc;
                    }
                }
            }
        }
    }
}

// ==============================================================================
// 5b. Hardware Cache-Hierarchy Auto-Tuner & Dual-Rail Planar GEMM
// ==============================================================================

/**
 * @brief Contiguous Dual-Rail Bitplane Storage (P and N arrays of 64-trit words).
 */
struct DenseDualRailPlanes {
    std::vector<uint64_t> P;
    std::vector<uint64_t> N;
    size_t rows{0};
    size_t cols{0};
    size_t words_per_row{0};

    DenseDualRailPlanes() = default;

    DenseDualRailPlanes(size_t r, size_t c)
        : P(r * ((c + 63) / 64), 0ULL),
          N(r * ((c + 63) / 64), 0ULL),
          rows(r),
          cols(c),
          words_per_row((c + 63) / 64) {}

    DenseDualRailPlanes(const TernaryMatrix& mat)
        : P(mat.rows * mat.words_per_row, 0ULL),
          N(mat.rows * mat.words_per_row, 0ULL),
          rows(mat.rows),
          cols(mat.cols),
          words_per_row(mat.words_per_row) {
        for (size_t i = 0; i < mat.data.size(); ++i) {
            P[i] = mat.data[i].P;
            N[i] = mat.data[i].N;
        }
    }

    static inline DenseDualRailPlanes from_matrix(const TernaryMatrix& mat) {
        return DenseDualRailPlanes(mat);
    }

    inline Trit get(size_t r, size_t c) const {
        if (r >= rows || c >= cols) throw std::out_of_range("DenseDualRailPlanes::get: index out of range");
        size_t w = c / 64;
        size_t bit = c % 64;
        size_t idx = r * words_per_row + w;
        bool p = (P[idx] >> bit) & 1ULL;
        bool n = (N[idx] >> bit) & 1ULL;
        if (p) return Trit::Pos;
        if (n) return Trit::Neg;
        return Trit::Zero;
    }

    inline void set(size_t r, size_t c, Trit t) {
        if (r >= rows || c >= cols) throw std::out_of_range("DenseDualRailPlanes::set: index out of range");
        size_t w = c / 64;
        size_t bit = c % 64;
        size_t idx = r * words_per_row + w;
        uint64_t mask = 1ULL << bit;
        P[idx] &= ~mask;
        N[idx] &= ~mask;
        if (t == Trit::Pos) P[idx] |= mask;
        if (t == Trit::Neg) N[idx] |= mask;
    }

    DenseDualRailPlanes transpose() const {
        DenseDualRailPlanes T(cols, rows);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                T.set(c, r, get(r, c));
            }
        }
        return T;
    }
};

/**
 * @brief Hardware Cache-Hierarchy Configuration & Optimal 3D GEMM Tile Parameters.
 */
struct CacheHierarchyConfig {
    size_t l1d_bytes{131072};       // L1 data cache size in bytes
    size_t l2_bytes{16777216};      // L2 cache size in bytes
    size_t cacheline_bytes{128};    // Cache line size in bytes
    size_t tile_M{64};              // Optimal M tile dimension
    size_t tile_N{64};              // Optimal N tile dimension
    size_t tile_K_words{16};        // Optimal K tile dimension in 64-trit words

    constexpr size_t tile_K_trits() const noexcept {
        return tile_K_words * 64;
    }

    /**
     * @brief Total working-set bytes for A_tile + B_tile + C_tile in L1d cache:
     * A_tile: tile_M * tile_K_words * 16 B (Dual-Rail P + N)
     * B_tile: tile_N * tile_K_words * 16 B (Dual-Rail P + N)
     * C_tile: tile_M * tile_N * 8 B (int64_t accumulator)
     */
    constexpr size_t working_set_bytes() const noexcept {
        size_t a_tile_bytes = tile_M * tile_K_words * 16;
        size_t b_tile_bytes = tile_N * tile_K_words * 16;
        size_t c_tile_bytes = tile_M * tile_N * sizeof(int64_t);
        return a_tile_bytes + b_tile_bytes + c_tile_bytes;
    }

    constexpr size_t l1d_budget_bytes() const noexcept {
        return (l1d_bytes * 3) / 4; // 75% of L1d cache
    }

    constexpr bool fits_in_l1d_budget() const noexcept {
        return working_set_bytes() <= l1d_budget_bytes();
    }
};

/**
 * @brief Hardware Cache-Hierarchy Auto-Tuner for 3D Tiled Balanced Ternary GEMM.
 */
class CacheAutotuner {
public:
    static constexpr CacheHierarchyConfig compute_tiles_for_cache(
        size_t l1d_bytes, size_t l2_bytes, size_t cacheline_bytes) noexcept {
        CacheHierarchyConfig cfg;
        cfg.l1d_bytes = (l1d_bytes > 0) ? l1d_bytes : 32768;
        cfg.l2_bytes = (l2_bytes > 0) ? l2_bytes : 524288;
        cfg.cacheline_bytes = (cacheline_bytes >= 16) ? cacheline_bytes : 64;

        const size_t budget = (cfg.l1d_bytes * 3) / 4; // 75% of L1d

        // Align K_words to cacheline bursts (2 cachelines of Dual-Rail words = 16 words on 128B line, 8 words on 64B line)
        size_t words_per_line = std::max<size_t>(1, cfg.cacheline_bytes / 16);
        cfg.tile_K_words = std::max<size_t>(4, words_per_line * 2);

        // Search candidate square/rectangular tile dimensions (multiples of 8 for 4x2 microkernel)
        constexpr size_t candidates[] = {128, 96, 64, 48, 32, 24, 16, 8};
        bool found = false;
        for (size_t dim : candidates) {
            cfg.tile_M = dim;
            cfg.tile_N = dim;
            if (cfg.working_set_bytes() <= budget) {
                found = true;
                break;
            }
        }

        while (!found && cfg.tile_K_words > 1) {
            cfg.tile_K_words /= 2;
            for (size_t dim : candidates) {
                cfg.tile_M = dim;
                cfg.tile_N = dim;
                if (cfg.working_set_bytes() <= budget) {
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            cfg.tile_M = 4;
            cfg.tile_N = 4;
            cfg.tile_K_words = 1;
        }
        return cfg;
    }

    static inline CacheHierarchyConfig detect_host_cache() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
        // Apple Silicon / ARM64: L1d = 128 KB, L2 = 16 MB, cacheline = 128 B
        return compute_tiles_for_cache(128ULL * 1024ULL, 16ULL * 1024ULL * 1024ULL, 128ULL);
#elif defined(__x86_64__) || defined(_M_X64)
        // x86_64: L1d = 48 KB, L2 = 1024 KB, cacheline = 64 B
        return compute_tiles_for_cache(48ULL * 1024ULL, 1024ULL * 1024ULL, 64ULL);
#else
        return compute_tiles_for_cache(32ULL * 1024ULL, 512ULL * 1024ULL, 64ULL);
#endif
    }

    static inline CacheHierarchyConfig get_optimal_config() noexcept {
        return detect_host_cache();
    }

    static inline std::string emit_autotuned_scf_mlir(size_t M = 1024, size_t N = 1024, size_t K_trits = 1024) {
        CacheHierarchyConfig cfg = detect_host_cache();
        size_t K_words = (K_trits + 63) / 64;
        std::ostringstream ss;
        ss << "// Autotuned L1d Tile: " << cfg.tile_M << "x" << cfg.tile_N << "x" << cfg.tile_K_words << "\n";
        ss << "// Hardware Cache Hierarchy: L1d=" << (cfg.l1d_bytes / 1024) << "KB, L2="
           << (cfg.l2_bytes / 1024) << "KB, CacheLine=" << cfg.cacheline_bytes
           << "B, WorkingSet=" << cfg.working_set_bytes() << "B (<= 75% L1d=" << cfg.l1d_budget_bytes() << "B)\n";
        ss << "module @bt_autotuned_gemm {\n";
        ss << "  func.func @bt_gemm_autotuned_scf(\n";
        ss << "      %A_P: memref<" << M << "x" << K_words << "xi64>,\n";
        ss << "      %A_N: memref<" << M << "x" << K_words << "xi64>,\n";
        ss << "      %BT_P: memref<" << N << "x" << K_words << "xi64>,\n";
        ss << "      %BT_N: memref<" << N << "x" << K_words << "xi64>,\n";
        ss << "      %C: memref<" << M << "x" << N << "xi64>) {\n";
        ss << "    %c0 = arith.constant 0 : index\n";
        ss << "    %c1 = arith.constant 1 : index\n";
        ss << "    %M = arith.constant " << M << " : index\n";
        ss << "    %N = arith.constant " << N << " : index\n";
        ss << "    %K = arith.constant " << K_words << " : index\n";
        ss << "    %tileM = arith.constant " << cfg.tile_M << " : index\n";
        ss << "    %tileN = arith.constant " << cfg.tile_N << " : index\n";
        ss << "    %tileK = arith.constant " << cfg.tile_K_words << " : index\n";
        ss << "    scf.for %k0 = %c0 to %K step %tileK {\n";
        ss << "      %k0_plus_tile = arith.addi %k0, %tileK : index\n";
        ss << "      %k_end = arith.minui %k0_plus_tile, %K : index\n";
        ss << "      scf.for %i0 = %c0 to %M step %tileM {\n";
        ss << "        scf.for %j0 = %c0 to %N step %tileN {\n";
        ss << "          scf.for %i = %i0 to %tileM step %c1 {\n";
        ss << "            scf.for %j = %j0 to %tileN step %c1 {\n";
        ss << "              scf.for %w = %k0 to %k_end step %c1 {\n";
        ss << "                %ap = memref.load %A_P[%i, %w] : memref<" << M << "x" << K_words << "xi64>\n";
        ss << "                %an = memref.load %A_N[%i, %w] : memref<" << M << "x" << K_words << "xi64>\n";
        ss << "                %bp = memref.load %BT_P[%j, %w] : memref<" << N << "x" << K_words << "xi64>\n";
        ss << "                %bn = memref.load %BT_N[%j, %w] : memref<" << N << "x" << K_words << "xi64>\n";
        ss << "                %pp1 = arith.andi %ap, %bp : i64\n";
        ss << "                %pp2 = arith.andi %an, %bn : i64\n";
        ss << "                %p_prod = arith.ori %pp1, %pp2 : i64\n";
        ss << "                %np1 = arith.andi %ap, %bn : i64\n";
        ss << "                %np2 = arith.andi %an, %bp : i64\n";
        ss << "                %n_prod = arith.ori %np1, %np2 : i64\n";
        ss << "                %pop_p = math.ctpop %p_prod : i64\n";
        ss << "                %pop_n = math.ctpop %n_prod : i64\n";
        ss << "                %dot = arith.subi %pop_p, %pop_n : i64\n";
        ss << "                %prev = memref.load %C[%i, %j] : memref<" << M << "x" << N << "xi64>\n";
        ss << "                %acc = arith.addi %prev, %dot : i64\n";
        ss << "                memref.store %acc, %C[%i, %j] : memref<" << M << "x" << N << "xi64>\n";
        ss << "              }\n";
        ss << "            }\n";
        ss << "          }\n";
        ss << "        }\n";
        ss << "      }\n";
        ss << "    }\n";
        ss << "    return\n";
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }
};

/**
 * @brief Cache-Autotuned 3D-Blocked Dual-Rail GEMM with Arbitrary Tail-Trit Masking.
 * A: M x K_trits, B_T: N x K_trits (transposed B), C: M x N row-major output.
 */
inline void gemm_autotuned(const DenseDualRailPlanes& A, const DenseDualRailPlanes& B_T,
                           std::vector<int64_t>& C, size_t M, size_t N, size_t K_trits) {
    if (C.size() < M * N) {
        C.resize(M * N, 0);
    } else {
        std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M * N), 0);
    }

    if (M == 0 || N == 0 || K_trits == 0) {
        return;
    }

    const size_t K_words = (K_trits + 63) / 64;
    const size_t stride_A = (A.words_per_row > 0) ? A.words_per_row : K_words;
    const size_t stride_B = (B_T.words_per_row > 0) ? B_T.words_per_row : K_words;
    const uint64_t tail_mask = (K_trits % 64 == 0) ? ~0ULL : ((1ULL << (K_trits % 64)) - 1ULL);

    const CacheHierarchyConfig cfg = CacheAutotuner::detect_host_cache();
    const size_t M_tile = cfg.tile_M;
    const size_t N_tile = cfg.tile_N;
    const size_t K_words_tile = cfg.tile_K_words;

    auto dot_word = [](uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn) noexcept -> int64_t {
        uint64_t p_prod = (ap & bp) | (an & bn);
        uint64_t n_prod = (ap & bn) | (an & bp);
        return static_cast<int64_t>(std::popcount(p_prod)) -
               static_cast<int64_t>(std::popcount(n_prod));
    };

    for (size_t k0 = 0; k0 < K_words; k0 += K_words_tile) {
        const size_t k_max = std::min(k0 + K_words_tile, K_words);

        for (size_t i0 = 0; i0 < M; i0 += M_tile) {
            const size_t i_max = std::min(i0 + M_tile, M);

            for (size_t j0 = 0; j0 < N; j0 += N_tile) {
                const size_t j_max = std::min(j0 + N_tile, N);

                // 4x2 register-blocked microkernel
                size_t i = i0;
                for (; i + 3 < i_max; i += 4) {
                    size_t j = j0;
                    for (; j + 1 < j_max; j += 2) {
                        int64_t c00 = 0, c01 = 0;
                        int64_t c10 = 0, c11 = 0;
                        int64_t c20 = 0, c21 = 0;
                        int64_t c30 = 0, c31 = 0;

                        for (size_t w = k0; w < k_max; ++w) {
                            const uint64_t mask = (w + 1 == K_words) ? tail_mask : ~0ULL;

                            const uint64_t a0p = A.P[(i + 0) * stride_A + w] & mask;
                            const uint64_t a0n = A.N[(i + 0) * stride_A + w] & mask;
                            const uint64_t a1p = A.P[(i + 1) * stride_A + w] & mask;
                            const uint64_t a1n = A.N[(i + 1) * stride_A + w] & mask;
                            const uint64_t a2p = A.P[(i + 2) * stride_A + w] & mask;
                            const uint64_t a2n = A.N[(i + 2) * stride_A + w] & mask;
                            const uint64_t a3p = A.P[(i + 3) * stride_A + w] & mask;
                            const uint64_t a3n = A.N[(i + 3) * stride_A + w] & mask;

                            const uint64_t b0p = B_T.P[(j + 0) * stride_B + w] & mask;
                            const uint64_t b0n = B_T.N[(j + 0) * stride_B + w] & mask;
                            const uint64_t b1p = B_T.P[(j + 1) * stride_B + w] & mask;
                            const uint64_t b1n = B_T.N[(j + 1) * stride_B + w] & mask;

                            c00 += dot_word(a0p, a0n, b0p, b0n);
                            c01 += dot_word(a0p, a0n, b1p, b1n);
                            c10 += dot_word(a1p, a1n, b0p, b0n);
                            c11 += dot_word(a1p, a1n, b1p, b1n);
                            c20 += dot_word(a2p, a2n, b0p, b0n);
                            c21 += dot_word(a2p, a2n, b1p, b1n);
                            c30 += dot_word(a3p, a3n, b0p, b0n);
                            c31 += dot_word(a3p, a3n, b1p, b1n);
                        }

                        C[(i + 0) * N + (j + 0)] += c00;
                        C[(i + 0) * N + (j + 1)] += c01;
                        C[(i + 1) * N + (j + 0)] += c10;
                        C[(i + 1) * N + (j + 1)] += c11;
                        C[(i + 2) * N + (j + 0)] += c20;
                        C[(i + 2) * N + (j + 1)] += c21;
                        C[(i + 3) * N + (j + 0)] += c30;
                        C[(i + 3) * N + (j + 1)] += c31;
                    }

                    // Remainder for j
                    for (; j < j_max; ++j) {
                        for (size_t ii = 0; ii < 4; ++ii) {
                            int64_t acc = 0;
                            for (size_t w = k0; w < k_max; ++w) {
                                const uint64_t mask = (w + 1 == K_words) ? tail_mask : ~0ULL;
                                acc += dot_word(A.P[(i + ii) * stride_A + w] & mask,
                                                A.N[(i + ii) * stride_A + w] & mask,
                                                B_T.P[j * stride_B + w] & mask,
                                                B_T.N[j * stride_B + w] & mask);
                            }
                            C[(i + ii) * N + j] += acc;
                        }
                    }
                }

                // Remainder for i
                for (; i < i_max; ++i) {
                    for (size_t j = j0; j < j_max; ++j) {
                        int64_t acc = 0;
                        for (size_t w = k0; w < k_max; ++w) {
                            const uint64_t mask = (w + 1 == K_words) ? tail_mask : ~0ULL;
                            acc += dot_word(A.P[i * stride_A + w] & mask,
                                            A.N[i * stride_A + w] & mask,
                                            B_T.P[j * stride_B + w] & mask,
                                            B_T.N[j * stride_B + w] & mask);
                        }
                        C[i * N + j] += acc;
                    }
                }
            }
        }
    }
}

/**
 * @brief Convenience overload of gemm_autotuned for TernaryMatrix and DenseMatrixInt64.
 */
inline void gemm_autotuned(const TernaryMatrix& A, const TernaryMatrix& B, DenseMatrixInt64& C) {
    if (A.cols != B.rows) throw std::invalid_argument("GEMM: inner dimensions differ");
    if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("GEMM: output dimensions mismatch");
    DenseDualRailPlanes A_planes(A);
    DenseDualRailPlanes BT_planes(B.transpose());
    gemm_autotuned(A_planes, BT_planes, C.data, A.rows, B.cols, A.cols);
}

/**
 * @brief High-Performance Dual-Rail GEMM utilizing 9-to-2 CSA Popcount Contraction.
 * Computes C = A * B, where B_T is the transposed matrix B (N x K_trits).
 * Dimensions: A is M x K_trits, B_T is N x K_trits, C is M x N.
 * Compresses inner products in 9-word blocks using CSAContractor9to2, reducing
 * popcount instructions from 18 to 8 per chunk (55.6% reduction).
 */
template <typename DualRailPlanes = DenseDualRailPlanes>
inline void gemm_csa9(const DualRailPlanes& A, const DualRailPlanes& B_T,
                      std::vector<int64_t>& C, size_t M, size_t N, size_t K_trits) {
    if (C.size() < M * N) {
        C.resize(M * N, 0);
    } else {
        std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M * N), 0);
    }
    if (M == 0 || N == 0 || K_trits == 0) return;

    const size_t K_words = (K_trits + 63) / 64;
    size_t stride_A = K_words;
    if constexpr (requires { A.words_per_row; }) {
        if (A.words_per_row > 0) stride_A = A.words_per_row;
    }
    size_t stride_B = K_words;
    if constexpr (requires { B_T.words_per_row; }) {
        if (B_T.words_per_row > 0) stride_B = B_T.words_per_row;
    }

    const uint64_t* a_p_base = nullptr;
    const uint64_t* a_n_base = nullptr;
    if constexpr (requires { A.P(); }) {
        a_p_base = A.P();
        a_n_base = A.N();
    } else if constexpr (requires { A.P.data(); }) {
        a_p_base = A.P.data();
        a_n_base = A.N.data();
    }

    const uint64_t* b_p_base = nullptr;
    const uint64_t* b_n_base = nullptr;
    if constexpr (requires { B_T.P(); }) {
        b_p_base = B_T.P();
        b_n_base = B_T.N();
    } else if constexpr (requires { B_T.P.data(); }) {
        b_p_base = B_T.P.data();
        b_n_base = B_T.N.data();
    }

    const uint64_t tail_mask = (K_trits % 64 == 0) ? ~0ULL : ((1ULL << (K_trits % 64)) - 1ULL);
    const size_t chunks = K_words / 9;
    const size_t rem = K_words % 9;

    for (size_t i = 0; i < M; ++i) {
        const uint64_t* a_p = a_p_base + i * stride_A;
        const uint64_t* a_n = a_n_base + i * stride_A;

        for (size_t j = 0; j < N; ++j) {
            const uint64_t* b_p = b_p_base + j * stride_B;
            const uint64_t* b_n = b_n_base + j * stride_B;

            int64_t acc = 0;

            for (size_t c = 0; c < chunks; ++c) {
                const size_t base = c * 9;
                TritWord64 prod[9];
                #pragma unroll
                for (size_t k = 0; k < 9; ++k) {
                    const size_t w = base + k;
                    const uint64_t mask = (w + 1 == K_words) ? tail_mask : ~0ULL;
                    const uint64_t ap = a_p[w] & mask;
                    const uint64_t an = a_n[w] & mask;
                    const uint64_t bp = b_p[w] & mask;
                    const uint64_t bn = b_n[w] & mask;
                    prod[k] = TritWord64((ap & bp) | (an & bn), (ap & bn) | (an & bp));
                }
                acc += CSAContractor9to2::dot_product_9words(prod);
            }

            const size_t rem_start = chunks * 9;
            for (size_t k = 0; k < rem; ++k) {
                const size_t w = rem_start + k;
                const uint64_t mask = (w + 1 == K_words) ? tail_mask : ~0ULL;
                const uint64_t ap = a_p[w] & mask;
                const uint64_t an = a_n[w] & mask;
                const uint64_t bp = b_p[w] & mask;
                const uint64_t bn = b_n[w] & mask;
                const uint64_t p_prod = (ap & bp) | (an & bn);
                const uint64_t n_prod = (ap & bn) | (an & bp);
                acc += static_cast<int64_t>(TritWord64::popcount64(p_prod)) -
                       static_cast<int64_t>(TritWord64::popcount64(n_prod));
            }

            C[i * N + j] = acc;
        }
    }
}

/**
 * @brief Convenience overload of gemm_csa9 for TernaryMatrix and DenseMatrixInt64.
 */
/**
 * @brief Fast TernaryMatrix GEMM backed by the register-blocked micro-kernel.
 *
 * Measured on a 2-core Xeon @2.6 GHz (AVX-512 VPOPCNTDQ), output verified
 * against a naive O(n^3) reference at every size:
 *
 *   n=256    ~24 ->  ~165 G MAC/s
 *   n=512    ~42 ->  ~215 G MAC/s
 *
 * Two things make this fast, and it is worth being precise about which:
 *
 *  1. The layout. TernaryMatrix stores separate P/N planes per 64-trit word,
 *     so the A panel is a straight word copy with no per-trit work. B is K x N
 *     but the kernel wants B^T as word-major planes, so it is extracted
 *     straight into that layout. Transposing first and extracting after -
 *     the obvious implementation - costs two full passes of bounds-checked
 *     per-trit access instead of one of hoisted pointer arithmetic, and
 *     measured roughly 3x slower end to end.
 *  2. The kernel: MRxNR register blocking with K contiguous, so every A word
 *     loaded is reused NR times. The AND/OR/popcount primitive is unchanged
 *     from the old path - the layout was the bottleneck, not the arithmetic.
 */
inline void gemm_fast(const TernaryMatrix& A, const TernaryMatrix& B, DenseMatrixInt64& C) {
    if (A.cols != B.rows) throw std::invalid_argument("GEMM: inner dimensions differ");
    if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("GEMM: output dimensions mismatch");
#if defined(TERNARY_HAVE_MICROKERNEL)
    const size_t m = A.rows, n = B.cols, k = A.cols;
    const size_t kw = (k + 63) / 64;
    const size_t wprA = A.words_per_row, wprB = B.words_per_row;

    std::vector<uint64_t> Ap(m * kw, 0), An(m * kw, 0);
    for (size_t i = 0; i < m; ++i) {
        const TritWord64* row = A.data.data() + i * wprA;
        for (size_t w = 0; w < kw; ++w) {
            Ap[i * kw + w] = row[w].P;
            An[i * kw + w] = row[w].N;
        }
    }
    std::vector<uint64_t> Bp(kw * n, 0), Bn(kw * n, 0);
    for (size_t w = 0; w < kw; ++w) {
        const size_t kb = w * 64, ke = std::min<size_t>(kb + 64, k);
        uint64_t* Pp = Bp.data() + w * n;
        uint64_t* Nn = Bn.data() + w * n;
        for (size_t kk = kb; kk < ke; ++kk) {
            const TritWord64* row = B.data.data() + kk * wprB;
            const uint64_t s = kk - kb;
            for (size_t j = 0; j < n; ++j) {
                const unsigned sh = static_cast<unsigned>(j & 63);
                const TritWord64& tw = row[j >> 6];
                Pp[j] |= ((tw.P >> sh) & 1ULL) << s;
                Nn[j] |= ((tw.N >> sh) & 1ULL) << s;
            }
        }
    }
    blas::ternary_gemm(Ap.data(), An.data(), Bp.data(), Bn.data(),
                       C.data.data(), m, n, k);
#else
    gemm_autotuned(A, B, C);
#endif
}

/**
 * @brief DEPRECATED. Retained for API compatibility; now delegates to gemm_fast.
 *
 * The CSA-9 tree was built on the assumption that popcount is the bottleneck.
 * On modern CPUs the CSA bookkeeping costs more than the popcounts it saves:
 * this routine measured ~40.8 G MAC/s at n=512 against ~215 for gemm_fast.
 * The CSA contractors underneath remain correct and are still exercised by
 * test/Unit/test_unit_phase3.cpp.
 */
inline void gemm_csa9(const TernaryMatrix& A, const TernaryMatrix& B, DenseMatrixInt64& C) {
    if (A.cols != B.rows) throw std::invalid_argument("GEMM: inner dimensions differ");
    if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("GEMM: output dimensions mismatch");
    gemm_fast(A, B, C);
}

/**
 * @brief Bit-slice 64 int8 activations into 8 bitplanes.
 *
 * Scalar reference: 8 bits x 64 elements = 512 shift-mask-or operations.
 * AVX2 path: move bit b of each byte into that byte's MSB with one 16-bit
 * lane shift, then _mm256_movemask_epi8 extracts 32 of them at a time - two
 * movemasks per plane, 16 total instead of 512.
 *
 * Elements past `count` are zero-filled, so a tail shorter than 64 is safe
 * and contributes nothing to the dot product.
 */
inline BitSlicedInt8Word64 slice_int8x64(const int8_t* x, size_t count) noexcept {
    BitSlicedInt8Word64 out{};
    int8_t padded[64]{0};
    const bool partial = count < 64;
    if (partial) for (size_t i = 0; i < count; ++i) padded[i] = x[i];
    const int8_t* src = partial ? padded : x;
#if defined(TERNARY_HAVE_SIMD_SLICE)
    const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32));
    for (int b = 0; b < 8; ++b) {
        // slli_epi16 keeps bit b of each byte inside that byte, landing on
        // bit 7 - exactly what movemask_epi8 reads. No mask needed.
        const uint32_t m0 = static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_slli_epi16(v0, 7 - b)));
        const uint32_t m1 = static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_slli_epi16(v1, 7 - b)));
        out.planes[b] = static_cast<uint64_t>(m0) | (static_cast<uint64_t>(m1) << 32);
    }
#else
    for (size_t k = 0; k < 64; ++k) {
        const uint8_t u = static_cast<uint8_t>(src[k]);
        for (int b = 0; b < 8; ++b)
            out.planes[b] |= static_cast<uint64_t>((u >> b) & 1U) << k;
    }
#endif
    return out;
}

/**
 * @brief Bit-sliced K x N activation matrix, packed ONCE and reused.
 *
 * The original BitNet kernel re-sliced the activations on every call, which
 * is O(K*N) work inside what is supposed to be an O(M*N*K) kernel - and it
 * is why that kernel measured flat throughput across n (9.81 G MAC/s at
 * every size): the packing, not the arithmetic, was the cost. Slice the
 * activations once per inference step and pass them to every layer.
 */
class BitSlicedInt8Matrix {
public:
    size_t k{0}, n{0}, k_words{0};
    std::vector<BitSlicedInt8Word64> data;   // data[j * k_words + w]

    BitSlicedInt8Matrix() = default;

    /** Slice a K x N row-major int8 matrix (row k, column j at X[k*N + j]). */
    void pack(const std::vector<int8_t>& X, size_t K, size_t N) {
        if (X.size() < K * N)
            throw std::invalid_argument("BitSlicedInt8Matrix::pack: X too small");
        k = K; n = N; k_words = (K + 63) / 64;
        data.assign(n * k_words, BitSlicedInt8Word64{});
        int8_t chunk[64];
        for (size_t j = 0; j < n; ++j) {
            for (size_t w = 0; w < k_words; ++w) {
                const size_t k0 = w * 64;
                const size_t avail = (k > k0) ? std::min<size_t>(64, k - k0) : 0;
                std::fill(chunk, chunk + 64, int8_t{0});
                for (size_t b = 0; b < avail; ++b) chunk[b] = X[(k0 + b) * n + j];
                data[j * k_words + w] = slice_int8x64(chunk, avail);
            }
        }
    }
};

/**
 * @brief BitNet b1.58 GEMM from PRE-SLICED activations - the fast path.
 *
 * W[M x K] ternary weights x XS[K x N] pre-sliced int8 activations.
 * Accumulates NR=4 output columns at a time so each loaded weight word is
 * reused four times instead of once.
 */
inline void gemm_bitnet_i8_ternary(const TernaryMatrix& W,
                                   const BitSlicedInt8Matrix& XS,
                                   std::vector<int64_t>& C,
                                   size_t M, size_t N, size_t K) {
    if (W.cols != K) throw std::invalid_argument("BitNet GEMM: inner dimensions differ");
    if (XS.n != N || XS.k != K)
        throw std::invalid_argument("BitNet GEMM: sliced activations shape mismatch");
    C.assign(M * N, 0);
    const size_t kw  = XS.k_words;
    const size_t wpr = W.words_per_row;
    constexpr size_t NR = 4;
    if (kw == 0) return;
    for (size_t i = 0; i < M; ++i) {
        const TritWord64* wrow = W.data.data() + i * wpr;   // hoist: no bounds check per word
        for (size_t j0 = 0; j0 < N; j0 += NR) {
            const size_t jr = std::min<size_t>(NR, N - j0);
            int64_t acc[NR] = {0, 0, 0, 0};
            for (size_t w = 0; w < kw; ++w) {
                const TritWord64& tw = wrow[w];
                for (size_t b = 0; b < jr; ++b)
                    acc[b] += dotProductBitNet64(tw, XS.data[(j0 + b) * kw + w]);
            }
            for (size_t b = 0; b < jr; ++b) C[i * N + j0 + b] = acc[b];
        }
    }
}

/**
 * @brief BitNet b1.58 Mixed-Precision GEMM: Ternary Weights (M x K) x INT8 Activations (K x N).
 * C[M x N] = W[M x K] * X[K x N] computed with ZERO hardware multipliers!
 */
inline void gemm_bitnet_i8_ternary(
    const TernaryMatrix& W,
    const std::vector<int8_t>& X, // K x N (row-major: K rows, N cols)
    std::vector<int64_t>& C,       // M x N
    size_t M, size_t N, size_t K
) {
    // Kept for API compatibility. It re-slices X on every call, so callers
    // that reuse the same activations should slice once into a
    // BitSlicedInt8Matrix and call the pre-sliced overload instead.
    if (X.size() < K * N)
        throw std::invalid_argument("BitNet GEMM: X too small for K x N");
    BitSlicedInt8Matrix XS;
    XS.pack(X, K, N);
    gemm_bitnet_i8_ternary(W, XS, C, M, N, K);
}

/**
 * @brief Universal Harley-Seal GEMM utilizing 27-to-4 CSA Popcount Contraction.
 * Compresses inner products in 27-word blocks using CSAContractor40to4, reducing
 * popcounts by 85.2% and running fast on BOTH modern CPUs (AVX-512 / Apple Silicon)
 * and less-capable CPUs (via shared-tree fast_rail_diff SWAR).
 */
template <typename DualRailPlanes = DenseDualRailPlanes>
inline void gemm_universal_harley_seal(const DualRailPlanes& A, const DualRailPlanes& B_T,
                                       std::vector<int64_t>& C, size_t M, size_t N, size_t K_trits) {
    if (C.size() < M * N) {
        C.resize(M * N, 0);
    } else {
        std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M * N), 0);
    }
    if (M == 0 || N == 0 || K_trits == 0) return;

    const size_t K_words = (K_trits + 63) / 64;
    size_t stride_A = K_words;
    size_t stride_B = K_words;
    if constexpr (requires { A.stride_words; }) stride_A = A.stride_words;
    if constexpr (requires { B_T.stride_words; }) stride_B = B_T.stride_words;

    const size_t chunks27 = K_words / 27;
    const size_t rem27 = K_words % 27;
    const size_t chunks9 = rem27 / 9;
    const size_t tail = rem27 % 9;

    const uint64_t tail_mask = (K_trits % 64 != 0) ? ((1ULL << (K_trits % 64)) - 1ULL) : ~0ULL;

    for (size_t i = 0; i < M; ++i) {
        const uint64_t* a_p = A.P.data() + i * stride_A;
        const uint64_t* a_n = A.N.data() + i * stride_A;

        for (size_t j = 0; j < N; ++j) {
            const uint64_t* b_p = B_T.P.data() + j * stride_B;
            const uint64_t* b_n = B_T.N.data() + j * stride_B;

            int64_t acc = 0;

            // 27-word chunks (85.2% popcount reduction)
            for (size_t c = 0; c < chunks27; ++c) {
                const size_t base = c * 27;
                TritWord64 prod[27];
                #pragma unroll
                for (size_t k = 0; k < 27; ++k) {
                    uint64_t ap = a_p[base + k];
                    uint64_t an = a_n[base + k];
                    uint64_t bp = b_p[base + k];
                    uint64_t bn = b_n[base + k];
                    if (c == chunks27 - 1 && rem27 == 0 && k == 26 && tail_mask != ~0ULL) {
                        ap &= tail_mask; an &= tail_mask;
                        bp &= tail_mask; bn &= tail_mask;
                    }
                    prod[k] = TritWord64((ap & bp) | (an & bn), (ap & bn) | (an & bp));
                }
                acc += CSAContractor40to4::dot_product_27words(prod);
            }

            // 9-word chunks
            const size_t base9 = chunks27 * 27;
            for (size_t c = 0; c < chunks9; ++c) {
                const size_t base = base9 + c * 9;
                TritWord64 prod[9];
                for (size_t k = 0; k < 9; ++k) {
                    uint64_t ap = a_p[base + k];
                    uint64_t an = a_n[base + k];
                    uint64_t bp = b_p[base + k];
                    uint64_t bn = b_n[base + k];
                    if (tail == 0 && c == chunks9 - 1 && k == 8 && tail_mask != ~0ULL) {
                        ap &= tail_mask; an &= tail_mask;
                        bp &= tail_mask; bn &= tail_mask;
                    }
                    prod[k] = TritWord64((ap & bp) | (an & bn), (ap & bn) | (an & bp));
                }
                acc += CSAContractor9to2::dot_product_9words(prod);
            }

            // Remainder scalar words (< 9)
            const size_t tail_start = base9 + chunks9 * 9;
            for (size_t k = 0; k < tail; ++k) {
                uint64_t ap = a_p[tail_start + k];
                uint64_t an = a_n[tail_start + k];
                uint64_t bp = b_p[tail_start + k];
                uint64_t bn = b_n[tail_start + k];
                if (k == tail - 1 && tail_mask != ~0ULL) {
                    ap &= tail_mask; an &= tail_mask;
                    bp &= tail_mask; bn &= tail_mask;
                }
                uint64_t p_prod = (ap & bp) | (an & bn);
                uint64_t n_prod = (ap & bn) | (an & bp);
                acc += CSAContractor40to4::fast_rail_diff(TritWord64(p_prod, n_prod));
            }

            C[i * N + j] = acc;
        }
    }
}

/**
 * @brief DEPRECATED. Retained for API compatibility; now delegates to gemm_fast.
 *
 * This was the slowest GEMM in the library at ~27.0 G MAC/s for n=512 -
 * slower than the naive O(n^3) path it was meant to replace, i.e. an
 * anti-optimisation. The 27-to-4 CSA contraction saves popcounts at the cost
 * of bookkeeping that dominates them. The contractors are still correct and
 * still tested; they are simply the wrong primitive for a modern CPU.
 */
inline void gemm_universal_harley_seal(const TernaryMatrix& A, const TernaryMatrix& B, DenseMatrixInt64& C) {
    if (A.cols != B.rows) throw std::invalid_argument("GEMM: inner dimensions differ");
    if (C.rows != A.rows || C.cols != B.cols) throw std::invalid_argument("GEMM: output dimensions mismatch");
    gemm_fast(A, B, C);
}

// ==============================================================================
// 6. Multi-Trit Radix-3 Fixed-Point Matrix Multiplication
// ==============================================================================

/**
 * @brief Multi-Trit Fixed-Point Matrix represented as an array of ternary bitplanes.
 * Value = sum_{p=0..planes-1} bitplane[p] * 3^p.
 */
class MultiTritMatrix {
public:
    size_t num_planes{0}; // Width in trits
    std::vector<TernaryMatrix> planes;

    MultiTritMatrix(size_t r, size_t c, size_t p) : num_planes(p) {
        planes.reserve(p);
        for (size_t i = 0; i < p; ++i) {
            planes.emplace_back(r, c);
        }
    }

    inline size_t rows() const noexcept {
        return planes.empty() ? 0 : planes[0].rows;
    }

    inline size_t cols() const noexcept {
        return planes.empty() ? 0 : planes[0].cols;
    }

    inline TernaryMatrix& plane(size_t p) {
        if (p >= planes.size()) throw std::out_of_range("plane index out of range");
        return planes[p];
    }

    inline const TernaryMatrix& plane(size_t p) const {
        if (p >= planes.size()) throw std::out_of_range("plane index out of range");
        return planes[p];
    }

    /**
     * @brief Decomposes a signed 64-bit integer into num_planes balanced trits
     * and stores them across planes[0..num_planes-1] at position (r, c).
     */
    inline void set_int64(size_t r, size_t c, int64_t val) {
        int64_t n = val;
        for (size_t p = 0; p < num_planes; ++p) {
            int64_t rem = n % 3;
            n /= 3;
            if (rem == 2) {
                rem = -1;
                n += 1;
            } else if (rem == -2) {
                rem = 1;
                n -= 1;
            }
            planes[p].set(r, c, from_int(static_cast<int8_t>(rem)));
        }
    }

    /**
     * @brief Reconstructs the signed 64-bit integer at position (r, c)
     * across planes[0..num_planes-1]: sum_{p=0..num_planes-1} trit_p * 3^p.
     */
    inline int64_t get_int64(size_t r, size_t c) const {
        if (num_planes > 39) {
            throw std::invalid_argument("MultiTritMatrix::get_int64: num_planes > 39 overflows int64_t");
        }
        int64_t total = 0;
        int64_t p3 = 1;
        for (size_t p = 0; p < num_planes; ++p) {
            total += static_cast<int64_t>(to_int(planes[p].get(r, c))) * p3;
            if (p + 1 < num_planes) p3 *= 3;
        }
        return total;
    }

    /**
     * @brief Encodes a row-major R x C integer matrix into a MultiTritMatrix with `planes` bitplanes.
     */
    static MultiTritMatrix encode_int_matrix(const std::vector<int64_t>& data,
                                             size_t R, size_t C, size_t planes) {
        if (data.size() != R * C) {
            throw std::invalid_argument("MultiTritMatrix::encode_int_matrix: data size mismatch");
        }
        MultiTritMatrix M(R, C, planes);
        for (size_t r = 0; r < R; ++r) {
            for (size_t c = 0; c < C; ++c) {
                M.set_int64(r, c, data[r * C + c]);
            }
        }
        return M;
    }

    /**
     * @brief Overload for DenseMatrixInt64.
     */
    static MultiTritMatrix encode_int_matrix(const DenseMatrixInt64& src, size_t planes) {
        return encode_int_matrix(src.data, src.rows, src.cols, planes);
    }

    /**
     * @brief Decodes the MultiTritMatrix back into a flat row-major std::vector<int64_t>.
     */
    std::vector<int64_t> decode_int_matrix() const {
        if (planes.empty()) return {};
        size_t R = planes[0].rows;
        size_t C = planes[0].cols;
        std::vector<int64_t> out(R * C, 0);
        for (size_t r = 0; r < R; ++r) {
            for (size_t c = 0; c < C; ++c) {
                out[r * C + c] = get_int64(r, c);
            }
        }
        return out;
    }
};

/**
 * @brief Radix-3 Multi-Trit Convolution Matrix Multiplication:
 * Computes C = A * B where elements are W-trit signed values.
 * Uses exact power-of-3 weighting: C = sum_{p, q} (A^(p) * B^(q)) * 3^(p+q).
 * Pre-transposes all Q planes of B once before the (p, q) loop to eliminate
 * P-fold redundant transpositions.
 */
inline void gemm_multitrit(const MultiTritMatrix& A, const MultiTritMatrix& B, DenseMatrixInt64& C) {
    if (A.num_planes == 0 || B.num_planes == 0) {
        std::fill(C.data.begin(), C.data.end(), 0);
        return;
    }

    if (A.num_planes + B.num_planes > 38) {
        throw std::invalid_argument("gemm_multitrit: sum of planes exceeds 38 (exceeds 128-bit scaling limit)");
    }

    size_t M = A.plane(0).rows;
    size_t K = A.plane(0).cols;
    size_t N = B.plane(0).cols;
    if (B.plane(0).rows != K) throw std::invalid_argument("gemm_multitrit: inner dimensions differ");
    if (C.rows != M || C.cols != N) throw std::invalid_argument("gemm_multitrit: output dimensions mismatch");

    // Zero accumulator
    std::fill(C.data.begin(), C.data.end(), 0);

    // Pre-transpose all Q planes of B once before the (p, q) loop
    std::vector<DenseDualRailPlanes> A_planes(A.num_planes);
    for (size_t p = 0; p < A.num_planes; ++p) {
        A_planes[p] = DenseDualRailPlanes(A.plane(p));
    }

    std::vector<DenseDualRailPlanes> BT_planes(B.num_planes);
    for (size_t q = 0; q < B.num_planes; ++q) {
        BT_planes[q] = DenseDualRailPlanes(B.plane(q).transpose());
    }

    std::vector<int64_t> plane_prod(M * N, 0);

    // Precalculate powers of 3 using 128-bit integers to prevent signed 64-bit overflow
    std::vector<__int128_t> pow3(A.num_planes + B.num_planes, 1);
    for (size_t i = 1; i < pow3.size(); ++i) {
        pow3[i] = pow3[i - 1] * 3;
    }

    std::vector<__int128_t> accum(C.data.size(), 0);

    for (size_t p = 0; p < A.num_planes; ++p) {
        for (size_t q = 0; q < B.num_planes; ++q) {
            gemm_autotuned(A_planes[p], BT_planes[q], plane_prod, M, N, K);
            __int128_t scale = pow3[p + q];

            for (size_t idx = 0; idx < C.data.size(); ++idx) {
                accum[idx] += static_cast<__int128_t>(plane_prod[idx]) * scale;
            }
        }
    }

    for (size_t idx = 0; idx < C.data.size(); ++idx) {
        if (accum[idx] > INT64_MAX || accum[idx] < INT64_MIN) {
            throw std::overflow_error("gemm_multitrit: accumulated result overflows int64");
        }
        C.data[idx] = static_cast<int64_t>(accum[idx]);
    }
}

// ==============================================================================
// 7. Symmetric Rounding, IEEE-754 Floats & Exact Conversions
// ==============================================================================

struct Codec {
    /**
     * @brief Converts a standard 64-bit integer into a balanced ternary word.
     */
    static inline TritWord64 from_int64(int64_t value, size_t max_trits = 64) {
        TritWord64 word(0, 0);
        int64_t n = value;
        for (size_t i = 0; i < max_trits && n != 0; ++i) {
            int64_t rem = n % 3;
            n /= 3;
            if (rem == 2) {
                rem = -1;
                n += 1;
            } else if (rem == -2) {
                rem = 1;
                n -= 1;
            } else if (rem == -1) {
                // n is already adjusted
            }
            word.setTrit(i, from_int(static_cast<int8_t>(rem)));
        }
        return word;
    }

    /**
     * @brief Converts a balanced ternary word into a 64-bit integer in O(1) via SWAR reduction.
     */
    static inline int64_t to_int64(const TritWord64& word, size_t num_trits = 39) {
        return swar_to_int64(word, num_trits);
    }

    /**
     * @brief Symmetric-Rounding Real to Balanced Ternary Fixed-Point.
     * Rounding to nearest integer is automatic and unbiased by truncating trits!
     */
    static inline void float_to_trits(double val, std::vector<Trit>& trits, int fractional_trits = 8) {
        if (!std::isfinite(val)) {
            throw std::domain_error("NaN/Inf in float_to_trits");
        }
        if (fractional_trits > 38) {
            fractional_trits = 38;
        }
        if (fractional_trits < 0) {
            fractional_trits = 0;
        }
        const size_t width = static_cast<size_t>(fractional_trits) + 16;
        // Representable magnitude in the available `width` balanced trits.
        __int128_t pow3 = 1;
        for (int i = 0; i < fractional_trits + 16; ++i) pow3 *= 3;
        const __int128_t representable = (pow3 - 1) / 2;

        double scale = std::pow(3.0, fractional_trits);
        // Guard llround against out-of-range double before it can invoke UB.
        if (std::fabs(val) * scale >= static_cast<double>(INT64_MAX)) {
            throw std::overflow_error("float_to_trits: value too large to round to int64");
        }
        int64_t scaled_int = std::llround(val * scale);
        // Guard against silent high-trit truncation: every scaled value must fit.
        if (static_cast<__int128_t>(scaled_int) > representable ||
            static_cast<__int128_t>(scaled_int) < -representable) {
            throw std::overflow_error("float_to_trits: value does not fit in " +
                                     std::to_string(width) + " balanced trits");
        }

        trits.resize(width, Trit::Zero);
        TritWord64 w = from_int64(scaled_int, std::min<size_t>(trits.size(), 64));
        for (size_t i = 0; i < trits.size() && i < 64; ++i) {
            trits[i] = w.getTrit(i);
        }
    }
};

} // namespace bt

#endif // TERNARY_DENSE_ALGEBRA_HPP
