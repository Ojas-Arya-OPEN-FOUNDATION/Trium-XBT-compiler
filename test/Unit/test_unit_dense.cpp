/**
 * @file test_unit_dense.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Dense Algebra & Tiled GEMM
 */

#include "../../ternary_dense_algebra.hpp"
#include <iostream>
#include <cassert>

using namespace bt;

void test_kleene_logic() {
    TritWord64 a(0b1010, 0b0101);
    TritWord64 b(0b1100, 0b0011);
    TritWord64 wand = kleene_and(a, b);
    TritWord64 wor = kleene_or(a, b);
    TritWord64 wnot = kleene_not(a);
    assert(wand.isValid());
    assert(wor.isValid());
    assert(wnot.isValid());
}

void test_tiled_gemm() {
    size_t M = 16, K = 64, N = 16;
    TernaryMatrix A(M, K);
    TernaryMatrix B(K, N);

    // Fill with known values
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            A.set(i, k, (i % 2 == 0) ? Trit::Pos : Trit::Neg);
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            B.set(k, j, (j % 2 == 0) ? Trit::Pos : Trit::Zero);
        }
    }

    DenseMatrixInt64 C_ref(M, N);
    gemm(A, B, C_ref);

    DenseMatrixInt64 C_tiled(M, N);
    gemm_tiled_l1_l2(A, B, C_tiled, 8, 8, 2);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            assert(C_ref(i, j) == C_tiled(i, j));
        }
    }
}

void test_codec_and_gemm_guards() {
    std::cout << "  [Dense] Testing Codec Guards & MultiTrit GEMM Edge Cases...\n";

    // 1. to_int64 with num_trits > 39
    TritWord64 w(0x55, 0xAA);
    bool caught_overflow = false;
    try {
        Codec::to_int64(w, 40);
    } catch (const std::invalid_argument&) {
        caught_overflow = true;
    }
    assert(caught_overflow && "to_int64 with num_trits > 39 must throw std::invalid_argument");

    // 2. float_to_trits with NaN and Inf
    std::vector<Trit> trits;
    bool caught_nan = false;
    try {
        Codec::float_to_trits(std::numeric_limits<double>::quiet_NaN(), trits);
    } catch (const std::domain_error&) {
        caught_nan = true;
    }
    assert(caught_nan && "float_to_trits with NaN must throw std::domain_error");

    bool caught_inf = false;
    try {
        Codec::float_to_trits(std::numeric_limits<double>::infinity(), trits);
    } catch (const std::domain_error&) {
        caught_inf = true;
    }
    assert(caught_inf && "float_to_trits with Inf must throw std::domain_error");

    // 3. float_to_trits with fractional_trits clamped
    Codec::float_to_trits(1.2345, trits, 45); // Should clamp to 38 without buffer overflow
    assert(!trits.empty());

    // 4. gemm_multitrit with empty planes
    MultiTritMatrix A_empty(4, 4, 0);
    MultiTritMatrix B_empty(4, 4, 0);
    DenseMatrixInt64 C(4, 4);
    C.data[0] = 42;
    gemm_multitrit(A_empty, B_empty, C);
    assert(C.data[0] == 0 && "gemm_multitrit on empty planes must zero C and safely return");
}

void test_cache_autotuner_and_gemm_autotuned() {
    std::cout << "  [Dense] Testing CacheAutotuner & gemm_autotuned (Aligned 128x128 & Unaligned 37x53x100)...\n";

    // 1. Verify CacheAutotuner host detection and L1d 75% working-set budget
    CacheHierarchyConfig host_cfg = CacheAutotuner::detect_host_cache();
    assert(host_cfg.l1d_bytes > 0);
    assert(host_cfg.l2_bytes >= host_cfg.l1d_bytes);
    assert(host_cfg.cacheline_bytes >= 64);
    assert(host_cfg.tile_M > 0 && host_cfg.tile_N > 0 && host_cfg.tile_K_words > 0);
    assert(host_cfg.working_set_bytes() <= (host_cfg.l1d_bytes * 3) / 4 &&
           "Working set A_tile + B_tile + C_tile must fit inside 75% of L1d cache!");

    // Verify Apple Silicon (128KB L1d) and x86_64 (48KB L1d) configurations explicitly
    CacheHierarchyConfig arm_cfg = CacheAutotuner::compute_tiles_for_cache(128 * 1024, 16 * 1024 * 1024, 128);
    assert(arm_cfg.fits_in_l1d_budget());
    assert(arm_cfg.tile_M == 64 && arm_cfg.tile_N == 64 && arm_cfg.tile_K_words == 16);

    CacheHierarchyConfig x86_cfg = CacheAutotuner::compute_tiles_for_cache(48 * 1024, 1024 * 1024, 64);
    assert(x86_cfg.fits_in_l1d_budget());

    // 2. Test gemm_autotuned on aligned 128x128x128 matrices
    {
        const size_t M = 128, N = 128, K = 128;
        TernaryMatrix A_mat(M, K);
        TernaryMatrix B_mat(K, N);
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                int v = static_cast<int>((i * 7 + k * 13) % 3) - 1;
                A_mat.set(i, k, from_int(static_cast<int8_t>(v)));
            }
        }
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                int v = static_cast<int>((k * 5 + j * 11) % 3) - 1;
                B_mat.set(k, j, from_int(static_cast<int8_t>(v)));
            }
        }

        DenseMatrixInt64 C_ref(M, N);
        gemm(A_mat, B_mat, C_ref);

        DenseDualRailPlanes A_planes(A_mat);
        DenseDualRailPlanes BT_planes(B_mat.transpose());
        std::vector<int64_t> C_auto(M * N, 0);
        gemm_autotuned(A_planes, BT_planes, C_auto, M, N, K);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                assert(C_auto[i * N + j] == C_ref(i, j));
            }
        }
    }

    // 3. Test gemm_autotuned on unaligned 37x53x100 matrices + tail-masking verification
    {
        const size_t M = 37, N = 53, K = 100;
        TernaryMatrix A_mat(M, K);
        TernaryMatrix B_mat(K, N);
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                int v = static_cast<int>((i * 3 + k * 17) % 3) - 1;
                A_mat.set(i, k, from_int(static_cast<int8_t>(v)));
            }
        }
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                int v = static_cast<int>((k * 19 + j * 23) % 3) - 1;
                B_mat.set(k, j, from_int(static_cast<int8_t>(v)));
            }
        }

        DenseMatrixInt64 C_ref(M, N);
        gemm(A_mat, B_mat, C_ref);

        DenseDualRailPlanes A_planes(A_mat);
        DenseDualRailPlanes BT_planes(B_mat.transpose());

        // Inject garbage bits into the unused tail bits [36..63] of the 2nd word (w=1, since 100 = 64 + 36)
        // to prove gemm_autotuned's tail_mask properly masks out bits >= K_trits!
        const uint64_t garbage_bits = ~((1ULL << (K % 64)) - 1ULL);
        for (size_t i = 0; i < M; ++i) {
            A_planes.P[i * A_planes.words_per_row + 1] |= garbage_bits;
        }
        for (size_t j = 0; j < N; ++j) {
            BT_planes.P[j * BT_planes.words_per_row + 1] |= garbage_bits;
        }

        std::vector<int64_t> C_auto(M * N, 0);
        gemm_autotuned(A_planes, BT_planes, C_auto, M, N, K);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                assert(C_auto[i * N + j] == C_ref(i, j) &&
                       "Unaligned 37x53x100 gemm_autotuned mismatch (tail mask failed)!");
            }
        }
    }
}

void test_multitrit_encode_decode_and_gemm() {
    std::cout << "  [Dense] Testing MultiTritMatrix encode_int_matrix, decode_int_matrix & gemm_multitrit...\n";

    // 1. Test set_int64, get_int64, encode_int_matrix, decode_int_matrix round-trip
    // With 5 planes, representable range is [-(3^5-1)/2, +(3^5-1)/2] = [-121, +121]
    const size_t M = 17, K = 23, N = 19;
    const size_t P_planes = 5, Q_planes = 6;

    std::vector<int64_t> A_int(M * K);
    std::vector<int64_t> B_int(K * N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int64_t val = static_cast<int64_t>((i * 31 + k * 17) % 243) - 121;
            A_int[i * K + k] = val;
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            int64_t val = static_cast<int64_t>((k * 29 + j * 13) % 729) - 364;
            B_int[k * N + j] = val;
        }
    }

    MultiTritMatrix A_mt = MultiTritMatrix::encode_int_matrix(A_int, M, K, P_planes);
    MultiTritMatrix B_mt = MultiTritMatrix::encode_int_matrix(B_int, K, N, Q_planes);

    std::vector<int64_t> A_dec = A_mt.decode_int_matrix();
    std::vector<int64_t> B_dec = B_mt.decode_int_matrix();
    assert(A_dec == A_int);
    assert(B_dec == B_int);

    // Verify individual get_int64 / set_int64
    A_mt.set_int64(0, 0, -121);
    assert(A_mt.get_int64(0, 0) == -121);
    A_mt.set_int64(0, 0, A_int[0]);
    assert(A_mt.get_int64(0, 0) == A_int[0]);

    // 2. Verify gemm_multitrit matches exact integer matrix multiplication 100% bit-exact
    DenseMatrixInt64 C_mt(M, N);
    gemm_multitrit(A_mt, B_mt, C_mt);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int64_t ref_sum = 0;
            for (size_t k = 0; k < K; ++k) {
                ref_sum += A_int[i * K + k] * B_int[k * N + j];
            }
            assert(C_mt(i, j) == ref_sum);
        }
    }

    // 3. Verify Codec::to_int64 O(1) SWAR delegation across full 39-trit signed range
    for (int64_t v : {0LL, 1LL, -1LL, 121LL, -121LL, 123456789012345LL, -123456789012345LL}) {
        TritWord64 w = Codec::from_int64(v, 39);
        assert(Codec::to_int64(w, 39) == v);
    }
}

void test_universal_harley_seal_and_bitnet() {
    std::cout << "  [Dense] Testing Universal Harley-Seal (CSA 27-to-4) & BitNet b1.58 INT8 GEMM...\n";

    // 1. Verify CSAContractor40to4::compress27 and dot_product_27words against 27 dotProduct64 calls
    TritWord64 words27[27];
    for (size_t i = 0; i < 27; ++i) {
        uint64_t p = (0x123456789ABCDEF0ULL ^ (i * 0x0101010101010101ULL));
        uint64_t n = (~p & 0x5555555555555555ULL);
        words27[i] = TritWord64(p & ~n, n);
    }
    int64_t ref_dot27 = 0;
    for (size_t i = 0; i < 27; ++i) {
        ref_dot27 += CSAContractor40to4::fast_rail_diff(words27[i]);
    }
    int64_t csa_dot27 = CSAContractor40to4::dot_product_27words(words27);
    assert(csa_dot27 == ref_dot27 && "CSAContractor40to4::dot_product_27words must match sum of 27 words!");

    // 2. Verify gemm_universal_harley_seal against standard gemm
    const size_t M = 32, K = 128, N = 32;
    TernaryMatrix A_mat(M, K);
    TernaryMatrix B_mat(K, N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < K; ++c) {
            A_mat.set(r, c, (r + c) % 3 == 0 ? Trit::Pos : ((r + c) % 3 == 1 ? Trit::Neg : Trit::Zero));
        }
    }
    for (size_t r = 0; r < K; ++r) {
        for (size_t c = 0; c < N; ++c) {
            B_mat.set(r, c, (r * 2 + c) % 3 == 0 ? Trit::Pos : ((r * 2 + c) % 3 == 1 ? Trit::Neg : Trit::Zero));
        }
    }
    DenseMatrixInt64 C_ref(M, N);
    gemm(A_mat, B_mat, C_ref);

    DenseMatrixInt64 C_harley(M, N);
    gemm_universal_harley_seal(A_mat, B_mat, C_harley);

    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            assert(C_harley(r, c) == C_ref(r, c) && "gemm_universal_harley_seal must match reference gemm!");
        }
    }

    // 3. Verify BitNet b1.58 Ternary-Weight x INT8-Activation GEMM (gemm_bitnet_i8_ternary)
    std::vector<int8_t> X_act(K * N);
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            X_act[k * N + j] = static_cast<int8_t>((static_cast<int>(k * 7 + j * 13) % 255) - 128);
        }
    }

    std::vector<int64_t> C_bitnet(M * N, 0);
    gemm_bitnet_i8_ternary(A_mat, X_act, C_bitnet, M, N, K);

    // Reference compute
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int64_t ref_val = 0;
            for (size_t k = 0; k < K; ++k) {
                int64_t w = to_int(A_mat.get(i, k));
                int64_t x = X_act[k * N + j];
                ref_val += w * x;
            }
            assert(C_bitnet[i * N + j] == ref_val && "BitNet b1.58 GEMM must match exact integer reference!");
        }
    }
}

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Dense Algebra & GEMM Tests...\n";
    test_kleene_logic();
    test_tiled_gemm();
    test_codec_and_gemm_guards();
    test_cache_autotuner_and_gemm_autotuned();
    test_multitrit_encode_decode_and_gemm();
    test_universal_harley_seal_and_bitnet();
    std::cout << "  -> PASSED: All Dense Algebra & CacheAutotuner Unit Tests Verified.\n";
    return 0;
}
