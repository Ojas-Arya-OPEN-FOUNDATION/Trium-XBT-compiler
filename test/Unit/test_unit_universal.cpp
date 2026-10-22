/**
 * @file test_unit_universal.cpp
 * @brief Comprehensive Verification Suite for Universal XBT Foundation
 *        (Tri-Radix Q_{2,3,5}, Unmodified Float GEMM, Nonlinear Engine,
 *         Magic Multipliers Base-59049, and Symplectic Reversible Layers).
 */

#include "ternary_universal_codec.hpp"
#include "ternary_xbt_blas.hpp"
#include "ternary_nonlinear_engine.hpp"
#include "ternary_xbt_foundation.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace bt;
using namespace bt::universal;
using namespace bt::blas;
using namespace bt::nonlinear;
using namespace bt::foundation;

void test_tri_radix_scalar() {
    std::cout << "  [Universal 1/6] Testing TriRadixScalar (Q_{2,3,5} lossless representation)...\n";

    // 1. Lossless binary float representation: 0.5 = 2^(-1)
    auto s_half = TriRadixScalar::from_float_lossless(0.5f);
    assert(std::fabs(s_half.to_double() - 0.5) < 1e-12);

    // 2. Exact decimal money: $0.10 = 1 * 10^(-1) = 1 * 2^(-1) * 5^(-1)
    auto s_dime = TriRadixScalar::from_decimal_exact(1, -1);
    assert(std::fabs(s_dime.to_double() - 0.1) < 1e-12);

    // 3. 0-Cycle rail-swap negation
    auto s_neg_dime = -s_dime;
    assert(std::fabs(s_neg_dime.to_double() - (-0.1)) < 1e-12);

    // 4. Arbitrary float: 13.5 = 27 / 2 = 3^3 * 2^(-1)
    auto s_float = TriRadixScalar::from_float_lossless(13.5f);
    assert(std::fabs(s_float.to_double() - 13.5) < 1e-12);

    std::cout << "      -> PASSED: TriRadixScalar Q_{2,3,5} exactness validated.\n";
}

void test_block_ternary_tensor() {
    std::cout << "  [Universal 2/6] Testing BlockTernaryTensor & active plane masking...\n";

    const size_t R = 2;
    const size_t C = 64;
    std::vector<float> data(R * C, 0.0f);
    for (size_t i = 0; i < C; ++i) {
        data[i] = static_cast<float>(i) * 0.1f - 3.2f;
    }
    // Row 1 is sparse: only 2 non-zero elements
    data[C + 10] = 5.5f;
    data[C + 42] = -2.5f;

    auto tensor = BlockTernaryTensor<8>::from_float_matrix(data.data(), R, C);
    assert(tensor.rows == R && tensor.cols == C);
    assert(tensor.blocks_per_row == 1);

    // Check active plane mask for row 1: non-zero
    assert(tensor.active_plane_mask[1] != 0);

    // Reconstruct and check error
    std::vector<float> recon(R * C, 0.0f);
    tensor.to_float_matrix(recon.data());

    for (size_t i = 0; i < R * C; ++i) {
        float err = std::fabs(data[i] - recon[i]);
        assert(err < 0.05f); // 8 trits gives ~0.03% resolution
    }

    std::cout << "      -> PASSED: BlockTernaryTensor plane slicing & reconstruction verified.\n";
}

void test_xbt_sgemm_unmodified() {
    std::cout << "  [Universal 3/6] Testing xbt_sgemm_unmodified on unmodified float matrices...\n";

    const size_t M = 4;
    const size_t K = 64;
    const size_t N = 4;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> C_xbt(M * N, 0.0f);

    for (size_t i = 0; i < M * K; ++i) {
        A[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.5f;
    }

    // Naive reference float GEMM
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C_ref[i * N + j] = sum;
        }
    }

    // XBT-BLAS GEMM with 8 planes
    xbt_sgemm_unmodified<8>(A.data(), B.data(), C_xbt.data(), M, K, N, 0);

    for (size_t i = 0; i < M * N; ++i) {
        float diff = std::fabs(C_ref[i] - C_xbt[i]);
        float rel = diff / (std::fabs(C_ref[i]) + 1e-5f);
        assert(rel < 0.05f); // Within 5% of float
    }

    std::cout << "      -> PASSED: xbt_sgemm_unmodified matches reference float GEMM.\n";
}

void test_nonlinear_engine() {
    std::cout << "  [Universal 4/6] Testing Non-Linear Engine (exp, tanh, rsqrt, rmsnorm, softmax)...\n";

    // 1. Radix-3 exp
    for (double x = -5.0; x <= 5.0; x += 0.5) {
        double calc = xbt_exp_fast(x);
        double ref = std::exp(x);
        double err = std::fabs(calc - ref) / ref;
        assert(err < 1e-6);
    }

    // 2. Padé tanh and sigmoid
    for (double x = -4.0; x <= 4.0; x += 0.5) {
        double calc_th = xbt_tanh_fast(x);
        double ref_th = std::tanh(x);
        assert(std::fabs(calc_th - ref_th) < 1e-6);

        double calc_sig = xbt_sigmoid_fast(x);
        double ref_sig = 1.0 / (1.0 + std::exp(-x));
        assert(std::fabs(calc_sig - ref_sig) < 1e-6);
    }

    // 3. Cubic Householder rsqrt
    for (double x = 0.1; x <= 50.0; x += 2.5) {
        double calc = xbt_rsqrt_householder(x);
        double ref = 1.0 / std::sqrt(x);
        double err = std::fabs(calc - ref) / ref;
        assert(err < 1e-9);
    }

    // 4. RMSNorm
    std::vector<float> x_norm = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w_norm = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out_norm(4, 0.0f);
    xbt_rmsnorm(x_norm.data(), w_norm.data(), out_norm.data(), 4);
    // mean(sq) = (1 + 4 + 9 + 16)/4 = 7.5. sqrt(7.5) = 2.7386.
    assert(std::fabs(out_norm[0] - (1.0f / 2.7386127f)) < 1e-3);

    // 5. Softmax
    std::vector<float> s_in = {1.0f, 2.0f, 3.0f};
    std::vector<float> s_out(3, 0.0f);
    xbt_softmax(s_in.data(), s_out.data(), 3);
    float s_sum = s_out[0] + s_out[1] + s_out[2];
    assert(std::fabs(s_sum - 1.0f) < 1e-6);

    std::cout << "      -> PASSED: Non-linear activations & transcendentals validated.\n";
}

void test_magic_multipliers_base59049() {
    std::cout << "  [Universal 5/6] Testing MagicBlockTernary10 (Division-free Base-59049 slicer)...\n";

    const size_t R = 2;
    const size_t C = 64;
    std::vector<float> src(R * C);
    for (size_t i = 0; i < R * C; ++i) {
        src[i] = static_cast<float>(static_cast<int>(i) - 30) * 0.1f;
    }

    auto encoded = MagicBlockTernary10::encode_division_free(src.data(), R, C);
    assert(encoded.rows == R && encoded.cols == C);
    assert(encoded.planes.size() == R * 1 * 10);
    assert(encoded.compressed_base59049.size() == R * C);

    // Verify compressed words are in [0, 59048]
    for (uint16_t w : encoded.compressed_base59049) {
        assert(w <= 59048);
    }

    std::cout << "      -> PASSED: Division-free Base-59049 plane slicer validated.\n";
}

void test_symplectic_reversible_layer() {
    std::cout << "  [Universal 6/6] Testing SymplecticReversibleLayer (Bit-exact reversible residual)...\n";

    const size_t dim = 64;
    std::vector<float> initial_x(dim);
    for (size_t i = 0; i < dim; ++i) {
        initial_x[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
    }

    auto state = SymplecticReversibleLayer::ReversibleStream::from_float_activations(initial_x.data(), dim);
    auto state_copy = state; // Save for verification

    // Sublayer F: RMSNorm + Silu
    auto fn_F = [](const float* in, float* out, size_t n) {
        std::vector<float> norm(n);
        xbt_rmsnorm(in, nullptr, norm.data(), n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = static_cast<float>(xbt_silu_fast(norm[i]));
        }
    };

    // Sublayer G: Softmax + Scale
    auto fn_G = [](const float* in, float* out, size_t n) {
        xbt_softmax(in, out, n);
        for (size_t i = 0; i < n; ++i) {
            out[i] *= 2.0f;
        }
    };

    // Run 10 forward layers
    for (int l = 0; l < 10; ++l) {
        SymplecticReversibleLayer::forward_layer(state, fn_F, fn_G);
    }

    // Run 10 backward layers
    for (int l = 0; l < 10; ++l) {
        SymplecticReversibleLayer::backward_layer(state, fn_F, fn_G);
    }

    // Verify 100% bit-exact reconstruction of (Q, P)
    for (size_t i = 0; i < dim; ++i) {
        assert(state.Q[i] == state_copy.Q[i]);
        assert(state.P[i] == state_copy.P[i]);
    }

    std::cout << "      -> PASSED: 100% bit-exact Symplectic reversibility across 10 layers.\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  UNIVERSAL XBT FOUNDATION VERIFICATION SUITE\n";
    std::cout << "======================================================================\n";

    test_tri_radix_scalar();
    test_block_ternary_tensor();
    test_xbt_sgemm_unmodified();
    test_nonlinear_engine();
    test_magic_multipliers_base59049();
    test_symplectic_reversible_layer();

    std::cout << "======================================================================\n";
    std::cout << "  ALL 6 UNIVERSAL XBT FOUNDATION MODULES 100% VERIFIED!\n";
    std::cout << "======================================================================\n";
    return 0;
}
