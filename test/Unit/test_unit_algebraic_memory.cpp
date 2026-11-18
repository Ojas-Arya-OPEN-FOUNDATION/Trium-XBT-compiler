/**
 * @file test_unit_algebraic_memory.cpp
 * @brief Comprehensive self-checking unit test suite for CirculantWeighing.hpp
 *        and AlgebraicVirtualMemory.hpp.
 */

#include "Ternary/Algebra/CirculantWeighing.hpp"
#include "Ternary/Algebra/AlgebraicVirtualMemory.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [%s:%d]: %s\n", file, line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

// Deterministic PRNG helper
std::mt19937_64 make_rng(uint64_t seed) {
    return std::mt19937_64(seed);
}

std::vector<int64_t> random_i64_vec(size_t n, int64_t lo, int64_t hi, std::mt19937_64& rng) {
    std::uniform_int_distribution<int64_t> dist(lo, hi);
    std::vector<int64_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

// ============================================================================
// Test 1: CW(120, 36) & CW(2m, 4) Orthogonality, Naive MatVec, Exact Inverse
// ============================================================================
void test_cw_orthogonality_and_inverse() {
    using bt::algebra::CirculantTernaryOperator;

    auto rng = make_rng(0x12036ULL);

    // 1a. Verify CW(120, 36)
    {
        const auto cw = CirculantTernaryOperator::cw_120_36();
        CHECK(cw.dim() == 120);
        CHECK(cw.weight_k == 36);
        CHECK(cw.verify_orthogonality());
        // O(n) seed storage vs 120^2 = 14,400 dense entries
        CHECK(cw.memory_bytes() < 1024);
        CHECK(cw.virtual_matrix_elements() == 120ULL * 120ULL);

        // Build explicit naive O(n^2) dense matrix
        const size_t n = cw.dim();
        std::vector<int8_t> dense(n * n, 0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                dense[i * n + j] = cw.entry(i, j);
            }
        }

        // Test int64_t apply vs naive dense MatVec & exact inverse
        const auto x = random_i64_vec(n, -100000, 100000, rng);
        std::vector<int64_t> y_op(n, 0);
        std::vector<int64_t> y_ref(n, 0);
        cw.apply(x, y_op);

        for (size_t i = 0; i < n; ++i) {
            int64_t acc = 0;
            for (size_t j = 0; j < n; ++j) {
                acc += static_cast<int64_t>(dense[i * n + j]) * x[j];
            }
            y_ref[i] = acc;
        }
        CHECK(y_op == y_ref);

        std::vector<int64_t> x_rec(n, 0);
        const bool exact = cw.apply_exact_inverse(y_op, x_rec);
        CHECK(exact);
        CHECK(x_rec == x);

        // Test double apply vs naive dense MatVec & inverse_double
        std::vector<double> xd(n), yd_op(n, 0.0), yd_ref(n, 0.0), xd_rec(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            xd[i] = static_cast<double>(x[i]);
        }
        cw.apply_double(xd, yd_op);
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < n; ++j) {
                acc += static_cast<double>(dense[i * n + j]) * xd[j];
            }
            yd_ref[i] = acc;
        }
        CHECK(yd_op == yd_ref); // integer-valued doubles in [-1e5, 1e5] are exact

        cw.apply_inverse_double(yd_op, xd_rec);
        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::abs(xd_rec[i] - xd[i]));
        }
        CHECK(max_err < 1e-10);
    }

    // 1b. Verify CW(2m, 4) for m in {2, 3, 5, 16, 64}
    const size_t ms[] = {2, 3, 5, 16, 64};
    for (size_t m : ms) {
        const auto cw = CirculantTernaryOperator::cw_even_weight4(m);
        const size_t n = 2 * m;
        CHECK(cw.dim() == n);
        CHECK(cw.weight_k == 4);
        CHECK(cw.verify_orthogonality());

        const auto x = random_i64_vec(n, -50000, 50000, rng);
        std::vector<int64_t> y_op(n, 0), y_ref(n, 0), x_rec(n, 0);
        cw.apply(x, y_op);
        for (size_t i = 0; i < n; ++i) {
            int64_t acc = 0;
            for (size_t j = 0; j < n; ++j) {
                acc += static_cast<int64_t>(cw.entry(i, j)) * x[j];
            }
            y_ref[i] = acc;
        }
        CHECK(y_op == y_ref);
        CHECK(cw.apply_exact_inverse(y_op, x_rec));
        CHECK(x_rec == x);
    }

    std::printf("PASS Test 1: CW(120,36) & CW(2m,4) Orthogonality & Exact Inverse\n");
}

// ============================================================================
// Test 2: GoethalsSeidelSkewHadamard Orders 212 & 220
// ============================================================================
void test_goethals_seidel_skew_hadamard() {
    using bt::algebra::GoethalsSeidelSkewHadamard;

    auto rng = make_rng(0x212220ULL);

    const std::vector<GoethalsSeidelSkewHadamard> ops = {
        GoethalsSeidelSkewHadamard::order_212(),
        GoethalsSeidelSkewHadamard::order_220()
    };

    for (const auto& H : ops) {
        CHECK(H.verify_orthogonality());
        CHECK(GoethalsSeidelSkewHadamard::seed_bytes() == 32);

        const size_t n = H.block_dim();
        const size_t N = H.dim();
        CHECK(N == 4 * n);

        // Construct explicit 4n x 4n dense matrix according to Goethals-Seidel block layout
        std::vector<int8_t> dense(N * N, 0);
        auto set_circ = [&](size_t br, size_t bc, size_t seq_idx, bool reversed, int8_t sign) {
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    const size_t idx = reversed ? ((2 * n - i - j) % n)
                                                : ((j >= i) ? (j - i) : (j + n - i));
                    dense[(br * n + i) * N + (bc * n + j)] =
                        static_cast<int8_t>(sign * H.seq_entry(seq_idx, idx));
                }
            }
        };

        // Row 0: [ A, BR, CR, DR ]
        set_circ(0, 0, 0, false,  1);
        set_circ(0, 1, 1, true,   1);
        set_circ(0, 2, 2, true,   1);
        set_circ(0, 3, 3, true,   1);
        // Row 1: [ -BR, A, DR, -CR ]
        set_circ(1, 0, 1, true,  -1);
        set_circ(1, 1, 0, false,  1);
        set_circ(1, 2, 3, true,   1);
        set_circ(1, 3, 2, true,  -1);
        // Row 2: [ -CR, -DR, A, BR ]
        set_circ(2, 0, 2, true,  -1);
        set_circ(2, 1, 3, true,  -1);
        set_circ(2, 2, 0, false,  1);
        set_circ(2, 3, 1, true,   1);
        // Row 3: [ -DR, CR, -BR, A ]
        set_circ(3, 0, 3, true,  -1);
        set_circ(3, 1, 2, true,   1);
        set_circ(3, 2, 1, true,  -1);
        set_circ(3, 3, 0, false,  1);

        const auto x = random_i64_vec(N, -10000, 10000, rng);
        std::vector<int64_t> y_op(N, 0), y_ref(N, 0), x_rec(N, 0);
        H.apply(x, y_op);

        for (size_t i = 0; i < N; ++i) {
            int64_t acc = 0;
            for (size_t j = 0; j < N; ++j) {
                acc += static_cast<int64_t>(dense[i * N + j]) * x[j];
            }
            y_ref[i] = acc;
        }
        CHECK(y_op == y_ref);
        CHECK(H.apply_exact_inverse(y_op, x_rec));
        CHECK(x_rec == x);
    }

    std::printf("PASS Test 2: GoethalsSeidelSkewHadamard Orders 212 & 220\n");
}

// ============================================================================
// Test 3: KroneckerTernaryOperator Sub-Linear Memory + Exact Slice Check
// ============================================================================
void test_kronecker_ternary_operator() {
    using bt::algebra::CirculantTernaryOperator;
    using bt::algebra::KroneckerTernaryOperator;

    auto rng = make_rng(0x96012000ULL);

    // 3a. 2-fold: CW(2*4, 4) (x) CW(120, 36), D = 960
    {
        KroneckerTernaryOperator kron({
            CirculantTernaryOperator::cw_even_weight4(4),
            CirculantTernaryOperator::cw_120_36()
        });
        const size_t D = kron.dim();
        CHECK(D == 960);
        CHECK(kron.total_weight() == 4ULL * 36ULL);
        CHECK(kron.memory_bytes() < 1024);

        const auto x = random_i64_vec(D, -1000, 1000, rng);
        std::vector<int64_t> y(D, 0), x_rec(D, 0);
        kron.apply(x, y);

        // Spot-check sampled rows against explicit eval_entry(r, c)
        const size_t sample_rows[] = {0, 1, 7, 119, 120, 479, 512, 959};
        for (size_t r : sample_rows) {
            int64_t dot = 0;
            for (size_t c = 0; c < D; ++c) {
                dot += static_cast<int64_t>(kron.eval_entry(r, c)) * x[c];
            }
            CHECK(y[r] == dot);
        }

        CHECK(kron.apply_exact_inverse(y, x_rec));
        CHECK(x_rec == x);
    }

    // 3b. 3-fold: CW(2*5, 4) (x) CW(120, 36) (x) CW(2*5, 4), D = 12,000
    {
        KroneckerTernaryOperator kron({
            CirculantTernaryOperator::cw_even_weight4(5),
            CirculantTernaryOperator::cw_120_36(),
            CirculantTernaryOperator::cw_even_weight4(5)
        });
        const size_t D = kron.dim();
        CHECK(D == 12000);
        CHECK(kron.total_weight() == 4ULL * 36ULL * 4ULL);
        CHECK(kron.memory_bytes() < 2048);
        // 12,000^2 * 4 = 576,000,000 bytes (576 MB FP32)
        CHECK(std::abs(kron.virtual_fp32_bytes() - 576000000.0) < 1.0);

        const auto x = random_i64_vec(D, -500, 500, rng);
        std::vector<int64_t> y(D, 0), x_rec(D, 0);
        kron.apply(x, y);

        // Spot-check a few rows via eval_entry
        const size_t sample_rows[] = {0, 11, 1200, 6001, 11999};
        for (size_t r : sample_rows) {
            int64_t dot = 0;
            for (size_t c = 0; c < D; ++c) {
                dot += static_cast<int64_t>(kron.eval_entry(r, c)) * x[c];
            }
            CHECK(y[r] == dot);
        }

        CHECK(kron.apply_exact_inverse(y, x_rec));
        CHECK(x_rec == x);
    }

    std::printf("PASS Test 3: KroneckerTernaryOperator Sub-Linear Memory & Exact Inverse\n");
}

// ============================================================================
// Test 4: GeneralAlgebraicOperator (Circulant + Rank-2 UV^T + Sparse Residual)
// ============================================================================
void test_general_algebraic_operator() {
    using bt::algebra::CirculantTernaryOperator;
    using bt::algebra::GeneralAlgebraicOperator;

    auto rng = make_rng(0x4E4E4EULL);
    const size_t n = 120;
    const size_t rank = 2;

    GeneralAlgebraicOperator op(n);
    const auto cw = CirculantTernaryOperator::cw_120_36();
    op.add_circulant(cw);

    std::uniform_int_distribution<int64_t> small_dist(-10, 10);
    std::vector<double> U(n * rank), V(n * rank);
    for (size_t i = 0; i < n * rank; ++i) {
        U[i] = static_cast<double>(small_dist(rng));
        V[i] = static_cast<double>(small_dist(rng));
    }
    op.add_low_rank(U, V, rank);

    // Add sparse residual triplets (including multiple entries on same/diff rows)
    std::uniform_int_distribution<uint32_t> idx_dist(0, static_cast<uint32_t>(n - 1));
    for (int t = 0; t < 40; ++t) {
        op.add_sparse_triplet(idx_dist(rng), idx_dist(rng),
                              static_cast<double>(small_dist(rng)));
    }

    // Build explicit 120 x 120 dense matrix M = W + U*V^T + S
    std::vector<double> M(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double val = static_cast<double>(cw.entry(i, j));
            for (size_t r = 0; r < rank; ++r) {
                val += U[i * rank + r] * V[j * rank + r];
            }
            M[i * n + j] = val;
        }
    }
    for (const auto& sp : op.sparse_residual) {
        M[sp.row * n + sp.col] += sp.val;
    }

    std::vector<double> x(n), y_op(n), y_ref(n, 0.0), xt_op(n), xt_ref(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(small_dist(rng));
    }

    op.apply(x, y_op);
    for (size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < n; ++j) {
            acc += M[i * n + j] * x[j];
        }
        y_ref[i] = acc;
    }
    CHECK(y_op == y_ref);

    op.apply_transpose(x, xt_op);
    for (size_t j = 0; j < n; ++j) {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i) {
            acc += M[i * n + j] * x[i];
        }
        xt_ref[j] = acc;
    }
    CHECK(xt_op == xt_ref);

    std::printf("PASS Test 4: GeneralAlgebraicOperator Forward & Transpose Bit-Exact\n");
}

// ============================================================================
// Test 5: ZeroSwapStreamStore<int64_t> & VirtualStencil3D<double>
// ============================================================================
void test_stream_store_and_virtual_stencil() {
    using bt::vmem::VirtualStencil3D;
    using bt::vmem::ZeroSwapStreamStore;

    // 5a. ZeroSwapStreamStore<int64_t> with N = 2,097,152 (16 MB virtual) in 512 KB slab
    {
        constexpr size_t N = 2097152;
        constexpr size_t SLAB_BYTES = 512 * 1024;
        ZeroSwapStreamStore<int64_t> src(N, SLAB_BYTES);
        ZeroSwapStreamStore<int64_t> dst(N, SLAB_BYTES);

        CHECK(src.virtual_bytes() == N * sizeof(int64_t));
        CHECK(src.resident_budget_bytes() == SLAB_BYTES);

        auto gen_fn = [](size_t i) -> int64_t {
            return static_cast<int64_t>((i * 7ULL + 13ULL) % 100003ULL) - 50000LL;
        };
        auto xform_fn = [](size_t i, int64_t v) -> int64_t {
            return v * 3LL - static_cast<int64_t>(i % 97ULL);
        };

        src.generate(gen_fn);
        src.transform_to(dst, xform_fn);

        const int64_t got_sum = dst.stream_reduce(int64_t{0}, [](int64_t acc, int64_t v) {
            return acc + v;
        });

        int64_t expected_sum = 0;
        for (size_t i = 0; i < N; ++i) {
            expected_sum += xform_fn(i, gen_fn(i));
        }
        CHECK(got_sum == expected_sum);

        const size_t spot_indices[] = {
            0, 1, 65535, 65536, 65537, 131072, 1048575, 1048576, N - 2, N - 1
        };
        for (size_t idx : spot_indices) {
            CHECK(src.read_element(idx) == gen_fn(idx));
            CHECK(dst.read_element(idx) == xform_fn(idx, gen_fn(idx)));
        }
    }

    // 5b. VirtualStencil3D<double> on 32 x 32 x 32 grid
    {
        constexpr size_t nx = 32, ny = 32, nz = 32;
        constexpr size_t total = nx * ny * nz;
        VirtualStencil3D<double> stencil(nx, ny, nz);
        CHECK(stencil.total_points() == total);
        CHECK(stencil.matrix_ram_bytes() == 0);

        ZeroSwapStreamStore<double> x_store(total, 256 * 1024);
        ZeroSwapStreamStore<double> y_store(total, 256 * 1024);

        std::vector<double> x_host(total);
        auto idx3 = [=](size_t ix, size_t iy, size_t iz) -> size_t {
            return (iz * ny + iy) * nx + ix;
        };

        for (size_t iz = 0; iz < nz; ++iz) {
            for (size_t iy = 0; iy < ny; ++iy) {
                for (size_t ix = 0; ix < nx; ++ix) {
                    const double val = std::sin(0.17 * static_cast<double>(ix + 1)) +
                                       std::cos(0.23 * static_cast<double>(iy + 1)) +
                                       0.05 * static_cast<double>(iz * iz);
                    x_host[idx3(ix, iy, iz)] = val;
                }
            }
        }
        x_store.write_range(0, x_host.data(), total);

        stencil.apply_laplacian_3d(x_store, y_store);

        std::vector<double> y_got(total);
        y_store.read_range(0, y_got.data(), total);

        double max_abs_err = 0.0;
        for (size_t iz = 0; iz < nz; ++iz) {
            for (size_t iy = 0; iy < ny; ++iy) {
                for (size_t ix = 0; ix < nx; ++ix) {
                    const size_t i = idx3(ix, iy, iz);
                    const double c  = x_host[i];
                    const double xm = (ix > 0) ? x_host[idx3(ix - 1, iy, iz)] : 0.0;
                    const double xp = (ix + 1 < nx) ? x_host[idx3(ix + 1, iy, iz)] : 0.0;
                    const double ym = (iy > 0) ? x_host[idx3(ix, iy - 1, iz)] : 0.0;
                    const double yp = (iy + 1 < ny) ? x_host[idx3(ix, iy + 1, iz)] : 0.0;
                    const double zm = (iz > 0) ? x_host[idx3(ix, iy, iz - 1)] : 0.0;
                    const double zp = (iz + 1 < nz) ? x_host[idx3(ix, iy, iz + 1)] : 0.0;
                    const double ref = -6.0 * c + xm + xp + ym + yp + zm + zp;
                    max_abs_err = std::max(max_abs_err, std::abs(y_got[i] - ref));
                }
            }
        }
        CHECK(max_abs_err < 1e-12);
    }

    std::printf("PASS Test 5: ZeroSwapStreamStore<int64_t> & VirtualStencil3D<double>\n");
}

// ============================================================================
// Test 6: ReversibleSymplecticTape (50,000 forward + 50,000 backward)
// ============================================================================
void test_reversible_symplectic_tape() {
    using bt::vmem::ReversibleSymplecticTape;
    using PhaseState = ReversibleSymplecticTape::PhaseState;

    const bt::TritWord64 q0(0x0F0F0F0F00FF00FFULL, 0xF0F00000FF000000ULL);
    const bt::TritWord64 p0(0x1234000056780000ULL, 0x0000ABCD00001234ULL);
    const bt::TritWord64 force(0x00AA005500AA0055ULL, 0x5500AA005500AA00ULL);

    CHECK(q0.isValid() && p0.isValid() && force.isValid());

    const PhaseState init_state(q0, p0);
    ReversibleSymplecticTape tape(init_state, 0);

    constexpr uint64_t STEPS = 50000;
    tape.step_forward(STEPS, force);
    CHECK(tape.step_count() == STEPS);
    CHECK(tape.history_ram_bytes() <= 64);

    tape.step_backward(STEPS, force);
    CHECK(tape.step_count() == 0);
    CHECK(tape.history_ram_bytes() <= 64);

    const PhaseState rec = tape.current_state();
    CHECK(rec.Q.P == init_state.Q.P && rec.Q.N == init_state.Q.N);
    CHECK(rec.P.P == init_state.P.P && rec.P.N == init_state.P.N);

    std::printf("PASS Test 6: ReversibleSymplecticTape (50,000 fwd/bwd O(1) RAM)\n");
}

} // namespace

int main() {
    test_cw_orthogonality_and_inverse();
    test_goethals_seidel_skew_hadamard();
    test_kronecker_ternary_operator();
    test_general_algebraic_operator();
    test_stream_store_and_virtual_stencil();
    test_reversible_symplectic_tape();

    if (g_failures == 0) {
        std::printf("\nALL 6 ALGEBRAIC VIRTUAL MEMORY UNIT TESTS PASSED.\n");
        return 0;
    }
    std::fprintf(stderr, "\nFAILED: %d assertion(s) failed.\n", g_failures);
    return 1;
}
