/**
 * @file test_unit_microkernel.cpp
 * @brief Correctness + performance tests for bt::blas::ternary_gemm.
 *
 * Correctness is established by differential comparison against a naive
 * scalar reference over many random shapes, including dimensions that are NOT
 * multiples of the register tile (4) or the word width (64) - these exercise
 * the padded-tile store guard and the scalar tail loop.
 */

#include "Ternary/Blas/TernaryMicroKernel.hpp"
#include "ternary_dense_algebra.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace bt;
using CK = std::chrono::high_resolution_clock;

static int g_failures = 0;

static void require(bool cond, const char* what) {
    if (!cond) { std::printf("  FAIL: %s\n", what); ++g_failures; }
}

// ---------------------------------------------------------------------------
// Reference: naive O(m*n*k) triple loop over trits. Slow and obviously correct.
// ---------------------------------------------------------------------------
struct Operands {
    size_t m, n, k, kw;
    std::vector<uint64_t> Ap, An;   // A: row-major words, Ap[i*kw + w]
    std::vector<uint64_t> Bp, Bn;   // B: word-major cols, Bp[w*n + j]
};

static Operands make_operands(size_t m, size_t n, size_t k, std::mt19937_64& g) {
    Operands o; o.m = m; o.n = n; o.k = k; o.kw = (k + 63) / 64;
    o.Ap.assign(m * o.kw, 0); o.An.assign(m * o.kw, 0);
    o.Bp.assign(o.kw * n, 0); o.Bn.assign(o.kw * n, 0);
    // Only fill trits < k so padding bits above k stay zero (documented
    // precondition of the packed word layout).
    for (size_t i = 0; i < m; ++i)
        for (size_t t = 0; t < k; ++t) {
            unsigned v = g() % 3; uint64_t bit = 1ULL << (t % 64);
            if (v == 0) o.Ap[i * o.kw + t / 64] |= bit;
            else if (v == 1) o.An[i * o.kw + t / 64] |= bit;
        }
    for (size_t j = 0; j < n; ++j)
        for (size_t t = 0; t < k; ++t) {
            unsigned v = g() % 3; uint64_t bit = 1ULL << (t % 64);
            if (v == 0) o.Bp[(t / 64) * n + j] |= bit;
            else if (v == 1) o.Bn[(t / 64) * n + j] |= bit;
        }
    return o;
}

static std::vector<int64_t> reference_gemm(const Operands& o) {
    std::vector<int64_t> C(o.m * o.n, 0);
    for (size_t i = 0; i < o.m; ++i)
        for (size_t j = 0; j < o.n; ++j) {
            int64_t acc = 0;
            for (size_t t = 0; t < o.k; ++t) {
                int a = ((o.Ap[i * o.kw + t / 64] >> (t % 64)) & 1)
                      - ((o.An[i * o.kw + t / 64] >> (t % 64)) & 1);
                int b = ((o.Bp[(t / 64) * o.n + j] >> (t % 64)) & 1)
                      - ((o.Bn[(t / 64) * o.n + j] >> (t % 64)) & 1);
                acc += int64_t(a) * int64_t(b);
            }
            C[i * o.n + j] = acc;
        }
    return C;
}

// ---------------------------------------------------------------------------
static void test_correctness() {
    std::printf("  [MicroKernel] differential correctness vs naive reference...\n");
    std::mt19937_64 g(20261103);

    // Deliberately awkward shapes: not multiples of MR/NR (4) nor of 64.
    const size_t shapes[][3] = {
        {1,1,1}, {1,1,7}, {3,5,9}, {4,4,64}, {5,3,65},
        {7,7,70}, {8,4,128}, {13,11,127}, {16,16,200},
        {17,5,256}, {32,32,384}, {33,31,129}, {64,64,64}, {10,10,513}
    };
    int cases = 0;
    for (auto& s : shapes) {
        Operands o = make_operands(s[0], s[1], s[2], g);
        std::vector<int64_t> ref = reference_gemm(o);

        for (int tile : {4, 8}) {
            for (int forced = 0; forced < 2; ++forced) {
                blas::GemmOptions opt;
                opt.mr = tile; opt.nr = 4;
                if (forced) opt.isa = blas::Isa::Scalar;   // else Auto
                std::vector<int64_t> C(o.m * o.n, -12345);
                blas::ternary_gemm(o.Ap.data(), o.An.data(), o.Bp.data(), o.Bn.data(),
                                   C.data(), o.m, o.n, o.k, opt);
                bool ok = true;
                for (size_t i = 0; i < o.m * o.n; ++i)
                    if (C[i] != ref[i]) { ok = false; break; }
                if (!ok) {
                    std::printf("    MISMATCH m=%zu n=%zu k=%zu mr=%d forced_scalar=%d\n",
                                s[0], s[1], s[2], tile, forced);
                    ++g_failures;
                }
                ++cases;
            }
        }
    }
    std::printf("    -> %d differential cases executed\n", cases);
}

// ---------------------------------------------------------------------------
static void test_packed_tensor() {
    std::printf("  [MicroKernel] PackedTensor round-trip and density...\n");
    std::mt19937_64 g(7);
    const size_t rows = 64, cols = 1285;   // deliberately not a multiple of 5
    blas::PackedTensor T(rows, cols);
    std::vector<std::vector<int>> gold(rows, std::vector<int>(cols, 0));
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            int v = int(g() % 3) - 1; gold[r][c] = v;
            T.set(r, c, v == 1 ? Trit::Pos : v == -1 ? Trit::Neg : Trit::Zero);
        }
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            int got = (T.get(r, c) == Trit::Pos) ? 1 : (T.get(r, c) == Trit::Neg) ? -1 : 0;
            if (got != gold[r][c]) { require(false, "PackedTensor round-trip"); return; }
        }
    double bpt = T.bits_per_trit();
    std::printf("    -> %zux%zu stored in %zu bytes = %.3f bits/trit "
                "(dual-rail would be 2.000, FP32 weight would be 32.000)\n",
                rows, cols, T.bytes(), bpt);
    require(bpt < 2.0, "packed storage must beat dual-rail density");

    // Unpack a row and confirm it matches.
    const size_t kw = (cols + 63) / 64;
    std::vector<uint64_t> P(kw, 0), N(kw, 0);
    T.unpack_row(3, P.data(), N.data(), kw);
    for (size_t c = 0; c < cols; ++c) {
        int got = ((P[c / 64] >> (c % 64)) & 1) - ((N[c / 64] >> (c % 64)) & 1);
        if (got != gold[3][c]) { require(false, "PackedTensor::unpack_row"); return; }
    }
}

// ---------------------------------------------------------------------------
static void test_checked_mode() {
    std::printf("  [MicroKernel] checked accumulator bound...\n");
    std::mt19937_64 g(5);
    Operands o = make_operands(8, 8, 128, g);
    std::vector<int64_t> C(o.m * o.n, 0);
    blas::GemmOptions opt; opt.checked = true; opt.acc_limit = 1;  // bound is tiny
    bool threw = false;
    try {
        blas::ternary_gemm(o.Ap.data(), o.An.data(), o.Bp.data(), o.Bn.data(),
                           C.data(), o.m, o.n, o.k, opt);
    } catch (const std::overflow_error&) { threw = true; }
    require(threw, "checked mode must throw when the accumulator exceeds the bound");

    // A generous bound must NOT throw.
    blas::GemmOptions ok; ok.checked = true; ok.acc_limit = 1000000;
    bool threw2 = false;
    try {
        blas::ternary_gemm(o.Ap.data(), o.An.data(), o.Bp.data(), o.Bn.data(),
                           C.data(), o.m, o.n, o.k, ok);
    } catch (const std::overflow_error&) { threw2 = true; }
    require(!threw2, "checked mode must not throw when within bound");
}

// ---------------------------------------------------------------------------
template <class F> static double best_of(F f, int reps) {
    double b = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = CK::now(); f();
        double d = std::chrono::duration<double>(CK::now() - t0).count();
        if (d < b) b = d;
    }
    return b;
}

static void benchmark() {
    std::printf("\n  [MicroKernel] throughput (higher is better)\n");
    std::printf("    %-34s %6s %10s %12s\n", "kernel", "n", "ms", "G MAC/s");
    std::mt19937_64 g(99);
    for (int n : {256, 512}) {
        Operands o = make_operands(n, n, n, g);
        std::vector<int64_t> C((size_t)n * n, 0);
        double mac = double(n) * double(n) * double(n);

        auto run_scalar = [&]{
            blas::GemmOptions opt; opt.isa = blas::Isa::Scalar; opt.mr = 4;
            blas::ternary_gemm(o.Ap.data(), o.An.data(), o.Bp.data(), o.Bn.data(),
                               C.data(), n, n, n, opt); };
        auto run_auto8 = [&]{
            blas::GemmOptions opt; opt.mr = 8;
            blas::ternary_gemm(o.Ap.data(), o.An.data(), o.Bp.data(), o.Bn.data(),
                               C.data(), n, n, n, opt); };

        double d1 = best_of(run_scalar, 3);
        double d2 = best_of(run_auto8, 3);
        std::printf("    %-34s %6d %10.2f %12.2f\n", "ternary_gemm scalar tile4x4", n, d1*1e3, mac/d1/1e9);

        blas::GemmOptions probe; blas::Isa used = blas::detect_isa();
        std::printf("    %-34s %6d %10.2f %12.2f   [%s, tile 8x4]\n",
                    "ternary_gemm auto", n, d2*1e3, mac/d2/1e9, blas::isa_name(used));
        (void)probe;

        // existing in-tree reference for comparison
        TernaryMatrix A(n, n), B(n, n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            A.set(i, j, ((o.Ap[(size_t)i*o.kw + j/64] >> (j%64)) & 1) ? Trit::Pos
                        : ((o.An[(size_t)i*o.kw + j/64] >> (j%64)) & 1) ? Trit::Neg : Trit::Zero);
            B.set(i, j, ((o.Bp[(size_t)(j/64)*n + i] >> (j%64)) & 1) ? Trit::Pos
                        : ((o.Bn[(size_t)(j/64)*n + i] >> (j%64)) & 1) ? Trit::Neg : Trit::Zero);
        }
        DenseMatrixInt64 D(n, n);
        double d3 = best_of([&]{ gemm_autotuned(A, B, D); }, 3);
        std::printf("    %-34s %6d %10.2f %12.2f\n", "gemm_autotuned (previous best)", n, d3*1e3, mac/d3/1e9);
        std::printf("\n");
    }
}

// ---------------------------------------------------------------------------
int main() {
    std::printf("=== bt::blas::ternary_gemm unit tests ===\n");
    std::printf("  detected ISA: %s\n\n", blas::isa_name(blas::detect_isa()));
    test_correctness();
    test_packed_tensor();
    test_checked_mode();
    benchmark();

    if (g_failures) {
        std::printf("\n  FAILED: %d assertion(s)\n", g_failures);
        return 1;
    }
    std::printf("  -> PASSED: all micro-kernel tests verified.\n");
    return 0;
}
