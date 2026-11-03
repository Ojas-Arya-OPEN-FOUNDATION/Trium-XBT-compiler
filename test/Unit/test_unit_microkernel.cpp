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
#include "Ternary/Blas/TernaryGemmGpu.hpp"
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

// ---------------------------------------------------------------------------
// Task 1: packed storage feeding the GEMM end to end.
// ---------------------------------------------------------------------------
static void test_packed_gemm() {
    std::printf("  [Task1] packed-storage GEMM (1.6 bits/trit) vs dual-rail...\n");
    std::mt19937_64 g(4242);
    const size_t m = 37, n = 29, k = 133;   // awkward sizes on purpose
    Operands o = make_operands(m, n, k, g);
    std::vector<int64_t> ref = reference_gemm(o);
    const size_t kw = o.kw;

    blas::PackedTensor A = blas::packed_from_dual_rail(o.Ap.data(), o.An.data(), m, k, kw);
    // BT packed N x K: row j of BT is column j of B. pack_bt consumes B as
    // Bp[w*n + j], so convert to N-major P[j*kw + w] first.
    std::vector<uint64_t> BTp(n * kw), BTn(n * kw);
    for (size_t j = 0; j < n; ++j)
        for (size_t w = 0; w < kw; ++w) {
            BTp[j * kw + w] = o.Bp[w * n + j];
            BTn[j * kw + w] = o.Bn[w * n + j];
        }
    blas::PackedTensor BT = blas::packed_from_dual_rail(BTp.data(), BTn.data(), n, k, kw);

    std::vector<int64_t> C(m * n, -1);
    blas::ternary_gemm_packed(A, BT, C.data());
    for (size_t i = 0; i < m * n; ++i)
        if (C[i] != ref[i]) { require(false, "packed GEMM must match dual-rail reference"); return; }

    const size_t dual_bytes = (m * kw + n * kw) * 2 * sizeof(uint64_t);
    const size_t pack_bytes = A.bytes() + BT.bytes();
    std::printf("    -> packed matches reference; storage %zu B vs %zu B dual-rail (%.1f%%), %.3f bits/trit\n",
                pack_bytes, dual_bytes, 100.0 * double(pack_bytes) / double(dual_bytes), A.bits_per_trit());
    require(pack_bytes < dual_bytes, "packed storage must be smaller than dual-rail");

    // Measure the tradeoff rather than assuming it: the packed path pays a
    // decompression pass, so it buys memory at some cost in speed. Both
    // outputs are checksummed OUTSIDE the timed region.
    const size_t pn = 512;
    Operands po = make_operands(pn, pn, pn, g);
    std::vector<uint64_t> PBTp(pn * po.kw), PBTn(pn * po.kw);
    for (size_t j = 0; j < pn; ++j)
        for (size_t w = 0; w < po.kw; ++w) {
            PBTp[j * po.kw + w] = po.Bp[w * pn + j];
            PBTn[j * po.kw + w] = po.Bn[w * pn + j];
        }
    blas::PackedTensor PA  = blas::packed_from_dual_rail(po.Ap.data(), po.An.data(), pn, pn, po.kw);
    blas::PackedTensor PBT = blas::packed_from_dual_rail(PBTp.data(),  PBTn.data(),  pn, pn, po.kw);
    std::vector<int64_t> Cdr(pn * pn), Cpk(pn * pn);
    double dd = best_of([&]{ blas::ternary_gemm(po.Ap.data(), po.An.data(), po.Bp.data(),
                                                po.Bn.data(), Cdr.data(), pn, pn, pn); }, 3);
    double dp = best_of([&]{ blas::ternary_gemm_packed(PA, PBT, Cpk.data()); }, 3);
    int64_t sd = 0, sp = 0;
    for (size_t i = 0; i < pn * pn; ++i) { sd += Cdr[i]; sp += Cpk[i]; }
    require(sd == sp, "packed and dual-rail GEMM must agree at n=512");

    const double mac = double(pn) * double(pn) * double(pn);
    const size_t pkb = PA.bytes() + PBT.bytes();
    const size_t fp32b = 2 * pn * pn * sizeof(float);
    std::printf("    -> n=512 dual-rail %.2f G MAC/s | packed %.2f G MAC/s\n",
                mac / dd / 1e9, mac / dp / 1e9);
    std::printf("    -> weights: packed is %.1fx smaller than FP32 (%.3f vs 32.000 bits/trit)\n",
                double(fp32b) / double(pkb), double(pkb) * 8.0 / double(2 * pn * pn));
}

// ---------------------------------------------------------------------------
// Task 2: GPU shaders present, CPU oracle correct, fallback well-behaved.
// ---------------------------------------------------------------------------
static void test_gpu_backend() {
    std::printf("  [Task2] GPU backend: shaders present, CPU oracle exact, fallback clean...\n");
    require(std::string(blas::ternary_gemm_glsl_native()).find("#version 450") == 0,
            "native shader must be GLSL 450");
    require(std::string(blas::ternary_gemm_glsl_portable()).find("#version 450") == 0,
            "portable shader must be GLSL 450");
    require(std::string(blas::ternary_gemm_glsl_portable()).find("uint64_t") == std::string::npos,
            "portable shader must not require 64-bit ints");

    std::mt19937_64 g(777);
    // Build the operands ONCE: make_operands advances the generator, so
    // calling it twice to "recreate" the same matrix yields different data.
    Operands o = make_operands(19, 23, 71, g);
    std::vector<int64_t> ref = reference_gemm(o);
    blas::GpuGemmOperands op;
    op.m = o.m; op.n = o.n; op.k = o.k; op.kw = o.kw;
    op.Ap = o.Ap; op.An = o.An;
    op.Bp.assign(o.n * o.kw, 0); op.Bn.assign(o.n * o.kw, 0);
    for (size_t j = 0; j < o.n; ++j)
        for (size_t w = 0; w < o.kw; ++w) {
            op.Bp[j * o.kw + w] = o.Bp[w * o.n + j];
            op.Bn[j * o.kw + w] = o.Bn[w * o.n + j];
        }

    std::vector<int64_t> cpu = blas::ternary_gemm_cpu_reference(op);
    for (size_t i = 0; i < ref.size(); ++i)
        if (cpu[i] != ref[i]) { require(false, "GPU CPU-oracle must match naive"); return; }
    std::printf("    -> CPU oracle matches naive (%zu elements)\n", ref.size());

    blas::GpuValidation v = blas::validate_gpu_vs_cpu(op);
    require(!v.note.empty(), "validation must report why the GPU path was not used");
    std::printf("    -> validate_gpu_vs_cpu: gpu=%d note=\"%s\"\n", (int)v.ran_on_gpu, v.note.c_str());
    bool threw = false;
    try { blas::ternary_gemm_gpu(op, true); } catch (const blas::GpuUnavailable&) { threw = true; }
    require(threw, "require_gpu=true must throw GpuUnavailable rather than fake a result");
}

// ---------------------------------------------------------------------------
// Task 3: BitNet b1.58 kernel - correctness and throughput.
// ---------------------------------------------------------------------------
static std::vector<int64_t> bitnet_reference(const TernaryMatrix& W,
                                             const std::vector<int8_t>& X,
                                             size_t M, size_t N, size_t K) {
    std::vector<int64_t> C(M * N, 0);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            int64_t acc = 0;
            for (size_t t = 0; t < K; ++t) {
                Trit w = W.get(i, t);
                int wv = (w == Trit::Pos) ? 1 : (w == Trit::Neg) ? -1 : 0;
                acc += int64_t(wv) * int64_t(X[t * N + j]);
            }
            C[i * N + j] = acc;
        }
    return C;
}

static void test_bitnet() {
    std::printf("  [Task3] BitNet b1.58 (ternary W x int8 X)...\n");
    std::mt19937_64 g(31337);
    const size_t M = 33, N = 27, K = 200;
    TernaryMatrix W(M, K);
    for (size_t i = 0; i < M; ++i)
        for (size_t t = 0; t < K; ++t) {
            unsigned v = g() % 3;
            W.set(i, t, v == 0 ? Trit::Pos : v == 1 ? Trit::Neg : Trit::Zero);
        }
    std::vector<int8_t> X(K * N);
    for (auto& x : X) x = static_cast<int8_t>(int(g() % 251) - 125);

    std::vector<int64_t> ref = bitnet_reference(W, X, M, N, K);
    std::vector<int64_t> C1, C2;
    gemm_bitnet_i8_ternary(W, X, C1, M, N, K);
    BitSlicedInt8Matrix XS; XS.pack(X, K, N);
    gemm_bitnet_i8_ternary(W, XS, C2, M, N, K);
    for (size_t i = 0; i < ref.size(); ++i) {
        if (C1[i] != ref[i]) { require(false, "bitnet (packing overload) mismatch"); return; }
        if (C2[i] != ref[i]) { require(false, "bitnet (pre-sliced overload) mismatch"); return; }
    }
    std::printf("    -> both overloads match naive reference (%zu elements)\n", ref.size());

    const size_t Mn = 256, Nn = 256, Kn = 512;
    TernaryMatrix Wb(Mn, Kn);
    for (size_t i = 0; i < Mn; ++i)
        for (size_t t = 0; t < Kn; ++t) {
            unsigned v = g() % 3;
            Wb.set(i, t, v == 0 ? Trit::Pos : v == 1 ? Trit::Neg : Trit::Zero);
        }
    std::vector<int8_t> Xb(Kn * Nn);
    for (auto& x : Xb) x = static_cast<int8_t>(int(g() % 251) - 125);
    BitSlicedInt8Matrix XSb; XSb.pack(Xb, Kn, Nn);
    std::vector<int64_t> Cb, Cr;
    // Head-to-head in the same process: the convenience overload re-slices
    // the activations on every call (the behaviour being replaced), the
    // pre-sliced overload does not. Comparing them here is far more honest
    // than quoting a number from an older benchmark harness.
    double d = best_of([&]{ gemm_bitnet_i8_ternary(Wb, XSb, Cb, Mn, Nn, Kn); }, 3);
    double dr = best_of([&]{ gemm_bitnet_i8_ternary(Wb, Xb, Cr, Mn, Nn, Kn); }, 3);
    int64_t chk = 0; for (int64_t v : Cb) chk += v;   // sink, outside the timed region
    for (int64_t v : Cr) chk -= v;
    require(chk == 0, "pre-sliced and re-packing BitNet must agree");
    const double mac = double(Mn) * double(Nn) * double(Kn);
    std::printf("    -> 256x512x256 pre-sliced : %.2f ms = %.2f G MAC/s\n",
                d * 1e3, mac / d / 1e9);
    std::printf("    -> 256x512x256 re-slicing : %.2f ms = %.2f G MAC/s  (%.2fx faster when sliced once)\n",
                dr * 1e3, mac / dr / 1e9, dr / d);
    require(Cb.size() == Mn * Nn, "bitnet output sized correctly");
    require(d < dr, "slicing the activations once must beat re-slicing per call");
}

// ---------------------------------------------------------------------------
// Task 4: the two deprecated GEMMs must delegate and be correct + fast.
// ---------------------------------------------------------------------------
static void test_delegation() {
    std::printf("  [Task4] gemm_csa9 / gemm_universal_harley_seal delegation...\n");
    std::mt19937_64 g(2024);
    const size_t n = 256;
    Operands o = make_operands(n, n, n, g);
    std::vector<int64_t> ref = reference_gemm(o);

    TernaryMatrix A(n, n), B(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            A.set(i, j, ((o.Ap[i * o.kw + j / 64] >> (j % 64)) & 1) ? Trit::Pos
                        : ((o.An[i * o.kw + j / 64] >> (j % 64)) & 1) ? Trit::Neg : Trit::Zero);
            B.set(i, j, ((o.Bp[(i / 64) * n + j] >> (i % 64)) & 1) ? Trit::Pos
                        : ((o.Bn[(i / 64) * n + j] >> (i % 64)) & 1) ? Trit::Neg : Trit::Zero);
        }
    DenseMatrixInt64 Cf(n, n), Cc(n, n), Ch(n, n);
    gemm_fast(A, B, Cf);
    gemm_csa9(A, B, Cc);
    gemm_universal_harley_seal(A, B, Ch);
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        if (Cf.data[i] != ref[i]) { require(false, "gemm_fast must match reference"); return; }
        if (Cc.data[i] != ref[i]) { require(false, "gemm_csa9 must now match gemm_fast"); return; }
        if (Ch.data[i] != ref[i]) { require(false, "harley_seal must now match gemm_fast"); return; }
    }
    const double mac = double(n) * double(n) * double(n);
    double dc = best_of([&]{ gemm_csa9(A, B, Cc); }, 3);
    double dh = best_of([&]{ gemm_universal_harley_seal(A, B, Ch); }, 3);
    DenseMatrixInt64 Ca(n, n);
    double da = best_of([&]{ gemm_autotuned(A, B, Ca); }, 3);
    std::printf("    -> csa9 %.2f | harley_seal %.2f | gemm_autotuned baseline %.2f G MAC/s\n",
                mac / dc / 1e9, mac / dh / 1e9, mac / da / 1e9);
    // Compare in-process rather than against a hard-coded constant: a
    // wall-clock threshold is flaky on a shared machine.
    require(dc < da, "delegated csa9 must beat gemm_autotuned measured here");
    require(dh < da, "delegated harley_seal must beat gemm_autotuned measured here");
}

int main() {
    std::printf("=== bt::blas::ternary_gemm unit tests ===\n");
    std::printf("  detected ISA: %s\n\n", blas::isa_name(blas::detect_isa()));
    test_correctness();
    test_packed_tensor();
    test_checked_mode();
    test_packed_gemm();
    test_gpu_backend();
    test_bitnet();
    test_delegation();
    benchmark();

    if (g_failures) {
        std::printf("\n  FAILED: %d assertion(s)\n", g_failures);
        return 1;
    }
    std::printf("  -> PASSED: all micro-kernel tests verified.\n");
    return 0;
}
