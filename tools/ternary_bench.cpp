/**
 * @file ternary_bench.cpp
 * @brief Honest benchmark runner for the ternary GEMM paths.
 *
 * Built on tools/bench_harness.hpp, which enforces the discipline:
 *   - verify bit-exact against a reference BEFORE timing,
 *   - report nothing but "UNVERIFIED" for a case that fails,
 *   - checksum outside the timed region so the compiler cannot elide work,
 *   - report best / median / worst, never a single sample,
 *   - exit nonzero if anything failed to verify.
 *
 * Usage:
 *   ternary_bench                 full sweep (256 / 512 / 1024)
 *   ternary_bench --check         correctness only, fast - for CI
 *   ternary_bench --json out.json also emit machine-readable results
 *
 * NOTE: this replaces the older benchmarks/ drivers, which embedded literal
 * result strings and discarded subprocess status. Nothing here is hardcoded:
 * every number printed was computed during this run on this machine.
 */
#include "bench_harness.hpp"

#include "Ternary/Blas/TernaryGemmGpu.hpp"
#include "Ternary/Blas/TernaryMicroKernel.hpp"
#include "ternary_dense_algebra.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace bt;
using blas::Isa;

// ---------------------------------------------------------------------------
// One problem, several representations, so every kernel solves the same thing.
// ---------------------------------------------------------------------------
struct Problem {
    size_t m{0}, n{0}, k{0}, kw{0};
    std::vector<uint64_t> Ap, An;     // A: m rows x kw words
    std::vector<uint64_t> Bp, Bn;     // B^T: kw words x n cols
    TernaryMatrix A, B;               // same trits, TernaryMatrix form
    std::vector<int64_t> ref;         // naive O(n^3) ground truth
};

static Problem make_problem(size_t m, size_t n, size_t k, uint64_t seed) {
    Problem p;
    p.m = m; p.n = n; p.k = k; p.kw = (k + 63) / 64;
    std::mt19937_64 g(seed);
    p.Ap.assign(m * p.kw, 0); p.An.assign(m * p.kw, 0);
    p.Bp.assign(p.kw * n, 0); p.Bn.assign(p.kw * n, 0);

    for (size_t i = 0; i < m; ++i)
        for (size_t t = 0; t < k; ++t) {
            const unsigned v = g() % 3;
            const uint64_t bit = 1ULL << (t % 64);
            if (v == 0) p.Ap[i * p.kw + t / 64] |= bit;
            else if (v == 1) p.An[i * p.kw + t / 64] |= bit;
        }
    for (size_t j = 0; j < n; ++j)
        for (size_t t = 0; t < k; ++t) {
            const unsigned v = g() % 3;
            const uint64_t bit = 1ULL << (t % 64);
            if (v == 0) p.Bp[(t / 64) * n + j] |= bit;
            else if (v == 1) p.Bn[(t / 64) * n + j] |= bit;
        }

    // Ground truth: naive triple loop, independent of any optimised path.
    p.ref.assign(m * n, 0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            int64_t acc = 0;
            for (size_t t = 0; t < k; ++t) {
                const int a = int((p.Ap[i * p.kw + t / 64] >> (t % 64)) & 1)
                            - int((p.An[i * p.kw + t / 64] >> (t % 64)) & 1);
                const int b = int((p.Bp[(t / 64) * n + j] >> (t % 64)) & 1)
                            - int((p.Bn[(t / 64) * n + j] >> (t % 64)) & 1);
                acc += int64_t(a) * int64_t(b);
            }
            p.ref[i * n + j] = acc;
        }

    // Same trits as TernaryMatrix, for the matrix-level APIs.
    p.A = TernaryMatrix(m, k);
    p.B = TernaryMatrix(k, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t t = 0; t < k; ++t)
            p.A.set(i, t, ((p.Ap[i * p.kw + t / 64] >> (t % 64)) & 1) ? Trit::Pos
                          : ((p.An[i * p.kw + t / 64] >> (t % 64)) & 1) ? Trit::Neg
                                                                       : Trit::Zero);
    for (size_t t = 0; t < k; ++t)
        for (size_t j = 0; j < n; ++j)
            p.B.set(t, j, ((p.Bp[(t / 64) * n + j] >> (t % 64)) & 1) ? Trit::Pos
                          : ((p.Bn[(t / 64) * n + j] >> (t % 64)) & 1) ? Trit::Neg
                                                                       : Trit::Zero);
    return p;
}

// ---------------------------------------------------------------------------
static void add_gemm_cases(btbench::Runner& r,
                          std::shared_ptr<const Problem> p,
                          bool include_naive) {
    const std::string tag = std::to_string(p->m) + "x" + std::to_string(p->n) +
                            "x" + std::to_string(p->k);

    // --- naive reference. Not timed as a competitor at large n; it exists so
    //     every other kernel has something true to be compared against. ---
    if (include_naive) {
        auto naive_out = std::make_shared<std::vector<int64_t>>();
        r.add({"naive O(n^3) [reference]", p->m, p->n, p->k,
               [p, naive_out] {
                   naive_out->assign(p->m * p->n, 0);
                   for (size_t i = 0; i < p->m; ++i)
                       for (size_t j = 0; j < p->n; ++j) {
                           int64_t acc = 0;
                           for (size_t t = 0; t < p->k; ++t) {
                               const int a = int((p->Ap[i * p->kw + t / 64] >> (t % 64)) & 1)
                                           - int((p->An[i * p->kw + t / 64] >> (t % 64)) & 1);
                               const int b = int((p->Bp[(t / 64) * p->n + j] >> (t % 64)) & 1)
                                           - int((p->Bn[(t / 64) * p->n + j] >> (t % 64)) & 1);
                               acc += int64_t(a) * int64_t(b);
                           }
                           (*naive_out)[i * p->n + j] = acc;
                       }
               },
               [naive_out]() -> const std::vector<int64_t>& { return *naive_out; },
               [p] { return p->ref; },
               true});   // it IS the reference
    }

    // --- gemm_autotuned: the previous best in this codebase ---
    {
        auto out = std::make_shared<DenseMatrixInt64>(p->m, p->n);
        r.add({"gemm_autotuned (previous best)", p->m, p->n, p->k,
               [p, out] { gemm_autotuned(p->A, p->B, *out); },
               [out]() -> const std::vector<int64_t>& { return out->data; },
               [p] { return p->ref; }});
    }
    // --- gemm_fast: micro-kernel, word-copy A + transposed extraction of B ---
    {
        auto out = std::make_shared<DenseMatrixInt64>(p->m, p->n);
        r.add({"gemm_fast (new micro-kernel)", p->m, p->n, p->k,
               [p, out] { gemm_fast(p->A, p->B, *out); },
               [out]() -> const std::vector<int64_t>& { return out->data; },
               [p] { return p->ref; }});
    }
    // --- the two repaired entry points, to prove the repair held ---
    {
        auto out = std::make_shared<DenseMatrixInt64>(p->m, p->n);
        r.add({"gemm_csa9 (now delegates)", p->m, p->n, p->k,
               [p, out] { gemm_csa9(p->A, p->B, *out); },
               [out]() -> const std::vector<int64_t>& { return out->data; },
               [p] { return p->ref; }});
    }
    {
        auto out = std::make_shared<DenseMatrixInt64>(p->m, p->n);
        r.add({"gemm_universal_harley_seal (now delegates)", p->m, p->n, p->k,
               [p, out] { gemm_universal_harley_seal(p->A, p->B, *out); },
               [out]() -> const std::vector<int64_t>& { return out->data; },
               [p] { return p->ref; }});
    }
    // --- raw micro-kernel, pre-packed operands (no per-call setup) ---
    {
        auto out = std::make_shared<std::vector<int64_t>>(p->m * p->n);
        r.add({"ternary_gemm tile 8x4 (pre-packed)", p->m, p->n, p->k,
               [p, out] { blas::ternary_gemm(p->Ap.data(), p->An.data(), p->Bp.data(),
                                              p->Bn.data(), out->data(), p->m, p->n, p->k); },
               [out]() -> const std::vector<int64_t>& { return *out; },
               [p] { return p->ref; }});
    }
    {
        auto out = std::make_shared<std::vector<int64_t>>(p->m * p->n);
        blas::GemmOptions o; o.isa = Isa::Scalar; o.mr = 4;
        r.add({"ternary_gemm tile 4x4 scalar (pre-packed)", p->m, p->n, p->k,
               [p, out, o] { blas::ternary_gemm(p->Ap.data(), p->An.data(), p->Bp.data(),
                                                 p->Bn.data(), out->data(), p->m, p->n, p->k, o); },
               [out]() -> const std::vector<int64_t>& { return *out; },
               [p] { return p->ref; }});
    }
    // --- packed storage: decompress once, then the same kernel ---
    {
        auto out = std::make_shared<std::vector<int64_t>>(p->m * p->n);
        // Build the N-major B^T planes the packed path wants, then pack both.
        std::vector<uint64_t> BTp(p->n * p->kw), BTn(p->n * p->kw);
        for (size_t j = 0; j < p->n; ++j)
            for (size_t w = 0; w < p->kw; ++w) {
                BTp[j * p->kw + w] = p->Bp[w * p->n + j];
                BTn[j * p->kw + w] = p->Bn[w * p->n + j];
            }
        auto PA  = std::make_shared<blas::PackedTensor>(
            blas::packed_from_dual_rail(p->Ap.data(), p->An.data(), p->m, p->k, p->kw));
        auto PBT = std::make_shared<blas::PackedTensor>(
            blas::packed_from_dual_rail(BTp.data(), BTn.data(), p->n, p->k, p->kw));
        r.add({"ternary_gemm_packed (1.6 bits/trit)", p->m, p->n, p->k,
               [PA, PBT, out] { blas::ternary_gemm_packed(*PA, *PBT, out->data()); },
               [out]() -> const std::vector<int64_t>& { return *out; },
               [p] { return p->ref; }});
    }
    (void)tag;
}

// ---------------------------------------------------------------------------
static void add_bitnet_cases(btbench::Runner& r, size_t M, size_t N, size_t K,
                             uint64_t seed) {
    std::mt19937_64 g(seed);
    auto W = std::make_shared<TernaryMatrix>(M, K);
    auto X = std::make_shared<std::vector<int8_t>>(K * N);
    for (size_t i = 0; i < M; ++i)
        for (size_t t = 0; t < K; ++t) {
            const unsigned v = g() % 3;
            W->set(i, t, v == 0 ? Trit::Pos : v == 1 ? Trit::Neg : Trit::Zero);
        }
    for (auto& x : *X) x = static_cast<int8_t>(int(g() % 251) - 125);

    std::vector<int64_t> ref(M * N, 0);
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            int64_t acc = 0;
            for (size_t t = 0; t < K; ++t) {
                const Trit w = W->get(i, t);
                const int wv = (w == Trit::Pos) ? 1 : (w == Trit::Neg) ? -1 : 0;
                acc += int64_t(wv) * int64_t((*X)[t * N + j]);
            }
            ref[i * N + j] = acc;
        }

    {
        auto out = std::make_shared<std::vector<int64_t>>();
        auto XS  = std::make_shared<BitSlicedInt8Matrix>();
        XS->pack(*X, K, N);
        r.add({"bitnet pre-sliced", M, N, K,
               [W, XS, out, M, N, K] { gemm_bitnet_i8_ternary(*W, *XS, *out, M, N, K); },
               [out]() -> const std::vector<int64_t>& { return *out; },
               [ref] { return ref; }});
    }
    // The ORIGINAL algorithm, reproduced verbatim, so the speedup is measured
    // against the real baseline in this same run rather than quoted from an
    // older harness. Scalar pack64, bounds-checked W.word() per access, one
    // output element at a time.
    {
        auto out = std::make_shared<std::vector<int64_t>>();
        r.add({"bitnet legacy (original algorithm)", M, N, K,
               [W, X, out, M, N, K] {
                   out->assign(M * N, 0);
                   const size_t K_words = (K + 63) / 64;
                   std::vector<BitSlicedInt8Word64> X_sliced(N * K_words);
                   for (size_t col = 0; col < N; ++col)
                       for (size_t kw = 0; kw < K_words; ++kw) {
                           int8_t chunk[64]{0};
                           for (size_t b = 0; b < 64 && (kw * 64 + b) < K; ++b)
                               chunk[b] = (*X)[(kw * 64 + b) * N + col];
                           X_sliced[col * K_words + kw] =
                               BitSlicedInt8Word64::pack64(chunk);
                       }
                   for (size_t i = 0; i < M; ++i)
                       for (size_t j = 0; j < N; ++j) {
                           int64_t sum = 0;
                           for (size_t kw = 0; kw < K_words; ++kw)
                               sum += dotProductBitNet64(W->word(i, kw),
                                                         X_sliced[j * K_words + kw]);
                           (*out)[i * N + j] = sum;
                       }
               },
               [out]() -> const std::vector<int64_t>& { return *out; },
               [ref] { return ref; }});
    }
    {
        auto out = std::make_shared<std::vector<int64_t>>();
        r.add({"bitnet re-slicing (old behaviour)", M, N, K,
               [W, X, out, M, N, K] { gemm_bitnet_i8_ternary(*W, *X, *out, M, N, K); },
               [out]() -> const std::vector<int64_t>& { return *out; },
               [ref] { return ref; }});
    }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    bool check_only = false;
    std::string json_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--check")) check_only = true;
        else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
        else { std::printf("usage: %s [--check] [--json FILE]\n", argv[0]); return 2; }
    }

    const std::vector<size_t> sizes = check_only
        ? std::vector<size_t>{64, 128}
        : std::vector<size_t>{256, 512, 1024};

    btbench::Runner runner(5);
    for (size_t n : sizes) {
        // n=1024 naive is ~10^9 MACs; include it only in the fast check sweep.
        // Shared, not a local: every registered lambda holds a reference to
        // it, and a stack Problem would dangle as soon as this iteration ends.
        auto p = std::make_shared<const Problem>(
            make_problem(n, n, n, 0x9E3779B97F4A7C15ULL + n));
        add_gemm_cases(runner, p, n <= 256);
    }
    if (check_only) add_bitnet_cases(runner, 64, 64, 128, 7);
    else            add_bitnet_cases(runner, 256, 256, 512, 7);

    const int rc = runner.run_all();

    if (!json_path.empty()) {
        const std::string js = runner.to_json(btbench::probe_machine());
        if (FILE* f = std::fopen(json_path.c_str(), "w")) {
            std::fwrite(js.data(), 1, js.size(), f);
            std::fclose(f);
            std::printf("wrote %s\n", json_path.c_str());
        } else {
            std::printf("could not write %s\n", json_path.c_str());
            return 2;
        }
    }
    return rc;
}
