/**
 * @file test_unit_kernel_isa_paths.cpp
 * @brief Differential test of every SIMD path against the scalar oracle.
 *
 * docs/SCALING_PLAN.md 5.1 requires: "every SIMD path differentially tested
 * against the scalar reference; CI runs all paths (and a forced-scalar run to
 * keep the oracle honest)."
 *
 * WHY THIS FILE EXISTS SEPARATELY
 * -------------------------------
 * The ISA lanes are guarded by `#if defined(__AVX2__)` etc, so they are only
 * compiled when the compiler targets the ISA. A normal build compiles NONE of
 * them, and `Isa::Auto` then silently resolves to Scalar.
 *
 * That means test_unit_microkernel.cpp's differential loop, which nominally
 * compares "Auto" against "Scalar", was in fact comparing Scalar against
 * Scalar on any default build -- it passed while testing nothing. A test that
 * cannot fail is not a test.
 *
 * So this TU is compiled with the ISA flags enabled (see CMakeLists), and it
 * forces EACH lane explicitly rather than trusting Auto. Each lane that is
 * both compiled in and present at runtime must agree with the scalar oracle
 * bit for bit. Anything else is reported as SKIPPED, never as passing.
 */

#include "Ternary/Blas/TernaryMicroKernel.hpp"

#include <cstdio>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace bt::blas;

namespace {

int g_checks = 0;
int g_fails = 0;
int g_skipped = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

void skip(const std::string& what) {
    ++g_skipped;
    std::printf("  [SKIP] %s\n", what.c_str());
}

#ifndef TERNARY_BLAS_HAVE_AVX2
#define TERNARY_BLAS_HAVE_AVX2 0
#endif
#ifndef TERNARY_BLAS_HAVE_AVX512VPOPCNT
#define TERNARY_BLAS_HAVE_AVX512VPOPCNT 0
#endif
#ifndef TERNARY_BLAS_HAVE_NEON
#define TERNARY_BLAS_HAVE_NEON 0
#endif

/** Naive reference. Independent of the kernel under test, on purpose. */
std::vector<int64_t> reference_gemm(const std::vector<uint64_t>& Ap,
                                    const std::vector<uint64_t>& An,
                                    const std::vector<uint64_t>& Bp,
                                    const std::vector<uint64_t>& Bn,
                                    std::size_t m, std::size_t n,
                                    std::size_t k) {
    const std::size_t kw = (k + 63) / 64;
    std::vector<int64_t> C(m * n, 0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            int64_t acc = 0;
            for (std::size_t w = 0; w < kw; ++w) {
                // Dual-rail dot: (P&~N) . (P&~N) - cross terms.
                const uint64_t ap = Ap[i * kw + w] & ~An[i * kw + w];
                const uint64_t an = An[i * kw + w] & ~Ap[i * kw + w];
                const uint64_t bp = Bp[w * n + j] & ~Bn[w * n + j];
                const uint64_t bn = Bn[w * n + j] & ~Bp[w * n + j];
                acc += static_cast<int64_t>(__builtin_popcountll(ap & bp));
                acc += static_cast<int64_t>(__builtin_popcountll(an & bn));
                acc -= static_cast<int64_t>(__builtin_popcountll(ap & bn));
                acc -= static_cast<int64_t>(__builtin_popcountll(an & bp));
            }
            C[i * n + j] = acc;
        }
    }
    return C;
}

struct Case {
    std::size_t m, n, k;
};

/** Pack random trits into dual-rail words. */
void pack_random(std::vector<uint64_t>& P, std::vector<uint64_t>& N,
                 std::size_t rows, std::size_t k, std::mt19937_64& rng) {
    const std::size_t kw = (k + 63) / 64;
    P.assign(rows * kw, 0);
    N.assign(rows * kw, 0);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t t = 0; t < k; ++t) {
            const int v = static_cast<int>(rng() % 3) - 1;
            if (v > 0) P[r * kw + t / 64] |= (1ull << (t % 64));
            else if (v < 0) N[r * kw + t / 64] |= (1ull << (t % 64));
        }
}

/**
 * Run one lane over many shapes and compare against the reference.
 * Returns the number of cases actually executed.
 */
int differential(Isa lane, const char* label, int mr) {
    std::mt19937_64 rng(0xC0FFEEu ^ static_cast<uint64_t>(lane));
    // The vectorised inner loop runs only while `w + W <= kw`, where W is the
    // lane width (AVX-512: 8, AVX2: 4, NEON: 2, scalar: 1). So a lane is dead
    // code unless kw >= W -- AVX-512 needs k >= 512, AVX2 needs k >= 256.
    // Cases below kw = 8 exercise ONLY the scalar tail, which is why an
    // earlier version of this test passed with the AVX-512 lane corrupted.
    const Case cases[] = {
        // small: scalar tail and edge handling
        {1, 1, 1}, {1, 1, 64}, {1, 1, 65}, {2, 3, 4},
        {4, 4, 128}, {7, 5, 130}, {8, 8, 192}, {3, 6, 257},
        {1, 9, 63}, {5, 1, 129}, {6, 6, 64}, {2, 2, 320},
        // kw >= 4: AVX2 vector body
        {4, 4, 256}, {4, 4, 300}, {2, 8, 384}, {8, 2, 448},
        // kw >= 8: AVX-512 vector body, plus a non-multiple tail
        {4, 4, 512}, {4, 4, 513}, {8, 8, 576}, {3, 7, 640},
        {6, 6, 1024}, {2, 4, 2048}, {1, 1, 512}, {5, 5, 519},
    };
    int executed = 0;
    GemmStats st{};
    for (const Case& c : cases) {
        const std::size_t kw = (c.k + 63) / 64;
        std::vector<uint64_t> Ap, An, Bp, Bn;
        pack_random(Ap, An, c.m, c.k, rng);
        // B is word-plane-major: kw words x n columns.
        Bp.assign(kw * c.n, 0);
        Bn.assign(kw * c.n, 0);
        for (std::size_t w = 0; w < kw; ++w)
            for (std::size_t j = 0; j < c.n; ++j)
                for (std::size_t t = 0; t < 64; ++t) {
                    const std::size_t idx = w * 64 + t;
                    if (idx >= c.k) break;
                    const int v = static_cast<int>(rng() % 3) - 1;
                    if (v > 0) Bp[w * c.n + j] |= (1ull << t);
                    else if (v < 0) Bn[w * c.n + j] |= (1ull << t);
                }

        const std::vector<int64_t> expected =
            reference_gemm(Ap, An, Bp, Bn, c.m, c.n, c.k);

        std::vector<int64_t> got(c.m * c.n, 0);
        GemmOptions opt;
        opt.isa = lane;
        opt.mr = mr;
        st = ternary_gemm(Ap.data(), An.data(), Bp.data(), Bn.data(),
                          got.data(), c.m, c.n, c.k, opt);

        ++executed;
        if (got != expected) {
            std::string where = std::string(label) + " m=" +
                                std::to_string(c.m) + " n=" +
                                std::to_string(c.n) + " k=" +
                                std::to_string(c.k) + " mr=" +
                                std::to_string(mr);
            // Show the first divergence; enough to debug, not a wall of text.
            for (std::size_t i = 0; i < expected.size(); ++i)
                if (got[i] != expected[i]) {
                    where += " first diff at [" + std::to_string(i) + "] exp=" +
                             std::to_string(expected[i]) + " got=" +
                             std::to_string(got[i]);
                    break;
                }
            check(false, where);
            return executed;   // one failure per lane is enough signal
        }
    }
    check(true, std::string(label) + " agrees with the scalar oracle on " +
                    std::to_string(executed) + " shapes");
    return executed;
}

}  // namespace

int main() {
    std::printf("micro-kernel ISA paths vs the scalar oracle\n");
    std::printf("  compiled in: AVX2=%d AVX512VPOPCNT=%d NEON=%d\n",
                TERNARY_BLAS_HAVE_AVX2, TERNARY_BLAS_HAVE_AVX512VPOPCNT,
                TERNARY_BLAS_HAVE_NEON);
    std::printf("  detected:    %s\n\n", isa_name(detect_isa()));

    // The oracle must be correct before anything is compared against it.
    std::printf("  oracle\n");
    differential(Isa::Scalar, "scalar", 4);

    struct Lane {
        Isa isa;
        const char* name;
        int compiled;
    };
    const Lane lanes[] = {
        {Isa::Avx2, "AVX2", TERNARY_BLAS_HAVE_AVX2},
        {Isa::Avx512Vpopcnt, "AVX-512+VPOPCNTDQ",
         TERNARY_BLAS_HAVE_AVX512VPOPCNT},
        {Isa::Neon, "NEON", TERNARY_BLAS_HAVE_NEON},
    };

    for (const Lane& l : lanes) {
        std::printf("  %s\n", l.name);
        if (!l.compiled) {
            skip(std::string(l.name) +
                 " is not compiled into this binary -- build with the ISA "
                 "flags to exercise it");
            continue;
        }
        // Availability is per-architecture, not a linear ordering: Neon is a
        // different machine, not a lower tier than AVX-512.
        const Isa detected = detect_isa();
        const bool present =
            (l.isa == Isa::Neon) ? (detected == Isa::Neon)
                                 : (static_cast<int>(detected) >=
                                    static_cast<int>(l.isa));
        if (!present) {
            skip(std::string(l.name) +
                 " is compiled in but this CPU does not report it");
            continue;
        }
        for (int mr : {1, 2, 4, 8}) differential(l.isa, l.name, mr);
    }

    std::printf("\n%d checks, %d failures, %d skipped\n",
                g_checks, g_fails, g_skipped);
    // Skipped does not fail the build, but it is reported loudly: a future
    // reader must not mistake "not run" for "passing".
    if (g_skipped)
        std::printf("WARNING: %d lane(s) were NOT exercised.\n", g_skipped);
    return g_fails == 0 ? 0 : 1;
}
