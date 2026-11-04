// Tile / ISA survey for the ternary GEMM micro-kernel.
//
// Why this exists: a tile "default" is only meaningful if tile and ISA are
// varied independently. Comparing a SCALAR 4x4 against an AVX-512 8x4 changes
// two variables at once and cannot justify either choice. This measures the
// full grid and reports each cell, so the two questions stay separate:
//
//   - given an ISA, which tile is faster?
//   - given a tile, which ISA is faster?
//
// Every measurement is verified bit-exact against a naive O(n^3) reference
// BEFORE it is timed, and re-verified after. An unverified cell prints
// UNVERIFIED instead of a number.
//
// Output is deliberately machine-readable (CSV) so results from several
// machines can be concatenated and compared. See docs/TILE_DEFAULT.md.
//
// Build:
//   g++ -O3 -march=native -std=c++20 -I. -Iinclude tools/tile_survey.cpp -o tile_survey

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "Ternary/Blas/TernaryMicroKernel.hpp"

using namespace bt;
using blas::GemmOptions;
using blas::GemmStats;
using blas::Isa;

namespace {

std::vector<int64_t> naive_reference(const std::vector<int8_t>& A,
                                     const std::vector<int8_t>& BT,
                                     size_t m, size_t n, size_t k) {
    std::vector<int64_t> C(m * n, 0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            long long s = 0;
            for (size_t t = 0; t < k; ++t)
                s += (long long)A[i * k + t] * (long long)BT[j * k + t];
            C[i * n + j] = s;
        }
    return C;
}

// Dual-rail word encoding.
//
// The two operands are NOT stored the same way:
//     A (m rows of k trits):  word w of row i  ->  p[i*kw + w]
//     B^T (n rows of k):      word w of row j  ->  p[w*n + j]   (word-plane-major)
// The orderings coincide only when k <= 64, so a row-major B looks fine on
// small tests and silently returns wrong values for larger K.
struct Rails {
    std::vector<uint64_t> p, n;
};

// A: row-major.
Rails encode_a(const std::vector<int8_t>& M, size_t rows, size_t k) {
    const size_t kw = (k + 63) / 64;
    Rails r{std::vector<uint64_t>(rows * kw, 0), std::vector<uint64_t>(rows * kw, 0)};
    for (size_t i = 0; i < rows; ++i)
        for (size_t t = 0; t < k; ++t) {
            const int8_t v = M[i * k + t];
            if (v > 0)      r.p[i * kw + t / 64] |= (uint64_t{1} << (t % 64));
            else if (v < 0) r.n[i * kw + t / 64] |= (uint64_t{1} << (t % 64));
        }
    return r;
}

// B^T: word-plane-major.
Rails encode_bt(const std::vector<int8_t>& M, size_t rows, size_t k) {
    const size_t kw = (k + 63) / 64;
    Rails r{std::vector<uint64_t>(rows * kw, 0), std::vector<uint64_t>(rows * kw, 0)};
    for (size_t j = 0; j < rows; ++j)
        for (size_t t = 0; t < k; ++t) {
            const int8_t v = M[j * k + t];
            if (v > 0)      r.p[(t / 64) * rows + j] |= (uint64_t{1} << (t % 64));
            else if (v < 0) r.n[(t / 64) * rows + j] |= (uint64_t{1} << (t % 64));
        }
    return r;
}

struct Cell {
    Isa  isa;
    int  mr;
    double best_gmacs = 0;
    double median_gmacs = 0;
    bool verified = true;
    bool compiled = true;
};

double gmacs_per_s(size_t m, size_t n, size_t k, double ms) {
    return (2.0 * double(m) * double(n) * double(k)) / (ms * 1e-3) / 1e9;
}

Cell measure(Isa isa, int mr, const std::vector<int8_t>& A,
             const std::vector<int8_t>& BT, size_t m, size_t n, size_t k,
             const std::vector<int64_t>& ref, int reps) {
    Cell c; c.isa = isa; c.mr = mr;

    const Rails ra = encode_a(A, m, k);
    const Rails rb = encode_bt(BT, n, k);
    std::vector<int64_t> C(m * n, 0);

    GemmOptions o;
    o.isa = isa;
    o.mr = mr;
    o.nr = 4;

    // Correctness BEFORE timing.
    {
        std::vector<int64_t> Cv(m * n, 0);
        try {
            blas::ternary_gemm(ra.p.data(), ra.n.data(), rb.p.data(), rb.n.data(),
                               Cv.data(), m, n, k, o);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  %s/%dx4 threw: %s\n", blas::isa_name(isa), mr, e.what());
            c.compiled = false; c.verified = false;
            return c;
        }
        if (std::memcmp(Cv.data(), ref.data(), ref.size() * sizeof(int64_t)) != 0) {
            c.verified = false;
            return c;
        }
    }

    std::vector<double> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        std::fill(C.begin(), C.end(), 0);
        const auto t0 = std::chrono::steady_clock::now();
        blas::ternary_gemm(ra.p.data(), ra.n.data(), rb.p.data(), rb.n.data(),
                           C.data(), m, n, k, o);
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (std::memcmp(C.data(), ref.data(), ref.size() * sizeof(int64_t)) != 0)
            c.verified = false;
    }
    std::sort(samples.begin(), samples.end());
    const double best = samples.front();
    const double med = samples[samples.size() / 2];
    c.best_gmacs = gmacs_per_s(m, n, k, best);
    c.median_gmacs = gmacs_per_s(m, n, k, med);
    return c;
}

bool isa_available(Isa i) {
    switch (i) {
        case Isa::Avx512Vpopcnt:
#if TERNARY_BLAS_HAVE_AVX512VPOPCNT
            return true;
#else
            return false;
#endif
        case Isa::Avx2:
#if TERNARY_BLAS_HAVE_AVX2
            return true;
#else
            return false;
#endif
        case Isa::Neon:
#if TERNARY_BLAS_HAVE_NEON
            return true;
#else
            return false;
#endif
        case Isa::Scalar: return true;
        default: return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<size_t> sizes = {256, 512, 1024};
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--sizes=", 0) == 0) {
            sizes.clear();
            const std::string v = a.substr(8);
            size_t pos = 0;
            while (pos < v.size()) {
                const size_t comma = v.find(',', pos);
                const std::string tok = v.substr(pos, comma == std::string::npos
                                                          ? std::string::npos : comma - pos);
                sizes.push_back(std::stoul(tok));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (a.rfind("--reps=", 0) == 0) {
            reps = std::stoi(a.substr(7));
        }
    }

    std::printf("# tile_survey: G MAC/s, higher is better\n");
    std::printf("# detected_isa=%s\n", isa_name(blas::detect_isa()));
    std::printf("# reps=%d\n", reps);
    std::printf("machine_isa,n,m,k,isa,tile,best_gmacs,median_gmacs,status\n");

    const Isa detected = blas::detect_isa();
    const std::vector<Isa> isas = {Isa::Avx512Vpopcnt, Isa::Avx2, Isa::Neon, Isa::Scalar};
    const std::vector<int> tiles = {4, 8};

    for (const size_t dim : sizes) {
        const size_t m = dim, n = dim, k = dim;
        std::mt19937 rng(static_cast<unsigned>(dim));
        std::uniform_int_distribution<int> trit(-1, 1);
        std::vector<int8_t> A(m * k), BT(n * k);
        for (auto& v : A)  v = static_cast<int8_t>(trit(rng));
        for (auto& v : BT) v = static_cast<int8_t>(trit(rng));

        // Sink: reference computed once, outside every timed region.
        const std::vector<int64_t> ref = naive_reference(A, BT, m, n, k);

        for (const Isa isa : isas) {
            if (!isa_available(isa)) continue;
            for (const int mr : tiles) {
                const Cell c = measure(isa, mr, A, BT, m, n, k, ref, reps);
                const char* status = !c.compiled ? "THREW"
                                   : !c.verified ? "UNVERIFIED" : "ok";
                std::printf("%s,%zu,%zu,%zu,%s,%dx4,%.2f,%.2f,%s\n",
                            blas::isa_name(detected), m, n, k, blas::isa_name(isa), mr,
                            c.best_gmacs, c.median_gmacs, status);
                std::fflush(stdout);
            }
        }
    }
    return 0;
}
