// Conformance test for bt_sdk_gemm_ternary's operand layout.
//
// A and B are NOT stored the same way, and the difference is invisible in any
// test with k <= 64, because then both orderings coincide:
//
//     A:  word w of row i        ->  ap[i*kw + w]
//     B:  word w of B^T row j    ->  bp[w*n + j]      (word-plane-major)
//
// A caller who lays B out row-major (bp[j*kw + w]) gets status 0 and wrong
// numbers as soon as k > 64 and n > 1. That is the failure this pins.
//
// Verification is differential against a naive O(n^3) reference over ternary
// values, never self-consistency.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "Ternary/SDK/sdk.h"

namespace {

constexpr int KW(int k) { return (k + 63) / 64; }

// trit t -> word t/64, bit t%64; +1 sets the P word, -1 the N word.
void encode_row(const std::vector<int8_t>& row, int k, uint64_t* p, uint64_t* n) {
    const int kw = KW(k);
    for (int w = 0; w < kw; ++w) { p[w] = 0; n[w] = 0; }
    for (int t = 0; t < k; ++t) {
        if (row[t] > 0)      p[t / 64] |= (uint64_t{1} << (t % 64));
        else if (row[t] < 0) n[t / 64] |= (uint64_t{1} << (t % 64));
    }
}

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
}

struct Case { int m, n, k; };

void run_case(const Case& c) {
    const int kw = KW(c.k);
    std::mt19937 rng(static_cast<unsigned>(c.m * 7919 + c.n * 104729 + c.k));
    std::uniform_int_distribution<int> trit(-1, 1);

    // A is m rows of k trits. Bt is B^T: row j is column j of B.
    std::vector<int8_t> A(static_cast<size_t>(c.m) * c.k);
    std::vector<int8_t> Bt(static_cast<size_t>(c.n) * c.k);
    for (auto& v : A)  v = static_cast<int8_t>(trit(rng));
    for (auto& v : Bt) v = static_cast<int8_t>(trit(rng));

    const size_t a_words = static_cast<size_t>(c.m) * kw;
    const size_t b_words = static_cast<size_t>(c.n) * kw;
    std::vector<uint64_t> ap(a_words, 0), an(a_words, 0);
    std::vector<uint64_t> bp(b_words, 0), bn(b_words, 0);

    // A: row-major
    for (int i = 0; i < c.m; ++i)
        encode_row({A.begin() + static_cast<size_t>(i) * c.k,
                    A.begin() + static_cast<size_t>(i + 1) * c.k},
                   c.k, &ap[static_cast<size_t>(i) * kw], &an[static_cast<size_t>(i) * kw]);

    // B: word-plane-major -- word w of B^T row j lives at index w*n + j
    for (int j = 0; j < c.n; ++j) {
        std::vector<uint64_t> pr(kw, 0), nr(kw, 0);
        encode_row({Bt.begin() + static_cast<size_t>(j) * c.k,
                    Bt.begin() + static_cast<size_t>(j + 1) * c.k},
                   c.k, pr.data(), nr.data());
        for (int w = 0; w < kw; ++w) {
            bp[static_cast<size_t>(w) * c.n + j] = pr[w];
            bn[static_cast<size_t>(w) * c.n + j] = nr[w];
        }
    }

    std::vector<int64_t> C(static_cast<size_t>(c.m) * c.n, 0);
    char err[512];
    const int rc = bt_sdk_gemm_ternary(ap.data(), an.data(), bp.data(), bn.data(),
                                       C.data(), c.m, c.n, c.k, err, sizeof err);

    // naive O(n^3) reference
    int bad = 0;
    for (int i = 0; i < c.m; ++i)
        for (int j = 0; j < c.n; ++j) {
            long long s = 0;
            for (int t = 0; t < c.k; ++t)
                s += static_cast<long long>(A[static_cast<size_t>(i) * c.k + t]) *
                     static_cast<long long>(Bt[static_cast<size_t>(j) * c.k + t]);
            if (s != C[static_cast<size_t>(i) * c.n + j]) {
                if (bad == 0)
                    std::printf("  [%d,%d] got %lld want %lld\n", i, j,
                                static_cast<long long>(C[static_cast<size_t>(i) * c.n + j]), s);
                ++bad;
            }
        }

    std::printf("  %2d x %2d x %3d (kw=%d): rc=%d mismatches=%d/%d %s\n",
                c.m, c.n, c.k, kw, rc, bad, c.m * c.n, bad ? "FAIL" : "ok");
    check(rc == 0, "gemm returned non-zero status");
    check(bad == 0, "gemm disagrees with the naive reference");

    // The row-major B layout MUST be wrong here, otherwise this test proves
    // nothing about the convention it claims to pin.
    if (kw > 1 && c.n > 1) {
        std::vector<uint64_t> bp_rm(b_words, 0), bn_rm(b_words, 0);
        for (int j = 0; j < c.n; ++j)
            encode_row({Bt.begin() + static_cast<size_t>(j) * c.k,
                        Bt.begin() + static_cast<size_t>(j + 1) * c.k},
                       c.k, &bp_rm[static_cast<size_t>(j) * kw],
                       &bn_rm[static_cast<size_t>(j) * kw]);
        std::vector<int64_t> C2(static_cast<size_t>(c.m) * c.n, 0);
        bt_sdk_gemm_ternary(ap.data(), an.data(), bp_rm.data(), bn_rm.data(),
                            C2.data(), c.m, c.n, c.k, err, sizeof err);
        const bool differs = std::memcmp(C.data(), C2.data(),
                                         C.size() * sizeof(int64_t)) != 0;
        check(differs, "row-major B gave the same answer, so this test cannot "
                       "distinguish the layouts");
    }
}

}  // namespace

int main() {
    std::printf("bt_sdk_gemm_ternary layout conformance\n");

    // Hand-computed case: A=[[1,-1],[0,1]], B^T=[[1,-1],[0,1]] -> [[2,-1],[-1,1]]
    {
        const int8_t a[4] = {1, -1, 0, 1};
        const int8_t bt[4] = {1, -1, 0, 1};
        uint64_t ap[2], an[2], bp[2], bn[2];
        encode_row({a, a + 2}, 2, &ap[0], &an[0]);
        encode_row({a + 2, a + 4}, 2, &ap[1], &an[1]);
        encode_row({bt, bt + 2}, 2, &bp[0], &bn[0]);
        encode_row({bt + 2, bt + 4}, 2, &bp[1], &bn[1]);
        int64_t C[4] = {0};
        char err[256];
        const int rc = bt_sdk_gemm_ternary(ap, an, bp, bn, C, 2, 2, 2, err, sizeof err);
        std::printf("  2x2x2 hand-computed: [%lld,%lld,%lld,%lld]\n",
                    (long long)C[0], (long long)C[1], (long long)C[2], (long long)C[3]);
        check(rc == 0 && C[0] == 2 && C[1] == -1 && C[2] == -1 && C[3] == 1,
              "hand-computed 2x2x2");
    }

    // Shapes that exercise multi-word K, and the single-word cases that do not.
    const Case cases[] = {
        {1, 1, 1},   {1, 1, 64},  {2, 2, 7},   {3, 3, 64},
        {1, 1, 65},  {1, 2, 65},  {2, 1, 65},  {3, 3, 70},
        {5, 4, 100}, {2, 7, 129}, {8, 8, 200}, {4, 4, 128},
        {6, 3, 192}, {7, 5, 256}, {3, 2, 63},  {5, 5, 65},
    };
    for (const auto& c : cases) run_case(c);

    // Rejections that must not silently succeed.
    {
        std::vector<uint64_t> ap(4, 0), an(4, 0), bp(2, 0), bn(2, 0);
        std::vector<int64_t> C(4, 0);
        char err[256];
        check(bt_sdk_gemm_ternary(ap.data(), an.data(), bp.data(), bn.data(),
                                  C.data(), 2, 2, 2, err, sizeof err) == 0,
              "well-formed 2x2x2 should succeed");
    }

    if (failures) {
        std::printf("\nFAIL: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nPASS: gemm layout conformance (%d shapes + hand-computed)\n",
                static_cast<int>(sizeof cases / sizeof *cases));
    return 0;
}
