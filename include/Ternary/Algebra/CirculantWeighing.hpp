#ifndef TERNARY_ALGEBRA_CIRCULANT_WEIGHING_HPP
#define TERNARY_ALGEBRA_CIRCULANT_WEIGHING_HPP

/**
 * @file CirculantWeighing.hpp
 * @brief Algebraic memory compression via Circulant Weighing Matrices,
 *        Goethals-Seidel Skew-Hadamard Matrices, Kronecker Products, and
 *        General Algebraic Operators (Circulant + Low-Rank + Sparse).
 *
 * All circulant operators store only O(n) seed rows (never O(n^2) matrices),
 * using dual-rail bit-sliced TritWord64 storage from ternary_core.hpp alongside
 * sparse support indices for zero-multiplier integer and floating-point apply.
 */

#include "../../../ternary_core.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace bt::algebra {

// ==============================================================================
// 1. CirculantTernaryOperator
// ==============================================================================

/**
 * @brief n x n circulant matrix W over {-1, 0, +1} stored as a single seed row.
 *
 * Convention:
 *   W(i, j) = row0[(j - i + n) % n]   for 0 <= i, j < n.
 *
 * Orthogonality condition (Circulant Weighing Matrix CW(n, k)):
 *   PAF(0) = k  and  PAF(s) = 0 for all 1 <= s < n,
 *   which is equivalent to W * W^T = W^T * W = k * I_n.
 */
class CirculantTernaryOperator {
public:
    std::vector<bt::TritWord64> dual_rail_words;
    std::vector<uint32_t> pos_idx;
    std::vector<uint32_t> neg_idx;
    uint32_t weight_k{0};

    CirculantTernaryOperator() noexcept = default;

    static CirculantTernaryOperator from_supports(
        size_t n,
        std::vector<uint32_t> pos,
        std::vector<uint32_t> neg,
        uint32_t expected_k = 0)
    {
        if (n == 0) {
            throw std::invalid_argument("CirculantTernaryOperator: dimension n must be > 0");
        }

        std::sort(pos.begin(), pos.end());
        std::sort(neg.begin(), neg.end());

        for (size_t i = 0; i < pos.size(); ++i) {
            if (pos[i] >= n) {
                throw std::invalid_argument("CirculantTernaryOperator: pos index out of range");
            }
            if (i > 0 && pos[i] == pos[i - 1]) {
                throw std::invalid_argument("CirculantTernaryOperator: duplicate index in pos");
            }
        }
        for (size_t i = 0; i < neg.size(); ++i) {
            if (neg[i] >= n) {
                throw std::invalid_argument("CirculantTernaryOperator: neg index out of range");
            }
            if (i > 0 && neg[i] == neg[i - 1]) {
                throw std::invalid_argument("CirculantTernaryOperator: duplicate index in neg");
            }
        }

        // Verify disjointness of pos and neg (dual-rail invariant: (P & N) == 0).
        size_t pi = 0;
        size_t ni = 0;
        while (pi < pos.size() && ni < neg.size()) {
            if (pos[pi] == neg[ni]) {
                throw std::invalid_argument("CirculantTernaryOperator: pos and neg supports overlap");
            }
            if (pos[pi] < neg[ni]) {
                ++pi;
            } else {
                ++ni;
            }
        }

        const uint32_t actual_k = static_cast<uint32_t>(pos.size() + neg.size());
        if (expected_k != 0 && actual_k != expected_k) {
            throw std::invalid_argument("CirculantTernaryOperator: support size does not match expected_k");
        }

        CirculantTernaryOperator op;
        op.n_ = n;
        op.pos_idx = std::move(pos);
        op.neg_idx = std::move(neg);
        op.weight_k = actual_k;
        op.dual_rail_words.assign((n + 63) / 64, bt::TritWord64{});

        for (uint32_t idx : op.pos_idx) {
            op.dual_rail_words[idx >> 6].setTrit(idx & 63U, bt::Trit::Pos);
        }
        for (uint32_t idx : op.neg_idx) {
            op.dual_rail_words[idx >> 6].setTrit(idx & 63U, bt::Trit::Neg);
        }

        return op;
    }

    /**
     * @brief Exact verified GSI-R-0005 proper CW(120, 36).
     */
    static CirculantTernaryOperator cw_120_36() {
        return from_supports(
            120,
            {17, 26, 30, 37, 42, 54, 57, 58, 60, 62, 68, 70, 77, 84, 94, 97, 98, 106, 112, 116, 117},
            {0, 2, 6, 8, 14, 18, 24, 50, 52, 56, 66, 74, 78, 102, 110},
            36
        );
    }

    /**
     * @brief Exact GSI-R-0005 proper CW(2m, 4) for any m >= 2.
     *        N = {0}, P = {1, m, m + 1}.
     */
    static CirculantTernaryOperator cw_even_weight4(size_t m) {
        if (m < 2) {
            throw std::invalid_argument("cw_even_weight4: requires m >= 2");
        }
        return from_supports(
            2 * m,
            {1U, static_cast<uint32_t>(m), static_cast<uint32_t>(m + 1)},
            {0U},
            4
        );
    }

    size_t dim() const noexcept { return n_; }

    int8_t seed_entry(size_t col) const noexcept {
        if (col >= n_) return 0;
        return bt::to_int(dual_rail_words[col >> 6].getTrit(col & 63U));
    }

    int8_t entry(size_t row, size_t col) const noexcept {
        if (row >= n_ || col >= n_) return 0;
        const size_t idx = (col >= row) ? (col - row) : (col + n_ - row);
        return seed_entry(idx);
    }

    int64_t periodic_autocorrelation(size_t shift) const {
        if (n_ == 0) return 0;
        const size_t s = shift % n_;
        int64_t acc = 0;
        for (uint32_t c : pos_idx) {
            const size_t idx = (c + s >= n_) ? (c + s - n_) : (c + s);
            acc += static_cast<int64_t>(seed_entry(idx));
        }
        for (uint32_t c : neg_idx) {
            const size_t idx = (c + s >= n_) ? (c + s - n_) : (c + s);
            acc -= static_cast<int64_t>(seed_entry(idx));
        }
        return acc;
    }

    bool verify_orthogonality() const {
        if (n_ == 0) return false;
        for (const auto& w : dual_rail_words) {
            if (!w.isValid()) {
                return false;
            }
        }
        if (periodic_autocorrelation(0) != static_cast<int64_t>(weight_k)) {
            return false;
        }
        for (size_t s = 1; s < n_; ++s) {
            if (periodic_autocorrelation(s) != 0) {
                return false;
            }
        }
        return true;
    }

    /// Zero-multiplier y = W * x (x and y must not alias).
    void apply(std::span<const int64_t> x, std::span<int64_t> y) const {
        if (x.size() != n_ || y.size() != n_) {
            throw std::invalid_argument("CirculantTernaryOperator::apply: size mismatch");
        }
        for (size_t i = 0; i < n_; ++i) {
            int64_t acc = 0;
            for (uint32_t c : pos_idx) {
                const size_t j = (i + c >= n_) ? (i + c - n_) : (i + c);
                acc += x[j];
            }
            for (uint32_t c : neg_idx) {
                const size_t j = (i + c >= n_) ? (i + c - n_) : (i + c);
                acc -= x[j];
            }
            y[i] = acc;
        }
    }

    /// Zero-multiplier x = W^T * y (y and x must not alias).
    void apply_transpose(std::span<const int64_t> y, std::span<int64_t> x) const {
        if (y.size() != n_ || x.size() != n_) {
            throw std::invalid_argument("CirculantTernaryOperator::apply_transpose: size mismatch");
        }
        for (size_t j = 0; j < n_; ++j) {
            int64_t acc = 0;
            for (uint32_t c : pos_idx) {
                const size_t i = (j >= c) ? (j - c) : (j + n_ - c);
                acc += y[i];
            }
            for (uint32_t c : neg_idx) {
                const size_t i = (j >= c) ? (j - c) : (j + n_ - c);
                acc -= y[i];
            }
            x[j] = acc;
        }
    }

    /// Computes x = (W^T * y) / k; returns true iff every element is exactly divisible by k.
    bool apply_exact_inverse(std::span<const int64_t> y, std::span<int64_t> x) const {
        if (y.size() != n_ || x.size() != n_) {
            throw std::invalid_argument("CirculantTernaryOperator::apply_exact_inverse: size mismatch");
        }
        if (weight_k == 0) {
            return false;
        }
        apply_transpose(y, x);
        const int64_t k = static_cast<int64_t>(weight_k);
        bool exact = true;
        for (size_t j = 0; j < n_; ++j) {
            if (x[j] % k != 0) {
                exact = false;
            }
            x[j] /= k;
        }
        return exact;
    }

    void apply_double(std::span<const double> x, std::span<double> y) const {
        if (x.size() != n_ || y.size() != n_) {
            throw std::invalid_argument("CirculantTernaryOperator::apply_double: size mismatch");
        }
        for (size_t i = 0; i < n_; ++i) {
            double acc = 0.0;
            for (uint32_t c : pos_idx) {
                const size_t j = (i + c >= n_) ? (i + c - n_) : (i + c);
                acc += x[j];
            }
            for (uint32_t c : neg_idx) {
                const size_t j = (i + c >= n_) ? (i + c - n_) : (i + c);
                acc -= x[j];
            }
            y[i] = acc;
        }
    }

    void apply_transpose_double(std::span<const double> y, std::span<double> x) const {
        if (y.size() != n_ || x.size() != n_) {
            throw std::invalid_argument("CirculantTernaryOperator::apply_transpose_double: size mismatch");
        }
        for (size_t j = 0; j < n_; ++j) {
            double acc = 0.0;
            for (uint32_t c : pos_idx) {
                const size_t i = (j >= c) ? (j - c) : (j + n_ - c);
                acc += y[i];
            }
            for (uint32_t c : neg_idx) {
                const size_t i = (j >= c) ? (j - c) : (j + n_ - c);
                acc -= y[i];
            }
            x[j] = acc;
        }
    }

    void apply_inverse_double(std::span<const double> y, std::span<double> x) const {
        if (weight_k == 0) {
            throw std::invalid_argument("CirculantTernaryOperator::apply_inverse_double: weight_k is zero");
        }
        apply_transpose_double(y, x);
        const double inv_k = 1.0 / static_cast<double>(weight_k);
        for (size_t j = 0; j < n_; ++j) {
            x[j] *= inv_k;
        }
    }

    size_t memory_bytes() const noexcept {
        return sizeof(*this)
             + dual_rail_words.size() * sizeof(bt::TritWord64)
             + (pos_idx.size() + neg_idx.size()) * sizeof(uint32_t);
    }

    uint64_t virtual_matrix_elements() const noexcept {
        return static_cast<uint64_t>(n_) * static_cast<uint64_t>(n_);
    }

private:
    size_t n_{0};
};

// ==============================================================================
// 2. GoethalsSeidelSkewHadamard (GSI-R-0006 Good(53) and Good(55))
// ==============================================================================

/**
 * @brief 4n x 4n skew-Hadamard matrix from 4 circulant +/-1 sequences (A, B, C, D)
 *        stored in only 4 uint64_t half-words (32 bytes of sequence storage).
 *
 * Block structure:
 *   H = [  A   BR   CR   DR ]
 *       [ -BR  A    DR  -CR ]
 *       [ -CR -DR   A    BR ]
 *       [ -DR  CR  -BR   A  ]
 * where R is the circulant reversal matrix R(i, j) = [i + j == 0 mod n].
 *
 * Half-word bit convention for odd n = 2m + 1:
 *   - A is skew-type: a[0] = +1; for 1 <= i <= m, bit (i - 1) of half_words[0]
 *     is 0 -> +1, 1 -> -1, and a[n - i] = -a[i].
 *   - B, C, D are symmetric: x[0] = +1; for 1 <= i <= m, bit (i - 1) of
 *     half_words[1..3] is 0 -> +1, 1 -> -1, and x[n - i] = x[i].
 */
class GoethalsSeidelSkewHadamard {
public:
    std::array<uint64_t, 4> half_words{0, 0, 0, 0};

    GoethalsSeidelSkewHadamard() noexcept = default;

    GoethalsSeidelSkewHadamard(size_t block_n, std::array<uint64_t, 4> hw)
        : half_words(hw), n_(block_n)
    {
        if (n_ == 0 || (n_ % 2) == 0 || ((n_ - 1) / 2) > 64) {
            throw std::invalid_argument("GoethalsSeidelSkewHadamard: invalid odd block dimension");
        }
    }

    static GoethalsSeidelSkewHadamard order_212() {
        return GoethalsSeidelSkewHadamard(
            53,
            {0x29f4daULL, 0xb79910ULL, 0x3171087ULL, 0x3576479ULL}
        );
    }

    static GoethalsSeidelSkewHadamard order_220() {
        return GoethalsSeidelSkewHadamard(
            55,
            {0x48f51dULL, 0x82e36eULL, 0x3a9fe69ULL, 0x41b7527ULL}
        );
    }

    size_t dim() const noexcept { return 4 * n_; }
    size_t block_dim() const noexcept { return n_; }

    static constexpr size_t seed_bytes() noexcept {
        return sizeof(half_words); // 32 bytes
    }

    int8_t seq_entry(size_t seq_idx, size_t pos) const noexcept {
        if (seq_idx >= 4 || pos >= n_) return 0;
        if (pos == 0) return 1;
        const size_t m = (n_ - 1) / 2;
        if (pos <= m) {
            const uint64_t bit = (half_words[seq_idx] >> (pos - 1)) & 1ULL;
            return bit ? -1 : 1;
        }
        const size_t mirror = n_ - pos;
        const uint64_t bit = (half_words[seq_idx] >> (mirror - 1)) & 1ULL;
        const int8_t val = bit ? -1 : 1;
        return (seq_idx == 0) ? static_cast<int8_t>(-val) : val;
    }

    bool verify_orthogonality() const {
        if (n_ == 0 || (n_ % 2) == 0) return false;
        // Skew condition: A + A^T = 2 I_n
        if (seq_entry(0, 0) != 1) return false;
        for (size_t i = 1; i < n_; ++i) {
            if (seq_entry(0, i) + seq_entry(0, n_ - i) != 0) {
                return false;
            }
        }
        // Hadamard orthogonality: sum_{q=0..3} PAF_q(s) == 0 for 1 <= s < n, and 4n for s == 0
        for (size_t s = 0; s < n_; ++s) {
            int64_t paf_sum = 0;
            for (size_t q = 0; q < 4; ++q) {
                for (size_t i = 0; i < n_; ++i) {
                    const size_t j = (i + s >= n_) ? (i + s - n_) : (i + s);
                    paf_sum += static_cast<int64_t>(seq_entry(q, i)) *
                               static_cast<int64_t>(seq_entry(q, j));
                }
            }
            const int64_t expected = (s == 0) ? static_cast<int64_t>(4 * n_) : 0;
            if (paf_sum != expected) {
                return false;
            }
        }
        return true;
    }

    void apply(std::span<const int64_t> x, std::span<int64_t> y) const {
        const size_t N = dim();
        if (x.size() != N || y.size() != N) {
            throw std::invalid_argument("GoethalsSeidelSkewHadamard::apply: size mismatch");
        }
        auto x0 = x.subspan(0, n_);
        auto x1 = x.subspan(n_, n_);
        auto x2 = x.subspan(2 * n_, n_);
        auto x3 = x.subspan(3 * n_, n_);

        auto y0 = y.subspan(0, n_);
        auto y1 = y.subspan(n_, n_);
        auto y2 = y.subspan(2 * n_, n_);
        auto y3 = y.subspan(3 * n_, n_);

        for (size_t i = 0; i < n_; ++i) {
            int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t j = 0; j < n_; ++j) {
                const size_t idx_circ = (j >= i) ? (j - i) : (j + n_ - i);
                const size_t idx_rev  = (2 * n_ - i - j) % n_;

                const int64_t a = seq_entry(0, idx_circ);
                const int64_t br = seq_entry(1, idx_rev);
                const int64_t cr = seq_entry(2, idx_rev);
                const int64_t dr = seq_entry(3, idx_rev);

                s0 +=  a * x0[j] + br * x1[j] + cr * x2[j] + dr * x3[j];
                s1 += -br * x0[j] +  a * x1[j] + dr * x2[j] - cr * x3[j];
                s2 += -cr * x0[j] - dr * x1[j] +  a * x2[j] + br * x3[j];
                s3 += -dr * x0[j] + cr * x1[j] - br * x2[j] +  a * x3[j];
            }
            y0[i] = s0;
            y1[i] = s1;
            y2[i] = s2;
            y3[i] = s3;
        }
    }

    /// Since H is skew-Hadamard (H + H^T = 2I), H^T * y = 2 * y - H * y.
    void apply_transpose(std::span<const int64_t> y, std::span<int64_t> x) const {
        const size_t N = dim();
        if (y.size() != N || x.size() != N) {
            throw std::invalid_argument("GoethalsSeidelSkewHadamard::apply_transpose: size mismatch");
        }
        apply(y, x);
        for (size_t i = 0; i < N; ++i) {
            x[i] = 2 * y[i] - x[i];
        }
    }

    bool apply_exact_inverse(std::span<const int64_t> y, std::span<int64_t> x) const {
        const size_t N = dim();
        if (y.size() != N || x.size() != N) {
            throw std::invalid_argument("GoethalsSeidelSkewHadamard::apply_exact_inverse: size mismatch");
        }
        if (N == 0) return false;
        apply_transpose(y, x);
        const int64_t scale = static_cast<int64_t>(N);
        bool exact = true;
        for (size_t i = 0; i < N; ++i) {
            if (x[i] % scale != 0) {
                exact = false;
            }
            x[i] /= scale;
        }
        return exact;
    }

private:
    size_t n_{0};
};

// ==============================================================================
// 3. KroneckerTernaryOperator
// ==============================================================================

/**
 * @brief Kronecker product W = W^(0) x W^(1) x ... x W^(d-1) of circulant
 *        ternary operators, stored in O(sum n_i) bytes and applied via
 *        factorized slice-by-slice Kronecker-vector products.
 */
class KroneckerTernaryOperator {
public:
    std::vector<CirculantTernaryOperator> factors;

    KroneckerTernaryOperator() noexcept = default;

    explicit KroneckerTernaryOperator(std::vector<CirculantTernaryOperator> f)
        : factors(std::move(f))
    {
        if (factors.empty()) {
            throw std::invalid_argument("KroneckerTernaryOperator: requires at least one factor");
        }
    }

    size_t dim() const noexcept {
        if (factors.empty()) return 0;
        size_t d = 1;
        for (const auto& f : factors) {
            d *= f.dim();
        }
        return d;
    }

    uint64_t total_weight() const noexcept {
        if (factors.empty()) return 0;
        uint64_t k = 1;
        for (const auto& f : factors) {
            k *= static_cast<uint64_t>(f.weight_k);
        }
        return k;
    }

    int8_t eval_entry(size_t row, size_t col) const {
        const size_t D = dim();
        if (factors.empty() || row >= D || col >= D) {
            throw std::out_of_range("KroneckerTernaryOperator::eval_entry: index out of range");
        }
        int8_t prod = 1;
        size_t r = row;
        size_t c = col;
        for (size_t l = factors.size(); l-- > 0;) {
            const size_t nl = factors[l].dim();
            const size_t rl = r % nl;
            const size_t cl = c % nl;
            r /= nl;
            c /= nl;
            prod = static_cast<int8_t>(prod * factors[l].entry(rl, cl));
            if (prod == 0) return 0;
        }
        return prod;
    }

    void apply(std::span<const int64_t> x, std::span<int64_t> y) const {
        apply_impl<int64_t, false>(x, y);
    }

    void apply_transpose(std::span<const int64_t> y, std::span<int64_t> x) const {
        apply_impl<int64_t, true>(y, x);
    }

    bool apply_exact_inverse(std::span<const int64_t> y, std::span<int64_t> x) const {
        const size_t D = dim();
        if (y.size() != D || x.size() != D) {
            throw std::invalid_argument("KroneckerTernaryOperator::apply_exact_inverse: size mismatch");
        }
        const uint64_t K = total_weight();
        if (K == 0) return false;
        apply_transpose(y, x);
        const int64_t k_signed = static_cast<int64_t>(K);
        bool exact = true;
        for (size_t i = 0; i < D; ++i) {
            if (x[i] % k_signed != 0) {
                exact = false;
            }
            x[i] /= k_signed;
        }
        return exact;
    }

    void apply_double(std::span<const double> x, std::span<double> y) const {
        apply_impl<double, false>(x, y);
    }

    void apply_transpose_double(std::span<const double> y, std::span<double> x) const {
        apply_impl<double, true>(y, x);
    }

    void apply_inverse_double(std::span<const double> y, std::span<double> y_out) const {
        const uint64_t K = total_weight();
        if (K == 0) {
            throw std::invalid_argument("KroneckerTernaryOperator::apply_inverse_double: total_weight is zero");
        }
        apply_transpose_double(y, y_out);
        const double inv_k = 1.0 / static_cast<double>(K);
        for (size_t i = 0; i < y_out.size(); ++i) {
            y_out[i] *= inv_k;
        }
    }

    size_t memory_bytes() const noexcept {
        size_t bytes = sizeof(*this);
        for (const auto& f : factors) {
            bytes += f.memory_bytes();
        }
        return bytes;
    }

    double virtual_fp32_bytes() const noexcept {
        const double d = static_cast<double>(dim());
        return d * d * 4.0;
    }

private:
    template <typename T, bool Transpose>
    void apply_impl(std::span<const T> in_vec, std::span<T> out_vec) const {
        const size_t D = dim();
        if (factors.empty() || in_vec.size() != D || out_vec.size() != D) {
            throw std::invalid_argument("KroneckerTernaryOperator::apply: size mismatch");
        }

        // Single scratch buffer of size D; ping-pong so final result lands in out_vec.
        std::vector<T> scratch(D);
        const size_t num_factors = factors.size();

        // If num_factors is odd:  in_vec -> out_vec (1st), or copy to scratch first if >1?
        // Actually: start with src = (num_factors % 2 == 1) ? scratch : out_vec?
        // Simpler: copy in_vec into (num_factors % 2 == 1 ? scratch : out_vec),
        // then step 0 writes to the other buffer, and after num_factors steps
        // the result is guaranteed to be in out_vec!
        T* buf_a = (num_factors % 2 == 1) ? scratch.data() : out_vec.data();
        T* buf_b = (num_factors % 2 == 1) ? out_vec.data() : scratch.data();
        std::copy(in_vec.begin(), in_vec.end(), buf_a);

        size_t stride_right = D;
        for (size_t l = 0; l < num_factors; ++l) {
            const auto& fl = factors[l];
            const size_t nl = fl.dim();
            stride_right /= nl;
            const size_t num_outer = D / (nl * stride_right);

            for (size_t outer = 0; outer < num_outer; ++outer) {
                const size_t block_base = outer * nl * stride_right;
                for (size_t inner = 0; inner < stride_right; ++inner) {
                    const size_t base = block_base + inner;
                    for (size_t i = 0; i < nl; ++i) {
                        T acc = T{0};
                        if constexpr (!Transpose) {
                            for (uint32_t c : fl.pos_idx) {
                                const size_t j = (i + c >= nl) ? (i + c - nl) : (i + c);
                                acc += buf_a[base + j * stride_right];
                            }
                            for (uint32_t c : fl.neg_idx) {
                                const size_t j = (i + c >= nl) ? (i + c - nl) : (i + c);
                                acc -= buf_a[base + j * stride_right];
                            }
                        } else {
                            for (uint32_t c : fl.pos_idx) {
                                const size_t j = (i >= c) ? (i - c) : (i + nl - c);
                                acc += buf_a[base + j * stride_right];
                            }
                            for (uint32_t c : fl.neg_idx) {
                                const size_t j = (i >= c) ? (i - c) : (i + nl - c);
                                acc -= buf_a[base + j * stride_right];
                            }
                        }
                        buf_b[base + i * stride_right] = acc;
                    }
                }
            }
            std::swap(buf_a, buf_b);
        }
    }
};

// ==============================================================================
// 4. GeneralAlgebraicOperator (Circulant + LowRank U*V^T + Sparse Residual)
// ==============================================================================

/**
 * @brief General-purpose linear operator combining:
 *        (1) Sum of CirculantTernaryOperator terms,
 *        (2) Low-rank factorizations U * V^T (U, V in R^{n x r}),
 *        (3) Sparse triplet residual.
 */
class GeneralAlgebraicOperator {
public:
    struct LowRankTerm {
        std::vector<double> U; // n x rank, row-major
        std::vector<double> V; // n x rank, row-major
        size_t rank{0};
    };

    struct SparseTriplet {
        uint32_t row{0};
        uint32_t col{0};
        double val{0.0};
    };

    std::vector<CirculantTernaryOperator> circulants;
    std::vector<LowRankTerm> low_rank_terms;
    std::vector<SparseTriplet> sparse_residual;

    GeneralAlgebraicOperator() noexcept = default;
    explicit GeneralAlgebraicOperator(size_t n) : n_(n) {
        if (n == 0) {
            throw std::invalid_argument("GeneralAlgebraicOperator: dimension n must be > 0");
        }
    }

    size_t dim() const noexcept { return n_; }

    void add_circulant(CirculantTernaryOperator op) {
        if (op.dim() != n_) {
            throw std::invalid_argument("GeneralAlgebraicOperator::add_circulant: dimension mismatch");
        }
        circulants.push_back(std::move(op));
    }

    void add_low_rank(std::vector<double> U, std::vector<double> V, size_t rank) {
        if (U.size() != n_ * rank || V.size() != n_ * rank) {
            throw std::invalid_argument("GeneralAlgebraicOperator::add_low_rank: dimension mismatch");
        }
        low_rank_terms.push_back(LowRankTerm{std::move(U), std::move(V), rank});
    }

    void add_sparse_triplet(uint32_t row, uint32_t col, double val) {
        if (row >= n_ || col >= n_) {
            throw std::invalid_argument("GeneralAlgebraicOperator::add_sparse_triplet: index out of range");
        }
        sparse_residual.push_back(SparseTriplet{row, col, val});
    }

    void apply(std::span<const double> x, std::span<double> y) const {
        if (x.size() != n_ || y.size() != n_) {
            throw std::invalid_argument("GeneralAlgebraicOperator::apply: size mismatch");
        }
        std::fill(y.begin(), y.end(), 0.0);

        if (!circulants.empty()) {
            std::vector<double> tmp(n_);
            for (const auto& c : circulants) {
                c.apply_double(x, tmp);
                for (size_t i = 0; i < n_; ++i) {
                    y[i] += tmp[i];
                }
            }
        }

        for (const auto& lr : low_rank_terms) {
            if (lr.rank == 0) continue;
            std::vector<double> vt_x(lr.rank, 0.0);
            for (size_t i = 0; i < n_; ++i) {
                const double xi = x[i];
                const size_t row_off = i * lr.rank;
                for (size_t r = 0; r < lr.rank; ++r) {
                    vt_x[r] += lr.V[row_off + r] * xi;
                }
            }
            for (size_t i = 0; i < n_; ++i) {
                double acc = 0.0;
                const size_t row_off = i * lr.rank;
                for (size_t r = 0; r < lr.rank; ++r) {
                    acc += lr.U[row_off + r] * vt_x[r];
                }
                y[i] += acc;
            }
        }

        for (const auto& t : sparse_residual) {
            y[t.row] += t.val * x[t.col];
        }
    }

    void apply_transpose(std::span<const double> y, std::span<double> x) const {
        if (y.size() != n_ || x.size() != n_) {
            throw std::invalid_argument("GeneralAlgebraicOperator::apply_transpose: size mismatch");
        }
        std::fill(x.begin(), x.end(), 0.0);

        if (!circulants.empty()) {
            std::vector<double> tmp(n_);
            for (const auto& c : circulants) {
                c.apply_transpose_double(y, tmp);
                for (size_t i = 0; i < n_; ++i) {
                    x[i] += tmp[i];
                }
            }
        }

        for (const auto& lr : low_rank_terms) {
            if (lr.rank == 0) continue;
            std::vector<double> ut_y(lr.rank, 0.0);
            for (size_t i = 0; i < n_; ++i) {
                const double yi = y[i];
                const size_t row_off = i * lr.rank;
                for (size_t r = 0; r < lr.rank; ++r) {
                    ut_y[r] += lr.U[row_off + r] * yi;
                }
            }
            for (size_t i = 0; i < n_; ++i) {
                double acc = 0.0;
                const size_t row_off = i * lr.rank;
                for (size_t r = 0; r < lr.rank; ++r) {
                    acc += lr.V[row_off + r] * ut_y[r];
                }
                x[i] += acc;
            }
        }

        for (const auto& t : sparse_residual) {
            x[t.col] += t.val * y[t.row];
        }
    }

    size_t memory_bytes() const noexcept {
        size_t bytes = sizeof(*this);
        for (const auto& c : circulants) {
            bytes += c.memory_bytes();
        }
        for (const auto& lr : low_rank_terms) {
            bytes += (lr.U.size() + lr.V.size()) * sizeof(double) + sizeof(LowRankTerm);
        }
        bytes += sparse_residual.size() * sizeof(SparseTriplet);
        return bytes;
    }

private:
    size_t n_{0};
};

} // namespace bt::algebra

#endif // TERNARY_ALGEBRA_CIRCULANT_WEIGHING_HPP
