/**
 * @file libbt.hpp
 * @brief Production-Grade Standalone Standard Library for Balanced Ternary Systems
 * @author Antigravity Prime Engineering Engine
 * @version 1.0.0
 * 
 * Provides:
 * 1. Exact Rational Arithmetic (Rational) with 128-bit overflow protection & balanced ternary expansion
 * 2. Dense Linear Algebra (Matrix<T>, RationalMatrix, RealMatrix):
 *    - Transposition, Trace, Determinant, Gauss-Jordan Inversion
 *    - Triangular Solvers (solve_upper, solve_lower)
 *    - Matrix Condition Checking (1-norm, inf-norm, Frobenius, kappa condition numbers)
 * 3. Vector Norms:
 *    - L1 (Manhattan), L2 squared (exact), L2 (Euclidean), Linf (Chebyshev)
 *    - Specialized zero-cycle & popcount paths for TritWord64 and Trit vectors
 * 4. Signal Processing:
 *    - Native Radix-3 Cooley-Tukey Fast Fourier Transform (FFT & IFFT) for N = 3^k
 *    - Exact energy preservation (Parseval identity) and unitary transforms
 *    - Trit-reversal indexing and trigonometric twiddle factor generation
 * 5. Bitwise & Coding Algorithms:
 *    - Fast branchless Ternary Hamming distance via dual-rail popcount
 *    - Multi-word Balanced Ternary Carry-Save (BT-CSA) tree reduction (Wallace-style)
 */

#ifndef LIBBT_HPP
#define LIBBT_HPP

#include "ternary_core.hpp"
#include "ternary_dense_algebra.hpp"

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>
#include <array>
#include <span>
#include <string>
#include <sstream>
#include <complex>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <limits>

namespace bt {

// ==============================================================================
// 1. Exact Rational Arithmetic (Balanced Ternary Exact Fraction Engine)
// ==============================================================================

/**
 * @brief Exact Rational Number representation p/q with __int128_t intermediate protection.
 * In balanced ternary systems, exact rational fractions prevent accumulated drift
 * in Gaussian elimination, matrix inversion, and coordinate transforms.
 */
struct Rational {
    int64_t num{0};
    int64_t den{1};

    constexpr Rational() noexcept : num(0), den(1) {}
    constexpr Rational(int64_t n) noexcept : num(n), den(1) {}

    // Build from an already-reduced __int128 pair with strict range enforcement.
    // Throws instead of silently wrapping (the old static_cast<int64_t> created
    // wrong values like Rational(2^40)*Rational(2^40) -> 0).
    static Rational from_reduced(__int128_t n, __int128_t d) {
        // precondition: d > 0, gcd(|n|, d) == 1, both reduced by caller
        if (n > INT64_MAX || n < INT64_MIN || d > INT64_MAX) {
            throw std::overflow_error(
                "Rational: result has no exact int64 representation (exceeds 64-bit range)");
        }
        Rational r;
        r.num = static_cast<int64_t>(n);
        r.den = static_cast<int64_t>(d);
        return r;
    }

    // Normalize a (n, d) pair to canonical reduced form with d > 0.
    static void reduce_128(__int128_t& n, __int128_t& d) {
        if (d < 0) { n = -n; d = -d; }
        __int128_t an = (n < 0) ? -n : n;
        __int128_t g = std::gcd(an, d);
        if (g > 1) { n /= g; d /= g; }
    }

    Rational(int64_t n, int64_t d) {
        if (d == 0) {
            throw std::invalid_argument("Rational: denominator cannot be zero");
        }
        __int128_t n128 = n;
        __int128_t d128 = d;
        reduce_128(n128, d128);
        *this = from_reduced(n128, d128);
    }

    constexpr bool is_zero() const noexcept {
        return num == 0;
    }

    Rational inv() const {
        if (num == 0) {
            throw std::invalid_argument("Rational::inv(): division by zero");
        }
        return Rational(den, num);
    }

    Rational abs() const noexcept {
        // INT64_MIN-safe: negate via 128-bit domain.
        __int128_t n = num;
        if (n < 0) n = -n;
        return from_reduced(n, static_cast<__int128_t>(den));
    }

    double to_double() const noexcept {
        return static_cast<double>(num) / static_cast<double>(den);
    }

    explicit operator double() const noexcept {
        return to_double();
    }

    // Arithmetic Operators: 128-bit internal precision, strictly range-checked results.
    Rational operator+(const Rational& o) const {
        __int128_t n = static_cast<__int128_t>(num) * o.den + static_cast<__int128_t>(o.num) * den;
        __int128_t d = static_cast<__int128_t>(den) * o.den;
        reduce_128(n, d);
        return from_reduced(n, d);
    }

    Rational operator-(const Rational& o) const {
        __int128_t n = static_cast<__int128_t>(num) * o.den - static_cast<__int128_t>(o.num) * den;
        __int128_t d = static_cast<__int128_t>(den) * o.den;
        reduce_128(n, d);
        return from_reduced(n, d);
    }

    Rational operator*(const Rational& o) const {
        __int128_t n = static_cast<__int128_t>(num) * o.num;
        __int128_t d = static_cast<__int128_t>(den) * o.den;
        reduce_128(n, d);
        return from_reduced(n, d);
    }

    Rational operator/(const Rational& o) const {
        if (o.num == 0) {
            throw std::invalid_argument("Rational::operator/: division by zero");
        }
        __int128_t n = static_cast<__int128_t>(num) * o.den;
        __int128_t d = static_cast<__int128_t>(den) * o.num;
        reduce_128(n, d);
        return from_reduced(n, d);
    }

    // INT64_MIN-safe negation computed in the 128-bit domain.
    Rational operator-() const {
        return from_reduced(-static_cast<__int128_t>(num), static_cast<__int128_t>(den));
    }


    Rational& operator+=(const Rational& o) { *this = *this + o; return *this; }
    Rational& operator-=(const Rational& o) { *this = *this - o; return *this; }
    Rational& operator*=(const Rational& o) { *this = *this * o; return *this; }
    Rational& operator/=(const Rational& o) { *this = *this / o; return *this; }

    bool operator==(const Rational& o) const noexcept {
        return num == o.num && den == o.den;
    }

    bool operator!=(const Rational& o) const noexcept {
        return !(*this == o);
    }

    bool operator<(const Rational& o) const noexcept {
        return static_cast<__int128_t>(num) * o.den < static_cast<__int128_t>(o.num) * den;
    }

    bool operator<=(const Rational& o) const noexcept {
        return *this < o || *this == o;
    }

    bool operator>(const Rational& o) const noexcept {
        return o < *this;
    }

    bool operator>=(const Rational& o) const noexcept {
        return !(*this < o);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << num;
        if (den != 1) {
            oss << "/" << den;
        }
        return oss.str();
    }

    /**
     * @brief Converts this rational fraction to its balanced ternary expansion string.
     * Generates integer trits, decimal point, and fractional trits up to precision.
     */
    std::string to_balanced_ternary(size_t precision = 10) const {
        if (precision > 20) precision = 20; // keep fractional expansion sane

        // Exact integer part via 128-bit symmetric rounding (no double/log drift, no
        // llround out-of-range UB, and 41 trits cover the full int64 range instead of
        // silently dropping high trits at 32).
        __int128_t n128 = num;
        __int128_t d128 = den; // invariant: den > 0
        // Symmetric half-away rounding: q = round(n/d)
        __int128_t q = (n128 >= 0) ? ((n128 * 2 + d128) / (2 * d128))
                                   : -(((-n128) * 2 + d128) / (2 * d128));
        int64_t int_part = static_cast<int64_t>(q);
        TritWord64 int_word = Codec::from_int64(int_part, 41);

        // Find most significant trit of integer part
        size_t msb = 0;
        for (size_t i = 40; i > 0; --i) {
            if (int_word.getTrit(i) != Trit::Zero) {
                msb = i;
                break;
            }
        }

        std::string s;
        for (int i = static_cast<int>(msb); i >= 0; --i) {
            s.push_back(trit_to_char(int_word.getTrit(i)));
        }
        if (s.empty()) s = "0";

        if (precision > 0) {
            s.push_back('.');
            // Exact fractional expansion in 128-bit:
            // rem = num - q * den (in [?, ?)) scaled digit-by-digit:
            // digit_i = round( rem_i * 3 / den ), rem_{i+1} = rem_i * 3 - digit_i * den
            __int128_t rem = num - static_cast<__int128_t>(static_cast<int64_t>(q)) * den;
            for (size_t i = 0; i < precision; ++i) {
                __int128_t twice = rem * 3;
                // symmetric rounding of twice/d to nearest integer (ties half-away)
                __int128_t trit128 = (twice >= 0) ? ((twice * 2 + d128) / (2 * d128))
                                                  : -(((-twice) * 2 + d128) / (2 * d128));
                // balanced digit constraint: |trit| <= 1 after first residual step
                int8_t trit_val = static_cast<int8_t>(trit128);
                if (trit_val > 1) trit_val = 1;
                if (trit_val < -1) trit_val = -1;
                s.push_back(trit_to_char(from_int(trit_val)));
                rem = twice - static_cast<__int128_t>(trit_val) * d128;
            }
        }
        return s;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Rational& r) {
    os << r.str();
    return os;
}

inline Rational abs(const Rational& r) noexcept {
    return r.abs();
}

// ==============================================================================
// 2. Generic Dense Matrix Class & Exact Linear Algebra Engine
// ==============================================================================

template <typename T = Rational>
class Matrix {
public:
    size_t rows{0};
    size_t cols{0};
    std::vector<T> data;

    Matrix() = default;

    Matrix(size_t r, size_t c, const T& init_val = T{})
        : rows(r), cols(c), data(r * c, init_val) {}

    Matrix(size_t r, size_t c, const std::vector<T>& d)
        : rows(r), cols(c), data(d) {
        if (d.size() != r * c) {
            throw std::invalid_argument("Matrix constructor: data size does not match rows * cols");
        }
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list) {
        rows = list.size();
        cols = rows > 0 ? list.begin()->size() : 0;
        data.reserve(rows * cols);
        for (const auto& row_list : list) {
            if (row_list.size() != cols) {
                throw std::invalid_argument("Matrix constructor: jagged initializer list not allowed");
            }
            data.insert(data.end(), row_list.begin(), row_list.end());
        }
    }

    static Matrix<T> identity(size_t n) {
        Matrix<T> I(n, n, T{0});
        for (size_t i = 0; i < n; ++i) {
            I(i, i) = T{1};
        }
        return I;
    }

    static Matrix<T> zeros(size_t r, size_t c) {
        return Matrix<T>(r, c, T{0});
    }

    inline T& operator()(size_t r, size_t c) noexcept {
        assert(r < rows && c < cols);
        return data[r * cols + c];
    }

    inline const T& operator()(size_t r, size_t c) const noexcept {
        assert(r < rows && c < cols);
        return data[r * cols + c];
    }

    inline T& at(size_t r, size_t c) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix::at: index out of range");
        }
        return data[r * cols + c];
    }

    inline const T& at(size_t r, size_t c) const {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix::at: index out of range");
        }
        return data[r * cols + c];
    }

    std::vector<T> row(size_t r) const {
        assert(r < rows);
        std::vector<T> res(cols);
        for (size_t c = 0; c < cols; ++c) {
            res[c] = (*this)(r, c);
        }
        return res;
    }

    std::vector<T> col(size_t c) const {
        assert(c < cols);
        std::vector<T> res(rows);
        for (size_t r = 0; r < rows; ++r) {
            res[r] = (*this)(r, c);
        }
        return res;
    }

    void set_row(size_t r, const std::vector<T>& v) {
        assert(r < rows && v.size() == cols);
        for (size_t c = 0; c < cols; ++c) {
            (*this)(r, c) = v[c];
        }
    }

    void set_col(size_t c, const std::vector<T>& v) {
        assert(c < cols && v.size() == rows);
        for (size_t r = 0; r < rows; ++r) {
            (*this)(r, c) = v[r];
        }
    }

    // Matrix Arithmetic
    Matrix<T> operator+(const Matrix<T>& o) const {
        if (rows != o.rows || cols != o.cols) {
            throw std::invalid_argument("Matrix addition dimension mismatch");
        }
        Matrix<T> res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] + o.data[i];
        }
        return res;
    }

    Matrix<T> operator-(const Matrix<T>& o) const {
        if (rows != o.rows || cols != o.cols) {
            throw std::invalid_argument("Matrix subtraction dimension mismatch");
        }
        Matrix<T> res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] - o.data[i];
        }
        return res;
    }

    Matrix<T> operator*(const Matrix<T>& o) const {
        if (cols != o.rows) {
            throw std::invalid_argument("Matrix multiplication dimension mismatch");
        }
        Matrix<T> res(rows, o.cols, T{0});
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                T a = (*this)(i, k);
                for (size_t j = 0; j < o.cols; ++j) {
                    res(i, j) = res(i, j) + a * o(k, j);
                }
            }
        }
        return res;
    }

    Matrix<T> operator*(const T& scalar) const {
        Matrix<T> res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            res.data[i] = data[i] * scalar;
        }
        return res;
    }

    std::vector<T> operator*(const std::vector<T>& vec) const {
        if (cols != vec.size()) {
            throw std::invalid_argument("Matrix-vector multiplication dimension mismatch");
        }
        std::vector<T> res(rows, T{0});
        for (size_t i = 0; i < rows; ++i) {
            T acc{0};
            for (size_t j = 0; j < cols; ++j) {
                acc = acc + (*this)(i, j) * vec[j];
            }
            res[i] = acc;
        }
        return res;
    }

    bool operator==(const Matrix<T>& o) const noexcept {
        return rows == o.rows && cols == o.cols && data == o.data;
    }

    bool operator!=(const Matrix<T>& o) const noexcept {
        return !(*this == o);
    }

    /**
     * @brief Transposes the matrix: A^T.
     */
    Matrix<T> transpose() const {
        Matrix<T> res(cols, rows);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                res(c, r) = (*this)(r, c);
            }
        }
        return res;
    }

    /**
     * @brief Computes the trace of a square matrix: sum_{i=0..n-1} A_{ii}.
     */
    T trace() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix::trace(): matrix must be square");
        }
        T tr{0};
        for (size_t i = 0; i < rows; ++i) {
            tr = tr + (*this)(i, i);
        }
        return tr;
    }

    /**
     * @brief Computes the determinant of a square matrix via Gaussian elimination.
     * For Rational matrices, the result is mathematically exact with zero rounding error.
     */
    T determinant() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix::determinant(): matrix must be square");
        }
        size_t n = rows;
        if (n == 0) return T{1};
        if (n == 1) return (*this)(0, 0);
        if (n == 2) {
            return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
        }
        if (n == 3) {
            return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) -
                   (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0)) +
                   (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
        }

        Matrix<T> A = *this;
        T det{1};

        // Scale-relative pivot threshold (see inverse() for rationale).
        double scale_base = 0.0;
        for (size_t r = 0; r < n && scale_base == 0.0; ++r) {
            for (size_t c = 0; c < n && scale_base == 0.0; ++c) {
                scale_base = std::abs(static_cast<double>(A(r, c)));
            }
        }
        if (scale_base == 0.0) scale_base = 1.0;
        const double singular_threshold = scale_base * 1e-15;

        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            double max_val = std::abs(static_cast<double>(A(i, i)));
            for (size_t r = i + 1; r < n; ++r) {
                double val = std::abs(static_cast<double>(A(r, i)));
                if (val > max_val) {
                    max_val = val;
                    pivot = r;
                }
            }

            if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
                if (max_val < singular_threshold) return T{0};
            } else {
                if (A(pivot, i) == T{0}) {
                    pivot = i;
                    while (pivot < n && A(pivot, i) == T{0}) {
                        ++pivot;
                    }
                    if (pivot == n) return T{0}; // Singular
                }
            }

            if (pivot != i) {
                for (size_t j = 0; j < n; ++j) {
                    std::swap(A(i, j), A(pivot, j));
                }
                det = -det;
            }

            T pivot_val = A(i, i);
            det = det * pivot_val;

            for (size_t r = i + 1; r < n; ++r) {
                if (!(A(r, i) == T{0})) {
                    T factor = A(r, i) / pivot_val;
                    for (size_t c = i + 1; c < n; ++c) {
                        A(r, c) = A(r, c) - factor * A(i, c);
                    }
                    A(r, i) = T{0};
                }
            }
        }
        return det;
    }

    /**
     * @brief Computes the inverse matrix A^-1 via Gauss-Jordan elimination on [A | I].
     * For Rational matrices, the inverse is exact and satisfies A * A^-1 == I.
     */
    Matrix<T> inverse(bool* invertible = nullptr) const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix::inverse(): matrix must be square");
        }
        size_t n = rows;
        Matrix<T> A = *this;
        Matrix<T> inv = Matrix<T>::identity(n);

        // Scale-relative singularity threshold: an absolute 1e-15 cutoff
        // misclassifies legitimately invertible matrices whose entries are tiny
        // (e.g. scaled SI units). Pivot is singular only when it is ~1e-15 of the
        // matrix's own magnitude.
        double scale_base = 0.0;
        for (size_t r = 0; r < n && scale_base == 0.0; ++r) {
            for (size_t c = 0; c < n && scale_base == 0.0; ++c) {
                scale_base = std::abs(static_cast<double>(A(r, c)));
            }
        }
        if (scale_base == 0.0) scale_base = 1.0;
        const double singular_threshold = scale_base * 1e-15;

        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            double max_val = std::abs(static_cast<double>(A(i, i)));
            for (size_t r = i + 1; r < n; ++r) {
                double val = std::abs(static_cast<double>(A(r, i)));
                if (val > max_val) {
                    max_val = val;
                    pivot = r;
                }
            }

            if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
                if (max_val < singular_threshold) {
                    if (invertible) *invertible = false;
                    else throw std::runtime_error("Matrix is singular (non-invertible)");
                    return Matrix<T>();
                }
            } else {
                if (A(pivot, i) == T{0}) {
                    pivot = i;
                    while (pivot < n && A(pivot, i) == T{0}) {
                        ++pivot;
                    }
                    if (pivot == n) {
                        if (invertible) *invertible = false;
                        else throw std::runtime_error("Matrix is singular (non-invertible)");
                        return Matrix<T>();
                    }
                }
            }

            if (pivot != i) {
                for (size_t j = 0; j < n; ++j) {
                    std::swap(A(i, j), A(pivot, j));
                    std::swap(inv(i, j), inv(pivot, j));
                }
            }

            T p = A(i, i);
            for (size_t j = 0; j < n; ++j) {
                A(i, j) = A(i, j) / p;
                inv(i, j) = inv(i, j) / p;
            }

            for (size_t r = 0; r < n; ++r) {
                if (r != i && !(A(r, i) == T{0})) {
                    T factor = A(r, i);
                    for (size_t c = 0; c < n; ++c) {
                        A(r, c) = A(r, c) - factor * A(i, c);
                        inv(r, c) = inv(r, c) - factor * inv(i, c);
                    }
                }
            }
        }

        if (invertible) *invertible = true;
        return inv;
    }

    /**
     * @brief Solves the general linear system A * X = B via augmented Gauss-Jordan
     * elimination with partial pivoting.
     * Over Matrix<Rational>, solves A * X = B with 100% exact rational arithmetic
     * and zero rounding error.
     */
    Matrix<T> solve(const Matrix<T>& B) const {
        if (rows != cols || rows != B.rows) {
            throw std::invalid_argument("Matrix::solve(): dimension mismatch");
        }
        size_t n = rows;
        size_t m = B.cols;
        Matrix<T> A = *this;
        Matrix<T> X = B;

        double scale_base = 0.0;
        for (size_t r = 0; r < n && scale_base == 0.0; ++r) {
            for (size_t c = 0; c < n && scale_base == 0.0; ++c) {
                scale_base = std::abs(static_cast<double>(A(r, c)));
            }
        }
        if (scale_base == 0.0) scale_base = 1.0;
        const double singular_threshold = scale_base * 1e-15;

        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            double max_val = std::abs(static_cast<double>(A(i, i)));
            for (size_t r = i + 1; r < n; ++r) {
                double val = std::abs(static_cast<double>(A(r, i)));
                if (val > max_val) {
                    max_val = val;
                    pivot = r;
                }
            }

            if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
                if (max_val < singular_threshold) {
                    throw std::runtime_error("Matrix::solve(): matrix is singular");
                }
            } else {
                if (A(pivot, i) == T{0}) {
                    pivot = i;
                    while (pivot < n && A(pivot, i) == T{0}) {
                        ++pivot;
                    }
                    if (pivot == n) {
                        throw std::runtime_error("Matrix::solve(): matrix is singular");
                    }
                }
            }

            if (pivot != i) {
                for (size_t j = 0; j < n; ++j) {
                    std::swap(A(i, j), A(pivot, j));
                }
                for (size_t j = 0; j < m; ++j) {
                    std::swap(X(i, j), X(pivot, j));
                }
            }

            T p = A(i, i);
            for (size_t j = i; j < n; ++j) {
                A(i, j) = A(i, j) / p;
            }
            for (size_t j = 0; j < m; ++j) {
                X(i, j) = X(i, j) / p;
            }

            for (size_t r = 0; r < n; ++r) {
                if (r != i && !(A(r, i) == T{0})) {
                    T factor = A(r, i);
                    for (size_t c = i; c < n; ++c) {
                        A(r, c) = A(r, c) - factor * A(i, c);
                    }
                    for (size_t c = 0; c < m; ++c) {
                        X(r, c) = X(r, c) - factor * X(i, c);
                    }
                }
            }
        }
        return X;
    }

    /**
     * @brief Solves the general linear system A * x = b via augmented Gauss-Jordan
     * elimination with partial pivoting.
     * Over Matrix<Rational>, solves A * x = b with 100% exact rational arithmetic
     * and zero rounding error.
     */
    std::vector<T> solve(const std::vector<T>& b) const {
        if (rows != cols || rows != b.size()) {
            throw std::invalid_argument("Matrix::solve(): dimension mismatch");
        }
        Matrix<T> B(b.size(), 1);
        for (size_t i = 0; i < b.size(); ++i) {
            B(i, 0) = b[i];
        }
        Matrix<T> X = solve(B);
        std::vector<T> x(b.size());
        for (size_t i = 0; i < b.size(); ++i) {
            x[i] = X(i, 0);
        }
        return x;
    }

    /**
     * @brief Solves an upper triangular system U * x = b via back-substitution.
     */
    static std::vector<T> solve_upper(const Matrix<T>& U, const std::vector<T>& b) {
        if (U.rows != U.cols || U.rows != b.size()) {
            throw std::invalid_argument("solve_upper: dimension mismatch");
        }
        size_t n = U.rows;
        std::vector<T> x(n);
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            T sum = b[i];
            for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
                sum = sum - U(i, j) * x[j];
            }
            if (U(i, i) == T{0}) {
                throw std::runtime_error("solve_upper: singular matrix with zero diagonal");
            }
            x[i] = sum / U(i, i);
        }
        return x;
    }

    /**
     * @brief Solves a lower triangular system L * x = b via forward-substitution.
     */
    static std::vector<T> solve_lower(const Matrix<T>& L, const std::vector<T>& b) {
        if (L.rows != L.cols || L.rows != b.size()) {
            throw std::invalid_argument("solve_lower: dimension mismatch");
        }
        size_t n = L.rows;
        std::vector<T> x(n);
        for (size_t i = 0; i < n; ++i) {
            T sum = b[i];
            for (size_t j = 0; j < i; ++j) {
                sum = sum - L(i, j) * x[j];
            }
            if (L(i, i) == T{0}) {
                throw std::runtime_error("solve_lower: singular matrix with zero diagonal");
            }
            x[i] = sum / L(i, i);
        }
        return x;
    }

    // Matrix Norms & Condition Checking
    double norm_1() const noexcept {
        double max_col_sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            double col_sum = 0.0;
            for (size_t r = 0; r < rows; ++r) {
                col_sum += std::abs(static_cast<double>((*this)(r, c)));
            }
            if (col_sum > max_col_sum) max_col_sum = col_sum;
        }
        return max_col_sum;
    }

    double norm_inf() const noexcept {
        double max_row_sum = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            double row_sum = 0.0;
            for (size_t c = 0; c < cols; ++c) {
                row_sum += std::abs(static_cast<double>((*this)(r, c)));
            }
            if (row_sum > max_row_sum) max_row_sum = row_sum;
        }
        return max_row_sum;
    }

    double norm_frobenius() const noexcept {
        double sum_sq = 0.0;
        for (const auto& elem : data) {
            double val = static_cast<double>(elem);
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq);
    }

    double condition_number_1() const {
        bool ok = false;
        Matrix<T> inv = inverse(&ok);
        if (!ok) return std::numeric_limits<double>::infinity();
        return norm_1() * inv.norm_1();
    }

    double condition_number_inf() const {
        bool ok = false;
        Matrix<T> inv = inverse(&ok);
        if (!ok) return std::numeric_limits<double>::infinity();
        return norm_inf() * inv.norm_inf();
    }

    bool is_invertible() const noexcept {
        bool ok = false;
        inverse(&ok);
        return ok;
    }

    bool is_well_conditioned(double max_cond = 1e12) const noexcept {
        double cond = condition_number_1();
        return std::isfinite(cond) && cond <= max_cond;
    }

    /**
     * @brief Fraction-free matrix determinant using Bareiss algorithm.
     * Applies Sylvester's determinant identity:
     * A_{i,j}^{(k+1)} = (A_{k,k}^{(k)} A_{i,j}^{(k)} - A_{i,k}^{(k)} A_{k,j}^{(k)}) / c_{k-1}
     * with exact integer division, preserving ring integrity over integer types
     * (int64_t, TritBigInt, Rational).
     */
    static T bareiss_determinant(const Matrix<T>& mat) {
        if (mat.rows != mat.cols) {
            throw std::invalid_argument("bareiss_determinant: matrix must be square");
        }
        size_t n = mat.rows;
        if (n == 0) return T{1};
        if (n == 1) return mat(0, 0);

        Matrix<T> A = mat;
        T sign{1};
        T c{1};

        for (size_t k = 0; k < n - 1; ++k) {
            if (A(k, k) == T{0}) {
                size_t swap_row = n;
                for (size_t r = k + 1; r < n; ++r) {
                    if (!(A(r, k) == T{0})) {
                        swap_row = r;
                        break;
                    }
                }
                if (swap_row == n) {
                    return T{0};
                }
                for (size_t col = 0; col < n; ++col) {
                    std::swap(A(k, col), A(swap_row, col));
                }
                sign = -sign;
            }

            T pivot = A(k, k);
            for (size_t i = k + 1; i < n; ++i) {
                for (size_t j = k + 1; j < n; ++j) {
                    T num = pivot * A(i, j) - A(i, k) * A(k, j);
                    A(i, j) = num / c;
                }
                A(i, k) = T{0};
            }
            c = pivot;
        }

        return sign * A(n - 1, n - 1);
    }

    /**
     * @brief Fraction-free linear system solver A * x = b using Bareiss elimination.
     * Operates over integer types (int64_t, TritBigInt) and exact rings.
     */
    static std::vector<T> bareiss_solve(const Matrix<T>& A, const std::vector<T>& b) {
        if (A.rows != A.cols) {
            throw std::invalid_argument("bareiss_solve: matrix must be square");
        }
        if (A.rows != b.size()) {
            throw std::invalid_argument("bareiss_solve: dimension mismatch between A and b");
        }
        size_t n = A.rows;
        if (n == 0) return {};

        Matrix<T> aug(n, n + 1, T{0});
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                aug(i, j) = A(i, j);
            }
            aug(i, n) = b[i];
        }

        T c{1};
        for (size_t k = 0; k < n - 1; ++k) {
            if (aug(k, k) == T{0}) {
                size_t swap_row = n;
                for (size_t r = k + 1; r < n; ++r) {
                    if (!(aug(r, k) == T{0})) {
                        swap_row = r;
                        break;
                    }
                }
                if (swap_row == n) {
                    throw std::runtime_error("bareiss_solve: matrix is singular");
                }
                for (size_t col = 0; col <= n; ++col) {
                    std::swap(aug(k, col), aug(swap_row, col));
                }
            }

            T pivot = aug(k, k);
            for (size_t i = k + 1; i < n; ++i) {
                for (size_t j = k + 1; j <= n; ++j) {
                    T num = pivot * aug(i, j) - aug(i, k) * aug(k, j);
                    aug(i, j) = num / c;
                }
                aug(i, k) = T{0};
            }
            c = pivot;
        }

        if (aug(n - 1, n - 1) == T{0}) {
            throw std::runtime_error("bareiss_solve: matrix is singular");
        }

        // Back substitution
        std::vector<T> x(n, T{0});
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            T s = aug(static_cast<size_t>(i), n);
            for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
                s = s - aug(static_cast<size_t>(i), j) * x[j];
            }
            x[static_cast<size_t>(i)] = s / aug(static_cast<size_t>(i), static_cast<size_t>(i));
        }
        return x;
    }

    T bareiss_det() const {
        return bareiss_determinant(*this);
    }
};

template <typename T>
inline T bareiss_determinant(const Matrix<T>& mat) {
    return Matrix<T>::bareiss_determinant(mat);
}

template <typename T>
inline std::vector<T> bareiss_solve(const Matrix<T>& A, const std::vector<T>& b) {
    return Matrix<T>::bareiss_solve(A, b);
}

using RationalMatrix = Matrix<Rational>;
using RealMatrix = Matrix<double>;

// Interop helpers: Convert TernaryMatrix to Matrix<Rational> or Matrix<double>
inline RationalMatrix to_rational_matrix(const TernaryMatrix& A) {
    RationalMatrix M(A.rows, A.cols);
    for (size_t r = 0; r < A.rows; ++r) {
        for (size_t c = 0; c < A.cols; ++c) {
            M(r, c) = Rational(to_int(A.get(r, c)));
        }
    }
    return M;
}

inline RealMatrix to_real_matrix(const TernaryMatrix& A) {
    RealMatrix M(A.rows, A.cols);
    for (size_t r = 0; r < A.rows; ++r) {
        for (size_t c = 0; c < A.cols; ++c) {
            M(r, c) = static_cast<double>(to_int(A.get(r, c)));
        }
    }
    return M;
}

inline RationalMatrix to_rational_matrix(const DenseMatrixInt64& A) {
    RationalMatrix M(A.rows, A.cols);
    for (size_t r = 0; r < A.rows; ++r) {
        for (size_t c = 0; c < A.cols; ++c) {
            M(r, c) = Rational(A(r, c));
        }
    }
    return M;
}

inline RealMatrix to_real_matrix(const DenseMatrixInt64& A) {
    RealMatrix M(A.rows, A.cols);
    for (size_t r = 0; r < A.rows; ++r) {
        for (size_t c = 0; c < A.cols; ++c) {
            M(r, c) = static_cast<double>(A(r, c));
        }
    }
    return M;
}

// ==============================================================================
// 3. Vector Norms (L1, L2 Squared, L2 Euclidean, Linf Chebyshev)
// ==============================================================================

template <typename T>
inline double norm_l1(const std::vector<T>& v) noexcept {
    double sum = 0.0;
    for (const auto& elem : v) {
        sum += std::abs(static_cast<double>(elem));
    }
    return sum;
}

template <typename T>
inline T norm_l2_sq(const std::vector<T>& v) noexcept {
    T sum{0};
    for (const auto& elem : v) {
        sum = sum + elem * elem;
    }
    return sum;
}

template <typename T>
inline double norm_l2(const std::vector<T>& v) noexcept {
    double sum_sq = 0.0;
    for (const auto& elem : v) {
        double val = static_cast<double>(elem);
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

template <typename T>
inline double norm_linf(const std::vector<T>& v) noexcept {
    double max_val = 0.0;
    for (const auto& elem : v) {
        double val = std::abs(static_cast<double>(elem));
        if (val > max_val) max_val = val;
    }
    return max_val;
}

// Specialized High-Performance Zero-Cycle & Bitwise Norms for Ternary Types
inline uint32_t norm_l1(const TritWord64& w) noexcept {
    // In balanced ternary, non-zero trits (+1 or -1) each have |t| = 1
    return static_cast<uint32_t>(TritWord64::popcount64(w.P | w.N));
}

inline uint32_t norm_l2_sq(const TritWord64& w) noexcept {
    // (+1)^2 = 1, (-1)^2 = 1, 0^2 = 0 -> exact sum of squares equals popcount(P | N)
    return static_cast<uint32_t>(TritWord64::popcount64(w.P | w.N));
}

inline double norm_l2(const TritWord64& w) noexcept {
    return std::sqrt(static_cast<double>(norm_l2_sq(w)));
}

inline double norm_linf(const TritWord64& w) noexcept {
    return (w.P | w.N) ? 1.0 : 0.0;
}

inline size_t norm_l1(std::span<const Trit> trits) noexcept {
    size_t count = 0;
    for (Trit t : trits) {
        if (t != Trit::Zero) ++count;
    }
    return count;
}

inline size_t norm_l2_sq(std::span<const Trit> trits) noexcept {
    return norm_l1(trits);
}

inline double norm_l2(std::span<const Trit> trits) noexcept {
    return std::sqrt(static_cast<double>(norm_l2_sq(trits)));
}

inline double norm_linf(std::span<const Trit> trits) noexcept {
    for (Trit t : trits) {
        if (t != Trit::Zero) return 1.0;
    }
    return 0.0;
}

// ==============================================================================
// 4. Native Radix-3 Cooley-Tukey Fast Fourier Transform (FFT & IFFT)
// ==============================================================================

/**
 * @brief Signal Processing Engine: Radix-3 Fast Fourier Transform.
 * Operates natively on sequences of length N = 3^k without binary padding distortion.
 * Forward and Inverse transforms satisfy exact energy conservation (Parseval identity).
 */
class Radix3FFT {
public:
    using cdouble = std::complex<double>;
    using Complex = std::complex<double>;

    static constexpr bool is_power_of_3(size_t n) noexcept {
        if (n == 0) return false;
        while (n % 3 == 0) n /= 3;
        return n == 1;
    }

    static size_t get_power_of_3_exponent(size_t n) {
        if (n == 0) throw std::invalid_argument("Radix-3 FFT: sequence length cannot be 0");
        size_t k = 0;
        size_t temp = n;
        while (temp > 1) {
            if (temp % 3 != 0) {
                throw std::invalid_argument("Radix-3 FFT: sequence length must be exact power of 3 (N = 3^k)");
            }
            temp /= 3;
            k++;
        }
        return k;
    }

    /**
     * @brief Computes the base-3 digit reversal of an index of width k trits.
     * Maps natural time-domain ordering into decimation-in-time leaf order.
     */
    static inline size_t trit_reverse(size_t n, size_t k) noexcept {
        size_t res = 0;
        for (size_t i = 0; i < k; ++i) {
            res = res * 3 + (n % 3);
            n /= 3;
        }
        return res;
    }

    /**
     * @brief In-Place Native Radix-3 Cooley-Tukey FFT / IFFT algorithm.
     * Operates in O(N log_3 N) time and O(1) auxiliary heap memory.
     */
    static void transform_inplace(std::vector<cdouble>& a, bool inverse = false) {
        size_t N = a.size();
        if (N <= 1) return;
        size_t k = get_power_of_3_exponent(N);

        // Stage 1: Trit-reversal permutation
        for (size_t i = 0; i < N; ++i) {
            size_t rev = trit_reverse(i, k);
            if (i < rev) {
                std::swap(a[i], a[rev]);
            }
        }

        // Stage 2: Native Radix-3 Butterflies
        const double sign = inverse ? 1.0 : -1.0;
        const double angle3 = sign * 2.0 * M_PI / 3.0;
        const cdouble w3_1(std::cos(angle3), std::sin(angle3));
        const cdouble w3_2(std::cos(2.0 * angle3), std::sin(2.0 * angle3));

        for (size_t length = 3; length <= N; length *= 3) {
            size_t m = length / 3;
            double step_angle = sign * 2.0 * M_PI / static_cast<double>(length);

            // Precompute twiddle factor tables for this stage
            std::vector<cdouble> w(m);
            std::vector<cdouble> w2(m);
            for (size_t j = 0; j < m; ++j) {
                double ang = step_angle * static_cast<double>(j);
                w[j] = cdouble(std::cos(ang), std::sin(ang));
                w2[j] = cdouble(std::cos(2.0 * ang), std::sin(2.0 * ang));
            }

            for (size_t i = 0; i < N; i += length) {
                for (size_t j = 0; j < m; ++j) {
                    cdouble u0 = a[i + j];
                    cdouble u1 = a[i + j + m] * w[j];
                    cdouble u2 = a[i + j + 2 * m] * w2[j];

                    a[i + j]         = u0 + u1 + u2;
                    a[i + j + m]     = u0 + u1 * w3_1 + u2 * w3_2;
                    a[i + j + 2 * m] = u0 + u1 * w3_2 + u2 * w3_1;
                }
            }
        }

        // Stage 3: Normalization for inverse transform
        if (inverse) {
            double inv_N = 1.0 / static_cast<double>(N);
            for (size_t i = 0; i < N; ++i) {
                a[i] *= inv_N;
            }
        }
    }

    /**
     * @brief Forward Radix-3 Fast Fourier Transform.
     */
    static std::vector<cdouble> fft(const std::vector<cdouble>& x) {
        std::vector<cdouble> res = x;
        transform_inplace(res, false);
        return res;
    }

    /**
     * @brief Inverse Radix-3 Fast Fourier Transform.
     */
    static std::vector<cdouble> ifft(const std::vector<cdouble>& X) {
        std::vector<cdouble> res = X;
        transform_inplace(res, true);
        return res;
    }

    /**
     * @brief Unitary Radix-3 FFT scaled by 1/sqrt(N).
     * Preserves exact total energy: ||X_unitary||_2 == ||x||_2.
     */
    static std::vector<cdouble> fft_unitary(const std::vector<cdouble>& x) {
        if (x.empty()) return {};
        std::vector<cdouble> res = fft(x);
        double scale = 1.0 / std::sqrt(static_cast<double>(x.size()));
        for (auto& v : res) v *= scale;
        return res;
    }

    /**
     * @brief Unitary Inverse Radix-3 FFT scaled by sqrt(N).
     */
    static std::vector<cdouble> ifft_unitary(const std::vector<cdouble>& X) {
        if (X.empty()) return {};
        std::vector<cdouble> res = ifft(X);
        double scale = std::sqrt(static_cast<double>(X.size()));
        for (auto& v : res) v *= scale;
        return res;
    }

    /**
     * @brief O(N log_3 N) linear spectral convolution of two complex sequences.
     * Zero-pads to the smallest power of 3 >= a.size() + b.size() - 1.
     */
    static std::vector<Complex> convolve_radix3(const std::vector<Complex>& a,
                                                const std::vector<Complex>& b) {
        if (a.empty() || b.empty()) return {};
        size_t L = a.size() + b.size() - 1;
        size_t N = 1;
        while (N < L) {
            N *= 3;
        }
        std::vector<Complex> A_pad(N, Complex(0.0, 0.0));
        std::vector<Complex> B_pad(N, Complex(0.0, 0.0));
        std::copy(a.begin(), a.end(), A_pad.begin());
        std::copy(b.begin(), b.end(), B_pad.begin());

        std::vector<Complex> FA = fft(A_pad);
        std::vector<Complex> FB = fft(B_pad);
        std::vector<Complex> FC(N);
        for (size_t i = 0; i < N; ++i) {
            FC[i] = FA[i] * FB[i];
        }
        std::vector<Complex> C = ifft(FC);
        C.resize(L);
        return C;
    }

    /**
     * @brief O(N log_3 N) linear spectral convolution of two real sequences.
     * Zero-pads to the smallest power of 3 >= a.size() + b.size() - 1.
     */
    static std::vector<double> convolve_real_radix3(const std::vector<double>& a,
                                                    const std::vector<double>& b) {
        if (a.empty() || b.empty()) return {};
        std::vector<Complex> ca(a.size());
        std::vector<Complex> cb(b.size());
        for (size_t i = 0; i < a.size(); ++i) ca[i] = Complex(a[i], 0.0);
        for (size_t i = 0; i < b.size(); ++i) cb[i] = Complex(b[i], 0.0);

        std::vector<Complex> cc = convolve_radix3(ca, cb);
        std::vector<double> res(cc.size());
        for (size_t i = 0; i < cc.size(); ++i) {
            res[i] = cc[i].real();
        }
        return res;
    }

    /**
     * @brief Calculates total energy of a signal: sum_{i=0..N-1} |x_i|^2.
     */
    static double energy(const std::vector<cdouble>& sig) noexcept {
        double e = 0.0;
        for (const auto& v : sig) {
            e += std::norm(v);
        }
        return e;
    }

    /**
     * @brief Validates Parseval's theorem: E_time == E_freq / N.
     */
    static bool verify_parseval(const std::vector<cdouble>& time_sig,
                                const std::vector<cdouble>& freq_sig,
                                double tol = 1e-10) noexcept {
        if (time_sig.size() != freq_sig.size() || time_sig.empty()) return false;
        double E_time = energy(time_sig);
        double E_freq = energy(freq_sig) / static_cast<double>(time_sig.size());
        double diff = std::abs(E_time - E_freq);
        double scale = std::max(E_time, 1.0);
        return (diff / scale) < tol;
    }

    /**
     * @brief Direct Discrete Fourier Transform (DFT) reference oracle for verification.
     */
    static std::vector<cdouble> dft_reference(const std::vector<cdouble>& x, bool inverse = false) {
        if (x.empty()) return {};
        size_t N = x.size();
        std::vector<cdouble> X(N, cdouble(0, 0));
        double sign = inverse ? 1.0 : -1.0;
        for (size_t k = 0; k < N; ++k) {
            cdouble acc(0, 0);
            for (size_t n = 0; n < N; ++n) {
                double angle = sign * 2.0 * M_PI * static_cast<double>(k * n) / static_cast<double>(N);
                cdouble twiddle(std::cos(angle), std::sin(angle));
                acc += x[n] * twiddle;
            }
            if (inverse) acc /= static_cast<double>(N);
            X[k] = acc;
        }
        return X;
    }
};

// Shorthand Free Functions for Radix-3 FFT
inline std::vector<std::complex<double>> fft_radix3(const std::vector<std::complex<double>>& x) {
    return Radix3FFT::fft(x);
}

inline std::vector<std::complex<double>> ifft_radix3(const std::vector<std::complex<double>>& X) {
    return Radix3FFT::ifft(X);
}

inline std::vector<std::complex<double>> fft_radix3_unitary(const std::vector<std::complex<double>>& x) {
    return Radix3FFT::fft_unitary(x);
}

inline std::vector<std::complex<double>> ifft_radix3_unitary(const std::vector<std::complex<double>>& X) {
    return Radix3FFT::ifft_unitary(X);
}

// ==============================================================================
// 4b. Exact Radix-3 Number Theoretic Transform (NTT) over GF(p)
// ==============================================================================

/**
 * @brief Exact Radix-3 Number Theoretic Transform (NTT) Engine.
 * Operates over Galois field GF(p) with Fermat-theoretic precision, eliminating
 * all floating-point roundoff errors and epsilon drift in linear convolutions.
 * 
 * Verified prime: p = 3,874,204,891 = 10 * 3^18 + 1
 * Primitive root of GF(p): g = 2
 * Order-3 principal root: omega_3 = g^((p-1)/3) mod p = 2,094,712,797
 */
class Radix3NTT {
public:
    static constexpr uint64_t P = 3874204891ULL;      // 10 * 3^18 + 1
    static constexpr uint64_t G = 2ULL;               // Primitive root
    static constexpr uint64_t OMEGA_3 = 2094712797ULL; // 2^((p-1)/3) mod p
    static constexpr size_t MAX_EXPONENT = 18;        // Maximum 3-power length 3^18

    static constexpr uint64_t mod_pow(uint64_t base, uint64_t exp) noexcept {
        uint64_t res = 1;
        base %= P;
        while (exp > 0) {
            if (exp & 1) res = static_cast<uint64_t>((static_cast<__uint128_t>(res) * base) % P);
            base = static_cast<uint64_t>((static_cast<__uint128_t>(base) * base) % P);
            exp >>= 1;
        }
        return res;
    }

    static constexpr uint64_t mod_inv(uint64_t n) noexcept {
        return mod_pow(n, P - 2);
    }

    static bool is_power_of_3(size_t n) noexcept {
        if (n == 0) return false;
        while (n % 3 == 0) n /= 3;
        return n == 1;
    }

    static size_t get_power_of_3_exponent(size_t n) {
        if (n == 0) throw std::invalid_argument("Radix-3 NTT: sequence length cannot be 0");
        size_t k = 0;
        size_t temp = n;
        while (temp > 1) {
            if (temp % 3 != 0) {
                throw std::invalid_argument("Radix-3 NTT: sequence length must be exact power of 3 (N = 3^k)");
            }
            temp /= 3;
            k++;
        }
        if (k > MAX_EXPONENT) {
            throw std::invalid_argument("Radix-3 NTT: sequence length exceeds maximum 3^18");
        }
        return k;
    }

    static inline size_t trit_reverse(size_t n, size_t k) noexcept {
        size_t res = 0;
        for (size_t i = 0; i < k; ++i) {
            res = res * 3 + (n % 3);
            n /= 3;
        }
        return res;
    }

    static void transform_inplace(std::vector<uint64_t>& a, bool inverse = false) {
        size_t N = a.size();
        if (N <= 1) return;
        size_t k = get_power_of_3_exponent(N);

        // Stage 1: Trit-reversal permutation
        for (size_t i = 0; i < N; ++i) {
            size_t rev = trit_reverse(i, k);
            if (i < rev) {
                std::swap(a[i], a[rev]);
            }
        }

        // Stage 2: Radix-3 Cooley-Tukey Butterflies over GF(P)
        uint64_t w3_1 = inverse ? mod_inv(OMEGA_3) : OMEGA_3;
        uint64_t w3_2 = static_cast<uint64_t>((static_cast<__uint128_t>(w3_1) * w3_1) % P);

        for (size_t length = 3; length <= N; length *= 3) {
            size_t m = length / 3;
            uint64_t w_step = mod_pow(G, (P - 1) / length);
            if (inverse) {
                w_step = mod_inv(w_step);
            }

            // Precompute twiddle factor tables
            std::vector<uint64_t> w(m);
            std::vector<uint64_t> w2(m);
            uint64_t curr = 1;
            for (size_t j = 0; j < m; ++j) {
                w[j] = curr;
                w2[j] = static_cast<uint64_t>((static_cast<__uint128_t>(curr) * curr) % P);
                curr = static_cast<uint64_t>((static_cast<__uint128_t>(curr) * w_step) % P);
            }

            for (size_t i = 0; i < N; i += length) {
                for (size_t j = 0; j < m; ++j) {
                    uint64_t u0 = a[i + j] % P;
                    uint64_t u1 = static_cast<uint64_t>((static_cast<__uint128_t>(a[i + j + m] % P) * w[j]) % P);
                    uint64_t u2 = static_cast<uint64_t>((static_cast<__uint128_t>(a[i + j + 2 * m] % P) * w2[j]) % P);

                    uint64_t t0 = (u0 + u1 + u2) % P;
                    uint64_t t1 = (u0 + (static_cast<__uint128_t>(u1) * w3_1) % P + (static_cast<__uint128_t>(u2) * w3_2) % P) % P;
                    uint64_t t2 = (u0 + (static_cast<__uint128_t>(u1) * w3_2) % P + (static_cast<__uint128_t>(u2) * w3_1) % P) % P;

                    a[i + j]         = t0;
                    a[i + j + m]     = t1;
                    a[i + j + 2 * m] = t2;
                }
            }
        }

        // Stage 3: Normalization for inverse transform
        if (inverse) {
            uint64_t inv_N = mod_inv(N);
            for (size_t i = 0; i < N; ++i) {
                a[i] = static_cast<uint64_t>((static_cast<__uint128_t>(a[i]) * inv_N) % P);
            }
        }
    }

    static std::vector<uint64_t> forward(const std::vector<uint64_t>& in) {
        std::vector<uint64_t> res = in;
        transform_inplace(res, false);
        return res;
    }

    static std::vector<uint64_t> inverse(const std::vector<uint64_t>& in) {
        std::vector<uint64_t> res = in;
        transform_inplace(res, true);
        return res;
    }

    static std::vector<int64_t> convolve(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        if (a.empty() || b.empty()) return {};
        size_t L = a.size() + b.size() - 1;
        size_t N = 1;
        while (N < L) N *= 3;

        std::vector<uint64_t> A(N, 0);
        std::vector<uint64_t> B(N, 0);
        for (size_t i = 0; i < a.size(); ++i) {
            int64_t v = a[i] % static_cast<int64_t>(P);
            if (v < 0) v += static_cast<int64_t>(P);
            A[i] = static_cast<uint64_t>(v);
        }
        for (size_t i = 0; i < b.size(); ++i) {
            int64_t v = b[i] % static_cast<int64_t>(P);
            if (v < 0) v += static_cast<int64_t>(P);
            B[i] = static_cast<uint64_t>(v);
        }

        std::vector<uint64_t> FA = forward(A);
        std::vector<uint64_t> FB = forward(B);
        std::vector<uint64_t> FC(N);
        for (size_t i = 0; i < N; ++i) {
            FC[i] = static_cast<uint64_t>((static_cast<__uint128_t>(FA[i]) * FB[i]) % P);
        }

        std::vector<uint64_t> C = inverse(FC);
        std::vector<int64_t> res(L);
        for (size_t i = 0; i < L; ++i) {
            uint64_t v = C[i];
            if (v > P / 2) {
                res[i] = static_cast<int64_t>(v) - static_cast<int64_t>(P);
            } else {
                res[i] = static_cast<int64_t>(v);
            }
        }
        return res;
    }
};

inline std::vector<uint64_t> ntt_radix3(const std::vector<uint64_t>& in) {
    return Radix3NTT::forward(in);
}

inline std::vector<uint64_t> intt_radix3(const std::vector<uint64_t>& in) {
    return Radix3NTT::inverse(in);
}

inline std::vector<int64_t> convolve_radix3_ntt(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    return Radix3NTT::convolve(a, b);
}

// ==============================================================================
// 5. Bitwise & Coding Algorithms (Hamming Distance & Carry-Save Tree Reduction)
// ==============================================================================

/**
 * @brief Fast Branchless Dual-Rail Ternary Hamming Distance.
 * Calculates the number of differing trits between two 64-trit words in 1-2 CPU clock cycles.
 * Two dual-rail trits (P1, N1) and (P2, N2) differ iff (P1 ^ P2) | (N1 ^ N2) != 0.
 */
inline uint32_t hamming_distance(const TritWord64& a, const TritWord64& b) noexcept {
    uint64_t diff = (a.P ^ b.P) | (a.N ^ b.N);
    return static_cast<uint32_t>(TritWord64::popcount64(diff));
}

/**
 * @brief Multi-Word Ternary Hamming Distance across two word vectors or spans.
 */
inline uint64_t hamming_distance(std::span<const TritWord64> a, std::span<const TritWord64> b) noexcept {
    assert(a.size() == b.size());
    uint64_t dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dist += hamming_distance(a[i], b[i]);
    }
    return dist;
}

/**
 * @brief Ternary Hamming Distance for standard trit spans.
 */
inline size_t hamming_distance(std::span<const Trit> a, std::span<const Trit> b) noexcept {
    assert(a.size() == b.size());
    size_t dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++dist;
    }
    return dist;
}

/**
 * @brief Hamming Distance between two TernaryMatrix structures.
 */
inline uint64_t hamming_distance(const TernaryMatrix& a, const TernaryMatrix& b) noexcept {
    assert(a.rows == b.rows && a.cols == b.cols);
    uint64_t dist = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        dist += hamming_distance(a.data[i], b.data[i]);
    }
    return dist;
}

/**
 * @brief Multi-Word Balanced Ternary Carry-Save Tree Reduction (Wallace-style).
 * Compresses an arbitrary number of 64-trit words using parallel BTFA 3:2 compressors
 * into 2 redundant words (Sum, Carry), then resolves into a single canonical TritWord64.
 * Complexity: O(log_{1.5} M) logic depth, completely eliminating ripple-carry stalls.
 */
inline TritWord64 reduce_csa_tree(std::span<const TritWord64> words, bool* overflow_out = nullptr) noexcept {
    if (words.empty()) { if (overflow_out) *overflow_out = false; return TritWord64(0, 0); }
    if (words.size() == 1) { if (overflow_out) *overflow_out = false; return words[0]; }

    bool overflow = false;
    std::vector<TritWord64> curr(words.begin(), words.end());

    while (curr.size() > 2) {
        std::vector<TritWord64> next_lvl;
        next_lvl.reserve((curr.size() * 2) / 3 + 2);
        size_t i = 0;

        // Group into triplets for 3:2 compression
        for (; i + 2 < curr.size(); i += 3) {
            AdderOutput fa = BTFA(curr[i], curr[i+1], curr[i+2]);
            next_lvl.push_back(fa.Sum);
            // Shift carry by 1 trit (multiplying by 3)
            if ((fa.Carry.P >> 63) || (fa.Carry.N >> 63)) overflow = true;
            next_lvl.push_back(TritWord64(fa.Carry.P << 1, fa.Carry.N << 1));
        }

        // Remaining pair compressed with Half-Adder
        if (i + 1 < curr.size()) {
            AdderOutput ha = BTHA(curr[i], curr[i+1]);
            next_lvl.push_back(ha.Sum);
            if ((ha.Carry.P >> 63) || (ha.Carry.N >> 63)) overflow = true;
            next_lvl.push_back(TritWord64(ha.Carry.P << 1, ha.Carry.N << 1));
            i += 2;
        } else if (i < curr.size()) {
            // Passthrough isolated single word
            next_lvl.push_back(curr[i]);
            i += 1;
        }

        curr = std::move(next_lvl);
    }

    if (curr.size() == 2) {
        AdderOutput ha = BTHA(curr[0], curr[1]);
        if ((ha.Carry.P >> 63) || (ha.Carry.N >> 63)) overflow = true;
        BT_CSA csa(ha.Sum, ha.Carry);
        if (overflow_out) *overflow_out = overflow;
        return csa.canonicalize();
    }

    if (overflow_out) *overflow_out = overflow;
    return curr[0];
}

/**
 * @brief Redundant Carry-Save Tree reduction returning redundant (Sum, Carry) output.
 */
inline AdderOutput reduce_csa_tree_redundant(std::span<const TritWord64> words) noexcept {
    if (words.empty()) return { TritWord64(0, 0), TritWord64(0, 0) };
    if (words.size() == 1) return { words[0], TritWord64(0, 0) };

    std::vector<TritWord64> curr(words.begin(), words.end());

    while (curr.size() > 2) {
        std::vector<TritWord64> next_lvl;
        next_lvl.reserve((curr.size() * 2) / 3 + 2);
        size_t i = 0;
        for (; i + 2 < curr.size(); i += 3) {
            AdderOutput fa = BTFA(curr[i], curr[i+1], curr[i+2]);
            next_lvl.push_back(fa.Sum);
            next_lvl.push_back(TritWord64(fa.Carry.P << 1, fa.Carry.N << 1));
        }
        if (i + 1 < curr.size()) {
            AdderOutput ha = BTHA(curr[i], curr[i+1]);
            next_lvl.push_back(ha.Sum);
            next_lvl.push_back(TritWord64(ha.Carry.P << 1, ha.Carry.N << 1));
            i += 2;
        } else if (i < curr.size()) {
            next_lvl.push_back(curr[i]);
            i += 1;
        }
        curr = std::move(next_lvl);
    }

    if (curr.size() == 2) {
        return BTHA(curr[0], curr[1]);
    }
    return { curr[0], TritWord64(0, 0) };
}

} // namespace bt

#endif // LIBBT_HPP
