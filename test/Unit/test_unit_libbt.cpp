/**
 * @file test_unit_libbt.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Standalone Standard Library (libbt) & Radix-3 FFT
 * @author Antigravity Prime Engineering Engine
 */

#include "../../libbt.hpp"

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using namespace bt;

// ==============================================================================
// 1. Rational Arithmetic Unit Tests
// ==============================================================================
void test_rational_arithmetic() {
    std::cout << "  [Test] Rational Arithmetic & Balanced Ternary String Expansion...\n";

    Rational r1(1, 3);
    Rational r2(1, 6);
    Rational sum = r1 + r2;
    assert(sum == Rational(1, 2));

    Rational diff = r1 - r2;
    assert(diff == Rational(1, 6));

    Rational prod = r1 * r2;
    assert(prod == Rational(1, 18));

    Rational quot = r1 / r2;
    assert(quot == Rational(2, 1));

    Rational neg_r = -r1;
    assert(neg_r == Rational(-1, 3));
    assert(neg_r.abs() == r1);

    // Negative denominators normalized
    Rational r_neg(3, -9);
    assert(r_neg.num == -1 && r_neg.den == 3);

    // Comparisons
    assert(Rational(1, 3) < Rational(1, 2));
    assert(Rational(2, 4) == Rational(1, 2));
    assert(Rational(3, 2) > Rational(1, 1));

    // Balanced Ternary Expansion
    std::string bt_str1 = Rational(1, 3).to_balanced_ternary(4);
    assert(!bt_str1.empty());

    // Division by zero throws
    bool threw = false;
    try {
        Rational bad(1, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// ==============================================================================
// 2. Linear Algebra Determinant & Trace Unit Tests
// ==============================================================================
void test_linear_algebra_determinant() {
    std::cout << "  [Test] Matrix Trace & Exact Determinant...\n";

    // 1x1 Matrix
    RationalMatrix M1(1, 1, {Rational(42)});
    assert(M1.trace() == Rational(42));
    assert(M1.determinant() == Rational(42));

    // 2x2 Matrix: [[1, 2], [3, 4]] -> det = 1*4 - 2*3 = -2
    RationalMatrix M2 = {
        {Rational(1), Rational(2)},
        {Rational(3), Rational(4)}
    };
    assert(M2.trace() == Rational(5));
    assert(M2.determinant() == Rational(-2));

    // 3x3 Matrix:
    // [[1, 2, 3],
    //  [0, 1, 4],
    //  [5, 6, 0]]
    // det = 1*(0 - 24) - 2*(0 - 20) + 3*(0 - 5) = -24 + 40 - 15 = 1
    RationalMatrix M3 = {
        {Rational(1), Rational(2), Rational(3)},
        {Rational(0), Rational(1), Rational(4)},
        {Rational(5), Rational(6), Rational(0)}
    };
    assert(M3.trace() == Rational(2));
    assert(M3.determinant() == Rational(1));

    // 4x4 Matrix with known determinant
    // Triangular matrix -> det is product of diagonals
    RationalMatrix M4 = {
        {Rational(2), Rational(3), Rational(1), Rational(5)},
        {Rational(0), Rational(3), Rational(7), Rational(2)},
        {Rational(0), Rational(0), Rational(5), Rational(1)},
        {Rational(0), Rational(0), Rational(0), Rational(4)}
    };
    assert(M4.determinant() == Rational(120)); // 2 * 3 * 5 * 4 = 120

    // Singular matrix -> det = 0
    RationalMatrix Msing = {
        {Rational(1), Rational(2)},
        {Rational(2), Rational(4)}
    };
    assert(Msing.determinant() == Rational(0));

    // Multiplicative identity: det(A * B) == det(A) * det(B)
    RationalMatrix A = {
        {Rational(2), Rational(1)},
        {Rational(1), Rational(3)}
    }; // det = 5
    RationalMatrix B = {
        {Rational(3), Rational(2)},
        {Rational(1), Rational(4)}
    }; // det = 10
    RationalMatrix AB = A * B;
    assert(AB.determinant() == A.determinant() * B.determinant());
    assert(AB.determinant() == Rational(50));
}

// ==============================================================================
// 3. Matrix Inversion (Gauss-Jordan Exact Fractions) Unit Tests
// ==============================================================================
void test_linear_algebra_inverse() {
    std::cout << "  [Test] Exact Gauss-Jordan Matrix Inversion...\n";

    // 2x2 Inversion
    RationalMatrix A2 = {
        {Rational(4), Rational(7)},
        {Rational(2), Rational(6)}
    }; // det = 24 - 14 = 10
    // A2^-1 = 1/10 * [[6, -7], [-2, 4]] = [[3/5, -7/10], [-1/5, 2/5]]
    bool ok = false;
    RationalMatrix inv2 = A2.inverse(&ok);
    assert(ok);
    assert(inv2(0, 0) == Rational(3, 5));
    assert(inv2(0, 1) == Rational(-7, 10));
    assert(inv2(1, 0) == Rational(-1, 5));
    assert(inv2(1, 1) == Rational(2, 5));

    // A * A^-1 == I
    RationalMatrix prod2 = A2 * inv2;
    assert(prod2 == RationalMatrix::identity(2));

    // 3x3 Inversion
    RationalMatrix A3 = {
        {Rational(1), Rational(2), Rational(3)},
        {Rational(0), Rational(1), Rational(4)},
        {Rational(5), Rational(6), Rational(0)}
    };
    RationalMatrix inv3 = A3.inverse(&ok);
    assert(ok);
    RationalMatrix prod3 = A3 * inv3;
    assert(prod3 == RationalMatrix::identity(3));
    RationalMatrix prod3_rev = inv3 * A3;
    assert(prod3_rev == RationalMatrix::identity(3));

    // Singular matrix fails gracefully
    RationalMatrix Asing = {
        {Rational(1), Rational(1)},
        {Rational(1), Rational(1)}
    };
    bool sing_ok = true;
    Asing.inverse(&sing_ok);
    assert(!sing_ok);
    assert(!Asing.is_invertible());

    // Double-precision Inversion
    RealMatrix R = {
        {2.0, 1.0},
        {5.0, 7.0}
    };
    bool real_ok = false;
    RealMatrix R_inv = R.inverse(&real_ok);
    assert(real_ok);
    RealMatrix R_prod = R * R_inv;
    assert(std::abs(R_prod(0, 0) - 1.0) < 1e-12);
    assert(std::abs(R_prod(0, 1)) < 1e-12);
    assert(std::abs(R_prod(1, 0)) < 1e-12);
    assert(std::abs(R_prod(1, 1) - 1.0) < 1e-12);
}

// ==============================================================================
// 4. Triangular Solvers (solve_upper, solve_lower) Unit Tests
// ==============================================================================
void test_triangular_solvers() {
    std::cout << "  [Test] Triangular Solvers (solve_upper, solve_lower)...\n";

    // Upper Triangular:
    // [ 3  2 -1 ] [ x0 ]   [ 1 ]
    // [ 0  2  4 ] [ x1 ] = [ 8 ]
    // [ 0  0  5 ] [ x2 ]   [ 10 ]
    // x2 = 10 / 5 = 2
    // 2 * x1 + 4 * 2 = 8 -> x1 = 0
    // 3 * x0 + 2 * 0 - 1 * 2 = 1 -> 3 * x0 = 3 -> x0 = 1
    RationalMatrix U = {
        {Rational(3), Rational(2), Rational(-1)},
        {Rational(0), Rational(2), Rational(4)},
        {Rational(0), Rational(0), Rational(5)}
    };
    std::vector<Rational> b_u = {Rational(1), Rational(8), Rational(10)};
    auto x_u = Matrix<Rational>::solve_upper(U, b_u);
    assert(x_u.size() == 3);
    assert(x_u[0] == Rational(1));
    assert(x_u[1] == Rational(0));
    assert(x_u[2] == Rational(2));

    // Verify U * x_u == b_u
    auto b_u_check = U * x_u;
    assert(b_u_check == b_u);

    // Lower Triangular:
    // [ 2  0  0 ] [ x0 ]   [ 6 ]
    // [ 1  4  0 ] [ x1 ] = [ 11 ]
    // [-1  2  3 ] [ x2 ]   [ 4 ]
    // 2 * x0 = 6 -> x0 = 3
    // 1 * 3 + 4 * x1 = 11 -> 4 * x1 = 8 -> x1 = 2
    // -1 * 3 + 2 * 2 + 3 * x2 = 4 -> 1 + 3 * x2 = 4 -> x2 = 1
    RationalMatrix L = {
        {Rational(2), Rational(0), Rational(0)},
        {Rational(1), Rational(4), Rational(0)},
        {Rational(-1), Rational(2), Rational(3)}
    };
    std::vector<Rational> b_l = {Rational(6), Rational(11), Rational(4)};
    auto x_l = Matrix<Rational>::solve_lower(L, b_l);
    assert(x_l.size() == 3);
    assert(x_l[0] == Rational(3));
    assert(x_l[1] == Rational(2));
    assert(x_l[2] == Rational(1));

    // Verify L * x_l == b_l
    auto b_l_check = L * x_l;
    assert(b_l_check == b_l);
}

// ==============================================================================
// 5. Vector & Matrix Norms and Condition Checking Unit Tests
// ==============================================================================
void test_norms_and_condition() {
    std::cout << "  [Test] Vector Norms & Matrix Condition Checking...\n";

    // Vector Norms (Rational)
    std::vector<Rational> v = {Rational(3), Rational(-4), Rational(0), Rational(12)};
    // L1: |3| + |-4| + |0| + |12| = 19
    assert(norm_l1(v) == 19.0);
    // L2 squared: 3^2 + (-4)^2 + 0^2 + 12^2 = 9 + 16 + 144 = 169
    assert(norm_l2_sq(v) == Rational(169));
    // L2 Euclidean: sqrt(169) = 13.0
    assert(std::abs(norm_l2(v) - 13.0) < 1e-12);
    // Linf Chebyshev: max(3, 4, 0, 12) = 12.0
    assert(norm_linf(v) == 12.0);

    // Balanced Ternary Word Norms
    // TritWord64: 5 positive trits, 3 negative trits -> 8 non-zero trits
    TritWord64 w(0b10101011ULL, 0b01010100ULL);
    assert(w.isValid());
    assert(norm_l1(w) == 8);
    assert(norm_l2_sq(w) == 8);
    assert(std::abs(norm_l2(w) - std::sqrt(8.0)) < 1e-12);
    assert(norm_linf(w) == 1.0);

    TritWord64 zero_w(0, 0);
    assert(norm_l1(zero_w) == 0);
    assert(norm_l2_sq(zero_w) == 0);
    assert(norm_l2(zero_w) == 0.0);
    assert(norm_linf(zero_w) == 0.0);

    // Matrix Norms:
    // [[ 1, -2 ],
    //  [ 3,  4 ]]
    RealMatrix M = {
        {1.0, -2.0},
        {3.0, 4.0}
    };
    // 1-norm: max column sum = max(|1|+|3|, |-2|+|4|) = max(4, 6) = 6
    assert(M.norm_1() == 6.0);
    // inf-norm: max row sum = max(|1|+|-2|, |3|+|4|) = max(3, 7) = 7
    assert(M.norm_inf() == 7.0);
    // Frobenius norm: sqrt(1^2 + (-2)^2 + 3^2 + 4^2) = sqrt(1 + 4 + 9 + 16) = sqrt(30)
    assert(std::abs(M.norm_frobenius() - std::sqrt(30.0)) < 1e-12);

    // Condition number
    assert(M.is_invertible());
    assert(M.is_well_conditioned(100.0));
    double k1 = M.condition_number_1();
    assert(k1 > 1.0 && k1 < 20.0);
}

// ==============================================================================
// 6. Native Radix-3 Cooley-Tukey FFT Unit Tests (N = 27, 81, 243)
// ==============================================================================
void test_radix3_fft() {
    std::cout << "  [Test] Native Radix-3 Cooley-Tukey FFT & Parseval Energy...\n";

    // Verify Power of 3 helper
    assert(Radix3FFT::is_power_of_3(1));
    assert(Radix3FFT::is_power_of_3(3));
    assert(Radix3FFT::is_power_of_3(9));
    assert(Radix3FFT::is_power_of_3(27));
    assert(Radix3FFT::is_power_of_3(81));
    assert(Radix3FFT::is_power_of_3(243));
    assert(!Radix3FFT::is_power_of_3(2));
    assert(!Radix3FFT::is_power_of_3(6));
    assert(!Radix3FFT::is_power_of_3(10));

    // Verify Trit Reversal
    // For k=2 (N=9): trit reversal should produce [0, 3, 6, 1, 4, 7, 2, 5, 8]
    std::vector<size_t> expected_rev9 = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    for (size_t i = 0; i < 9; ++i) {
        assert(Radix3FFT::trit_reverse(i, 2) == expected_rev9[i]);
    }

    // Comparison against DFT reference oracle for N=3, 9, 27
    for (size_t N : {3, 9, 27}) {
        std::vector<std::complex<double>> x(N);
        for (size_t i = 0; i < N; ++i) {
            x[i] = std::complex<double>(std::sin(0.4 * i + 0.1), std::cos(0.8 * i - 0.2));
        }

        auto X_fft = Radix3FFT::fft(x);
        auto X_ref = Radix3FFT::dft_reference(x, false);

        double max_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double diff = std::abs(X_fft[i] - X_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        assert(max_diff < 1e-12);
    }

    // High-Density Round-Trip & Parseval Tests across N = 27, 81, 243
    for (size_t N : {27, 81, 243}) {
        std::vector<std::complex<double>> x(N);
        std::mt19937_64 rng(42 + N);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (size_t i = 0; i < N; ++i) {
            x[i] = std::complex<double>(dist(rng), dist(rng));
        }

        // 1. Forward FFT
        auto X = Radix3FFT::fft(x);

        // 2. Inverse FFT
        auto x_rec = Radix3FFT::ifft(X);

        // 3. Exact Round-Trip Invariant: ifft(fft(x)) == x
        double max_rec_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double err = std::abs(x[i] - x_rec[i]);
            if (err > max_rec_err) max_rec_err = err;
        }
        assert(max_rec_err < 1e-14);

        // 4. Exact Parseval Energy Conservation: 1/N * sum |X_k|^2 == sum |x_n|^2
        bool parseval_ok = Radix3FFT::verify_parseval(x, X, 1e-12);
        assert(parseval_ok);

        // 5. Unitary FFT Round-Trip
        auto X_u = Radix3FFT::fft_unitary(x);
        auto x_u_rec = Radix3FFT::ifft_unitary(X_u);
        double max_u_err = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double err = std::abs(x[i] - x_u_rec[i]);
            if (err > max_u_err) max_u_err = err;
        }
        assert(max_u_err < 1e-14);

        // Unitary energy preservation: ||X_u||^2 == ||x||^2
        double E_time = Radix3FFT::energy(x);
        double E_unitary = Radix3FFT::energy(X_u);
        assert(std::abs(E_time - E_unitary) / E_time < 1e-12);

        std::cout << "    -> N=" << std::setw(3) << N 
                  << " | Max Round-Trip Error: " << std::scientific << max_rec_err
                  << " | Parseval Preserved: 100% (diff < 1e-12)\n";
    }
}

// ==============================================================================
// 7. Bitwise & Coding Algorithms (Hamming & Carry-Save Tree Reduction)
// ==============================================================================
void test_bitwise_and_coding() {
    std::cout << "  [Test] Fast Ternary Hamming Distance & CSA Tree Reduction...\n";

    // 1. Hamming Distance
    TritWord64 a(0b1100ULL, 0b0011ULL); // trits 0,1 are -1; trits 2,3 are +1
    TritWord64 b(0b1010ULL, 0b0101ULL); // trits 0,2 are -1; trits 1,3 are +1
    // Trit 0: a=-1, b=-1 -> match (diff=0)
    // Trit 1: a=-1, b=+1 -> mismatch (diff=1)
    // Trit 2: a=+1, b=-1 -> mismatch (diff=1)
    // Trit 3: a=+1, b=+1 -> match (diff=0)
    // Hamming distance across trits 0..3 = 2
    assert(hamming_distance(a, b) == 2);

    // Same word distance = 0
    assert(hamming_distance(a, a) == 0);

    // Span Hamming distance
    std::vector<Trit> t1 = {Trit::Pos, Trit::Zero, Trit::Neg, Trit::Pos};
    std::vector<Trit> t2 = {Trit::Pos, Trit::Pos,  Trit::Neg, Trit::Zero};
    // diff at index 1 and 3 -> dist = 2
    assert(hamming_distance(std::span<const Trit>(t1), std::span<const Trit>(t2)) == 2);

    // 2. Multi-word Carry-Save Tree Reduction
    std::mt19937_64 rng(2026);
    std::uniform_int_distribution<int64_t> dist(-300, 300);

    for (size_t count : {1, 2, 3, 4, 5, 8, 16, 27, 32, 64}) {
        std::vector<TritWord64> words;
        words.reserve(count);
        int64_t expected_sum = 0;

        for (size_t i = 0; i < count; ++i) {
            int64_t val = dist(rng);
            expected_sum += val;
            words.push_back(Codec::from_int64(val, 20));
        }

        // Execute tree reduction
        TritWord64 reduced = reduce_csa_tree(words);
        int64_t actual_sum = Codec::to_int64(reduced, 25);

        assert(actual_sum == expected_sum);
    }
    std::cout << "    -> Multi-word CSA tree reduction verified across sizes 1 to 64.\n";
}

// ==============================================================================
// Main Runner
// ==============================================================================
int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Standalone libbt.hpp & Radix-3 FFT Suite...\n";

    test_rational_arithmetic();
    test_linear_algebra_determinant();
    test_linear_algebra_inverse();
    test_triangular_solvers();
    test_norms_and_condition();
    test_radix3_fft();
    test_bitwise_and_coding();

    std::cout << "  -> PASSED: All libbt Standard Library & Radix-3 FFT Unit Tests 100% Verified.\n";
    return 0;
}
