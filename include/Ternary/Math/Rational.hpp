#ifndef TERNARY_MATH_RATIONAL_HPP
#define TERNARY_MATH_RATIONAL_HPP
#include "Integer.hpp"
namespace bt::math {
// Exact, reduced rational values with bounded canonical components. Arithmetic
// reduces BEFORE checking final bounds; no floating-point approximation is used.
template<unsigned Width = 39> class Rational {
    using Integer = BalancedInteger<Width>;
    int64_t numerator_ = 0, denominator_ = 1;
    struct Wide {};
    static constexpr __int128 magnitude(__int128 n) { return n < 0 ? -n : n; }
    static constexpr __int128 gcd(__int128 a, __int128 b) {
        a = magnitude(a); b = magnitude(b);
        while (b) { __int128 r = a % b; a = b; b = r; }
        return a;
    }
    constexpr Rational(__int128 n, __int128 d, Wide) {
        if (!d) throw std::domain_error("rational denominator is zero");
        if (d < 0) { n = -n; d = -d; }
        auto divisor = gcd(n,d);
        n /= divisor; d /= divisor;
        numerator_ = Integer::check(n); denominator_ = Integer::check(d);
    }
public:
    constexpr Rational(int64_t n = 0, int64_t d = 1) : Rational(n,d,Wide{}) {}
    constexpr int64_t numerator() const { return numerator_; }
    constexpr int64_t denominator() const { return denominator_; }
    friend constexpr bool operator==(Rational, Rational) = default;
    constexpr Rational operator-() const { return Rational(-numerator_, denominator_); }
    constexpr Rational operator+(Rational b) const {
        return Rational(static_cast<__int128>(numerator_)*b.denominator_ +
                        static_cast<__int128>(b.numerator_)*denominator_,
                        static_cast<__int128>(denominator_)*b.denominator_, Wide{});
    }
    constexpr Rational operator-(Rational b) const { return *this + (-b); }
    constexpr Rational operator*(Rational b) const {
        return Rational(static_cast<__int128>(numerator_)*b.numerator_,
                        static_cast<__int128>(denominator_)*b.denominator_, Wide{});
    }
    constexpr Rational operator/(Rational b) const {
        return Rational(static_cast<__int128>(numerator_)*b.denominator_,
                        static_cast<__int128>(denominator_)*b.numerator_, Wide{});
    }
    constexpr bool operator<(Rational b) const {
        return static_cast<__int128>(numerator_)*b.denominator_ <
               static_cast<__int128>(b.numerator_)*denominator_;
    }
    constexpr int64_t round() const { return round_quotient(numerator_, denominator_); }
};
} // namespace bt::math
#endif
