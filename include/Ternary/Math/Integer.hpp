#ifndef TERNARY_MATH_INTEGER_HPP
#define TERNARY_MATH_INTEGER_HPP
#include <cstdint>
#include <stdexcept>

// Hardware-independent value semantics. __int128 is an implementation strategy,
// not part of the public mathematical domain or a future hardware requirement.
namespace bt::math {
struct DivMod { int64_t quotient; int64_t remainder; };

// Euclidean division: a = b*q+r, 0 <= r < |b|, including negative operands.
inline constexpr DivMod divmod_euclidean(int64_t a, int64_t b) {
    if (b == 0) throw std::domain_error("integer division by zero");
    __int128 q = static_cast<__int128>(a) / b;
    __int128 r = static_cast<__int128>(a) % b;
    if (r < 0) {
        if (b > 0) { --q; r += b; }
        else { ++q; r -= b; }
    }
    if (q < INT64_MIN || q > INT64_MAX) throw std::overflow_error("integer quotient overflow");
    return {static_cast<int64_t>(q), static_cast<int64_t>(r)};
}

// Nearest integer quotient; exact half-way ties go away from zero.
inline constexpr int64_t round_quotient(int64_t a, int64_t b) {
    if (b == 0) throw std::domain_error("rounded division by zero");
    __int128 n = a, d = b;
    const bool negative = (n < 0) != (d < 0);
    if (n < 0) n = -n;
    if (d < 0) d = -d;
    __int128 q = (n + d / 2) / d;
    if (negative) q = -q;
    if (q < INT64_MIN || q > INT64_MAX) throw std::overflow_error("rounded quotient overflow");
    return static_cast<int64_t>(q);
}

template <unsigned Width> class BalancedInteger {
    static_assert(Width >= 1 && Width <= 40, "int64 storage supports 1..40 balanced trits");
    int64_t value_;
public:
    static constexpr int64_t maximum = [] {
        __int128 p = 1;
        for (unsigned i = 0; i < Width; ++i) p *= 3;
        return static_cast<int64_t>((p - 1) / 2);
    }();
    static constexpr int64_t minimum = -maximum;
    static constexpr int64_t check(__int128 value) {
        if (value < minimum || value > maximum)
            throw std::overflow_error("integer overflow: outside balanced scalar domain");
        return static_cast<int64_t>(value);
    }
    explicit constexpr BalancedInteger(int64_t value = 0) : value_(check(value)) {}
    constexpr int64_t value() const noexcept { return value_; }
    friend constexpr bool operator==(BalancedInteger, BalancedInteger) = default;
    constexpr BalancedInteger operator-() const { return BalancedInteger(-value_); }
    constexpr BalancedInteger operator+(BalancedInteger b) const {
        return BalancedInteger(check(static_cast<__int128>(value_) + b.value_));
    }
    constexpr BalancedInteger operator-(BalancedInteger b) const {
        return BalancedInteger(check(static_cast<__int128>(value_) - b.value_));
    }
    constexpr BalancedInteger operator*(BalancedInteger b) const {
        return BalancedInteger(check(static_cast<__int128>(value_) * b.value_));
    }
    constexpr BalancedInteger operator/(BalancedInteger b) const {
        return BalancedInteger(divmod_euclidean(value_, b.value_).quotient);
    }
    constexpr BalancedInteger operator%(BalancedInteger b) const {
        return BalancedInteger(divmod_euclidean(value_, b.value_).remainder);
    }
};
using ScalarInteger = BalancedInteger<39>;
} // namespace bt::math
#endif
