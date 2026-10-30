#ifndef TRIUM_MATH_FLOAT_HPP
#define TRIUM_MATH_FLOAT_HPP
#include "Interchange.hpp"
#include "../Detail/ternary_fp.hpp"
namespace bt::math {
// Exact decimal-to-ratio input, one BTFP rounding. No binary64 intermediate.
// The bounded decimal ratio must fit the explicit Rational<40> input profile.
template<int W,int E=0> bt::BTFP<W,E> float_from_decimal(std::string_view text) {
    auto value=parse_decimal<40>(text);return bt::BTFP<W,E>::from_ratio(value.numerator(),value.denominator());
}
}
#endif
