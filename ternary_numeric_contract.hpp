#ifndef TERNARY_NUMERIC_CONTRACT_HPP
#define TERNARY_NUMERIC_CONTRACT_HPP
#include <cstdint>
#include <stdexcept>
#include "include/Ternary/Math/Integer.hpp"

namespace bt::contract {
// Language scalar integers use 39 balanced positional trits. This is NOT
// the range of a packed 64-lane vector or of the arbitrary-precision library.
inline constexpr int64_t integer_max = math::ScalarInteger::maximum;
inline constexpr int64_t integer_min = math::ScalarInteger::minimum;
inline int64_t checked_integer(__int128 value) { return math::ScalarInteger::check(value); }
inline int64_t add(int64_t a, int64_t b) { return checked_integer(static_cast<__int128>(a) + b); }
inline int64_t sub(int64_t a, int64_t b) { return checked_integer(static_cast<__int128>(a) - b); }
inline int64_t mul(int64_t a, int64_t b) { return checked_integer(static_cast<__int128>(a) * b); }
} // namespace bt::contract
#endif
