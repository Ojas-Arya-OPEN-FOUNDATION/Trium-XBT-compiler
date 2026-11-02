#ifndef TRIUM_TERNARY_WIDE_HPP
#define TRIUM_TERNARY_WIDE_HPP
#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <utility>
namespace bt::fp_detail {
struct NumericalLimit:std::runtime_error{using std::runtime_error::runtime_error;};
template<int W,int E> struct Numerical;
inline constexpr std::array<uint64_t, 41> POW3_TABLE_64 = []() constexpr {
    std::array<uint64_t, 41> arr{};
    arr[0] = 1;
    for (size_t i = 1; i <= 40; ++i) {
        arr[i] = arr[i - 1] * 3;
    }
    return arr;
}();

// Internal checked unsigned scratch integers, not a public arbitrary-precision
// datatype. 512 bits cover close FMA alignment; 1280 cover every binary64 input.
// No allocation, external library, or floating-point arithmetic is involved.
template<size_t N> struct Wide {
    static_assert(N>=2);
    std::array<uint64_t,N> words{};
    constexpr Wide(unsigned __int128 n=0) { words[0]=uint64_t(n);words[1]=uint64_t(n>>64); }
    constexpr bool zero() const {for(auto w:words)if(w)return false;return true;}
    constexpr int compare(const Wide& b) const {for(size_t i=N;i-->0;)if(words[i]!=b.words[i])return words[i]<b.words[i]?-1:1;return 0;}
    constexpr bool can_multiply(uint64_t b) const {
        if (b == 0 || zero()) return true;
        unsigned __int128 carry = 0;
        for (size_t i = 0; i < N; ++i) {
            carry = (static_cast<unsigned __int128>(words[i]) * b + carry) >> 64;
        }
        return carry == 0;
    }
    constexpr Wide times(uint64_t b) const {
        Wide r;unsigned __int128 carry=0;
        for(size_t i=0;i<N;++i){auto p=static_cast<unsigned __int128>(words[i])*b+carry;r.words[i]=uint64_t(p);carry=p>>64;}
        if(carry)throw std::overflow_error("BTFP scratch multiplication capacity");return r;
    }
    constexpr Wide times_pow3(uint64_t count) const {
        Wide r = *this;
        while (count > 0) {
            uint64_t step = (count > 40) ? 40 : count;
            r = r.times(POW3_TABLE_64[step]);
            count -= step;
        }
        return r;
    }
    constexpr Wide plus(const Wide& b) const {
        Wide r;unsigned __int128 carry=0;
        for(size_t i=0;i<N;++i){auto p=static_cast<unsigned __int128>(words[i])+b.words[i]+carry;r.words[i]=uint64_t(p);carry=p>>64;}
        if(carry)throw std::overflow_error("BTFP scratch addition capacity");return r;
    }
    constexpr Wide minus(const Wide& b) const {
        if(compare(b)<0)throw std::logic_error("negative unsigned BTFP scratch subtraction");
        Wide r;uint64_t borrow=0;
        for(size_t i=0;i<N;++i){auto rhs=static_cast<unsigned __int128>(b.words[i])+borrow;r.words[i]=uint64_t(static_cast<unsigned __int128>(words[i])-rhs);borrow=static_cast<unsigned __int128>(words[i])<rhs;}
        return r;
    }
    constexpr std::pair<Wide,uint64_t> divmod(uint64_t divisor) const {
        if(!divisor)throw std::domain_error("zero scratch divisor");
        Wide quotient;unsigned __int128 remainder=0;
        for(size_t i=N;i-->0;){auto value=(remainder<<64)|words[i];quotient.words[i]=uint64_t(value/divisor);remainder=value%divisor;}
        return {quotient,uint64_t(remainder)};
    }
    constexpr Wide shifted(unsigned count) const {
        if (count == 0 || zero()) return *this;
        Wide r;
        const size_t word_shift = count / 64;
        const unsigned bit_shift = count % 64;
        if (word_shift >= N) throw std::overflow_error("BTFP scratch shift capacity");
        for (size_t i = N - word_shift; i < N; ++i) {
            if (words[i]) throw std::overflow_error("BTFP scratch shift capacity");
        }
        if (bit_shift == 0) {
            for (size_t i = 0; i < N - word_shift; ++i) r.words[i + word_shift] = words[i];
        } else {
            const unsigned rshift = 64 - bit_shift;
            if (words[N - word_shift - 1] >> rshift) throw std::overflow_error("BTFP scratch shift capacity");
            r.words[word_shift] = words[0] << bit_shift;
            for (size_t i = 1; i < N - word_shift; ++i) {
                r.words[i + word_shift] = (words[i] << bit_shift) | (words[i - 1] >> rshift);
            }
        }
        return r;
    }
    constexpr unsigned bits() const {for(size_t i=N;i-->0;)if(words[i]){unsigned n=0;auto v=words[i];while(v){++n;v>>=1;}return unsigned(i*64)+n;}return 0;}
};
template<size_t N> constexpr uint64_t quotient(const Wide<N>& n,const Wide<N>& d,uint64_t maximum) {
    if(d.zero())throw std::domain_error("zero scratch divisor");
    uint64_t lo=0,hi=maximum;
    while(lo<hi){
        auto mid=lo+(hi-lo)/2+1;
        if(d.can_multiply(mid) && d.times(mid).compare(n)<=0) lo=mid;
        else hi=mid-1;
    }
    return lo;
}
}
#endif
