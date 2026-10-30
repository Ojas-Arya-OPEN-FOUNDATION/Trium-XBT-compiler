#ifndef TRIUM_TERNARY_WIDE_HPP
#define TRIUM_TERNARY_WIDE_HPP
#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace bt::fp_detail {
// Internal checked unsigned scratch integers, not a public arbitrary-precision
// datatype. 512 bits cover close FMA alignment; 1280 cover every binary64 input.
// No allocation, external library, or floating-point arithmetic is involved.
template<size_t N> struct Wide {
    static_assert(N>=2);
    std::array<uint64_t,N> words{};
    constexpr Wide(unsigned __int128 n=0) { words[0]=uint64_t(n);words[1]=uint64_t(n>>64); }
    constexpr bool zero() const {for(auto w:words)if(w)return false;return true;}
    constexpr int compare(const Wide& b) const {for(size_t i=N;i-->0;)if(words[i]!=b.words[i])return words[i]<b.words[i]?-1:1;return 0;}
    constexpr Wide times(uint64_t b) const {
        Wide r;unsigned __int128 carry=0;
        for(size_t i=0;i<N;++i){auto p=static_cast<unsigned __int128>(words[i])*b+carry;r.words[i]=uint64_t(p);carry=p>>64;}
        if(carry)throw std::overflow_error("BTFP scratch multiplication capacity");return r;
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
    constexpr Wide shifted(unsigned count) const {Wide r=*this;for(unsigned i=0;i<count;++i)r=r.times(2);return r;}
    constexpr unsigned bits() const {for(size_t i=N;i-->0;)if(words[i]){unsigned n=0;auto v=words[i];while(v){++n;v>>=1;}return unsigned(i*64)+n;}return 0;}
};
template<size_t N> constexpr uint64_t quotient(const Wide<N>& n,const Wide<N>& d,uint64_t maximum) {
    if(d.zero())throw std::domain_error("zero scratch divisor");
    uint64_t lo=0,hi=maximum;
    while(lo<hi){auto mid=lo+(hi-lo)/2+1;if(d.times(mid).compare(n)<=0)lo=mid;else hi=mid-1;}
    return lo;
}
}
#endif
