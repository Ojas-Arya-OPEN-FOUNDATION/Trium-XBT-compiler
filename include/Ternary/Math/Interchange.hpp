#ifndef TRIUM_MATH_INTERCHANGE_HPP
#define TRIUM_MATH_INTERCHANGE_HPP
#include "Integer.hpp"
#include "Rational.hpp"
#include <array>
#include <algorithm>
#include <new>
#include <utility>
#include <bit>
#include <charconv>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace bt::math {
enum class MathStatus { ok, invalid, domain, overflow, allocation, internal };
template<class T> struct MathResult { T value{}; MathStatus status=MathStatus::ok; explicit operator bool() const {return status==MathStatus::ok;} };
template<class F> auto attempt(F fn) noexcept -> MathResult<decltype(fn())> {
    try{return {fn(),MathStatus::ok};}
    catch(const std::overflow_error&){return {{},MathStatus::overflow};}
    catch(const std::domain_error&){return {{},MathStatus::domain};}
    catch(const std::invalid_argument&){return {{},MathStatus::invalid};}
    catch(const std::bad_alloc&){return {{},MathStatus::allocation};}
    catch(...){return {{},MathStatus::internal};}
}
inline int64_t parse_integer(std::string_view text) {
    if(text.empty() || text.size()>64) throw std::invalid_argument("invalid bounded decimal integer");
    int64_t n;auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),n);
    if(error==std::errc::result_out_of_range)throw std::overflow_error("decimal integer outside int64");
    if(error!=std::errc{} || end!=text.data()+text.size())throw std::invalid_argument("invalid decimal integer");
    return n;
}
template<unsigned W=39> Rational<W> parse_rational(std::string_view text) {
    if(text.size()>130)throw std::invalid_argument("rational text limit exceeded");
    auto pos=text.find('/');
    return pos==text.npos?Rational<W>(parse_integer(text)):Rational<W>(parse_integer(text.substr(0,pos)),parse_integer(text.substr(pos+1)));
}
template<unsigned W> std::string format(Rational<W> value) {return std::to_string(value.numerator())+"/"+std::to_string(value.denominator());}
// Bounded exact decimal grammar: optional '-', digits, optional '.' + digits.
// At most 18 fractional places, reduced before final component bounds.
template<unsigned W=39> Rational<W> parse_decimal(std::string_view text) {
    if(text.empty() || text.size()>64)throw std::invalid_argument("decimal text limit");
    bool minus=text.front()=='-';if(minus)text.remove_prefix(1);
    auto dot=text.find('.');auto whole=text.substr(0,dot);
    if(whole.empty() || (whole.size()>1 && whole.front()=='0'))throw std::invalid_argument("noncanonical decimal integer part");
    auto fraction=dot==text.npos?std::string_view{}:text.substr(dot+1);
    if(dot!=text.npos && fraction.empty())throw std::invalid_argument("missing fractional digits");
    if(fraction.size()>18)throw std::invalid_argument("decimal fractional capacity is 18 digits");
    __int128 n=0,d=1;const __int128 limit=(__int128(1)<<126)-1;
    for(char c:text){if(c=='.')continue;if(c<'0'||c>'9')throw std::invalid_argument("invalid decimal digit");if(n>(limit-(c-'0'))/10)throw std::overflow_error("decimal accumulator capacity");n=n*10+(c-'0');}
    // Reject a second decimal point explicitly (it is not another separator).
    if(dot!=text.npos && fraction.find('.')!=text.npos)throw std::invalid_argument("multiple decimal points");
    for(size_t i=0;i<fraction.size();++i)d*=10;
    auto a=n,b=d;while(b){auto r=a%b;a=b;b=r;}n/=a;d/=a;if(minus)n=-n;
    return Rational<W>(BalancedInteger<W>::check(n),BalancedInteger<W>::check(d));
}
// Positional dual-rail representation. Invalid 11 rails are never a fourth trit.
struct DualRail { uint64_t positive=0,negative=0; friend bool operator==(DualRail,DualRail)=default; };
inline void validate(DualRail w,unsigned width=64) {
    if(width<1 || width>64 || (w.positive&w.negative))throw std::invalid_argument("invalid dual-rail width/overlap");
    if(width<64 && ((w.positive|w.negative)>>width))throw std::invalid_argument("nonzero rails outside declared width");
}
inline DualRail encode_integer(int64_t n,unsigned width=64) {
    validate({},width);DualRail w;
    for(unsigned i=0;i<width;++i) {
        auto q=n/3,r=n%3;if(r==2){++q;r=-1;}else if(r==-2){--q;r=1;}
        if(r>0)w.positive|=uint64_t(1)<<i;else if(r<0)w.negative|=uint64_t(1)<<i;n=q;
    }
    if(n)throw std::overflow_error("integer does not fit requested trit width");
    return w;
}
inline int64_t decode_integer(DualRail w,unsigned width=64) {
    validate(w,width);__int128 n=0;
    for(unsigned i=width;i--;) {
        n=n*3+int((w.positive>>i)&1)-int((w.negative>>i)&1);
        // A balanced suffix cannot bring an overflowing prefix back into int64.
        if(n<INT64_MIN || n>INT64_MAX)throw std::overflow_error("dual-rail positional value outside int64");
    }
    return static_cast<int64_t>(n);
}
inline DualRail parse_trits(std::string_view text) {
    if(text.empty() || text.size()>64)throw std::invalid_argument("trit text requires 1..64 digits");
    DualRail w;for(char c:text) {
        w.positive<<=1;w.negative<<=1;
        if(c=='+')w.positive|=1;else if(c=='-')w.negative|=1;else if(c!='0')throw std::invalid_argument("trit digits are +,0,-");
    }return w;
}
inline std::string format_trits(DualRail w,unsigned width) {
    validate(w,width);std::string s;for(unsigned i=width;i--;)s+=((w.positive>>i)&1)?'+':((w.negative>>i)&1)?'-':'0';return s;
}
inline void put64(std::span<uint8_t> out,uint64_t n) {for(unsigned i=0;i<8;++i)out[i]=uint8_t(n>>(8*i));}
inline uint64_t get64(std::span<const uint8_t> in) {uint64_t n=0;for(unsigned i=0;i<8;++i)n|=uint64_t(in[i])<<(8*i);return n;}
// Wire v1: BTR1, width, 3 reserved zero bytes, positive LE64, negative LE64.
inline std::array<uint8_t,24> serialize(DualRail w,unsigned width) {
    validate(w,width);std::array<uint8_t,24> out{'B','T','R',1,uint8_t(width)};
    put64(std::span(out).subspan(8,8),w.positive);put64(std::span(out).subspan(16,8),w.negative);return out;
}
inline std::pair<DualRail,unsigned> deserialize(std::span<const uint8_t> in) {
    if(in.size()!=24 || in[0]!='B' || in[1]!='T' || in[2]!='R' || in[3]!=1 || in[5] || in[6] || in[7])throw std::invalid_argument("invalid dual-rail wire v1");
    DualRail w{get64(in.subspan(8,8)),get64(in.subspan(16,8))};validate(w,in[4]);return {w,in[4]};
}
template<unsigned W> std::array<uint8_t,24> serialize(Rational<W> value) {
    std::array<uint8_t,24> out{'B','T','Q',1,uint8_t(W)};
    put64(std::span(out).subspan(8,8),std::bit_cast<uint64_t>(value.numerator()));
    put64(std::span(out).subspan(16,8),uint64_t(value.denominator()));return out;
}
template<unsigned W> Rational<W> deserialize_rational(std::span<const uint8_t> in) {
    if(in.size()!=24 || in[0]!='B'||in[1]!='T'||in[2]!='Q'||in[3]!=1||in[4]!=W||in[5]||in[6]||in[7])throw std::invalid_argument("invalid rational wire v1");
    auto n=std::bit_cast<int64_t>(get64(in.subspan(8,8))),d=std::bit_cast<int64_t>(get64(in.subspan(16,8)));
    Rational<W> value(n,d);if(n!=value.numerator()||d!=value.denominator())throw std::invalid_argument("noncanonical rational wire");return value;
}
// Dense row-major int39 matmul. Increasing-k checked multiply/add order is part
// of the contract. No floating quantization or parallel reassociation is hidden.
inline void matmul(std::span<const int64_t> a,std::span<const int64_t> b,std::span<int64_t> out,size_t m,size_t k,size_t n) {
    auto product=[](size_t x,size_t y){if(y && x>SIZE_MAX/y)throw std::overflow_error("matrix extent overflow");return x*y;};
    if(a.size()!=product(m,k) || b.size()!=product(k,n) || out.size()!=product(m,n))throw std::invalid_argument("matrix storage/shape mismatch");
    if(product(product(m,k),n)>100000000 || out.size()>1000000)throw std::invalid_argument("matrix resource limit exceeded");
    for(auto x:a)ScalarInteger::check(x);for(auto x:b)ScalarInteger::check(x);
    if(out.empty())return; // avoid iterating an enormous m when n==0
    std::vector<int64_t> result(out.size());
    for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j){int64_t sum=0;for(size_t p=0;p<k;++p){auto term=ScalarInteger::check(__int128(a[i*k+p])*b[p*n+j]);sum=ScalarInteger::check(__int128(sum)+term);}result[i*n+j]=sum;}
    if(!result.empty())std::copy(result.begin(),result.end(),out.begin());
}
} // namespace bt::math
#endif
