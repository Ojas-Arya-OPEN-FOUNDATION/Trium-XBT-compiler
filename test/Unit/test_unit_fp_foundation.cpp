#include "ternary_fp.hpp"
#include <cassert>
#include <climits>
#include <iostream>
using F=bt::BTFP<3>;
constexpr auto fused=F::fma(F(6),F(7),F(-5,-200));
static_assert(fused.mantissa()==13 && fused.exponent()==1);
int main(){
    assert(F::fma(F(6),F(7),F(5,-200))==F(5,2));
    assert(F::fma(F(-6),F(7),F(5,-200))==F(-13,1));
    assert(F::fma(F(-6),F(7),F(-5,-200))==F(-5,2));
    assert(F::fma(F(5,INT_MAX),F(3),F(-5,INT_MAX))==F(10,INT_MAX));
    using G=bt::BTFP<40>;auto h=G::max_mantissa();
    assert(G(h).add(G(h,-39))==G(G::min_norm_mantissa(),1));
    assert(G(h).sub(G(h,-39))==G(h-1));
    assert(G(9).sqrt()==G(3));
    assert(G(1).sqrt()==G(1));
    assert(G::from_ratio(9007199254740993LL,1).to_double()==9007199254740992.0);
    assert(G::from_ratio(9007199254740995LL,1).to_double()==9007199254740996.0);
    using C=bt::BTFP<3,1>;
    assert(C(13,1).sqrt().is_positive());
    bool underflow=false;try{(void)C(5,-1).sqrt();}catch(const std::overflow_error&){underflow=true;}assert(underflow);
    bool overflow=false;try{(void)C(13,1).mul(C(13,1));}catch(const std::overflow_error&){overflow=true;}assert(overflow);
    using S=bt::fp_detail::Wide<8>;S carry(~static_cast<unsigned __int128>(0));
    auto next=carry.plus(S(1));assert(next.words[0]==0 && next.words[1]==0 && next.words[2]==1);
    assert(next.minus(S(1)).compare(carry)==0);
    assert(carry.times(3).compare(carry.plus(carry).plus(carry))==0);
    std::cout<<"PASS constexpr sticky ties, W40 sqrt/alignment, fused cancellation, binary ties and wide carries\n";
}
