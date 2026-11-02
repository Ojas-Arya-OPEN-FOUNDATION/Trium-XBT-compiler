#ifndef TRIUM_FLOAT_RUNTIME_HPP
#define TRIUM_FLOAT_RUNTIME_HPP
#include "Ternary/SDK/sdk.h"
#include "Ternary/Math/Float.hpp"
namespace bt::sdk::detail {
template<int W=1,class F> void float_width(uint32_t width,F&& fn){
    if(width==W)fn.template operator()<W>();
    else if constexpr(W<40)float_width<W+1>(width,std::forward<F>(fn));
    else throw std::invalid_argument("BTFP width must be 1..40");
}
template<int W> bt::BTFP<W> read_float(const bt_sdk_float_value* v){
    if(!(v && v->width==W && !v->reserved && !v->reserved2))throw std::invalid_argument("invalid BTFP record/profile");
    const auto lo=bt::BTFP<W>::min_norm_mantissa(),hi=bt::BTFP<W>::max_mantissa();
    if(!(v->mantissa==0?v->exponent==0:
        ((v->mantissa>=lo && v->mantissa<=hi)||(v->mantissa<=-lo && v->mantissa>=-hi))))throw std::invalid_argument("noncanonical BTFP record");
    return bt::BTFP<W>(v->mantissa,v->exponent);
}
template<int W> bt_sdk_float_value write_float(bt::BTFP<W> v){return {W,0,v.mantissa(),v.exponent(),0};}

}
#endif
