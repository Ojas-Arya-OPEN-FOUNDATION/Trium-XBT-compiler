#include "NumericalRuntime.hpp"
#include "FloatRuntime.hpp"
#include "Ternary/Math/Numerical.hpp"
namespace bt::sdk::detail {
bt_sdk_float_value float_extended(uint32_t op,const bt_sdk_float_value* a,const bt_sdk_float_value* b){
    if(!a)throw std::invalid_argument("null BTFP input");bt_sdk_float_value result{};
    float_width(a->width,[&]<int W>(){auto x=read_float<W>(a);bt::BTFP<W> value;
        switch(op){case BT_SDK_FP_RECIPROCAL:value=x.reciprocal();break;case BT_SDK_FP_CBRT:value=x.cbrt();break;
        case BT_SDK_FP_HYPOT:value=x.hypot(read_float<W>(b));break;default:throw std::invalid_argument("unknown extended BTFP operation");}
        result=write_float(value);
    });return result;
}
bt_sdk_float_value float_powi(const bt_sdk_float_value* a,int32_t power){
    if(!a)throw std::invalid_argument("null BTFP input");bt_sdk_float_value result{};
    float_width(a->width,[&]<int W>(){result=write_float(read_float<W>(a).pow_int(power));});return result;
}
bt_sdk_float_value float_reduce(uint32_t width,uint32_t op,const bt_sdk_float_value* a,const bt_sdk_float_value* b,uint64_t count){
    if(op<BT_SDK_FP_SUM || op>BT_SDK_FP_NORM2)throw std::invalid_argument("unknown BTFP reduction");
    if(count>4096)throw bt::fp_detail::NumericalLimit("BTFP reduction count exceeds 4096");
    if(count && (!a || (op==BT_SDK_FP_DOT && !b)))throw std::invalid_argument("null BTFP array");
    bt_sdk_float_value result{};
    float_width(width,[&]<int W>(){using F=bt::BTFP<W>;std::vector<F> x,y;x.reserve(count);if(op==BT_SDK_FP_DOT)y.reserve(count);
        for(size_t i=0;i<count;++i){x.push_back(read_float<W>(a+i));if(op==BT_SDK_FP_DOT)y.push_back(read_float<W>(b+i));}
        F value;if(op==BT_SDK_FP_SUM)value=bt::math::sum_exact<W,0>(x);
        else if(op==BT_SDK_FP_DOT)value=bt::math::dot_exact<W,0>(x,y);
        else value=bt::math::norm2<W,0>(x);
        result=write_float(value);
    });return result;
}
bt_sdk_float_value float_polynomial(const bt_sdk_float_value* x,const bt_sdk_float_value* coefficients,uint64_t count){
    if(!x || (count && !coefficients))throw std::invalid_argument("null BTFP polynomial input");
    if(count>17)throw bt::fp_detail::NumericalLimit("BTFP polynomial degree exceeds 16");
    bt_sdk_float_value result{};
    float_width(x->width,[&]<int W>(){auto value=read_float<W>(x);std::vector<bt::BTFP<W>> c;c.reserve(count);
        for(size_t i=0;i<count;++i)c.push_back(read_float<W>(coefficients+i));
        result=write_float(bt::math::polynomial_exact<W,0>(value,c));
    });return result;
}
}
