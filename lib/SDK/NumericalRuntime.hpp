#ifndef TRIUM_NUMERICAL_RUNTIME_HPP
#define TRIUM_NUMERICAL_RUNTIME_HPP
#include "Ternary/SDK/sdk.h"
namespace bt::sdk::detail {
bt_sdk_float_value float_extended(uint32_t,const bt_sdk_float_value*,const bt_sdk_float_value*);
bt_sdk_float_value float_powi(const bt_sdk_float_value*,int32_t);
bt_sdk_float_value float_reduce(uint32_t,uint32_t,const bt_sdk_float_value*,const bt_sdk_float_value*,uint64_t);
bt_sdk_float_value float_polynomial(const bt_sdk_float_value*,const bt_sdk_float_value*,uint64_t);
}
#endif
