#include <Ternary/SDK/SDK.hpp>
#include <Ternary/Math/Float.hpp>
#include <iostream>
int main(){
 auto nine=bt::sdk::TernaryFloat::from_ratio(40,9);
 if(nine.apply(BT_SDK_FP_SQRT).to_double()!=3.0)return 2;
bt::sdk::Context context;auto module=context.compile("func transform(x:int)->int{return x*3;}");auto output=module.map("transform",{-2,0,2});if(output!=std::vector<int64_t>{-6,0,6})return 1;auto half=bt::math::float_from_decimal<20>("0.5");std::cout<<"CPU SDK and exact-ratio BTFP input: "<<half.to_double()<<"\n";}
