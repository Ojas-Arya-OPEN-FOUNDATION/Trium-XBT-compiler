#include <Ternary/SDK/SDK.hpp>
#include <Ternary/Math/Float.hpp>
#include <Ternary/Math/Numerical.hpp>
#include <iostream>
int main(){
 auto nine=bt::sdk::TernaryFloat::from_ratio(40,9);
 if(nine.apply(BT_SDK_FP_SQRT).to_double()!=3.0)return 2;
auto three=bt::sdk::TernaryFloat::from_ratio(40,3),four=bt::sdk::TernaryFloat::from_ratio(40,4);
 if(three.apply(BT_SDK_FP_HYPOT,&four).to_double()!=5 || bt::sdk::TernaryFloat::reduce(40,BT_SDK_FP_NORM2,{three,four}).to_double()!=5)return 3;
 std::vector<bt::BTFP<40>> terms{bt::BTFP<40>(100),bt::BTFP<40>(1),bt::BTFP<40>(-100)};
 if(bt::math::sum_exact<40,0>(terms)!=bt::BTFP<40>(1))return 4;
bt::sdk::Context context;auto module=context.compile("func transform(x:int)->int{return x*3;}");auto output=module.map("transform",{-2,0,2});if(output!=std::vector<int64_t>{-6,0,6})return 1;auto half=bt::math::float_from_decimal<20>("0.5");std::cout<<"CPU SDK and exact-ratio BTFP input: "<<half.to_double()<<"\n";}
