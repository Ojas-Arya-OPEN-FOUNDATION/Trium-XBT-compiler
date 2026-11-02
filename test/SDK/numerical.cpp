#include <Ternary/Math/Numerical.hpp>
#include <Ternary/SDK/SDK.hpp>
#include <cassert>
#include <bit>
#include <cmath>
#include <limits>
#include <iostream>
#include <random>
using namespace bt;
int main(){
 using F=BTFP<40>;using S=bt::sdk::TernaryFloat;
 std::vector<F> v{F(100),F(1),F(-100)};
 assert((bt::math::sum_exact<40,0>(v)==F(1)));
 assert((bt::math::dot_exact<40,0>(v,v)==F(20001)));
 std::vector<F> tri{F(3),F(4)};assert((bt::math::norm2<40,0>(tri)==F(5)));
 std::vector<F> poly{F(1),F(2),F(3)};assert((bt::math::polynomial_exact<40,0>(F(2),poly)==F(17)));
 assert(F(-27).cbrt()==F(-3));assert(F(3).hypot(F(4))==F(5));
 assert(F(1).pow_int(-16)==F(1));assert(F(0).pow_int(0)==F(1));
 auto a=S::from_ratio(40,3),b=S::from_ratio(40,4);
 assert(a.apply(BT_SDK_FP_HYPOT,&b).to_double()==5);
 assert(S::from_ratio(40,-27).apply(BT_SDK_FP_CBRT).to_double()==-3);
 assert(S::from_ratio(40,1).pow_int(-16).to_double()==1);
 assert(S::reduce(40,BT_SDK_FP_SUM,{a,b}).to_double()==7);
 assert(S::reduce(40,BT_SDK_FP_DOT,{a,b},{a,b}).to_double()==25);
 assert(S::reduce(40,BT_SDK_FP_NORM2,{a,b}).to_double()==5);
 assert(S::from_ratio(40,2).polynomial({S::from_ratio(40,1),S::from_ratio(40,2),S::from_ratio(40,3)}).to_double()==17);
 // New scratch division: exact reconstruction at full limb range and all u64 divisors.
 std::mt19937_64 rng(122);
 for(int n=0;n<500;++n){fp_detail::Wide<4> big;for(auto& word:big.words)word=rng();uint64_t d=rng()|1;
  auto [q,r]=big.divmod(d);assert(r<d && q.times(d).plus(fp_detail::Wide<4>(r)).compare(big)==0);}
 // Older power-of-three helpers remain safe even at signed minima/maxima.
 auto max128=static_cast<__int128>((static_cast<unsigned __int128>(1)<<127)-1);
 for(__int128 x:{-max128-1,-max128,__int128(-1),__int128(0),__int128(1),max128})for(int k=-2;k<=85;++k){
  auto expected=x;for(int i=0;i<k;++i)expected=fp_detail::round_div3_128(expected);
  assert(fp_detail::round_div_pow3_fast128(x,k)==expected);
 }
 for(int64_t x:{INT64_MIN,INT64_MIN+1,int64_t(-1),int64_t(0),int64_t(1),INT64_MAX})for(int k=-2;k<=45;++k){
  auto expected=x;for(int i=0;i<k;++i)expected=fp_detail::round_div3(expected);
  assert(fp_detail::round_div_pow3_fast64(x,k)==expected);
 }
 // Approximate helpers: robustness and sampled error, not universal rounding proof.
 double s,c;fp_detail::cordic_circ_sincos(-0.0,s,c);assert(s==0 && std::signbit(s) && c==1);
 fp_detail::cordic_circ_sincos(INFINITY,s,c);assert(std::isnan(s)&&std::isnan(c));
 fp_detail::cordic_hyp_sincos(-0.0,s,c);assert(s==0 && std::signbit(s) && c==1);
 fp_detail::cordic_hyp_sincos(1.0,s,c);assert(std::isnan(s)&&std::isnan(c));
 assert(std::isnan(fp_detail::cordic_hyp_ln(0)));assert(std::isnan(fp_detail::cordic_hyp_ln(INFINITY)));
 for(int i=-1000;i<=1000;++i){double x=i/10.0;fp_detail::cordic_circ_sincos(x,s,c);
  assert(std::abs(s-std::sin(x))<1e-11 && std::abs(c-std::cos(x))<1e-11);}
 for(int i=0;i<=1000;++i){double x=0.5+i/1000.0;assert(std::abs(fp_detail::cordic_hyp_ln(x)-std::log(x))<1e-11);
  x=(i-500)*fp_detail::CORDIC_LN3/1000;fp_detail::cordic_hyp_sincos(x,s,c);
  assert(std::abs(s-std::sinh(x))<1e-11 && std::abs(c-std::cosh(x))<1e-11);}
 const double inf=std::numeric_limits<double>::infinity();
 assert(std::isnan(fp_detail::cordic_circ_atan2(NAN,1.0)));
 for(double x:{-inf,-std::numeric_limits<double>::max(),-1.0,-std::numeric_limits<double>::denorm_min(),-0.0,0.0,std::numeric_limits<double>::denorm_min(),1.0,std::numeric_limits<double>::max(),inf})
 for(double y:{-inf,-std::numeric_limits<double>::max(),-1.0,-std::numeric_limits<double>::denorm_min(),-0.0,0.0,std::numeric_limits<double>::denorm_min(),1.0,std::numeric_limits<double>::max(),inf}){
  auto actual=fp_detail::cordic_circ_atan2(y,x),expected=std::atan2(y,x);
  assert(std::abs(actual-expected)<1e-11);if(actual==0 && expected==0)assert(std::signbit(actual)==std::signbit(expected));
 }
 // Near half-bin reductions in the legacy approximate exp path remain finite.
 for(int k:{-1000,-1,0,1,1000})for(double d:{-1e-12,0.0,1e-12}){
  double x=(k+0.5)*fp_detail::CORDIC_LN3+d;auto result=BTFP<20>::from_double(x).exp();assert(result.is_positive());
 }
 std::cout<<"PASS numerical C++/SDK consumers, scratch divmod, signed-extreme legacy rounding and sampled CORDIC robustness\n";
}
