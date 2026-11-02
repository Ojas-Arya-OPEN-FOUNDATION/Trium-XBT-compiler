// Deterministic sampled error survey, not a global transcendental error proof.
#include "ternary_fp.hpp"
#include <cstdio>
int main(){
 double circular=0,where=0,s,c,sinh_error=0,cosh_error=0,log_error=0,atan_error=0;
 for(int i=-1000;i<=1000;++i){double x=i/10.0;bt::fp_detail::cordic_circ_sincos(x,s,c);
  auto error=std::max(std::abs(s-std::sin(x)),std::abs(c-std::cos(x)));
  if(error>circular){circular=error;where=x;}
 }
 for(int i=0;i<=1000;++i){double x=(i-500)*bt::fp_detail::CORDIC_LN3/1000;
  bt::fp_detail::cordic_hyp_sincos(x,s,c);
  sinh_error=std::max(sinh_error,std::abs(s-std::sinh(x)));
  cosh_error=std::max(cosh_error,std::abs(c-std::cosh(x)));
  x=0.5+i/1000.0;log_error=std::max(log_error,std::abs(bt::fp_detail::cordic_hyp_ln(x)-std::log(x)));
 }
 for(int i=0;i<1000;++i){double x=(i+1)/31.0,y=(i%17+1)/9.0;
  atan_error=std::max(atan_error,std::abs(bt::fp_detail::cordic_circ_atan2(y,x)-std::atan2(y,x)));
 }
 std::printf("{\"circular_samples\":2001,\"circular_max_abs_error\":%.17g,\"circular_worst_input\":%.17g,\"sinh_max_abs_error\":%.17g,\"cosh_max_abs_error\":%.17g,\"log_max_abs_error\":%.17g,\"atan_max_abs_error\":%.17g}\n",circular,where,sinh_error,cosh_error,log_error,atan_error);
}
