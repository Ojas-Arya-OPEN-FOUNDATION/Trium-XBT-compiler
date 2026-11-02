#ifndef TRIUM_MATH_NUMERICAL_HPP
#define TRIUM_MATH_NUMERICAL_HPP
#include "Float.hpp"
#include <span>
#include <vector>
#include <algorithm>

namespace bt::fp_detail {
// Exact represented-input reductions. Scratch is fixed-capacity, NOT a public
// arbitrary-precision type. Limits are part of numerical-v1, not approximations.
template<int W,int E> struct Numerical {
    using F=bt::BTFP<W,E>;using Big=Wide<32>;
    struct Term {Big magnitude;int64_t exponent;bool negative;};
    static constexpr size_t maximum_terms=4096;
    static constexpr int64_t maximum_window=256;
    static constexpr size_t maximum_coefficients=17;
    static uint64_t magnitude(F a){return a.mantissa()<0?uint64_t(-a.mantissa()):uint64_t(a.mantissa());}
    static void count(size_t n){if(n>maximum_terms)throw NumericalLimit("numerical-v1 term limit: 4096");}
    static F finish(std::vector<Term> terms,bool root=false){
        // Strip exact factors of 3 first: powers of the represented value 1
        // must not consume a large artificial exponent-alignment window.
        for(auto& t:terms)if(!t.magnitude.zero())for(;;){auto [q,r]=t.magnitude.divmod(3);if(r)break;t.magnitude=q;++t.exponent;}
        int64_t low=INT64_MAX,high=INT64_MIN;
        for(auto& t:terms)if(!t.magnitude.zero()){low=std::min(low,t.exponent);high=std::max(high,t.exponent);}
        if(low==INT64_MAX)return F();
        if(high-low>maximum_window)throw NumericalLimit("numerical-v1 active exponent window: 256 trits");
        Big acc;bool negative=false;
        for(auto& t:terms){
            if(t.magnitude.zero())continue;
            Big value=t.magnitude.times_pow3(t.exponent-low);
            if(acc.zero()){acc=value;negative=t.negative;}
            else if(negative==t.negative)acc=acc.plus(value);
            else if(acc.compare(value)>=0)acc=acc.minus(value);
            else{acc=value.minus(acc);negative=t.negative;}
        }
        if(root && negative && !acc.zero())throw std::domain_error("negative numerical radicand");
        if(root)return F::template round_root<2>(acc,low);
        return F::round_ratio(acc,Big(1),low,negative);
    }
    static F sum(std::span<const F> values){
        count(values.size());std::vector<Term> terms;terms.reserve(values.size());
        for(auto a:values)terms.push_back({Big(magnitude(a)),a.exponent(),a.is_negative()});
        return finish(std::move(terms));
    }
    static F dot(std::span<const F> a,std::span<const F> b,bool root=false){
        count(a.size());if(a.size()!=b.size())throw std::invalid_argument("numerical dot shape mismatch");
        std::vector<Term> terms;terms.reserve(a.size());
        for(size_t i=0;i<a.size();++i)terms.push_back({Big(magnitude(a[i])).times(magnitude(b[i])),int64_t(a[i].exponent())+b[i].exponent(),a[i].is_negative()!=b[i].is_negative()});
        return finish(std::move(terms),root);
    }
    static F polynomial(F x,std::span<const F> coefficients){
        if(coefficients.size()>maximum_coefficients)throw NumericalLimit("numerical-v1 polynomial degree limit: 16");
        std::vector<Term> terms;terms.reserve(coefficients.size());Big power(1);
        for(size_t i=0;i<coefficients.size();++i){auto c=coefficients[i];
            terms.push_back({power.times(magnitude(c)),int64_t(c.exponent())+int64_t(i)*x.exponent(),c.is_negative()!=(x.is_negative() && i%2)});
            if(i+1<coefficients.size())power=power.times(magnitude(x));
        }
        return finish(std::move(terms));
    }
};
}
namespace bt::math {
template<int W,int E=0> bt::BTFP<W,E> sum_exact(std::span<const bt::BTFP<W,E>> a){return fp_detail::Numerical<W,E>::sum(a);}
template<int W,int E=0> bt::BTFP<W,E> dot_exact(std::span<const bt::BTFP<W,E>> a,std::span<const bt::BTFP<W,E>> b){return fp_detail::Numerical<W,E>::dot(a,b);}
template<int W,int E=0> bt::BTFP<W,E> norm2(std::span<const bt::BTFP<W,E>> a){return fp_detail::Numerical<W,E>::dot(a,a,true);}
// Ascending coefficients, one final rounding of the ENTIRE exact polynomial.
template<int W,int E=0> bt::BTFP<W,E> polynomial_exact(bt::BTFP<W,E> x,std::span<const bt::BTFP<W,E>> c){return fp_detail::Numerical<W,E>::polynomial(x,c);}
}
#endif
