#include "ternary_fp.hpp"
#include "include/Ternary/Math/Rational.hpp"
#include <iostream>
#include <string>
template<int W> void fp(const std::string& op, int64_t m, int e, int64_t n, int f, int64_t c, int g) {
    using F=bt::BTFP<W>;
    F a(m,e), b(n,f), z(c,g), r;
    if(op=="ratio") r=F::from_ratio(m,n,e);
    else if(op=="norm") r=a;
    else if(op=="add") r=a+b;
    else if(op=="mul") r=a*b;
    else if(op=="div") r=a/b;
    else if(op=="fma") r=F::fma(a,b,z);
    else if(op=="sqrt") r=a.sqrt();
    else throw std::invalid_argument("unknown operation");
    std::cout<<r.mantissa()<<' '<<r.exponent()<<'\n';
}
int main() {
    std::string family,op;
    while(std::cin>>family>>op) {
        try {
            if(family=="rational") {
                int64_t a,b,c,d; std::cin>>a>>b>>c>>d;
                bt::math::Rational<> x(a,b), y(c,d), r;
                if(op=="add") r=x+y; else if(op=="sub") r=x-y;
                else if(op=="mul") r=x*y; else if(op=="div") r=x/y;
                else throw std::invalid_argument("unknown rational operation");
                std::cout<<r.numerator()<<' '<<r.denominator()<<'\n';
            } else {
                int w,e,f,g; int64_t m,n,c; std::cin>>w>>m>>e>>n>>f>>c>>g;
                switch(w) {
                case 1: fp<1>(op,m,e,n,f,c,g); break;
                case 2: fp<2>(op,m,e,n,f,c,g); break;
                case 3: fp<3>(op,m,e,n,f,c,g); break;
                case 10: fp<10>(op,m,e,n,f,c,g); break;
                case 20: fp<20>(op,m,e,n,f,c,g); break;
                case 40: fp<40>(op,m,e,n,f,c,g); break;
                default: throw std::invalid_argument("unknown width");
                }
            }
        } catch(const std::overflow_error&) { std::cout<<"overflow\n"; }
          catch(const std::domain_error&) { std::cout<<"domain\n"; }
          catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
        if(!std::cin) return 2;
    }
}
