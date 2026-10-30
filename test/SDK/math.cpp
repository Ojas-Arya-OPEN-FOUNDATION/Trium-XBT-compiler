#include "Ternary/Math/Interchange.hpp"
#include "Ternary/Math/Float.hpp"
#include <cassert>
#include <random>
#include <iostream>
using namespace bt::math;
template<unsigned W> void width_check(){
 auto high=BalancedInteger<W>::maximum;assert(decode_integer(encode_integer(high,W),W)==high);
 auto value=Rational<W>(high,1);assert(deserialize_rational<W>(serialize(value))==value);
 assert(parse_decimal<W>("0")==Rational<W>());
 assert(float_from_decimal<W>("0.5")==bt::BTFP<W>::from_ratio(1,2));
}
template<size_t... I> void widths(std::index_sequence<I...>){(width_check<I+1>(),...);}
int main(){
 widths(std::make_index_sequence<40>{});
 for(int64_t i=-100000;i<=100000;++i){auto w=encode_integer(i);assert(decode_integer(w)==i);auto bytes=serialize(w,64);assert(deserialize(bytes).first==w);}
 for(auto n:{INT64_MIN,INT64_MIN+1,INT64_MAX,int64_t(0)})assert(decode_integer(encode_integer(n))==n);
 assert(!attempt([]{return decode_integer({1,1});}));assert(!attempt([]{return encode_integer(2,1);}));
 assert(!attempt([]{return decode_integer({uint64_t(1)<<63,0});}));
 for(int w=1;w<=8;++w){int bound=1;for(int i=0;i<w;++i)bound*=3;bound=(bound-1)/2;for(int n=-bound;n<=bound;++n){auto rail=encode_integer(n,w);assert(decode_integer(rail,w)==n);assert(parse_trits(format_trits(rail,w))==rail);}}
 assert(parse_decimal<39>("-12.500")==Rational<39>(-25,2));assert(parse_decimal<39>("0.000")==Rational<39>());
 for(auto s:{"", ".5", "1.","1.2.3","01","1e3","--1","1.1234567890123456789"})assert(!attempt([&]{return parse_decimal<39>(s);}));
 assert(parse_rational<39>("2/4")==Rational<39>(1,2));
 auto q=Rational<40>(-17,9);auto wire=serialize(q);assert(deserialize_rational<40>(wire)==q);wire[6]=1;assert(!attempt([&]{return deserialize_rational<40>(wire);}));
 assert(float_from_decimal<20>("0.5")==bt::BTFP<20>::from_ratio(1,2));
 std::vector<int64_t> a{1,2,3,4},b{5,6,7,8},out(4);matmul(a,b,out,2,2,2);assert((out==std::vector<int64_t>{19,22,43,50}));
 std::vector<int64_t> empty;matmul(empty,empty,empty,SIZE_MAX,0,0);
 auto H=ScalarInteger::maximum;std::vector<int64_t> bad{H,H},v{1,1},dest{123};assert(!attempt([&]{matmul(bad,v,dest,1,2,1);return 0;}));assert(dest[0]==123);
 std::mt19937 rng(12345);for(unsigned c=0;c<1000;++c){std::vector<int64_t>x(12),y(20),z(15);for(auto& n:x)n=int(rng()%201)-100;for(auto& n:y)n=int(rng()%201)-100;matmul(x,y,z,3,4,5);for(int i=0;i<3;++i)for(int j=0;j<5;++j){int64_t ref=0;for(int k=0;k<4;++k)ref+=x[i*4+k]*y[k*5+j];assert(z[i*5+j]==ref);}}
 std::cout<<"PASS dual-rail full-int64, canonical wires, exact decimal/BTFP conversion, checked matmul\n";
}
