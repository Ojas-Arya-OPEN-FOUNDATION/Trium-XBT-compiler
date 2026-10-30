#ifndef TRIUM_KERNEL_VULKAN_HPP
#define TRIUM_KERNEL_VULKAN_HPP
#include "Map.hpp"
#include <set>
#include <sstream>
namespace bt::kernel {
// Shader ABI v1: std430 uvec2 int39 elements, binding 0 input, 1 output,
// 2 uint lane status; push constant uint element count. No Int64 capability.
class VulkanEmitter {
    std::ostringstream out;unsigned serial=0;
    std::string tmp(){return "t"+std::to_string(++serial);}
    std::string value(const std::string& code){auto t=tmp();out<<"uvec2 "<<t<<"="<<code<<"; if(failure!=0u)return uvec2(0u);\n";return t;}
    static std::string literal(int64_t n){auto bits=static_cast<uint64_t>(n);return "uvec2("+std::to_string(uint32_t(bits))+"u,"+std::to_string(uint32_t(bits>>32))+"u)";}
    std::string expr(const ir::Expr& e) {
        using ir::Op;
        if(e.op==Op::Integer)return literal(e.integer);
        if(e.op==Op::Parameter)return "v"+std::to_string(e.symbol);
        auto a=expr(*e.operands[0]);
        if(e.op==Op::IntCast)return a;
        if(e.op==Op::Negate)return value("neg64("+a+")");
        if(e.op==Op::Abs)return value("(negative("+a+")?neg64("+a+"):"+a+")");
        if(e.op==Op::SignSelect) {
            auto t=tmp();out<<"uvec2 "<<t<<";if(positive("<<a<<")){\n";auto b=expr(*e.operands[1]);out<<t<<"="<<b<<";}else if(negative("<<a<<")){\n";b=expr(*e.operands[3]);out<<t<<"="<<b<<";}else{\n";b=expr(*e.operands[2]);out<<t<<"="<<b<<";}\n";return t;
        }
        std::vector<std::string> args{a};for(size_t i=1;i<e.operands.size();++i)args.push_back(expr(*e.operands[i]));auto b=args[1];
        switch(e.op) {
        case Op::Add:return value("checked(add64("+a+","+b+"))");
        case Op::Sub:return value("checked(add64("+a+",neg64("+b+")))");
        case Op::Mul:return value("multiply("+a+","+b+")");
        case Op::DigitAnd:case Op::DigitOr:case Op::DigitXor:return value("digits("+a+","+b+","+std::to_string(e.op==Op::DigitAnd?0:e.op==Op::DigitOr?1:2)+")");
        case Op::Intrinsic:{
            if(e.name=="planar_dot64")return value("from32(bitCount(("+a+"&"+args[2]+")|("+b+"&"+args[3]+"))-bitCount(("+a+"&"+args[3]+")|("+b+"&"+args[2]+")))");
            auto c=args[2],d=args[3];std::string cp="(("+a+"&"+c+")|("+b+"&"+d+"))";
            std::string code=e.name=="btha_carry_p"?"("+a+"&"+c+")":e.name=="btha_carry_n"?"("+b+"&"+d+")":e.name=="btha_sum_p"?"(("+a+"^"+c+")&~("+b+"|"+d+")|("+b+"&"+d+"))":"(("+b+"^"+d+")&~("+a+"|"+c+")|("+a+"&"+c+"))";
            return value("checked("+code+")");}
        default:break;
        }
        std::string cond;
        if(e.op==Op::Equal)cond="all(equal("+a+","+b+"))";
        else if(e.op==Op::NotEqual)cond="any(notEqual("+a+","+b+"))";
        else if(e.op==Op::Less)cond="less64("+a+","+b+")";
        else if(e.op==Op::Greater)cond="less64("+b+","+a+")";
        else if(e.op==Op::LessEqual)cond="!less64("+b+","+a+")";
        else if(e.op==Op::GreaterEqual)cond="!less64("+a+","+b+")";
        else if(e.op==Op::Spaceship)return value("(less64("+a+","+b+")?from32(-1):less64("+b+","+a+")?from32(1):from32(0))");
        else throw std::invalid_argument("unsupported Vulkan map opcode");
        return value("("+cond+"?from32(1):from32(-1))");
    }
    void collect(const std::vector<ir::Statement>& ss,std::set<uint32_t>& ids){for(auto& s:ss){if(s.kind==ir::StatementKind::Let)ids.insert(s.symbol);collect(s.body,ids);collect(s.alternative,ids);collect(s.negative,ids);}}
    void body(const std::vector<ir::Statement>& ss) {
        using ir::StatementKind;
        for(auto& s:ss) {
            if(s.kind==StatementKind::Block){body(s.body);continue;}
            auto a=expr(*s.value);
            if(s.kind==StatementKind::Return){out<<"return "<<a<<";\n";continue;}
            if(s.kind==StatementKind::Let || s.kind==StatementKind::Assign){out<<"v"<<s.symbol<<"="<<a<<";\n";continue;}
            if(s.kind==StatementKind::Evaluate)continue;
            out<<"if(positive("<<a<<")){\n";body(s.body);
            if(s.kind==StatementKind::SignSelect){out<<"}else if(negative("<<a<<")){\n";body(s.negative);}
            out<<"}else{\n";body(s.alternative);out<<"}\n";
        }
    }
public:
    std::string generate(const ir::Program& p,const std::string& name,unsigned group=64,bool optimize=false,bool native64=false) {
        if(group!=32 && group!=64 && group!=128)throw std::invalid_argument("workgroup must be 32,64 or128");
        auto f=select(p,name,optimize);out.str("");out.clear();serial=0;
        out<<"#version 450\n";if(native64)out<<"#extension GL_ARB_gpu_shader_int64 : require\n";
        out<<"layout(local_size_x="<<group<<") in;\n"<<R"GLSL(
layout(std430,binding=0) readonly buffer Input {uvec2 input_data[];};
layout(std430,binding=1) writeonly buffer Output {uvec2 output_data[];};
layout(std430,binding=2) buffer Status {uint errors[];};
layout(push_constant) uniform Parameters {uint count;};
uint failure=0u;
bool negative(uvec2 a){return (a.y&0x80000000u)!=0u;}
bool positive(uvec2 a){return !negative(a)&&any(notEqual(a,uvec2(0u)));}
uvec2 from32(int a){return uvec2(uint(a),a<0?0xffffffffu:0u);}
// popcount on uvec2 returns ivec2; scalar contraction is explicit.
uvec2 from32(ivec2 a){return from32(a.x+a.y);}
uvec2 add64(uvec2 a,uvec2 b){uint lo=a.x+b.x;return uvec2(lo,a.y+b.y+uint(lo<a.x));}
uvec2 neg64(uvec2 a){return add64(~a,uvec2(1u,0u));}
bool uless(uvec2 a,uvec2 b){return a.y<b.y||(a.y==b.y&&a.x<b.x);}
bool less64(uvec2 a,uvec2 b){return int(a.y)<int(b.y)||(a.y==b.y&&a.x<b.x);}
uvec2 checked(uvec2 a){
)GLSL";
        out<<"if(less64(a,"<<literal(math::ScalarInteger::minimum)<<")||less64("<<literal(math::ScalarInteger::maximum)<<",a))failure=1u;return a;}\n";
        out<<"const uvec2 magnitude_limit="<<literal(math::ScalarInteger::maximum)<<";\n";
        out<<R"GLSL(
void addword(inout uvec4 p,int k,uint x){for(;k<4;++k){uint old=p[k];p[k]+=x;x=uint(p[k]<old);}}
uvec2 multiply(uvec2 a,uvec2 b){bool minus=negative(a)!=negative(b);if(negative(a))a=neg64(a);if(negative(b))b=neg64(b);
uvec4 p=uvec4(0u);for(int i=0;i<2;++i)for(int j=0;j<2;++j){uint hi,lo;umulExtended(a[i],b[j],hi,lo);addword(p,i+j,lo);addword(p,i+j+1,hi);}
if(p.z!=0u||p.w!=0u){failure=1u;return uvec2(0u);}uvec2 r=p.xy;if(uless(magnitude_limit,r)){failure=1u;return uvec2(0u);}return minus?neg64(r):r;}
uvec2 divthree(uvec2 a,out uint rem){uvec2 q=uvec2(0u);rem=0u;for(int i=63;i>=0;--i){rem=2u*rem+((a[i/32]>>uint(i%32))&1u);if(rem>=3u){rem-=3u;q[i/32]|=1u<<uint(i%32);}}return q;}
int take_digit(inout uvec2 a){bool minus=negative(a);uint rem;uvec2 q=divthree(minus?neg64(a):a,rem);int digit=int(rem);if(digit==2){q=add64(q,uvec2(1u,0u));digit=-1;}a=minus?neg64(q):q;return minus?-digit:digit;}
uvec2 digits(uvec2 a,uvec2 b,int op){uvec2 r=uvec2(0u),place=uvec2(1u,0u);for(int i=0;i<39;++i){int x=take_digit(a),y=take_digit(b);int d=op==0?min(x,y):op==1?max(x,y):x*y;if(d!=0)r=add64(r,d<0?neg64(place):place);if(i<38)place=add64(add64(place,place),place);}return checked(r);}
)GLSL";
        std::set<uint32_t> ids;collect(f.body,ids);out<<"uvec2 transform(uvec2 v"<<f.parameters[0].symbol<<"){\n";
        for(auto id:ids)out<<"uvec2 v"<<id<<"=uvec2(0u);\n";
        body(f.body);out<<"return uvec2(0u);}\nvoid main(){uint lane=gl_GlobalInvocationID.x;if(lane>=count)return;uvec2 x=checked(input_data[lane]);if(failure!=0u){errors[lane]=failure;return;}uvec2 y=transform(x);errors[lane]=failure;if(failure==0u)output_data[lane]=y;}\n";auto source=out.str();
        if(native64){
            auto begin=source.find("uvec2 multiply(");auto end=source.find("uvec2 divthree(",begin);
            source.replace(begin,end-begin,R"GLSL(
uint64_t pack_native(uvec2 a){return uint64_t(a.x)|(uint64_t(a.y)<<32);}
uvec2 multiply(uvec2 a,uvec2 b){bool minus=negative(a)!=negative(b);if(negative(a))a=neg64(a);if(negative(b))b=neg64(b);
uint64_t x=pack_native(a),y=pack_native(b),bound=pack_native(magnitude_limit);
uint64_t divisor=x==uint64_t(0)?uint64_t(1):x;
if(x!=uint64_t(0)&&y>bound/divisor){failure=1u;return uvec2(0u);}
uint64_t product=x*y;uvec2 r=uvec2(uint(product),uint(product>>32));return minus?neg64(r):r;}
)GLSL");
        }
        return source;
    }
};
} // namespace bt::kernel
#endif
