#ifndef TERNARY_IR_EXECUTE_HPP
#define TERNARY_IR_EXECUTE_HPP
#include "Program.hpp"
#include <variant>
#include <ostream>
#include <iomanip>
#include <bit>
#include <array>
#include <algorithm>
namespace bt::ir {
using Value=std::variant<std::monostate,int64_t,double>;
class Executor {
    const Program program;
    std::ostream& output;
    uint64_t steps=0, limit;
    unsigned calls=0;
    std::unordered_map<std::string,size_t> function_indices;
    using Frame=std::unordered_map<uint32_t,Value>;
    struct Flow { bool returned=false; Value value; };
    void tick() { if(limit && ++steps>limit) throw std::runtime_error("typed core execution budget exceeded"); }
    static double real(const Value& v) {
        return std::holds_alternative<double>(v)?std::get<double>(v):static_cast<double>(std::get<int64_t>(v));
    }
    static int sign(const Value& v) { double x=real(v); return (x>0)-(x<0); }
    static int64_t logic(int64_t a,int64_t b,Op op) {
        __int128 value=0,place=1;
        while(a || b) {
            int64_t qa=a/3,ra=a%3,qb=b/3,rb=b%3;
            if(ra==2){++qa;ra=-1;} else if(ra==-2){--qa;ra=1;}
            if(rb==2){++qb;rb=-1;} else if(rb==-2){--qb;rb=1;}
            value+=(op==Op::DigitAnd?std::min(ra,rb):op==Op::DigitOr?std::max(ra,rb):ra*rb)*place;
            a=qa;b=qb;place*=3;
        }
        return math::ScalarInteger::check(value);
    }
    Value evaluate(const Expr& e,Frame& frame) {
        tick();
        switch(e.op) {
        case Op::Integer:return e.integer;
        case Op::Floating:return e.floating;
        case Op::Packed:throw std::invalid_argument("packed64 is outside the scalar execution profile");
        case Op::Parameter:return frame.at(e.symbol);
        case Op::SignSelect: {
            int s=sign(evaluate(*e.operands[0],frame));
            return evaluate(*e.operands[s>0?1:s<0?3:2],frame);
        }
        case Op::Call: {
            std::vector<Value> arguments;
            for(auto& x:e.operands) arguments.push_back(evaluate(*x,frame));
            if(calls>=512) throw std::runtime_error("typed core call depth exceeded");
            const auto& f=program.functions.at(function_indices.at(e.name));
            Frame local;
            for(size_t i=0;i<arguments.size();++i) local.emplace(f.parameters[i].symbol,arguments[i]);
            ++calls;
            try { auto flow=execute(f.body,local);--calls;return flow.value; }
            catch(...) { --calls; throw; }
        }
        case Op::Intrinsic: {
            std::array<uint64_t,4> rails;
            for(size_t i=0;i<4;++i) rails[i]=static_cast<uint64_t>(std::get<int64_t>(evaluate(*e.operands[i],frame)));
            auto [ap,an,bp,bn]=rails;
            if(e.name=="planar_dot64") return static_cast<int64_t>(std::popcount((ap&bp)|(an&bn))-std::popcount((ap&bn)|(an&bp)));
            uint64_t bits=e.name=="btha_sum_p"?(((ap^bp)&~(an|bn))|(an&bn)):
                          e.name=="btha_sum_n"?(((an^bn)&~(ap|bp))|(ap&bp)):
                          e.name=="btha_carry_p"?(ap&bp):(an&bn);
            __int128 n=bits; if(bits>>63) n-=(static_cast<__int128>(1)<<64);
            return math::ScalarInteger::check(n);
        }
        default: break;
        }
        Value a=evaluate(*e.operands[0],frame);
        if(e.op==Op::Print) {
            if(std::holds_alternative<double>(a)) output<<std::setprecision(17)<<std::get<double>(a)<<" [core float64]\n";
            else output<<std::get<int64_t>(a)<<" [core int39]\n";
            return std::monostate{};
        }
        if(e.op==Op::IntToFloat) return real(a);
        if(e.op==Op::IntCast) {
            auto n=std::get<int64_t>(a);
            if(e.type==Type::Trit && (n < -1 || n > 1)) throw std::overflow_error("core trit conversion overflow");
            return n;
        }
        if(e.op==Op::Negate) return e.type==Type::Float64?Value(-real(a)):Value(-std::get<int64_t>(a));
        if(e.op==Op::Abs) return e.type==Type::Float64?Value(std::abs(real(a))):Value(std::get<int64_t>(a)<0?-std::get<int64_t>(a):std::get<int64_t>(a));
        if(e.op==Op::Sqrt) return std::sqrt(real(a));
        Value b=evaluate(*e.operands[1],frame);
        if(e.op==Op::Fma) { Value c=evaluate(*e.operands[2],frame); return std::fma(real(a),real(b),real(c)); }
        if(e.op==Op::DigitAnd || e.op==Op::DigitOr || e.op==Op::DigitXor)
            return logic(std::get<int64_t>(a),std::get<int64_t>(b),e.op);
        bool floating=std::holds_alternative<double>(a);
        if(e.op==Op::Add || e.op==Op::Sub || e.op==Op::Mul) {
            if(floating) return e.op==Op::Add?real(a)+real(b):e.op==Op::Sub?real(a)-real(b):real(a)*real(b);
            __int128 x=std::get<int64_t>(a),y=std::get<int64_t>(b);
            return math::ScalarInteger::check(e.op==Op::Add?x+y:e.op==Op::Sub?x-y:x*y);
        }
        // Do not round wide integer comparisons through binary64.
        auto compare=[&](auto x,auto y)->int64_t {
            switch(e.op) {
            case Op::Equal:return x==y?1:-1; case Op::NotEqual:return x!=y?1:-1;
            case Op::Less:return x<y?1:-1; case Op::Greater:return x>y?1:-1;
            case Op::LessEqual:return x<=y?1:-1; case Op::GreaterEqual:return x>=y?1:-1;
            case Op::Spaceship:return (x>y)-(x<y);
            default:throw std::invalid_argument("unsupported core expression at execution");
            }
        };
        return floating?compare(real(a),real(b)):compare(std::get<int64_t>(a),std::get<int64_t>(b));
    }
    // Return is ordinary control flow, not a C++ exception on every call.
    // Actual failures still unwind through exceptions and preserve call depth.
    Flow execute(const std::vector<Statement>& body,Frame& frame) {
        for(auto& s:body) {
            tick();Flow flow;
            switch(s.kind) {
            case StatementKind::Let:case StatementKind::Assign:frame[s.symbol]=evaluate(*s.value,frame);break;
            case StatementKind::Evaluate:evaluate(*s.value,frame);break;
            case StatementKind::Block:flow=execute(s.body,frame);break;
            case StatementKind::Return:return {true,s.value?evaluate(*s.value,frame):Value(std::monostate{})};
            case StatementKind::If:flow=execute(sign(evaluate(*s.value,frame))>0?s.body:s.alternative,frame);break;
            case StatementKind::While:
                while(sign(evaluate(*s.value,frame))>0) {tick();flow=execute(s.body,frame);if(flow.returned) break;}
                break;
            case StatementKind::SignSelect: {
                int signum=sign(evaluate(*s.value,frame));
                flow=execute(signum>0?s.body:signum<0?s.negative:s.alternative,frame);break;
            }
            }
            if(flow.returned) return flow;
        }
        return {};
    }
public:
    explicit Executor(Program p,std::ostream& output,uint64_t budget=10000000)
        :program(std::move(p)),output(output),limit(budget) {
        verify_program(program);
        for(size_t i=0;i<program.functions.size();++i) function_indices.emplace(program.functions[i].name,i);
    }
    int64_t run() {
        steps=0;calls=0;Frame frame;
        auto flow=execute(program.body,frame);
        return flow.returned?std::get<int64_t>(flow.value):0;
    }
};
} // namespace bt::ir
#endif
