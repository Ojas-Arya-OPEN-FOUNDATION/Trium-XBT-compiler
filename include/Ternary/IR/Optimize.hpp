#ifndef TERNARY_IR_OPTIMIZE_HPP
#define TERNARY_IR_OPTIMIZE_HPP
#include "Program.hpp"
#include <algorithm>
#include <optional>
namespace bt::ir {
struct IntegerRange { __int128 low, high; };
// Ranges of successful operand evaluations are always bounded by their type.
// Keep an operation's RAW range for its own overflow-check decision. Clamping
// that raw range before deciding whether to remove a check would be unsound.
inline IntegerRange type_range(Type t) {
    return t==Type::Trit?IntegerRange{-1,1}:IntegerRange{math::ScalarInteger::minimum,math::ScalarInteger::maximum};
}
inline bool fits(IntegerRange r) {
    return r.low>=math::ScalarInteger::minimum && r.high<=math::ScalarInteger::maximum;
}
inline IntegerRange integer_range(const Expr& e);
inline IntegerRange accepted_range(const Expr& e) {
    auto r=integer_range(e); return fits(r)?r:type_range(e.type);
}
inline IntegerRange integer_range(const Expr& e) {
    if(e.op==Op::Integer) return {e.integer,e.integer};
    if(e.op==Op::IntCast) return e.type==Type::Trit?IntegerRange{-1,1}:accepted_range(*e.operands[0]);
    if(e.op==Op::Negate) { auto a=accepted_range(*e.operands[0]);return {-a.high,-a.low}; }
    if(e.op==Op::Add || e.op==Op::Sub || e.op==Op::Mul) {
        auto a=accepted_range(*e.operands[0]),b=accepted_range(*e.operands[1]);
        if(e.op==Op::Add) return {a.low+b.low,a.high+b.high};
        if(e.op==Op::Sub) return {a.low-b.high,a.high-b.low};
        __int128 p[]={a.low*b.low,a.low*b.high,a.high*b.low,a.high*b.high};
        return {*std::min_element(p,p+4),*std::max_element(p,p+4)};
    }
    return type_range(e.type);
}
struct OptimizationStats { size_t folded=0, removed_statements=0; };
class Optimizer {
    OptimizationStats stats_;
    static std::optional<bool> positive(Expression x) {
        if(x->op==Op::Integer) return x->integer>0;
        if(x->op==Op::Floating) return x->floating>0;
        return {};
    }
    static Expression widen(Expression x,Type type) {
        if(x->type==type) return x;
        return std::make_shared<Expr>(Expr{Op::IntCast,type,{x}});
    }
    Expression expression(Expression x) {
        auto e=std::make_shared<Expr>(*x);
        for(auto& a:e->operands) a=expression(a);
        if(e->op==Op::SignSelect && (e->operands[0]->op==Op::Integer || e->operands[0]->op==Op::Floating)) {
            auto c=e->operands[0]; double n=c->op==Op::Integer?static_cast<double>(c->integer):c->floating;
            ++stats_.folded;return e->operands[n>0?1:n<0?3:2];
        }
        bool constants=integral(e->type) && !e->operands.empty();
        for(auto& a:e->operands) constants=constants && a->op==Op::Integer;
        if(constants) {
            __int128 a=e->operands[0]->integer,b=e->operands.size()>1?e->operands[1]->integer:0,n=0;
            bool fold=true;
            switch(e->op) {
            case Op::Add:n=a+b;break;case Op::Sub:n=a-b;break;case Op::Mul:n=a*b;break;
            case Op::Negate:n=-a;break;case Op::Abs:n=a<0?-a:a;break;case Op::IntCast:n=a;break;
            case Op::Equal:n=a==b?1:-1;break;case Op::NotEqual:n=a!=b?1:-1;break;
            case Op::Less:n=a<b?1:-1;break;case Op::Greater:n=a>b?1:-1;break;
            case Op::LessEqual:n=a<=b?1:-1;break;case Op::GreaterEqual:n=a>=b?1:-1;break;
            case Op::Spaceship:n=(a>b)-(a<b);break;
            default:fold=false;
            }
            if(fold && fits({n,n}) && (e->type!=Type::Trit || (n>=-1 && n<=1))) {
                e->op=Op::Integer;e->integer=static_cast<int64_t>(n);e->operands.clear();++stats_.folded;return e;
            }
        }
        // These identities preserve evaluation of the nonconstant operand. In
        // particular x*0 is NOT used to erase a failing/observable operand.
        if(e->type==Type::Int39 && (e->op==Op::Add || e->op==Op::Sub || e->op==Op::Mul)) {
            auto a=e->operands[0],b=e->operands[1];
            int64_t identity=e->op==Op::Mul?1:0;
            if(b->op==Op::Integer && b->integer==identity) {++stats_.folded;return widen(a,e->type);}
            if(e->op!=Op::Sub && a->op==Op::Integer && a->integer==identity) {++stats_.folded;return widen(b,e->type);}
        }
        return e;
    }
    static bool returns(const Statement& s) {
        if(s.kind==StatementKind::Return) return true;
        if(s.kind==StatementKind::Block) return returns(s.body);
        if(s.kind==StatementKind::If) return returns(s.body) && returns(s.alternative);
        if(s.kind==StatementKind::SignSelect) return returns(s.body) && returns(s.alternative) && returns(s.negative);
        return false;
    }
    static bool returns(const std::vector<Statement>& body) {
        for(auto& s:body) if(returns(s)) return true;
        return false;
    }
    std::vector<Statement> statements(const std::vector<Statement>& input) {
        std::vector<Statement> out;
        size_t visited=0;
        for(auto s:input) {
            ++visited;
            if(s.value) s.value=expression(s.value);
            s.body=statements(s.body);s.alternative=statements(s.alternative);s.negative=statements(s.negative);
            auto condition=s.value?positive(s.value):std::optional<bool>{};
            if(s.kind==StatementKind::While && condition && !*condition) {++stats_.removed_statements;continue;}
            if(s.kind==StatementKind::If && condition) {
                auto chosen=*condition?std::move(s.body):std::move(s.alternative);
                s.kind=StatementKind::Block;s.value={};s.body=std::move(chosen);s.alternative.clear();++stats_.folded;
            }
            if(s.kind==StatementKind::SignSelect && condition) {
                double n=s.value->op==Op::Integer?static_cast<double>(s.value->integer):s.value->floating;
                auto chosen=n>0?std::move(s.body):n<0?std::move(s.negative):std::move(s.alternative);
                s.kind=StatementKind::Block;s.value={};s.body=std::move(chosen);s.alternative.clear();s.negative.clear();++stats_.folded;
            }
            if(s.kind==StatementKind::Evaluate && (s.value->op==Op::Integer || s.value->op==Op::Floating)) {++stats_.removed_statements;continue;}
            out.push_back(std::move(s));
            if(returns(out.back())) {stats_.removed_statements+=input.size()-visited;break;}
        }
        return out;
    }
public:
    Program run(const Program& p) {
        verify_program(p);stats_={};Program result=p;
        for(auto& f:result.functions) f.body=statements(f.body);
        result.body=statements(result.body);verify_program(result);return result;
    }
    OptimizationStats stats() const { return stats_; }
};
} // namespace bt::ir
#endif
