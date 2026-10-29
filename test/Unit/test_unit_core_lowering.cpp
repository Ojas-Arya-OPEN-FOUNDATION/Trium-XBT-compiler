#include "Ternary/IR/LLVM.hpp"
#include "Ternary/IR/Execute.hpp"
#include <iostream>
#include <random>
using namespace bt::ir;
static void check(bool ok){if(!ok) throw std::runtime_error("core lowering test failed");}
template<class F> void rejects(F fn){bool threw=false;try{fn();}catch(const std::exception&){threw=true;}check(threw);}
Expression number(int64_t n){auto e=std::make_shared<Expr>(Expr{Op::Integer,Type::Int39});e->integer=n;return e;}
Expression operation(Op op,Expression a,Expression b){return std::make_shared<Expr>(Expr{op,Type::Int39,{a,b}});}
Statement ret(Expression x){Statement s{StatementKind::Return};s.value=x;return s;}
int main(){
    auto h=bt::math::ScalarInteger::maximum;
    auto fail=operation(Op::Add,number(h),number(1));
    Program p{{},{ret(operation(Op::Mul,number(0),fail))}};
    std::ostringstream output;
    auto optimized=Optimizer{}.run(p);
    rejects([&]{Executor(p,output).run();});rejects([&]{Executor(optimized,output).run();});
    p.body={ret(operation(Op::Sub,fail,number(1)))};
    rejects([&]{Executor(Optimizer{}.run(p),output).run();});
    // Never clamp an unsafe raw interval into the type before deciding safety.
    check(!fits(integer_range(*fail)));
    auto parent=operation(Op::Mul,fail,number(0));check(fits(integer_range(*parent)));
    p.body={ret(parent)};rejects([&]{Executor(Optimizer{}.run(p),output).run();});
    auto selected=std::make_shared<Expr>(Expr{Op::SignSelect,Type::Int39,{number(0),fail,number(7),fail}});
    p.body={ret(selected)};auto q=Optimizer{}.run(p);check(Executor(q,output).run()==7);
    check(q.body[0].value->op==Op::Integer);
    std::mt19937_64 rng(0x434f5245);
    for(int i=0;i<10000;++i){
        auto a=number(int64_t(rng()%200000)-100000),b=number(int64_t(rng()%200000)-100000);
        auto x=operation(i%3==0?Op::Add:i%3==1?Op::Sub:Op::Mul,a,b);
        auto range=integer_range(*x);p.body={ret(x)};
        auto value=Executor(p,output).run();check(range.low==value && range.high==value);
        check(Executor(Optimizer{}.run(p),output).run()==value);
    }
    auto unknown=std::make_shared<Expr>(Expr{Op::Parameter,Type::Int39});unknown->symbol=1;
    auto plus=operation(Op::Add,unknown,number(1));check(!fits(integer_range(*plus)));
    auto safe=operation(Op::Mul,unknown,number(1));check(fits(integer_range(*safe)));
    // Shared DAGs must not turn verifier work into an exponential denial of service.
    Expression dag=number(0);for(int i=0;i<20;++i)dag=operation(Op::Add,dag,dag);
    rejects([&]{verify(*dag);});
    // Source names never become LLVM identifiers: manually constructed IR may
    // contain names that could otherwise inject text into an LLVM module.
    auto call=std::make_shared<Expr>(Expr{Op::Call,Type::Int39});call->name="unsafe\"\nname";
    p.functions={{call->name,Type::Int39,{}, {ret(number(3))}}};p.body={ret(call)};
    auto llvm=LLVMEmitter{}.generate(p,"x86_64-unknown-linux-gnu","");
    check(llvm.find("@bt.core.fn.0")!=std::string::npos);
    check(llvm.find(call->name)==std::string::npos);
    std::cout<<"PASS: 10000 optimizer/range cases, effect/error regressions, verifier budget, safe symbols\n";
}
