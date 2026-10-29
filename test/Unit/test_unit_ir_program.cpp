#include "Ternary/IR/Execute.hpp"
#include "Ternary/Math/Rational.hpp"
#include <sstream>
#include <iostream>
using namespace bt::ir;
void require(bool b) { if(!b) throw std::runtime_error("IR program test failed"); }
template<class F> void rejects(F f) { bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed); }
Expression integer(int64_t n) { auto e=std::make_shared<Expr>(Expr{Op::Integer,Type::Int39});e->integer=n;return e; }
Expression param(uint32_t id) { auto e=std::make_shared<Expr>(Expr{Op::Parameter,Type::Int39});e->symbol=id;return e; }
Expression op(Op kind,Type type,std::vector<Expression> args) { return std::make_shared<Expr>(Expr{kind,type,std::move(args)}); }
Statement ret(Expression value) { Statement s{StatementKind::Return};s.value=value;return s; }
Statement eval(Expression value) { Statement s{StatementKind::Evaluate};s.value=value;return s; }
int main() {
    std::ostringstream output;
    auto call=std::make_shared<Expr>(Expr{Op::Call,Type::Int39,{integer(7)}});call->name="identity";
    Program p{{Function{"identity",Type::Int39,{{9,Type::Int39}},{ret(param(9))}}},{ret(call)}};
    Executor executor(p,output); require(executor.run()==7 && executor.run()==7);
    auto bad=p;bad.functions[0].parameters.push_back({9,Type::Int39});rejects([&]{verify_program(bad);});
    bad=p;bad.functions[0].body.clear();rejects([&]{verify_program(bad);});
    bad=p;bad.functions.push_back(bad.functions[0]);rejects([&]{verify_program(bad);});
    bad=p;bad.body={ret(param(999))};rejects([&]{verify_program(bad);});
    bad=p;bad.body={ret(op(Op::Floating,Type::Float64,{}))};rejects([&]{verify_program(bad);});
    Statement local{StatementKind::Let};local.symbol=1;local.type=Type::Int39;local.value=integer(2);
    Statement block{StatementKind::Block};block.body={local};
    bad=p;bad.body={block,ret(param(1))};rejects([&]{verify_program(bad);});
    bad=p;bad.body={local,local};rejects([&]{verify_program(bad);});
    auto packed=op(Op::Packed,Type::Packed64,{});
    bad=p;bad.body={eval(packed)};rejects([&]{verify_program(bad);});
    auto print=op(Op::Print,Type::Void,{integer(42)});
    require(verify(*print)==Effect::Observable);
    auto printcall=std::make_shared<Expr>(Expr{Op::Call,Type::Int39});printcall->name="effect";
    auto overflow=op(Op::Add,Type::Int39,{integer(bt::math::ScalarInteger::maximum),integer(1)});
    Function effect{"effect",Type::Int39,{}, {eval(print),ret(integer(5))}};
    auto select=op(Op::SignSelect,Type::Int39,{integer(0),printcall,integer(12),overflow});
    bad.functions={effect};bad.body={ret(select)};
    require(Executor(bad,output).run()==12 && output.str().empty());
    bad.body={ret(op(Op::SignSelect,Type::Int39,{integer(1),printcall,integer(12),overflow}))};
    require(Executor(bad,output).run()==5 && output.str()=="42 [core int39]\n");
    bad.body={ret(overflow)};rejects([&]{Executor(bad,output).run();});
    Statement loop{StatementKind::While};loop.value=integer(1);
    bad.body={loop};rejects([&]{Executor(bad,output,100).run();});
    auto recursive=std::make_shared<Expr>(Expr{Op::Call,Type::Int39});recursive->name="rec";
    bad.functions={Function{"rec",Type::Int39,{}, {ret(recursive)}}};bad.body={ret(recursive)};
    rejects([&]{Executor(bad,output).run();});
    auto h=bt::math::ScalarInteger::maximum;
    bad.functions.clear();bad.body={ret(op(Op::Greater,Type::Int39,{integer(h),integer(h-1)}))};
    require(Executor(bad,output).run()==1);
    using R=bt::math::Rational<>;
    require(R(2,-4)==R(-1,2));require(R(0,INT64_MIN)==R());require(R(INT64_MIN,INT64_MIN)==R(1));
    require(R(h,h)==R(1));require(R(h,2)+R(h,2)==R(h));
    require((R(2,3)/R(4,5))==R(5,6));require(R(-1,2).round()==-1);
    rejects([]{R(1,0);});rejects([]{R(1)/R();});rejects([&]{auto r=R(h)*R(h);(void)r;});
    // Width-40 cross-products fit signed i128 even at both positive extremes.
    using W=bt::math::Rational<40>;auto k=bt::math::BalancedInteger<40>::maximum;
    require(W(k,k-1)-W(k,k-1)==W());
    std::cout<<"PASS: whole-program IR, observable/lazy effects, budgets, exact rational edges\n";
}
