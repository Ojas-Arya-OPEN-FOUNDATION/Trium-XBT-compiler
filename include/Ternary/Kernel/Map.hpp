#ifndef TRIUM_KERNEL_MAP_HPP
#define TRIUM_KERNEL_MAP_HPP
#include "../IR/Optimize.hpp"
namespace bt::kernel {
inline size_t expression(const ir::Expr& e) {
    if(!ir::integral(e.type) || e.op==ir::Op::Call || e.op==ir::Op::Print || e.op==ir::Op::Packed)
        throw std::invalid_argument("map-v1 requires pure integral expressions; no calls/IO");
    size_t work=1;for(auto& child:e.operands)work+=expression(*child);return work;
}
inline size_t statements(const std::vector<ir::Statement>& body) {
    size_t work=0;for(auto& s:body) {
        if(s.kind==ir::StatementKind::While)throw std::invalid_argument("map-v1 rejects source loops");
        if(s.value)work+=expression(*s.value);
        ++work;work+=statements(s.body)+statements(s.alternative)+statements(s.negative);
    }return work;
}
inline ir::Function select(const ir::Program& p,const std::string& name,bool optimize=false) {
    ir::verify_program(p);
    if(!p.body.empty())throw std::invalid_argument("map-v1 module cannot contain top-level execution");
    auto f=std::find_if(p.functions.begin(),p.functions.end(),[&](auto& f){return f.name==name;});
    if(f==p.functions.end() || f->parameters.size()!=1 || f->parameters[0].type!=ir::Type::Int39 || f->result!=ir::Type::Int39)
        throw std::invalid_argument("map-v1 signature must be (int39)->int39");
    statements(f->body);
    if(optimize){auto q=ir::Optimizer{}.run(p);return *std::find_if(q.functions.begin(),q.functions.end(),[&](auto& f){return f.name==name;});}
    return *f;
}
} // namespace bt::kernel
#endif
