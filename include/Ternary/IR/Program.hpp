#ifndef TERNARY_IR_PROGRAM_HPP
#define TERNARY_IR_PROGRAM_HPP
#include "Core.hpp"
#include <unordered_set>

namespace bt::ir {
using Expression=std::shared_ptr<const Expr>;
enum class StatementKind { Let, Assign, Evaluate, Block, If, While, Return, SignSelect };
struct Statement {
    StatementKind kind;
    uint32_t symbol=0;
    Type type=Type::Void;
    Expression value;
    std::vector<Statement> body, alternative, negative;
};
struct Parameter { uint32_t symbol; Type type; };
struct Function { std::string name; Type result; std::vector<Parameter> parameters; std::vector<Statement> body; };
struct Program { std::vector<Function> functions; std::vector<Statement> body; };
inline bool verify_statements(const std::vector<Statement>& body, Symbols symbols,
                              std::unordered_set<uint32_t>& allocated, const Signatures& functions,
                              Type result, unsigned depth=0, size_t* budget=nullptr) {
    size_t local_budget=100000;
    if(!budget) budget=&local_budget;
    if(depth>512) throw std::invalid_argument("core statement nesting limit exceeded");
    bool returned=false;
    for(const auto& s:body) {
        if(!*budget) throw std::invalid_argument("core program work budget exceeded");
        --*budget;
        auto require=[](bool ok){ if(!ok) throw std::invalid_argument("invalid core statement"); };
        if(s.kind!=StatementKind::Block && s.kind!=StatementKind::If &&
           s.kind!=StatementKind::While && s.kind!=StatementKind::SignSelect)
            require(s.body.empty() && s.alternative.empty() && s.negative.empty());
        if(s.kind==StatementKind::Block || s.kind==StatementKind::While)
            require(s.alternative.empty() && s.negative.empty());
        if(s.kind==StatementKind::If) require(s.negative.empty());
        if(s.value) {
            require(numeric(s.value->type) || s.value->type==Type::Void);
            verify(*s.value,symbols,0,&functions,budget);
        }
        bool terminates=false;
        switch(s.kind) {
        case StatementKind::Let:
            require(s.value && numeric(s.type) && s.value->type==s.type && allocated.insert(s.symbol).second);
            symbols.emplace(s.symbol,s.type); break;
        case StatementKind::Assign:
            require(s.value && symbols.count(s.symbol) && symbols.at(s.symbol)==s.type && s.value->type==s.type); break;
        case StatementKind::Evaluate: require(bool(s.value)); break;
        case StatementKind::Return:
            require((result==Type::Void && (!s.value || s.value->type==Type::Void)) ||
                    (s.value && s.value->type==result)); terminates=true; break;
        case StatementKind::Block:
            require(!s.value);
            terminates=verify_statements(s.body,symbols,allocated,functions,result,depth+1,budget); break;
        case StatementKind::If: case StatementKind::While: case StatementKind::SignSelect: {
            require(s.value && numeric(s.value->type));
            bool p=verify_statements(s.body,symbols,allocated,functions,result,depth+1,budget);
            bool z=verify_statements(s.alternative,symbols,allocated,functions,result,depth+1,budget);
            bool n=verify_statements(s.negative,symbols,allocated,functions,result,depth+1,budget);
            terminates=s.kind==StatementKind::If?(p&&z):s.kind==StatementKind::SignSelect?(p&&z&&n):false;
            break;
        }
        default: throw std::invalid_argument("unknown core statement opcode");
        }
        returned=returned||terminates;
    }
    return returned;
}
inline void verify_program(const Program& program) {
    size_t budget=100000;
    if(program.functions.size()>10000) throw std::invalid_argument("core function count exceeded");
    Signatures signatures;
    for(const auto& f:program.functions) {
        if(f.name.empty() || !signatures.emplace(f.name,Signature{f.result,{}}).second)
            throw std::invalid_argument("duplicate or empty core function name");
        auto& params=signatures.at(f.name).parameters;
        if(f.parameters.size()>1000) throw std::invalid_argument("core parameter count exceeded");
        for(auto p:f.parameters) {
            if(!numeric(p.type)) throw std::invalid_argument("unsupported core parameter type");
            params.push_back(p.type);
        }
        if(f.result!=Type::Void && !numeric(f.result)) throw std::invalid_argument("unsupported core result type");
    }
    for(const auto& f:program.functions) {
        Symbols symbols; std::unordered_set<uint32_t> allocated;
        for(auto p:f.parameters) {
            if(!allocated.insert(p.symbol).second) throw std::invalid_argument("duplicate core parameter symbol");
            symbols.emplace(p.symbol,p.type);
        }
        bool returns=verify_statements(f.body,symbols,allocated,signatures,f.result,0,&budget);
        if(f.result!=Type::Void && !returns) throw std::invalid_argument("core function missing return path");
    }
    std::unordered_set<uint32_t> allocated;
    verify_statements(program.body,{},allocated,signatures,Type::Int39,0,&budget);
}
} // namespace bt::ir
#endif
