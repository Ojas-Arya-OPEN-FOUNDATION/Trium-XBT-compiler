#ifndef TERNARY_IR_LOWERING_HPP
#define TERNARY_IR_LOWERING_HPP
#include "ternary_frontend_ast.hpp"
#include "include/Ternary/IR/Program.hpp"
namespace bt::ir {
class Lowering {
    struct Binding { uint32_t id; Type type; };
    std::vector<std::unordered_map<std::string,Binding>> scopes;
    Signatures functions;
    uint32_t next=1;
    Type result=Type::Int39;
    static Type type(frontend::BTType t) {
        using T=frontend::BTType;
        switch(t) {
        case T::Int:return Type::Int39; case T::Trit:return Type::Trit;
        case T::TritWord:return Type::Positional39; case T::Float:return Type::Float64;
        case T::Void:return Type::Void;
        default:throw std::invalid_argument("typed core currently supports scalar programs only");
        }
    }
    Binding lookup(const std::string& name) {
        for(auto i=scopes.rbegin();i!=scopes.rend();++i) if(i->count(name)) return i->at(name);
        throw std::invalid_argument("unresolved core symbol: "+name);
    }
    static std::shared_ptr<Expr> make(Op op,Type t,std::vector<Expression> inputs={}) {
        return std::make_shared<Expr>(Expr{op,t,std::move(inputs)});
    }
    static Expression coerce(Expression x,Type t) {
        if(x->type==t) return x;
        if(integral(x->type) && integral(t)) return make(Op::IntCast,t,{x});
        if(t==Type::Float64 && integral(x->type))
            return make(Op::IntToFloat,t,{coerce(x,Type::Int39)});
        throw std::invalid_argument("unsupported core conversion");
    }
    Expression expression(const frontend::ASTNodePtr& node) {
        using namespace frontend;
        if(!node) return {};
        std::shared_ptr<Expr> out;
        if(auto n=std::dynamic_pointer_cast<LiteralNode>(node)) {
            out=make(Op::Integer,type(n->checked_type)); out->integer=n->int_val;
        } else if(auto n=std::dynamic_pointer_cast<FloatLiteralExprAST>(node)) {
            out=make(Op::Floating,Type::Float64); out->floating=n->value;
        } else if(auto n=std::dynamic_pointer_cast<IdentifierNode>(node)) {
            auto b=lookup(n->name); out=make(Op::Parameter,b.type); out->symbol=b.id;
        } else if(auto n=std::dynamic_pointer_cast<UnaryOpNode>(node)) {
            auto x=expression(n->operand); out=make(Op::Negate,x->type,{x});
        } else if(auto n=std::dynamic_pointer_cast<BinaryOpNode>(node)) {
            auto a=expression(n->left),b=expression(n->right);
            if(a->type==Type::Float64 || b->type==Type::Float64) { a=coerce(a,Type::Float64); b=coerce(b,Type::Float64); }
            Op op;
            switch(n->op) {
            case BinaryOp::Add:op=Op::Add;break; case BinaryOp::Sub:op=Op::Sub;break; case BinaryOp::Mul:op=Op::Mul;break;
            case BinaryOp::KleeneAnd:op=Op::DigitAnd;break; case BinaryOp::KleeneOr:op=Op::DigitOr;break;
            case BinaryOp::TernaryXor:op=Op::DigitXor;break; case BinaryOp::Equal:op=Op::Equal;break;
            case BinaryOp::NotEqual:op=Op::NotEqual;break; case BinaryOp::LessThan:op=Op::Less;break;
            case BinaryOp::GreaterThan:op=Op::Greater;break; case BinaryOp::LessEqual:op=Op::LessEqual;break;
            case BinaryOp::GreaterEqual:op=Op::GreaterEqual;break; case BinaryOp::Spaceship:op=Op::Spaceship;break;
            default:throw std::invalid_argument("unknown source binary opcode");
            }
            out=make(op,type(n->checked_type),{a,b});
        } else if(auto n=std::dynamic_pointer_cast<SelectNode>(node)) {
            auto t=type(n->checked_type);
            out=make(Op::SignSelect,t,{expression(n->condition),coerce(expression(n->pos_branch),t),
                                     coerce(expression(n->zero_branch),t),coerce(expression(n->neg_branch),t)});
        } else if(auto n=std::dynamic_pointer_cast<CallNode>(node)) {
            std::vector<Expression> args;
            for(auto& a:n->arguments) args.push_back(expression(a));
            Type t=type(n->checked_type); Op op=Op::Call;
            if(n->callee=="print") op=Op::Print;
            else if(n->callee=="sqrt" || n->callee=="fma") {
                op=n->callee=="sqrt"?Op::Sqrt:Op::Fma;
                for(auto& a:args) a=coerce(a,Type::Float64);
            } else if(n->callee=="abs") op=Op::Abs;
            else if(n->callee=="planar_dot64" || n->callee.rfind("btha_",0)==0) op=Op::Intrinsic;
            else {
                if(!functions.count(n->callee)) throw std::invalid_argument("call outside scalar core: "+n->callee+" (aggregate compatibility requires --legacy-llvm)");
                const auto& signature=functions.at(n->callee);
                for(size_t i=0;i<args.size();++i) args[i]=coerce(args[i],signature.parameters[i]);
            }
            out=make(op,t,std::move(args)); out->name=n->callee;
        } else throw std::invalid_argument("expression is outside typed scalar core");
        out->line=node->location.line; out->column=node->location.column;
        return out;
    }
    Statement statement(const frontend::ASTNodePtr& node) {
        using namespace frontend;
        Statement s{StatementKind::Evaluate};
        if(auto n=std::dynamic_pointer_cast<VarDeclNode>(node)) {
            Type t=type(n->binding_type);
            Expression value=n->init_expr?expression(n->init_expr):make(Op::Integer,Type::Int39);
            value=coerce(value,t);
            Binding binding{next++,t}; scopes.back().emplace(n->name,binding);
            s.kind=StatementKind::Let; s.symbol=binding.id; s.type=t; s.value=value;
        } else if(auto n=std::dynamic_pointer_cast<AssignNode>(node)) {
            auto b=lookup(n->target); s.kind=StatementKind::Assign; s.symbol=b.id; s.type=b.type;
            s.value=coerce(expression(n->value),b.type);
        } else if(auto n=std::dynamic_pointer_cast<BlockNode>(node)) {
            s.kind=StatementKind::Block; scopes.emplace_back();
            for(auto& x:n->statements) s.body.push_back(statement(x));
            scopes.pop_back();
        } else if(auto n=std::dynamic_pointer_cast<IfNode>(node)) {
            s.kind=StatementKind::If; s.value=expression(n->condition);
            s.body={statement(n->then_block)};
            if(n->else_block) s.alternative={statement(n->else_block)};
        } else if(auto n=std::dynamic_pointer_cast<WhileNode>(node)) {
            s.kind=StatementKind::While; s.value=expression(n->condition); s.body={statement(n->body)};
        } else if(auto n=std::dynamic_pointer_cast<ReturnNode>(node)) {
            s.kind=StatementKind::Return;
            if(n->expr) s.value=expression(n->expr);
            if(result==Type::Int39 && (!s.value || s.value->type==Type::Void)) {
                // A void expression's effects must be preserved before main returns zero.
                if(s.value) { Statement effect{StatementKind::Evaluate}; effect.value=s.value;
                    Statement ret{StatementKind::Return}; ret.value=make(Op::Integer,Type::Int39);
                    s.kind=StatementKind::Block; s.value={}; s.body={effect,ret}; return s; }
                s.value=make(Op::Integer,Type::Int39);
            }
            if(s.value) s.value=coerce(s.value,result);
        } else if(auto n=std::dynamic_pointer_cast<ExpressionStmtNode>(node)) {
            s.value=expression(n->expr);
        } else if(auto n=std::dynamic_pointer_cast<SelectNode>(node);
                  n && std::dynamic_pointer_cast<BlockNode>(n->pos_branch)) {
            s.kind=StatementKind::SignSelect; s.value=expression(n->condition);
            s.body={statement(n->pos_branch)}; s.alternative={statement(n->zero_branch)};
            s.negative={statement(n->neg_branch)};
        } else s.value=expression(node);
        return s;
    }
public:
    Program lower(frontend::ProgramNode& ast) {
        frontend::TypeChecker checker(true); checker.typecheck(ast);
        functions.clear(); scopes={{}}; next=1;
        for(auto& item:ast.declarations) if(auto f=std::dynamic_pointer_cast<frontend::FunctionDefNode>(item)) {
            Signature signature{type(frontend::string_to_bttype(f->return_type)),{}};
            for(auto& p:f->params) signature.parameters.push_back(type(frontend::string_to_bttype(p.type)));
            functions.emplace(f->name,std::move(signature));
        }
        Program program;
        for(auto& item:ast.declarations) if(auto f=std::dynamic_pointer_cast<frontend::FunctionDefNode>(item)) {
            scopes={{}}; result=functions.at(f->name).result;
            Function function{f->name,result,{},{}};
            for(size_t i=0;i<f->params.size();++i) {
                Binding b{next++,functions.at(f->name).parameters[i]};
                scopes.back().emplace(f->params[i].name,b); function.parameters.push_back({b.id,b.type});
            }
            function.body={statement(f->body)}; program.functions.push_back(std::move(function));
        }
        scopes={{}}; result=Type::Int39;
        for(auto& item:ast.declarations) if(!std::dynamic_pointer_cast<frontend::FunctionDefNode>(item))
            program.body.push_back(statement(item));
        verify_program(program); return program;
    }
};
} // namespace bt::ir
#endif
