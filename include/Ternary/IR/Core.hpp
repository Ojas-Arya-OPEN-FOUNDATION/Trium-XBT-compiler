#ifndef TERNARY_IR_CORE_HPP
#define TERNARY_IR_CORE_HPP
#include "../Math/Integer.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <string>

// Backend-neutral scalar expression IR schema, version 0.1. This verifier is
// used by whole-program lowering/execution; LLVM still consumes the checked AST.
namespace bt::ir {
enum class Type { Int39, Trit, Positional39, Packed64, Float64, Void };
enum class Op { Integer, Floating, Packed, Parameter, Add, Sub, Mul, Negate,
                IntToFloat, Equal, NotEqual, Less, Spaceship, SignSelect, IntCast, Greater, LessEqual, GreaterEqual,
                DigitAnd, DigitOr, DigitXor, Call, Print, Sqrt, Abs, Fma, Intrinsic };
enum class Effect { Pure, MayFail, Observable };
struct Expr {
    Op op;
    Type type;
    std::vector<std::shared_ptr<const Expr>> operands;
    int64_t integer = 0;
    double floating = 0;
    uint64_t positive = 0, negative = 0;
    uint32_t symbol = 0;
    std::string name;
    size_t line = 0, column = 0;
};
struct Signature { Type result; std::vector<Type> parameters; };
using Signatures = std::unordered_map<std::string,Signature>;
using Symbols = std::unordered_map<uint32_t, Type>;
inline bool integral(Type t) {
    return t == Type::Int39 || t == Type::Trit || t == Type::Positional39;
}
inline bool numeric(Type t) { return integral(t) || t == Type::Float64; }
inline Effect verify(const Expr& e, const Symbols& symbols = {}, unsigned depth = 0, const Signatures* functions = nullptr, size_t* budget = nullptr) {
    size_t local_budget=100000;
    if(!budget) budget=&local_budget;
    if(!*budget) throw std::invalid_argument("core verification work budget exceeded");
    --*budget;
    if (depth > 512) throw std::invalid_argument("core IR nesting limit exceeded");
    if (!numeric(e.type) && e.type != Type::Packed64 && e.type != Type::Void)
        throw std::invalid_argument("unknown core IR type");
    Effect effect = Effect::Pure;
    for (const auto& input : e.operands) {
        if (!input) throw std::invalid_argument("null core IR operand");
        auto child=verify(*input,symbols,depth+1,functions,budget);
        if(static_cast<int>(child)>static_cast<int>(effect)) effect=child;
    }
    const auto require = [](bool condition) {
        if (!condition) throw std::invalid_argument("invalid typed core IR signature or constant");
    };
    auto arity = [&](size_t n) { require(e.operands.size() == n); };
    auto fail=[&] { if(effect==Effect::Pure) effect=Effect::MayFail; };
    switch (e.op) {
    case Op::Integer:
        arity(0); require(integral(e.type));
        require(e.integer >= math::ScalarInteger::minimum && e.integer <= math::ScalarInteger::maximum);
        if (e.type == Type::Trit) require(e.integer >= -1 && e.integer <= 1);
        break;
    case Op::Floating:
        arity(0); require(e.type == Type::Float64 && std::isfinite(e.floating)); break;
    case Op::Packed:
        arity(0); require(e.type == Type::Packed64 && !(e.positive & e.negative)); break;
    case Op::Parameter: {
        arity(0); auto it = symbols.find(e.symbol);
        require(it != symbols.end() && it->second == e.type); break;
    }
    case Op::IntCast:
        arity(1); require(integral(e.type) && integral(e.operands[0]->type));
        if (e.type == Type::Trit) fail();
        break;
    case Op::IntToFloat:
        arity(1); require(e.operands[0]->type == Type::Int39 && e.type == Type::Float64); break;
    case Op::Add: case Op::Sub: case Op::Mul:
        arity(2);
        if (e.type == Type::Int39) {
            require(integral(e.operands[0]->type) && integral(e.operands[1]->type));
            fail();
        } else require(e.type == Type::Float64 && e.operands[0]->type == e.type && e.operands[1]->type == e.type);
        break;
    case Op::Negate:
        arity(1); require(numeric(e.type) && e.operands[0]->type == e.type); break;
    case Op::Equal: case Op::NotEqual: case Op::Less: case Op::Greater: case Op::LessEqual: case Op::GreaterEqual: case Op::Spaceship:
        arity(2); require(e.type == Type::Int39);
        require((integral(e.operands[0]->type) && integral(e.operands[1]->type)) ||
                (e.operands[0]->type == Type::Float64 && e.operands[1]->type == Type::Float64));
        break;
    case Op::SignSelect:
        // Operands 1,2,3 are regions/thunks: NEVER eagerly materialize all three.
        arity(4); require(numeric(e.operands[0]->type));
        for (size_t i = 1; i < 4; ++i) require(e.operands[i]->type == e.type);
        break;
    case Op::DigitAnd: case Op::DigitOr: case Op::DigitXor:
        arity(2); require(e.type == Type::Int39 && integral(e.operands[0]->type) && integral(e.operands[1]->type)); break;
    case Op::Call: {
        require(functions && functions->count(e.name));
        const auto& f=functions->at(e.name); arity(f.parameters.size()); require(e.type==f.result);
        for(size_t i=0;i<e.operands.size();++i) require(e.operands[i]->type==f.parameters[i]);
        effect=Effect::Observable; break;
    }
    case Op::Print:
        arity(1); require(e.type==Type::Void && numeric(e.operands[0]->type)); effect=Effect::Observable; break;
    case Op::Abs:
        arity(1); require(numeric(e.type) && e.operands[0]->type==e.type); break;
    case Op::Sqrt: case Op::Fma:
        arity(e.op==Op::Sqrt?1:3); require(e.type==Type::Float64);
        for(auto& operand:e.operands) require(operand->type==Type::Float64); break;
    case Op::Intrinsic:
        arity(4); require(e.type==Type::Int39);
        for(auto& operand:e.operands) require(integral(operand->type));
        require(e.name=="planar_dot64" || e.name=="btha_sum_p" || e.name=="btha_sum_n" ||
                e.name=="btha_carry_p" || e.name=="btha_carry_n");
        fail(); break;
    default: throw std::invalid_argument("unknown core IR opcode");
    }
    return effect;
}
} // namespace bt::ir
#endif
