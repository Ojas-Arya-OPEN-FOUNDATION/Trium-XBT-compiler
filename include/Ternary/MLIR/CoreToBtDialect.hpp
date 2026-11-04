#ifndef TERNARY_MLIR_CORE_TO_BT_DIALECT_HPP
#define TERNARY_MLIR_CORE_TO_BT_DIALECT_HPP

/**
 * @file include/Ternary/MLIR/CoreToBtDialect.hpp
 * @brief Lowers the backend-neutral core IR into the MLIR `bt` dialect.
 *
 * This is the connection the readiness audit recorded as missing. The
 * repository had two compilers that never met:
 *
 *     .bt  ->  AST  ->  ir::Program  ->  { C++20 | LLVM IR | GLSL }
 *
 *     hand-written .mlir  ->  bt dialect  ->  5 passes  ->  linalg / vector
 *
 * The dialect, its five lowering passes and bt-opt were reachable only from
 * checked-in .mlir files. User code could not use them, so the most modular
 * part of the compiler was also the least used.
 *
 * This lowering makes the dialect the path for real programs:
 *
 *     .bt  ->  AST  ->  ir::Program  ->  bt dialect  ->  passes  ->  target
 *
 * Representation
 * --------------
 * The dialect carries a ternary scalar as a PAIR of i64 SSA values, the P and
 * N rails. So one core-IR scalar becomes two SSA values throughout, and every
 * emitted op follows that convention:
 *
 *     bt.const       -> (P, N)          materialise a constant
 *     bt.neg         (P,N) -> (P,N)     rail swap
 *     bt.lane_mul    (4)   -> (P,N)     lane-wise product
 *     bt.btha        (4)   -> (4)       sum rails + carry rails
 *     bt.csa_reduce  (4)   -> (P,N)     fold carry into sum
 *
 * `Add` is therefore two ops: a half adder producing (sum, carry), then a
 * carry-save reduction folding the carry back in. That is the honest shape of
 * a ternary add, and it is what the lowering passes already know how to
 * handle.
 *
 * Honest failure
 * --------------
 * Constructs with no dialect representation are reported, never emitted as
 * something approximately right. Lane-wise multiply is refused for multi-trit
 * operands for exactly the reason documented in docs/XBT_ISA_SPEC.md.
 */

#include "../IR/Core.hpp"
#include "../IR/Program.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

// The dialect's op definitions (namespace mlir::bt) and the standalone
// (out-of-tree) MLIR representation (namespace bt::mlir). Both are needed:
// the former names the operations, the latter is what we build.
#include "../TernaryDialect.h"
#include "../../../ternary_mlir_lowering.hpp"

namespace bt::mlir::bridge {

/** A core-IR scalar, as the two SSA values holding its rails. */
struct RailPair {
    ::bt::mlir::MLIRValue p;
    ::bt::mlir::MLIRValue n;
};

/** Outcome of lowering one program. */
struct BtModule {
    ::bt::mlir::MLIRModule module;
    std::vector<std::string> unsupported;   // functions that could not be lowered
    int functions{0};
    int ops{0};

    bool complete() const { return unsupported.empty(); }
    std::string to_string() const { return module.to_string(); }
};

namespace detail {

inline ::bt::mlir::MLIRType i64_type() {
    return ::bt::mlir::MLIRType::getI64();
}

inline bool is_single_trit(const ir::Expression& e) {
    return e && e->op == ir::Op::Integer && e->integer >= -1 && e->integer <= 1;
}

class FunctionLowering {
public:
    explicit FunctionLowering(::bt::mlir::MLIRFunction& fn) : fn_(fn) {}

    RailPair lower(const ir::Expression& e) {
        if (!e) throw std::invalid_argument("bt dialect: null expression");

        switch (e->op) {
        case ir::Op::Integer:
            return emit_const(e->integer);

        case ir::Op::Parameter: {
            auto it = args_.find(e->symbol);
            if (it == args_.end())
                throw std::runtime_error("bt dialect: unknown parameter symbol #" +
                                         std::to_string(e->symbol));
            return it->second;
        }

        case ir::Op::Negate: {
            auto a = lower(e->operands.at(0));
            return emit_neg(a);
        }

        case ir::Op::Add: {
            auto a = lower(e->operands.at(0));
            auto b = lower(e->operands.at(1));
            return emit_add(a, b);
        }

        case ir::Op::Sub: {
            // a - b  ==  a + (-b); negation is a free rail swap.
            auto a = lower(e->operands.at(0));
            auto b = lower(e->operands.at(1));
            return emit_add(a, emit_neg(b));
        }

        case ir::Op::Mul: {
            // Lane-wise only: carries do not cross lanes.
            if (!is_single_trit(e->operands.at(0)) ||
                !is_single_trit(e->operands.at(1)))
                throw std::runtime_error(
                    "bt dialect: scalar multiplication is not implemented "
                    "(bt.lane_mul drops inter-lane carries); refusing to emit "
                    "incorrect code");
            auto a = lower(e->operands.at(0));
            auto b = lower(e->operands.at(1));
            return emit_lane_mul(a, b);
        }

        default:
            throw std::runtime_error(
                "bt dialect: no lowering for IR op #" +
                std::to_string(static_cast<int>(e->op)) +
                " -- refusing to emit a stub");
        }
    }

    void bind_parameter(uint32_t symbol, const RailPair& r) {
        args_[symbol] = r;
    }

private:
    ::bt::mlir::MLIRFunction& fn_;
    std::unordered_map<uint32_t, RailPair> args_;
    int counter_{0};

    std::string fresh(const char* tag) {
        return std::string("%") + tag + std::to_string(counter_++);
    }

    void push(::bt::mlir::MLIROperation op) { fn_.operations.push_back(op); }

    RailPair emit_const(int64_t v) {
        auto [P, N] = ::mlir::bt::ConstOp::encode(v);
        ::bt::mlir::MLIROperation op;
        op.name = ::mlir::bt::ConstOp::kOpName;
        op.attributes.push_back({"value", std::to_string(v)});
        op.results.push_back({fresh("cp"), i64_type()});
        op.results.push_back({fresh("cn"), i64_type()});
        // The encoded rails are carried as a second attribute so the emitted
        // text is self-describing and re-parsable without re-deriving them.
        op.attributes.push_back({"P", std::to_string(P)});
        op.attributes.push_back({"N", std::to_string(N)});
        push(op);
        return {op.results[0], op.results[1]};
    }

    RailPair emit_neg(const RailPair& a) {
        ::bt::mlir::MLIROperation op;
        op.name = ::mlir::bt::NegOp::kOpName;
        op.operands = {a.p, a.n};
        op.results.push_back({fresh("np"), i64_type()});
        op.results.push_back({fresh("nn"), i64_type()});
        push(op);
        return {op.results[0], op.results[1]};
    }

    RailPair emit_lane_mul(const RailPair& a, const RailPair& b) {
        ::bt::mlir::MLIROperation op;
        op.name = ::mlir::bt::LaneMulOp::kOpName;
        op.operands = {a.p, a.n, b.p, b.n};
        op.results.push_back({fresh("mp"), i64_type()});
        op.results.push_back({fresh("mn"), i64_type()});
        push(op);
        return {op.results[0], op.results[1]};
    }

    /** A ternary add is a half adder followed by a carry-save reduction. */
    RailPair emit_add(const RailPair& a, const RailPair& b) {
        ::bt::mlir::MLIROperation ha;
        ha.name = ::mlir::bt::BTHAOp::kOpName;
        ha.operands = {a.p, a.n, b.p, b.n};
        ha.results.push_back({fresh("sp"), i64_type()});
        ha.results.push_back({fresh("sn"), i64_type()});
        ha.results.push_back({fresh("kp"), i64_type()});
        ha.results.push_back({fresh("kn"), i64_type()});
        push(ha);

        ::bt::mlir::MLIROperation red;
        red.name = ::mlir::bt::CSAReduceOp::kOpName;
        red.operands = {ha.results[0], ha.results[1], ha.results[2], ha.results[3]};
        red.results.push_back({fresh("rp"), i64_type()});
        red.results.push_back({fresh("rn"), i64_type()});
        push(red);

        return {red.results[0], red.results[1]};
    }
};

} // namespace detail

/** Build the terminator returning both rails. */
inline ::bt::mlir::MLIROperation make_return(const RailPair& r) {
    ::bt::mlir::MLIROperation op;
    op.name = "return";
    op.operands = {r.p, r.n};
    return op;
}

/**
 * @brief Lower a whole core-IR program into the `bt` dialect.
 *
 * Functions that cannot be lowered are named in `unsupported` rather than
 * silently emitted. Callers should treat a non-empty list as an error.
 */
inline BtModule lower_program_to_bt_dialect(const ir::Program& program,
                                            const std::string& module_name = "bt_module") {
    BtModule out;
    out.module.name = module_name;

    for (const auto& f : program.functions) {
        ++out.functions;
        ::bt::mlir::MLIRFunction fn;
        fn.name = f.name;

        // Each core-IR scalar parameter becomes two i64 arguments: P and N.
        // SSA names must begin with a letter or underscore, so the symbol id
        // is prefixed rather than used raw.
        std::vector<RailPair> bound;
        for (size_t i = 0; i < f.parameters.size(); ++i) {
            const auto& p = f.parameters[i];
            ::bt::mlir::MLIRValue pv("%arg" + std::to_string(i) + "_P",
                                     detail::i64_type());
            ::bt::mlir::MLIRValue nv("%arg" + std::to_string(i) + "_N",
                                     detail::i64_type());
            fn.arguments.push_back(pv);
            fn.arguments.push_back(nv);
            bound.push_back({pv, nv});
        }

        try {
            detail::FunctionLowering lowering(fn);
            for (size_t i = 0; i < f.parameters.size(); ++i)
                lowering.bind_parameter(f.parameters[i].symbol, bound[i]);

            // Walk the body recursively: the frontend wraps statements in
            // blocks, so a Return is not necessarily at the top level. The
            // last Return encountered wins, matching the scalar backends.
            bool returned = false;
            std::function<void(const std::vector<ir::Statement>&)> walk =
                [&](const std::vector<ir::Statement>& body) {
                    for (const auto& s : body) {
                        switch (s.kind) {
                        case ir::StatementKind::Let:
                        case ir::StatementKind::Assign:
                            // `let t = expr` binds an SSA rails pair to the
                            // symbol, so later expressions can reference it.
                            if (s.value) {
                                lowering.bind_parameter(s.symbol,
                                                        lowering.lower(s.value));
                            }
                            break;
                        case ir::StatementKind::Return:
                            if (s.value) {
                                auto r = lowering.lower(s.value);
                                fn.operations.push_back(make_return(r));
                                returned = true;
                            }
                            break;
                        case ir::StatementKind::Block:
                            // Only descend into plain blocks; branches need
                            // real control flow, which the dialect does not
                            // carry at this level.
                            if (!s.body.empty()) walk(s.body);
                            break;
                        default:
                            throw std::runtime_error(
                                "bt dialect: no lowering for statement kind #" +
                                std::to_string(static_cast<int>(s.kind)));
                        }
                    }
                };
            walk(f.body);
            if (f.result != ir::Type::Void && !returned) {
                throw std::runtime_error("function has no return path");
            }
            if (f.result != ir::Type::Void) {
                fn.returnTypes.push_back(detail::i64_type());
                fn.returnTypes.push_back(detail::i64_type());
            }

            out.ops += static_cast<int>(fn.operations.size());
            out.module.functions.push_back(fn);
        } catch (const std::exception& e) {
            out.unsupported.push_back(f.name + ": " + e.what());
        }
    }

    return out;
}

} // namespace bt::mlir::bridge

#endif // TERNARY_MLIR_CORE_TO_BT_DIALECT_HPP
