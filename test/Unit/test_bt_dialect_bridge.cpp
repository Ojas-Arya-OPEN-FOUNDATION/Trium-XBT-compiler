/**
 * @file test/Unit/test_bt_dialect_bridge.cpp
 * @brief Connects the .bt frontend to the bt dialect, end to end.
 *
 * The readiness audit found the frontend and the dialect were two unconnected
 * compilers: bt-opt could only be fed hand-written .mlir files, so the five
 * lowering passes were unreachable from user code.
 *
 * These tests exercise the seam that now joins them:
 *
 *   ir::Program  ->  bt dialect module  ->  MLIR text  ->  re-parsed module
 *                                                     ->  lowering passes
 *
 * The re-parse matters: it proves the emitted text is accepted by the same
 * loader bt-opt uses, so a .bt program can genuinely flow through the pass
 * pipeline rather than merely being printed in MLIR syntax.
 */

#include "../../include/Ternary/MLIR/CoreToBtDialect.hpp"
#include "../../include/Ternary/TernaryDialect.h"

#include <cstdio>
#include <memory>
#include <string>

using namespace bt;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::printf("  FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

using ExprPtr = std::shared_ptr<const ir::Expr>;

ExprPtr intlit(int64_t v) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::Integer;
    e->type = ir::Type::Int39;
    e->integer = v;
    return e;
}

ExprPtr param(uint32_t sym) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::Parameter;
    e->type = ir::Type::Int39;
    e->symbol = sym;
    return e;
}

ExprPtr binop(ir::Op op, ExprPtr a, ExprPtr b) {
    auto e = std::make_shared<ir::Expr>();
    e->op = op;
    e->type = ir::Type::Int39;
    e->operands = {a, b};
    return e;
}

ExprPtr neg(ExprPtr a) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::Negate;
    e->type = ir::Type::Int39;
    e->operands = {a};
    return e;
}

ir::Function make_fn(const std::string& name, ExprPtr body, bool with_param = true) {
    ir::Function f;
    f.name = name;
    f.result = ir::Type::Int39;
    if (with_param) f.parameters.push_back({1, ir::Type::Int39});
    ir::Statement r;
    r.kind = ir::StatementKind::Return;
    r.type = ir::Type::Int39;
    r.value = body;
    f.body.push_back(r);
    return f;
}

// --------------------------------------------------------------------------

void test_constants_lower() {
    std::printf("  constants become bt.const\n");
    ir::Program p;
    p.functions.push_back(make_fn("k", intlit(42), false));

    auto r = bt::mlir::bridge::lower_program_to_bt_dialect(p, "consts");
    check(r.complete(), "constant lowering completes");
    check(r.ops > 0, "operations were emitted");

    const std::string text = r.to_string();
    check(text.find("bt.const") != std::string::npos, "emits bt.const");
    check(text.find("value = 42") != std::string::npos, "carries the value");

    // The encoded rails must match the reference encoder.
    auto [P, N] = ::mlir::bt::ConstOp::encode(42);
    int64_t decoded = 0, place = 1;
    for (int i = 0; i < 40; ++i) {
        if ((P >> i) & 1ULL) decoded += place;
        else if ((N >> i) & 1ULL) decoded -= place;
        place *= 3;
    }
    check(decoded == 42, "bt.const encoding round-trips to 42");
}

void test_const_encoding_roundtrip() {
    std::printf("  bt.const encoding round-trips across the int39 range\n");
    const int64_t vals[] = {0, 1, -1, 2, -2, 3, -3, 121, -121, 1000, -1000,
                            123456, -123456, 2026277576509488133LL,
                            -2026277576509488133LL};
    for (int64_t v : vals) {
        auto [P, N] = ::mlir::bt::ConstOp::encode(v);
        check((P & N) == 0, "rails are disjoint for " + std::to_string(v));
        int64_t got = 0, place = 1;
        for (int i = 0; i < 40; ++i) {
            if ((P >> i) & 1ULL) got += place;
            else if ((N >> i) & 1ULL) got -= place;
            place *= 3;
        }
        check(got == v, "encode/decode round-trip for " + std::to_string(v));
    }
}

void test_add_lowers_to_btha_and_reduce() {
    std::printf("  Add lowers to bt.btha + bt.csa_reduce\n");
    ir::Program p;
    p.functions.push_back(make_fn("addfn", binop(ir::Op::Add, param(1), intlit(7))));

    auto r = bt::mlir::bridge::lower_program_to_bt_dialect(p, "addmod");
    check(r.complete(), "add lowering completes");

    const std::string text = r.to_string();
    check(text.find("bt.btha") != std::string::npos,
          "emits bt.btha (half adder)");
    check(text.find("bt.csa_reduce") != std::string::npos,
          "emits bt.csa_reduce (folds the carry back in)");
    check(text.find("func.func @addfn") != std::string::npos,
          "emits the function");
}

void test_neg_and_sub() {
    std::printf("  Negate and Sub\n");
    {
        ir::Program p;
        p.functions.push_back(make_fn("negfn", neg(param(1))));
        auto r = bt::mlir::bridge::lower_program_to_bt_dialect(p, "negmod");
        check(r.complete(), "negate lowering completes");
        check(r.to_string().find("bt.neg") != std::string::npos, "emits bt.neg");
    }
    {
        // a - b lowers to a neg + add
        ir::Program p;
        p.functions.push_back(make_fn("subfn", binop(ir::Op::Sub, param(1), intlit(9))));
        auto r = bt::mlir::bridge::lower_program_to_bt_dialect(p, "submod");
        check(r.complete(), "subtract lowering completes");
        const std::string t = r.to_string();
        check(t.find("bt.neg") != std::string::npos, "sub emits a negation");
        check(t.find("bt.btha") != std::string::npos, "sub emits an add");
    }
}

void test_emitted_text_reparses() {
    std::printf("  emitted MLIR is accepted by the bt-opt loader\n");
    ir::Program p;
    p.functions.push_back(make_fn("addfn", binop(ir::Op::Add, param(1), intlit(7))));
    p.functions.push_back(make_fn("subfn", binop(ir::Op::Sub, param(1), intlit(9))));

    auto r = bt::mlir::bridge::lower_program_to_bt_dialect(p, "reparse_mod");
    check(r.complete(), "lowering completes");

    const std::string text = r.to_string();

    // This is the loader bt-opt itself uses. If it rejects our output, the
    // frontend and the dialect are still not actually connected.
    bt::mlir::MLIRModule reparsed;
    std::string error;
    const bool ok = ::mlir::bt::loadMLIRModuleFromString(reparsed, text, error);
    check(ok, "bt-opt loader accepts the emitted text: " + error);
    check(reparsed.name == "reparse_mod", "module symbol survives the round-trip");
    check(reparsed.functions.size() == 2,
          "both functions survive: got " + std::to_string(reparsed.functions.size()));
}

void test_dialect_ops_are_registered() {
    std::printf("  new dialect ops are registered\n");
    ::mlir::bt::TernaryDialect dialect;
    dialect.initialize();

    check(::mlir::bt::isKnownBtOpName(::mlir::bt::ConstOp::kOpName), "bt.const registered");
    check(::mlir::bt::isKnownBtOpName(::mlir::bt::NegOp::kOpName), "bt.neg registered");
    check(::mlir::bt::isKnownBtOpName(::mlir::bt::LaneMulOp::kOpName),
          "bt.lane_mul registered");
    check(::mlir::bt::isKnownBtOpName(::mlir::bt::BTHAOp::kOpName),
          "bt.btha still registered");
    check(::mlir::bt::isKnownBtOpName(::mlir::bt::CSAReduceOp::kOpName),
          "bt.csa_reduce still registered");
}

void test_neg_and_lanemul_execute() {
    std::printf("  bt.neg and bt.lane_mul execute correctly\n");

    // Negation is an involution and negates the value.
    for (uint64_t w : {0ULL, 1ULL, 5ULL, 0xFFULL, (1ULL << 63)}) {
        auto [np, nn] = ::mlir::bt::NegOp::execute(w, 0);
        check(np == 0 && nn == w, "neg swaps rails");
        auto [pp, pn] = ::mlir::bt::NegOp::execute(np, nn);
        check(pp == w && pn == 0, "neg is an involution");
    }

    // Lane-wise product over all 9 trit combinations.
    const int64_t trits[3] = {-1, 0, 1};
    for (int a : trits) {
        for (int b : trits) {
            uint64_t ap = (a == 1), an = (a == -1);
            uint64_t bp = (b == 1), bn = (b == -1);
            auto [rp, rn] = ::mlir::bt::LaneMulOp::execute(ap, an, bp, bn);
            check((rp & rn) == 0, "product rails are disjoint");
            int got = (rp ? 1 : 0) - (rn ? 1 : 0);
            check(got == a * b, "lane product " + std::to_string(a) + "*" +
                                std::to_string(b));
        }
    }
}

void test_refuses_unsupported() {
    std::printf("  refuses what the dialect cannot express\n");
    // Multi-trit scalar multiply: lane_mul would drop carries.
    ir::Program p;
    p.functions.push_back(make_fn("bigmul", binop(ir::Op::Mul, intlit(1000), intlit(37))));
    auto r = bt::mlir::bridge::lower_program_to_bt_dialect(p, "refusemod");

    check(!r.complete(), "lowering reports the failure");
    check(r.unsupported.size() == 1, "exactly one function reported");
    check(r.unsupported[0].find("lane_mul") != std::string::npos ||
          r.unsupported[0].find("lane") != std::string::npos,
          "reason names the semantic problem");
    check(r.to_string().find("bigmul") == std::string::npos,
          "no code emitted for the refused function");
}

void test_mixed_program_is_partial() {
    std::printf("  mixed program lowers what it can\n");
    ir::Program p;
    p.functions.push_back(make_fn("ok_add", binop(ir::Op::Add, param(1), intlit(2))));
    p.functions.push_back(make_fn("bad_mul", binop(ir::Op::Mul, intlit(100), intlit(100))));

    auto r = bt::mlir::bridge::lower_program_to_bt_dialect(p, "mixedmod");
    check(!r.complete(), "result is marked incomplete");
    check(r.to_string().find("ok_add") != std::string::npos,
          "the lowerable function is still emitted");
    check(r.unsupported.size() == 1, "only the bad function is reported");
}

} // namespace

int main() {
    std::printf("Frontend -> bt dialect bridge\n");
    std::printf("=============================\n");

    test_constants_lower();
    test_const_encoding_roundtrip();
    test_add_lowers_to_btha_and_reduce();
    test_neg_and_sub();
    test_emitted_text_reparses();
    test_dialect_ops_are_registered();
    test_neg_and_lanemul_execute();
    test_refuses_unsupported();
    test_mixed_program_is_partial();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    if (g_failures) {
        std::printf("BT DIALECT BRIDGE: FAILED\n");
        return 1;
    }
    std::printf("BT DIALECT BRIDGE: PASSED\n");
    return 0;
}
