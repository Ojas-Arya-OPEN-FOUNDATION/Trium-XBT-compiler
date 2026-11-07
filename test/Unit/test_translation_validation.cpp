/**
 * @file test_translation_validation.cpp
 * @brief Differential translation validation for the real C++ optimizer.
 *
 * Layer 2 of two. Layer 1 (verification/translation_validation.py) proves each
 * rewrite rule sound symbolically in Z3. That is necessary but NOT sufficient:
 * a rule can be sound and still be *applied* wrongly -- the classic case is an
 * identity rule firing on a non-commutative op, so that `0 - x` rewrites to
 * `x`. Optimize.hpp guards exactly that with `e->op != Op::Sub`.
 *
 * So this file does the other half: run the ACTUAL Optimizer on random
 * programs and check that evaluation is preserved, including the *failure*
 * behaviour. If the input overflows, the optimized program must overflow too --
 * the optimizer must not delete an overflow check.
 *
 * This is the harness docs/SCALING_PLAN.md 3.3 asks for: obligations are
 * emitted by the pass harness, not hand-written per test, and coverage is
 * reported rather than assumed.
 */

#include "Ternary/IR/Execute.hpp"
#include "Ternary/IR/Optimize.hpp"

#include <cstdint>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace bt::ir;

namespace {

int g_checks = 0;
int g_fails = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

Expression number(int64_t n) {
    auto e = std::make_shared<Expr>(Expr{Op::Integer, Type::Int39});
    e->integer = n;
    return e;
}
Expression param(uint32_t sym) {
    auto e = std::make_shared<Expr>(Expr{Op::Parameter, Type::Int39});
    e->symbol = sym;
    return e;
}
Expression operation(Op op, Expression a, Expression b) {
    return std::make_shared<Expr>(Expr{op, Type::Int39, {a, b}});
}
Expression negate(Expression a) {
    return std::make_shared<Expr>(Expr{Op::Negate, Type::Int39, {a}});
}
Expression select(Expression c, Expression p, Expression z, Expression n) {
    return std::make_shared<Expr>(Expr{Op::SignSelect, Type::Int39, {c, p, z, n}});
}

/** Random expression tree mixing constants and parameters. */
Expression random_expr(std::mt19937_64& rng, int depth) {
    if (depth <= 0 || (rng() % 100) < 30) {
        if (rng() % 2) return number(int64_t(rng() % 41) - 20);
        return param(uint32_t(1 + (rng() % 2)));
    }
    switch (rng() % 6) {
        case 0: return operation(Op::Add, random_expr(rng, depth - 1),
                                          random_expr(rng, depth - 1));
        case 1: return operation(Op::Sub, random_expr(rng, depth - 1),
                                          random_expr(rng, depth - 1));
        case 2: return operation(Op::Mul, random_expr(rng, depth - 1),
                                          random_expr(rng, depth - 1));
        case 3: return negate(random_expr(rng, depth - 1));
        case 4: return select(random_expr(rng, depth - 1),
                              random_expr(rng, depth - 1),
                              random_expr(rng, depth - 1),
                              random_expr(rng, depth - 1));
        default: {
            // Bias towards the identity shapes the optimizer rewrites, so the
            // rules actually fire during the fuzz rather than rarely.
            static const int64_t idents[] = {0, 1, -1, 2};
            int64_t k = idents[rng() % 4];
            Op op = (rng() % 2) ? Op::Add : Op::Mul;
            if (rng() % 2)
                return operation(op, random_expr(rng, depth - 1), number(k));
            return operation(op, number(k), random_expr(rng, depth - 1));
        }
    }
}

/**
 * Run one expression through the optimizer and compare evaluation, including
 * whether it throws. Returns false on any divergence.
 */
bool one_case(Expression e, int64_t a1, int64_t a2, std::string* detail) {
    Function f;
    f.name = "f";
    f.result = Type::Int39;
    f.parameters.push_back({1u, Type::Int39});
    f.parameters.push_back({2u, Type::Int39});
    Statement s;
    s.kind = StatementKind::Return;
    s.value = e;
    f.body.push_back(s);

    Program p;
    p.functions.push_back(f);

    Program opt;
    try {
        opt = Optimizer{}.run(p);
    } catch (const std::exception& ex) {
        *detail = "Optimizer threw: " + std::string(ex.what());
        return false;
    }

    std::ostringstream o1, o2;
    std::vector<Value> args{Value(a1), Value(a2)};

    bool before_threw = false, after_threw = false;
    int64_t before = 0, after = 0;
    try {
        Value v = Executor(p, o1).invoke("f", args);
        before = std::get<int64_t>(v);
    } catch (const std::exception&) { before_threw = true; }

    try {
        Value v = Executor(opt, o2).invoke("f", args);
        after = std::get<int64_t>(v);
    } catch (const std::exception&) { after_threw = true; }

    if (before_threw != after_threw) {
        *detail = before_threw ? "optimizer REMOVED an overflow failure"
                               : "optimizer INTRODUCED a failure";
        return false;
    }
    if (!before_threw && before != after) {
        *detail = "value changed: " + std::to_string(before) + " -> " +
                  std::to_string(after);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    std::printf("translation validation (differential, real Optimizer)\n");

    std::mt19937_64 rng(0x5456414C);   // "TVAL"
    const int kTrials = 20000;
    int mismatches = 0;

    for (int i = 0; i < kTrials; ++i) {
        Expression e = random_expr(rng, 4);
        int64_t a1 = int64_t(rng() % 61) - 30;
        int64_t a2 = int64_t(rng() % 61) - 30;

        Program p;
        Function f;
        f.name = "f";
        f.result = Type::Int39;
        f.parameters.push_back({1u, Type::Int39});
        f.parameters.push_back({2u, Type::Int39});
        Statement s;
        s.kind = StatementKind::Return;
        s.value = e;
        f.body.push_back(s);
        p.functions.push_back(f);

        std::string detail;
        if (!one_case(e, a1, a2, &detail)) {
            if (mismatches < 5)
                std::printf("      %s\n", detail.c_str());
            ++mismatches;
        }
    }

    check(mismatches == 0,
          "optimizer preserves evaluation on " + std::to_string(kTrials) +
          " random programs");

    // ---- the specific hazard the symbolic layer cannot catch --------------
    // `0 - x` must NOT become `x`. Sub is not commutative and Optimize.hpp
    // guards the left-operand case explicitly. If someone removes that guard,
    // this is the test that fails.
    {
        std::string detail;
        check(one_case(operation(Op::Sub, number(0), param(1)), 7, 0, &detail),
              "0 - x is not rewritten to x (Sub is not commutative): " + detail);
        check(one_case(operation(Op::Sub, param(1), number(0)), 7, 0, &detail),
              "x - 0 still equals x: " + detail);
        check(one_case(operation(Op::Sub, number(0), param(1)), -7, 0, &detail),
              "0 - (-7) == 7 survives optimization: " + detail);
    }

    // ---- overflow checks must survive -------------------------------------
    {
        const int64_t h = bt::math::ScalarInteger::maximum;
        std::string detail;
        // A failing sub-expression multiplied by 0 must still fail: the
        // optimizer may not propagate "x*0 == 0" through a value that never
        // successfully evaluates. Checked directly just below.
        Program p;
        Function f;
        f.name = "f";
        f.result = Type::Int39;
        Statement s;
        s.kind = StatementKind::Return;
        s.value = operation(Op::Mul, number(0),
                            operation(Op::Add, number(h), number(1)));
        f.body.push_back(s);
        p.functions.push_back(f);
        std::ostringstream o;
        // NOTE: invoke(), not run(). run() executes `program.body`, which is
        // empty here because the code lives in a function -- using it would
        // make this test pass vacuously.
        const std::vector<Value> no_args;
        bool threw_before = false, threw_after = false;
        try { Executor(p, o).invoke("f", no_args); }
        catch (const std::exception&) { threw_before = true; }
        try { Executor(Optimizer{}.run(p), o).invoke("f", no_args); }
        catch (const std::exception&) { threw_after = true; }
        check(threw_before, "the unoptimized program does overflow");
        check(threw_after, "the optimized program still overflows");
    }

    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
