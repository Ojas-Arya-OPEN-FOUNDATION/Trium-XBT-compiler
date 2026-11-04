/**
 * @file test/Unit/test_xbt_selector.cpp
 * @brief End-to-end: core IR -> Xbt machine code -> execution on the model.
 *
 * This closes the loop that `docs/HARDWARE_READINESS_AUDIT.md` recorded as
 * open: the compiler previously parsed `--target=riscv64-xbt` and then threw
 * "instruction selection is not implemented". Here the IR is lowered to real
 * encoded instructions, those instructions are executed on the cycle-accurate
 * coprocessor model, and the result is compared against plain C++ arithmetic.
 *
 * The instructions themselves are proved correct against balanced-ternary
 * mathematics in verification/verify_isa.py.
 */

#include "../../hw/xbt_alu_model.hpp"
#include "../../include/Ternary/Target/XbtInstructionSelector.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace bt;
using namespace bt::hw;

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

// ---- IR construction helpers ---------------------------------------------

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

/** Encode a lowered instruction into a 32-bit word the coprocessor executes. */
uint32_t encode(const bt::target::xbt::Insn& i) {
    if (i.is_itype)
        return encode_xbt_i(static_cast<uint16_t>(i.imm), i.rs1, i.funct3, i.rd);
    return encode_xbt_r(i.funct7, i.rs2, i.rs1, i.funct3, i.rd);
}

/** Value of a trit word as an ordinary integer. */
int64_t word_value(const TritWord64& w) {
    int64_t v = 0, p = 1;
    for (int i = 0; i < 40; ++i) {
        bool pp = (w.P >> i) & 1ULL;
        bool nn = (w.N >> i) & 1ULL;
        if (pp) v += p;
        else if (nn) v -= p;
        p *= 3;
    }
    return v;
}

/**
 * Run a function body through the selector and the coprocessor.
 * `params` are pre-loaded into t1..tN in order.
 */
int64_t run(ExprPtr expr, const std::vector<int64_t>& params) {
    // Build:  func(sym1..symN) -> Int39 { return expr; }
    ir::Function f;
    f.name = "testfn";
    f.result = ir::Type::Int39;
    for (size_t i = 0; i < params.size(); ++i)
        f.parameters.push_back({static_cast<uint32_t>(i + 1), ir::Type::Int39});

    ir::Statement ret;
    ret.kind = ir::StatementKind::Return;
    ret.type = ir::Type::Int39;
    ret.value = expr;
    f.body.push_back(ret);

    bt::target::xbt::InstructionSelector sel;
    auto lowered = sel.lower_function(f);

    RiscvXbtCoprocessor cpu;
    for (size_t i = 0; i < params.size(); ++i) {
        // materialise each parameter the way the selector would for a constant
        bt::ir::Function one;
        one.name = "p";
        one.result = bt::ir::Type::Int39;
        ir::Statement r;
        r.kind = bt::ir::StatementKind::Return;
        r.type = bt::ir::Type::Int39;
        r.value = intlit(params[i]);
        one.body.push_back(r);
        bt::target::xbt::InstructionSelector s2;
        auto lp = s2.lower_function(one);
        for (const auto& ins : lp.code) cpu.execute(encode(ins));
        cpu.treg[static_cast<size_t>(i) + 1] = cpu.treg[1];
    }

    for (const auto& ins : lowered.code) cpu.execute(encode(ins));
    return word_value(cpu.treg[1]);
}

// ---- tests ---------------------------------------------------------------

void test_constants() {
    std::printf("  constants (LI5 materialisation)\n");
    const int64_t vals[] = {0, 1, -1, 2, -2, 5, -5, 121, -121, 364, -364,
                            1000, -1000, 123456, -123456,
                            2026277576509488133LL, -2026277576509488133LL};
    for (int64_t v : vals) {
        int64_t got = run(intlit(v), {});
        check(got == v, "li5 materialise " + std::to_string(v) + " -> " +
                        std::to_string(got));
    }
}

void test_arithmetic() {
    std::printf("  arithmetic (Add / Sub / Mul / Negate)\n");
    struct Case { int64_t a, b; };
    const Case cases[] = {{0,0},{1,1},{1,-1},{121,121},{-121,-121},
                          {1000,37},{-1000,37},{243,81},{12345,-6789}};
    for (const auto& c : cases) {
        {
            int64_t got = run(binop(ir::Op::Add, intlit(c.a), intlit(c.b)), {});
            check(got == c.a + c.b, "add " + std::to_string(c.a) + " + " +
                                    std::to_string(c.b) + " = " + std::to_string(got));
        }
        {
            int64_t got = run(binop(ir::Op::Sub, intlit(c.a), intlit(c.b)), {});
            check(got == c.a - c.b, "sub " + std::to_string(c.a) + " - " +
                                    std::to_string(c.b) + " = " + std::to_string(got));
        }
        // Lane-wise MUL is only safe for single-trit operands; multi-trit
        // operands must be refused (see test_refuses_unsupported).
        if (c.a >= -1 && c.a <= 1 && c.b >= -1 && c.b <= 1) {
            int64_t got = run(binop(ir::Op::Mul, intlit(c.a), intlit(c.b)), {});
            check(got == c.a * c.b, "mul " + std::to_string(c.a) + " * " +
                                    std::to_string(c.b) + " = " + std::to_string(got));
        }
        {
            int64_t got = run(neg(intlit(c.a)), {});
            check(got == -c.a, "neg " + std::to_string(c.a) + " = " +
                               std::to_string(got));
        }
    }
}

void test_nested() {
    std::printf("  nested expressions\n");
    // (a + b) - c
    auto e = binop(ir::Op::Sub,
                   binop(ir::Op::Add, intlit(100), intlit(50)),
                   intlit(30));
    check(run(e, {}) == 120, "(100+50)-30");

    // -(a) + (b * c)   with single-trit b, c so the lane-wise MUL is exact
    auto e2 = binop(ir::Op::Add,
                    neg(intlit(7)),
                    binop(ir::Op::Mul, intlit(-1), intlit(1)));
    check(run(e2, {}) == -8, "-7 + (-1*1)");

    // deeply nested add chain
    auto e3 = intlit(1);
    for (int i = 0; i < 8; ++i) e3 = binop(ir::Op::Add, e3, intlit(11));
    check(run(e3, {}) == 1 + 8 * 11, "nested add chain of 9 terms");
}

void test_assembly_output() {
    std::printf("  assembly emission\n");
    ir::Function f;
    f.name = "addfn";
    f.result = ir::Type::Int39;
    ir::Statement ret;
    ret.kind = ir::StatementKind::Return;
    ret.type = ir::Type::Int39;
    ret.value = binop(ir::Op::Add, intlit(5), intlit(3));
    f.body.push_back(ret);

    bt::target::xbt::InstructionSelector sel;
    auto lowered = sel.lower_function(f);
    std::string asm_text = bt::target::xbt::emit_assembly(lowered);

    check(asm_text.find(".globl addfn") != std::string::npos,
          "assembly declares the function global");
    check(asm_text.find(".insn") != std::string::npos,
          "assembly contains encoded Xbt instructions");
    check(!lowered.code.empty(), "lowering produced instructions");
}

void test_refuses_unsupported() {
    std::printf("  honest refusal for unsupported IR\n");
    // Float64 has no Xbt lowering; the selector must refuse, not emit a stub.
    auto f_expr = std::make_shared<ir::Expr>();
    f_expr->op = ir::Op::Floating;
    f_expr->type = ir::Type::Float64;
    f_expr->floating = 1.5;

    ir::Function f;
    f.name = "f";
    f.result = ir::Type::Float64;
    ir::Statement ret;
    ret.kind = ir::StatementKind::Return;
    ret.type = ir::Type::Float64;
    ret.value = f_expr;
    f.body.push_back(ret);

    bool threw = false;
    try {
        bt::target::xbt::InstructionSelector sel;
        sel.lower_function(f);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "unsupported IR op must raise rather than emit a stub");

    // Multi-trit scalar multiplication must also be refused, because the Xbt
    // MUL is lane-wise and would silently drop inter-lane carries.
    ir::Function g;
    g.name = "g";
    g.result = ir::Type::Int39;
    ir::Statement r;
    r.kind = ir::StatementKind::Return;
    r.type = ir::Type::Int39;
    r.value = binop(ir::Op::Mul, intlit(1000), intlit(37));
    g.body.push_back(r);
    bool mul_threw = false;
    try {
        bt::target::xbt::InstructionSelector sel;
        sel.lower_function(g);
    } catch (const std::runtime_error&) {
        mul_threw = true;
    }
    check(mul_threw, "multi-trit scalar Mul must be refused (MUL is lane-wise)");
}

} // namespace

int main() {
    std::printf("Xbt instruction selection (IR -> machine code -> execution)\n");
    std::printf("===========================================================\n");

    test_constants();
    test_arithmetic();
    test_nested();
    test_assembly_output();
    test_refuses_unsupported();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    if (g_failures) {
        std::printf("XBT SELECTOR: FAILED\n");
        return 1;
    }
    std::printf("XBT SELECTOR: PASSED\n");
    return 0;
}
