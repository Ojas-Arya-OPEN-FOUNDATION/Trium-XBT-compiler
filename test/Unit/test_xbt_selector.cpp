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
    // Load highest-numbered parameter first: every materialisation lands in
    // t1, so going in reverse means t1 is only clobbered after the higher
    // slots have already been copied out of it.
    for (size_t k = params.size(); k-- > 0;) {
        // materialise each parameter the way the selector would for a constant
        bt::ir::Function one;
        one.name = "p";
        one.result = bt::ir::Type::Int39;
        ir::Statement r;
        r.kind = bt::ir::StatementKind::Return;
        r.type = bt::ir::Type::Int39;
        r.value = intlit(params[k]);
        one.body.push_back(r);
        bt::target::xbt::InstructionSelector s2;
        auto lp = s2.lower_function(one);
        for (const auto& ins : lp.code) cpu.execute(encode(ins));
        cpu.treg[k + 1] = cpu.treg[1];
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

/**
 * Build a SignSelect:  select(cond, pos, zero, neg).
 * Operand 0 is the condition, 1..3 are the arms (thunks in the core IR).
 */
ExprPtr sign_select(ExprPtr cond, ExprPtr pos, ExprPtr zero, ExprPtr negv) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::SignSelect;
    e->type = ir::Type::Int39;
    e->operands = {cond, pos, zero, negv};
    return e;
}

ExprPtr int_cast(ExprPtr a) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::IntCast;
    e->type = ir::Type::Int39;
    e->operands = {a};
    return e;
}

void test_comparisons() {
    std::printf("  comparisons (six relational ops, all lowered via SIGN+CMOV)\n");
    // Core comparisons yield 1 when true and -1 when false -- never 0.
    struct Case { int64_t a, b; };
    const Case cases[] = {{0,0},{1,0},{0,1},{-1,1},{1,-1},{5,5},{-5,5},
                          {1000,999},{999,1000},{-1000,-999},{243,81},{81,243}};

    struct Rel { ir::Op op; const char* name; };
    const Rel rels[] = {
        {ir::Op::Greater,      ">" },
        {ir::Op::Equal,        "=="},
        {ir::Op::Less,         "<" },
        {ir::Op::NotEqual,     "!="},
        {ir::Op::GreaterEqual, ">="},
        {ir::Op::LessEqual,    "<="},
    };

    for (const auto& c : cases) {
        for (const auto& r : rels) {
            bool truth;
            switch (r.op) {
                case ir::Op::Greater:      truth = c.a >  c.b; break;
                case ir::Op::Equal:        truth = c.a == c.b; break;
                case ir::Op::Less:         truth = c.a <  c.b; break;
                case ir::Op::NotEqual:     truth = c.a != c.b; break;
                case ir::Op::GreaterEqual: truth = c.a >= c.b; break;
                default:                   truth = c.a <= c.b; break;
            }
            int64_t want = truth ? 1 : -1;
            int64_t got  = run(binop(r.op, intlit(c.a), intlit(c.b)), {});
            check(got == want, std::string(r.name) + " on " + std::to_string(c.a) +
                               "," + std::to_string(c.b) + " -> " +
                               std::to_string(got) + " want " + std::to_string(want));
        }
    }
}

void test_comparisons_with_parameters() {
    std::printf("  comparisons against function parameters (not just constants)\n");
    const int64_t vals[] = {-2,-1,0,1,2,7,-7,100,-100};
    for (int64_t x : vals) {
        for (int64_t y : vals) {
            int64_t got = run(binop(ir::Op::Less, param(1), param(2)), {x, y});
            check(got == (x < y ? 1 : -1),
                  "param<param " + std::to_string(x) + "<" + std::to_string(y) +
                  " -> " + std::to_string(got));
        }
    }
}

void test_spaceship() {
    std::printf("  spaceship (<=>) yields -1 / 0 / +1\n");
    const int64_t vals[] = {-121,-5,-1,0,1,5,121,1000,-1000};
    for (int64_t a : vals) {
        for (int64_t b : vals) {
            int64_t want = (a > b) - (a < b);
            int64_t got  = run(binop(ir::Op::Spaceship, intlit(a), intlit(b)), {});
            check(got == want, std::to_string(a) + " <=> " + std::to_string(b) +
                               " -> " + std::to_string(got) + " want " +
                               std::to_string(want));
        }
    }
    // And with parameters, which is the shape the .bt frontend actually emits.
    const int64_t pv[] = {-3,-1,0,1,3};
    for (int64_t x : pv) {
        int64_t want = (x > 0) - (x < 0);
        int64_t got  = run(binop(ir::Op::Spaceship, param(1), intlit(0)), {x});
        check(got == want, std::to_string(x) + " <=> 0 -> " + std::to_string(got));
    }
}

void test_sign_select() {
    std::printf("  sign-select (three-way select, lowered to CMOVZ/CMOVN)\n");
    const int64_t conds[] = {-9,-1,0,1,9};

    // Constant arms: select(c, 1, 0, -1) is exactly sign(c).
    for (int64_t c : conds) {
        int64_t want = (c > 0) ? 1 : ((c == 0) ? 0 : -1);
        int64_t got  = run(sign_select(intlit(c), intlit(1), intlit(0), intlit(-1)), {});
        check(got == want, "select(" + std::to_string(c) + ",1,0,-1) -> " +
                           std::to_string(got) + " want " + std::to_string(want));
    }

    // Distinct, asymmetric arms so a mis-wired cmov cannot accidentally pass.
    for (int64_t c : conds) {
        int64_t want = (c > 0) ? 77 : ((c == 0) ? 42 : -13);
        int64_t got  = run(sign_select(intlit(c), intlit(77), intlit(42), intlit(-13)), {});
        check(got == want, "select(" + std::to_string(c) + ",77,42,-13) -> " +
                           std::to_string(got) + " want " + std::to_string(want));
    }

    // Condition driven by a parameter via <=> 0, i.e. the .bt frontend shape.
    for (int64_t x : conds) {
        auto cond = binop(ir::Op::Spaceship, param(1), intlit(0));
        int64_t want = (x > 0) ? 77 : ((x == 0) ? 42 : -13);
        int64_t got  = run(sign_select(cond, intlit(77), intlit(42), intlit(-13)), {x});
        check(got == want, "select(" + std::to_string(x) + "<=>0, ...) -> " +
                           std::to_string(got) + " want " + std::to_string(want));
    }
}

void test_int_cast() {
    std::printf("  IntCast forwards its operand (value-preserving on Int39)\n");
    const int64_t vals[] = {0,1,-1,5,-5,1000,-1000,123456,-123456};
    for (int64_t v : vals) {
        int64_t got = run(int_cast(intlit(v)), {});
        check(got == v, "intcast " + std::to_string(v) + " -> " + std::to_string(got));
    }
    for (int64_t x : vals) {
        int64_t got = run(int_cast(param(1)), {x});
        check(got == x, "intcast(param " + std::to_string(x) + ") -> " +
                        std::to_string(got));
    }
}

void test_sign_and_cmov_encoded() {
    std::printf("  SIGN and CMOV appear in the emitted assembly\n");
    ir::Function f;
    f.name = "classify";
    f.result = ir::Type::Int39;
    f.parameters.push_back({1, ir::Type::Int39});
    ir::Statement ret;
    ret.kind = ir::StatementKind::Return;
    ret.type = ir::Type::Int39;
    ret.value = sign_select(binop(ir::Op::Spaceship, param(1), intlit(0)),
                            intlit(1), intlit(0), intlit(-1));
    f.body.push_back(ret);

    bt::target::xbt::InstructionSelector sel;
    auto lowered = sel.lower_function(f);
    std::string asm_text = bt::target::xbt::emit_assembly(lowered);

    check(asm_text.find("xbt.sign") != std::string::npos,
          "assembly contains xbt.sign");
    check(asm_text.find("cmovz") != std::string::npos, "assembly contains cmovz");
    check(asm_text.find("cmovn") != std::string::npos, "assembly contains cmovn");

    // The lowering must be branch-free. Check the instruction comments only,
    // so the function's own `label:` line does not trip the assertion.
    bool branch_free = true;
    for (const auto& ins : lowered.code) {
        const std::string& c = ins.comment;
        if (c.rfind("j ", 0) == 0 || c.rfind("beq", 0) == 0 ||
            c.rfind("bne", 0) == 0 || c.rfind("blt", 0) == 0 ||
            c.rfind("bge", 0) == 0 || c.rfind("jal", 0) == 0 ||
            c.rfind("branch", 0) == 0)
            branch_free = false;
    }
    check(branch_free, "select lowering is branch-free");

    // Every emitted instruction must be a legal Xbt encoding.
    bool all_valid = true;
    for (const auto& ins : lowered.code) {
        auto d = decode_xbt_r(encode(ins));
        if (!d.is_valid_xbt()) all_valid = false;
    }
    check(all_valid, "all emitted instructions decode as valid Xbt");
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
    test_comparisons();
    test_comparisons_with_parameters();
    test_spaceship();
    test_sign_select();
    test_int_cast();
    test_sign_and_cmov_encoded();
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
