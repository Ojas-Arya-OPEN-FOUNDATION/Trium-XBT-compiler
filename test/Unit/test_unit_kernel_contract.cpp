// Kernel contract: the seam between the GLSL backend and any device runtime.
//
// The Vulkan runtime hard-codes 3 storage bindings and a 4-byte push range.
// That is enough for the elementwise map kernel and not enough for the GEMM
// kernel (5 bindings, 20 bytes), which is why a real kernel cannot be
// dispatched today. KernelContract turns that layout into data so the runtime
// can allocate what a kernel actually declares.
//
// These tests run with no GPU and no Vulkan SDK. They check the part of "can
// this be launched" that is checkable on the host: that emitted GLSL agrees
// with the contract it claims to satisfy. Each check has a negative control,
// so a check that cannot fail is not trusted.

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "Ternary/IR/Core.hpp"
#include "Ternary/IR/GLSL.hpp"
#include "Ternary/IR/Program.hpp"

using namespace bt;
using namespace bt::ir;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("  FAIL: %s\n", what.c_str()); ++failures; }
    else     { std::printf("  ok:   %s\n", what.c_str()); }
}

std::shared_ptr<const Expr> sym(uint32_t id, Type t) {
    auto e = std::make_shared<Expr>();
    e->op = Op::Parameter; e->type = t; e->symbol = id;
    return e;
}
std::shared_ptr<const Expr> konst(int64_t v) {
    auto e = std::make_shared<Expr>();
    e->op = Op::Integer; e->type = Type::Int39; e->integer = v;
    return e;
}
std::shared_ptr<const Expr> binary(Op op, std::shared_ptr<const Expr> a,
                                   std::shared_ptr<const Expr> b) {
    auto e = std::make_shared<Expr>();
    e->op = op; e->type = Type::Int39;
    e->operands = {a, b};
    return e;
}

// func transform(x) { var scaled = x * 3; return scaled + 1; }
Program build_program() {
    auto x = sym(1, Type::Int39);
    auto three = konst(3);
    auto one = konst(1);

    Statement let;
    let.kind = StatementKind::Let;
    let.symbol = 2;
    let.type = Type::Int39;
    let.value = binary(Op::Mul, x, three);

    Statement ret;
    ret.kind = StatementKind::Return;
    ret.type = Type::Int39;
    ret.value = binary(Op::Add, sym(2, Type::Int39), one);

    Function f;
    f.name = "transform";
    f.result = Type::Int39;
    f.parameters = {{1, Type::Int39}};
    f.body = {let, ret};

    Program p;
    p.functions = {f};
    return p;
}

void test_elementwise_matches_contract() {
    std::printf("\n[contract] elementwise map kernel matches its contract\n");
    const Program p = build_program();
    const std::string glsl = GLSLEmitter{}.generate(p, "transform");

    const KernelContract c = KernelContract::elementwise_map("bt_transform", 64);
    const auto problems = validate_against_contract(glsl, c);
    for (const auto& s : problems) std::printf("    problem: %s\n", s.c_str());
    check(problems.empty(), "emitted GLSL satisfies the elementwise contract");
    check(c.push_bytes() == 4, "elementwise push block is 4 bytes");
    check(c.bindings.size() == 2, "elementwise declares 2 bindings");
}

void test_contract_catches_disagreements() {
    std::printf("\n[contract] negative controls: each mutation must be detected\n");
    const Program p = build_program();
    const std::string glsl = GLSLEmitter{}.generate(p, "transform");
    const KernelContract base = KernelContract::elementwise_map("bt_transform", 64);

    struct Mut { const char* what; KernelContract c; };
    std::vector<Mut> mutants;

    { auto c = base; c.bindings[0].name = "wrong_name";          mutants.push_back({"renamed binding", c}); }
    { auto c = base; c.bindings.push_back({2, "extra", "int64_t", true});
                                                                   mutants.push_back({"extra binding", c}); }
    { auto c = base; c.bindings[0].readonly = false;              mutants.push_back({"in buffer made writable", c}); }
    { auto c = base; c.bindings[1].readonly = true;               mutants.push_back({"out buffer made readonly", c}); }
    { auto c = base; c.push_fields.push_back({"uint", "extra"});  mutants.push_back({"extra push field", c}); }
    { auto c = base; c.workgroup = 128;                           mutants.push_back({"wrong workgroup size", c}); }
    { auto c = base; c.entry_point = "not_main";                  mutants.push_back({"wrong entry point", c}); }
    { auto c = base; c.needs_fp64 = true;                         mutants.push_back({"unmet fp64 requirement", c}); }

    for (const auto& m : mutants) {
        const auto problems = validate_against_contract(glsl, m.c);
        check(!problems.empty(), std::string("detected: ") + m.what);
    }
}

void test_gemm_contract_shape() {
    std::printf("\n[contract] GEMM contract shape (not yet emitted)\n");
    const KernelContract g = KernelContract::packed_gemm();
    check(g.bindings.size() == 5, "GEMM declares 5 bindings");
    check(g.push_bytes() == 20, "GEMM push block is 20 bytes");
    check(g.needs_int64, "GEMM requires int64");

    // The whole reason this contract exists: it must NOT fit the map binding
    // layout the Vulkan runtime hard-codes today.
    const KernelContract map = KernelContract::elementwise_map("bt_transform");
    check(g.bindings.size() > map.bindings.size(),
          "GEMM needs more bindings than the elementwise map");
    check(g.push_bytes() > map.push_bytes(),
          "GEMM needs more push bytes than the elementwise map");

    // And it must not validate against a shader that was not built for it.
    const Program p = build_program();
    const std::string glsl = GLSLEmitter{}.generate(p, "transform");
    const auto problems = validate_against_contract(glsl, g);
    check(!problems.empty(),
          "GEMM contract is correctly rejected against the elementwise shader");
}

void test_contract_is_honest_about_size() {
    std::printf("\n[contract] push_bytes accounts for field widths\n");
    KernelContract c;
    c.push_fields = {{"uint", "a"}, {"uint64_t", "b"}};
    check(c.push_bytes() == 12, "uint + uint64_t is 12 bytes");
    c.push_fields = {{"double", "d"}};
    check(c.push_bytes() == 8, "double is 8 bytes");
    c.push_fields = {};
    check(c.push_bytes() == 0, "no fields is 0 bytes");
}

}  // namespace

int main() {
    std::printf("Kernel contract tests\n");
    test_elementwise_matches_contract();
    test_contract_catches_disagreements();
    test_gemm_contract_shape();
    test_contract_is_honest_about_size();
    if (failures) {
        std::printf("\nFAIL: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nPASS: kernel contract\n");
    return 0;
}
