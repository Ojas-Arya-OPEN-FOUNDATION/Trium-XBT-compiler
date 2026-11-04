/**
 * @file test/Unit/test_backend_registry.cpp
 * @brief Tests the retargeting seam: target descriptions and backend registry.
 *
 * The point of these tests is that a backend is *pluggable*. They check that
 * a backend can be discovered without the driver knowing its name, that the
 * same target description drives codegen decisions, and -- most importantly --
 * that a backend reports what it cannot lower instead of emitting something
 * plausible and wrong.
 */

#include "../../include/Ternary/Target/Backend.hpp"
#include "../../include/Ternary/Target/TargetDescription.hpp"
#include "../../include/Ternary/Target/XbtBackend.hpp"

#include <cstdio>
#include <memory>
#include <string>

using namespace bt;
using namespace bt::target;

namespace {

// Static registration. The point of the design is that any translation unit
// can make a backend available; the driver has no list of names. This test
// registers the Xbt backend the same way the driver does.
static bt::target::StaticBackend<bt::target::XbtBackend> xbt_backend_registration;

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::printf("  FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

// ---- IR helpers ----------------------------------------------------------

using ExprPtr = std::shared_ptr<const ir::Expr>;

ExprPtr intlit(int64_t v) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::Integer;
    e->type = ir::Type::Int39;
    e->integer = v;
    return e;
}

ExprPtr binop(ir::Op op, ExprPtr a, ExprPtr b) {
    auto e = std::make_shared<ir::Expr>();
    e->op = op;
    e->type = ir::Type::Int39;
    e->operands = {a, b};
    return e;
}

ir::Function make_fn(const std::string& name, ExprPtr body) {
    ir::Function f;
    f.name = name;
    f.result = ir::Type::Int39;
    f.parameters.push_back({1, ir::Type::Int39});
    ir::Statement r;
    r.kind = ir::StatementKind::Return;
    r.type = ir::Type::Int39;
    r.value = body;
    f.body.push_back(r);
    return f;
}

// ---- a second backend, to prove the registry is not hardcoded ------------

/**
 * A deliberately minimal backend defined only in this test. If the registry
 * were hardcoded to the Xbt backend, this could not be discovered.
 */
class FakeTargetBackend final : public Backend {
public:
    std::string name() const override { return "faketarget"; }
    const TargetDescription& target() const override {
        static TargetDescription t;
        if (t.name.empty()) {
            t.name = "fake-machine";
            t.aliases = {"fake"};
            t.native_trit_width = 27;
            t.native_layouts = {Layout::TCSR};
            t.preferred_layout = Layout::TCSR;
        }
        return t;
    }
    BackendResult lower(const ir::Program&) const override {
        BackendResult r;
        r.output = "// fake";
        r.ok = true;
        return r;
    }
};

// ---- tests ---------------------------------------------------------------

void test_registry_discovery() {
    std::printf("  registry discovery\n");
    auto& reg = BackendRegistry::instance();

    // The Xbt backend registers itself statically; nothing in the driver
    // names it at the point of registration.
    check(reg.find("xbt") != nullptr, "find by canonical name");
    check(reg.find("riscv64-xbt") != nullptr, "find by target name");
    check(reg.find("riscv-xbt") != nullptr, "find by alias");
    check(reg.find("riscv64-unknown-linux-gnu") != nullptr, "find by triple");
    check(reg.find("XBT") != nullptr, "lookup is case-insensitive");
    check(reg.find("nonexistent-target") == nullptr, "unknown target yields null");
    check(!reg.empty(), "registry is populated");
}

void test_registry_is_extensible() {
    std::printf("  registry accepts new backends\n");
    auto& reg = BackendRegistry::instance();

    check(reg.find("faketarget") == nullptr, "test backend not registered yet");
    reg.register_backend(std::make_unique<FakeTargetBackend>());

    auto* b = reg.find("faketarget");
    check(b != nullptr, "newly registered backend is discoverable");
    check(reg.find("fake") == b, "alias resolves to the same backend");
    check(b && b->target().native_trit_width == 27,
          "target description drives the backend's own parameters");
    check(b && b->target().preferred_layout == Layout::TCSR,
          "target description carries its own layout preference");
}

void test_target_description() {
    std::printf("  target description\n");
    const auto t = xbt_target_v2();

    check(t.native_trit_width == 64, "Xbt word is 64 trits");
    check(t.ternary_registers == 32, "Xbt has 32 ternary registers");
    check(t.supports_layout(Layout::DualRail), "supports dual-rail");
    check(t.supports_layout(Layout::Base243), "supports base-243");
    check(!t.supports_layout(Layout::TCSR), "does not claim T-CSR support");
    check(t.has_word_add, "has native word add");
    check(t.has_lane_mul, "has lane-wise multiply");
    check(!t.has_scalar_mul,
          "does NOT have scalar multiply (lane-wise only) -- this is what "
          "makes the selector refuse general Mul");

    const std::string d = t.describe();
    check(d.find("64 trits") != std::string::npos, "description reports word width");
    check(d.find("dual-rail") != std::string::npos, "description reports layouts");
}

void test_backend_lowers() {
    std::printf("  Xbt backend lowers IR\n");
    auto* b = BackendRegistry::instance().find("xbt");
    check(b != nullptr, "xbt backend present");

    ir::Program p;
    p.functions.push_back(make_fn("add_const", binop(ir::Op::Add, intlit(100), intlit(5))));

    auto r = b->lower(p);
    check(r.ok, "lowering succeeded");
    check(r.unsupported.empty(), "nothing unsupported");
    check(r.output.find(".insn") != std::string::npos,
          "output contains encoded instructions");
    check(r.output.find("add_const") != std::string::npos,
          "output names the function");
}

void test_backend_refuses() {
    std::printf("  Xbt backend refuses what it cannot lower\n");
    auto* b = BackendRegistry::instance().find("xbt");

    // Multi-trit scalar multiplication: the Xbt MUL is lane-wise and would
    // silently drop inter-lane carries.
    ir::Program p;
    p.functions.push_back(make_fn("bigmul", binop(ir::Op::Mul, intlit(1000), intlit(37))));

    auto r = b->lower(p);
    check(!r.unsupported.empty(), "unsupported construct is reported");
    check(r.unsupported.size() == 1, "exactly one function reported");
    check(r.output.find(".globl bigmul") == std::string::npos,
          "no code is emitted for a function that cannot be lowered");
    check(r.unsupported[0].find("lane-wise") != std::string::npos,
          "the reason names the actual semantic problem");
}

void test_mixed_program_reports_partial() {
    std::printf("  mixed program reports partial lowering\n");
    auto* b = BackendRegistry::instance().find("xbt");

    ir::Program p;
    p.functions.push_back(make_fn("ok_add", binop(ir::Op::Add, intlit(1), intlit(2))));
    p.functions.push_back(make_fn("bad_mul", binop(ir::Op::Mul, intlit(100), intlit(100))));

    auto r = b->lower(p);
    check(r.output.find("ok_add") != std::string::npos,
          "the lowerable function is still emitted");
    check(r.unsupported.size() == 1, "only the bad function is reported");
    check(!r.complete(), "result is marked incomplete");
}

} // namespace

int main() {
    std::printf("Backend registry and target descriptions\n");
    std::printf("========================================\n");

    test_registry_discovery();
    test_registry_is_extensible();
    test_target_description();
    test_backend_lowers();
    test_backend_refuses();
    test_mixed_program_reports_partial();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    if (g_failures) {
        std::printf("BACKEND REGISTRY: FAILED\n");
        return 1;
    }
    std::printf("BACKEND REGISTRY: PASSED\n");
    return 0;
}
