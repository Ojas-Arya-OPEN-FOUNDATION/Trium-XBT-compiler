/**
 * @file test_unit_scalar_mul.cpp
 * @brief P1 scalar multiply: SHL3, the shift-add lowering, and target routing.
 *
 * `xbt.mul` is lane-wise and drops inter-lane carries, so the selector used to
 * refuse general multiplication outright. This closes the constant-multiplier
 * case with a shift-add lowering and routes on the target's `has_scalar_mul`.
 *
 * Counterpart proofs: verification/verify_scalar_mul.py (S1-S7, 9/9).
 */

#include "../../hw/xbt_alu_model.hpp"
#include "Ternary/Target/GeneratedTargets.hpp"
#include "Ternary/Target/XbtInstructionSelector.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bt;
using namespace bt::hw;
using namespace bt::target;
using namespace bt::target::xbt;

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

// ---- IR helpers -----------------------------------------------------------

using ExprPtr = std::shared_ptr<const ir::Expr>;

ExprPtr intlit(int64_t v) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::Integer; e->type = ir::Type::Int39; e->integer = v;
    return e;
}
ExprPtr param(uint32_t sym) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::Parameter; e->type = ir::Type::Int39; e->symbol = sym;
    return e;
}
ExprPtr mul(ExprPtr a, ExprPtr b) {
    auto e = std::make_shared<ir::Expr>();
    e->op = ir::Op::Mul; e->type = ir::Type::Int39; e->operands = {a, b};
    return e;
}

uint32_t encode(const Insn& i) {
    if (i.is_itype)
        return encode_xbt_i(static_cast<uint16_t>(i.imm), i.rs1, i.funct3, i.rd);
    return encode_xbt_r(i.funct7, i.rs2, i.rs1, i.funct3, i.rd);
}

int64_t word_value(const TritWord64& w) {
    int64_t v = 0, p = 1;
    for (int i = 0; i < 40; ++i) {
        if ((w.P >> i) & 1ULL) v += p;
        else if ((w.N >> i) & 1ULL) v -= p;
        if (i < 39) p *= 3;
    }
    return v;
}

/** Full 64-trit value. word_value() above stops at 40 trits because int39
 *  scalars only need that; SHL3 works on the whole word, so it needs all 64. */
__int128 word_value_full(const TritWord64& w) {
    __int128 v = 0, p = 1;
    for (int i = 0; i < 64; ++i) {
        if ((w.P >> i) & 1ULL) v += p;
        else if ((w.N >> i) & 1ULL) v -= p;
        p *= 3;
    }
    return v;
}

TritWord64 encode_trits(const int* t, int w) {
    uint64_t P = 0, N = 0;
    for (int i = 0; i < w; ++i) {
        if (t[i] > 0) P |= (1ULL << i);
        else if (t[i] < 0) N |= (1ULL << i);
    }
    return TritWord64(P, N);
}

// ---- T1: shl3() against a reference ---------------------------------------

void test_shl3_reference() {
    std::printf("  shl3() exactness and overflow (vs integer reference)\n");
    //
    // IMPORTANT: shl3() guards the 64-TRIT WORD, not the logical width of the
    // value being shifted. A 12-trit value shifted by 12 does not overflow a
    // 64-trit register; only a trit leaving bit 63 does. That is the right
    // level for a word-level primitive -- the scalar-range check belongs to
    // the lowering, which must refuse a product that exceeds int39.
    const int W = 64;
    int bad = 0, exact_cases = 0, overflow_cases = 0;

    // Single-trit words at every position, both signs -- this is what
    // actually determines the overflow boundary.
    for (int pos = 0; pos < W; ++pos) {
        for (int sign = -1; sign <= 1; sign += 2) {
            int trits[W] = {0};
            trits[pos] = sign;
            TritWord64 in = encode_trits(trits, W);
            int64_t va = sign * 1;             // 3^pos handled below
            (void)va;

            for (int k = 0; k <= 64; ++k) {
                TritWord64 out;
                bool ok = shl3(in, k, out);
                const bool truth_overflow = (pos + k >= W);

                if (ok != !truth_overflow) { ++bad; continue; }
                if (truth_overflow) { ++overflow_cases; continue; }

                // Exactness: the shifted word must equal sign * 3^(pos+k).
                // Compare against the model's own word value.
                __int128 want = 1, p = 3;
                for (int j = 0; j < pos + k; ++j) { want = p; p *= 3; }
                want *= sign;
                __int128 got = word_value_full(out);
                if (got != want) ++bad;
                ++exact_cases;
            }
        }
    }
    check(bad == 0, "shl3 == multiplication by 3^k, overflow detected exactly");
    check(exact_cases > 1000, "the sweep exercised exactness cases");
    check(overflow_cases > 100, "the sweep exercised overflow cases");
}

void test_shl3_executes() {
    std::printf("  SHL3 executes on the coprocessor model\n");
    RiscvXbtCoprocessor cpu;
    // Materialise 5 in t1 (5 = 9 - 3 - 1 -> trits +1 at 0? no: 5 = 1*9 -1*3 -1*1)
    // Simpler: build t1 directly.
    cpu.treg[1] = TritWord64(0b000000001001ULL, 0b000000000010ULL); // 9 - 3 - ... let's just use 3
    cpu.treg[1] = TritWord64(1ULL << 1, 0ULL);                       // value 3
    // SHL3 t2, t1, 2  ->  3 * 9 = 27
    Insn i;
    i.funct3 = static_cast<uint8_t>(Funct3::GROUP2);
    i.funct7 = static_cast<uint8_t>(Group2::SHL3);
    i.rd = 2; i.rs1 = 1; i.rs2 = 2;
    cpu.execute(encode_xbt_r(i.funct7, i.rs2, i.rs1, i.funct3, i.rd));
    check(word_value(cpu.treg[2]) == 27, "SHL3 t2, t1, 2 gives 3*9 == 27");

    // Overflow must throw, not wrap. A trit at position 60 shifted by 5
    // leaves the 64-trit word.
    cpu.treg[3] = TritWord64(1ULL << 60, 0ULL);
    Insn j;
    j.funct3 = static_cast<uint8_t>(Funct3::GROUP2);
    j.funct7 = static_cast<uint8_t>(Group2::SHL3);
    j.rd = 4; j.rs1 = 3; j.rs2 = 5;
    bool threw = false;
    try { cpu.execute(encode_xbt_r(j.funct7, j.rs2, j.rs1, j.funct3, j.rd)); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "SHL3 throws on overflow instead of wrapping");

    // And just inside the boundary it must succeed.
    cpu.treg[5] = TritWord64(1ULL << 60, 0ULL);
    Insn m;
    m.funct3 = static_cast<uint8_t>(Funct3::GROUP2);
    m.funct7 = static_cast<uint8_t>(Group2::SHL3);
    m.rd = 6; m.rs1 = 5; m.rs2 = 3;            // 60 + 3 == 63, still in range
    bool ok63 = true;
    try { cpu.execute(encode_xbt_r(m.funct7, m.rs2, m.rs1, m.funct3, m.rd)); }
    catch (const std::invalid_argument&) { ok63 = false; }
    check(ok63, "SHL3 succeeds when the top trit lands exactly at bit 63");
}

// ---- T2: end-to-end constant multiply -------------------------------------

/** Run `expr` through a target-aware selector and the model; return the value. */
int64_t run(ExprPtr expr, int64_t param_value,
            const MachineDescription* machine = nullptr) {
    ir::Function f;
    f.name = "testfn";
    f.result = ir::Type::Int39;
    f.parameters.push_back({1u, ir::Type::Int39});

    ir::Statement ret;
    ret.kind = ir::StatementKind::Return;
    ret.type = ir::Type::Int39;
    ret.value = expr;
    f.body.push_back(ret);

    InstructionSelector sel = machine ? InstructionSelector(*machine)
                                      : InstructionSelector();
    auto lowered = sel.lower_function(f);

    RiscvXbtCoprocessor cpu;
    // Materialise the parameter value into t1 via a constant-only function.
    ir::Function one;
    one.name = "p"; one.result = ir::Type::Int39;
    ir::Statement r;
    r.kind = ir::StatementKind::Return;
    r.type = ir::Type::Int39;
    r.value = intlit(param_value);
    one.body.push_back(r);
    InstructionSelector s2;
    auto lp = s2.lower_function(one);
    for (const auto& ins : lp.code) cpu.execute(encode(ins));

    for (const auto& ins : lowered.code) cpu.execute(encode(ins));
    return word_value(cpu.treg[1]);
}

void test_constant_multiply() {
    std::printf("  constant multiply: shift-add lowering vs C++ arithmetic\n");

    const int64_t kvals[] = {0, 1, -1, 2, -2, 3, -3, 4, 5, -5, 7, -7,
                             9, -9, 10, -10, 13, 40, -40, 121, -121};
    const int64_t avals[] = {0, 1, -1, 2, -2, 3, -3, 5, -5, 12, -12,
                             100, -100, 1234, -1234};

    int bad = 0, tested = 0;
    for (int64_t k : kvals) {
        for (int64_t a : avals) {
            int64_t want = a * k;
            // Stay inside int39: the reference word_value only reads 40 trits.
            const int64_t lim = 4052555153018976267LL / 3;   // 3^39 / 3
            if (want > lim || want < -lim) continue;
            int64_t got = run(mul(param(1), intlit(k)), a);
            ++tested;
            if (got != want) {
                if (bad < 5)
                    std::printf("      a=%lld k=%lld want=%lld got=%lld\n",
                                (long long)a, (long long)k,
                                (long long)want, (long long)got);
                ++bad;
            }
        }
    }
    check(bad == 0, "constant multiply is exact on the model");
    check(tested > 200, "the sweep covered a meaningful number of cases");

    // Constant on the left must work identically (commuted).
    int bad2 = 0;
    for (int64_t k : {2LL, -3LL, 7LL, 40LL, -121LL}) {
        for (int64_t a : {1LL, -5LL, 12LL, -100LL}) {
            int64_t want = a * k;
            int64_t got = run(mul(intlit(k), param(1)), a);
            if (got != want) ++bad2;
        }
    }
    check(bad2 == 0, "constant on either side gives the same result");
}

void test_target_routing() {
    std::printf("  has_scalar_mul routing (Xbt vs ternary-torus)\n");
    const MachineDescription* xbt = find_machine("riscv64-xbt");
    const MachineDescription* torus = find_machine("ternary-torus");
    check(xbt && torus, "both targets exist");
    if (!xbt || !torus) return;
    check(!xbt->has_scalar_mul, "Xbt declares no scalar multiply");
    check(torus->has_scalar_mul, "ternary-torus declares a scalar multiply");

    // Inspect the emitted code rather than trusting the result alone.
    auto lower_with = [](const MachineDescription* m, int64_t k) {
        ir::Function f;
        f.name = "f"; f.result = ir::Type::Int39;
        f.parameters.push_back({1u, ir::Type::Int39});
        ir::Statement r;
        r.kind = ir::StatementKind::Return; r.type = ir::Type::Int39;
        r.value = mul(param(1), intlit(k));
        f.body.push_back(r);
        InstructionSelector sel(*m);
        return sel.lower_function(f);
    };

    auto has_group2 = [](const LoweredFunction& lf, Group2 g) {
        for (const auto& i : lf.code)
            if (i.funct3 == static_cast<uint8_t>(Funct3::GROUP2) &&
                i.funct7 == static_cast<uint8_t>(g)) return true;
        return false;
    };

    LoweredFunction on_xbt = lower_with(xbt, 7);
    LoweredFunction on_torus = lower_with(torus, 7);
    check(has_group2(on_xbt, Group2::SHL3),
          "Xbt lowers  a*7  to SHL3 partial products");
    check(!has_group2(on_torus, Group2::SHL3),
          "ternary-torus does not need the shift-add path");
    check(has_group2(on_torus, Group2::MUL),
          "ternary-torus emits the native MUL");
}

void test_runtime_refusal() {
    std::printf("  runtime x runtime is refused, not silently wrong\n");
    // Two runtime values cannot be decomposed: extracting a runtime operand's
    // trits is impossible without a trit-extract instruction the ISA lacks.
    bool threw = false;
    std::string msg;
    try {
        ir::Function f;
        f.name = "f"; f.result = ir::Type::Int39;
        f.parameters.push_back({1u, ir::Type::Int39});
        f.parameters.push_back({2u, ir::Type::Int39});
        ir::Statement r;
        r.kind = ir::StatementKind::Return; r.type = ir::Type::Int39;
        r.value = mul(param(1), param(2));
        f.body.push_back(r);
        InstructionSelector sel;
        (void)sel.lower_function(f);
    } catch (const std::exception& e) {
        threw = true;
        msg = e.what();
    }
    check(threw, "runtime x runtime multiply is refused");
    check(msg.find("constant") != std::string::npos,
          "the refusal explains what would have to be true");
}

}  // namespace

int main() {
    std::printf("P1 scalar multiply\n");
    test_shl3_reference();
    test_shl3_executes();
    test_constant_multiply();
    test_target_routing();
    test_runtime_refusal();
    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
