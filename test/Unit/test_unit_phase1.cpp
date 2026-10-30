#include "ternary_frontend_ast.hpp"
#include "ternary_jit_engine.hpp"
#include "ternary_llvm_codegen.hpp"
#include "ternary_nonlinear_engine.hpp"
#include "ternary_universal_codec.hpp"
#include "include/Ternary/ternary_c_abi.h"
#include "bt_process.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

// Always-active checks, including in Release builds.
#define CHECK(c) do { if (!(c)) throw std::runtime_error("check failed: " #c); } while (false)
using namespace bt;
using namespace bt::frontend;
static auto parse(const std::string& s) { Parser p(s); return p.parse_program(); }
static void rejects(const std::string& src, bool native = false) {
    bool rejected = false;
    try { auto p = parse(src); TypeChecker c(native); c.typecheck(*p); }
    catch (const std::exception&) { rejected = true; }
    CHECK(rejected);
}
static std::string execute(const std::string& src) {
    std::ostringstream out;
    jit::JITConfig cfg; cfg.out = &out;
    jit::TernaryJITEngine engine(cfg); engine.execute_source(src);
    return out.str();
}
static void semantics() {
    for (bool native : {false, true}) {
        for (const auto& src : {
            "print(missing);", "x = 1;", "print(no_such_function(42));",
            "func f(x: int) -> int { return x; } print(f());",
            "func f(x: int) -> int { return x; } print(f(1, 2));",
            "func f(x: int) -> int { return x; } print(f(1.5));",
            "func f() -> int { return 1.5; }", "func f() -> int { if (1) { return 1; } }",
            "func f() { return 1; }", "var x = 1; var x = 2;",
            "func f(x: int, x: int) -> int { return x; }",
            "func bt_main() -> int { return 0; }",
            "if (1) { var local = 1; } print(local);",
            "var x = print(1);", "print(select(1, 1, print(2), 3));",
            "print(2026277576509488134);", "print(9999999999999999999999999);",
            "var a = sparse_matrix(1, 1, 2);", "var a = sparse_matrix(1, 1, missing);",
            "var a = sparse_matrix(2, 2, -1, 0, 1);"}) rejects(src, native);
    }
    rejects("var a = matrix_init(1,1,1); print(matmul(a,a));", true);
    rejects("var a = 1; func f() -> int { return a; }", true);
    auto good = parse("func f(x: float) -> float { return x; } print(f(1));");
    TypeChecker checker(true); checker.typecheck(*good);
    codegen::LLVMCodeGenVisitor generator;
    auto ir = generator.generate(*good);
    CHECK(ir.find("sitofp i64 1 to double") != std::string::npos);
    CHECK(ir.find("call double @bt.user.f(double") != std::string::npos);
    auto sel = parse("print(select(1, 1.5, 2.5, 3.5));");
    CHECK(generator.generate(*sel).find("phi double") != std::string::npos);
    CHECK(execute("func f(x: int) -> int { print(x); return x; } print(select(1, f(7), f(8), f(9)));").find("8 [") == std::string::npos);
    CHECK(execute("print(f(6)); func f(x: int) -> int { return x * 7; }").find("42 [") != std::string::npos);
    CHECK(execute("var x = 1; if (1) { var x = 2; } print(x);").find("1 [") != std::string::npos);
    CHECK(execute("print(select(-0.5, 1, 2, 3));").find("3 [") != std::string::npos);
    // Failed analysis and runtime errors must not mutate a session's root bindings.
    jit::TernaryJITEngine engine;
    engine.execute_source("var okay = 7;");
    try { engine.execute_source("okay = 9; print(missing);"); } catch (const TypeError&) {}
    CHECK(engine.get_env()->get("okay").as_int64() == 7);
    try { engine.execute_source("okay = 9; var x = 2026277576509488133 + 1;"); } catch (const std::overflow_error&) {}
    CHECK(engine.get_env()->get("okay").as_int64() == 7);
}
static void integers() {
    constexpr auto max = contract::integer_max;
    CHECK(contract::add(max, -1) == max - 1);
    CHECK(contract::sub(-max, -1) == -max + 1);
    CHECK(contract::mul(-max, -1) == max);
    for (auto src : {"var x = 2026277576509488133 + 1;", "var x = -2026277576509488133 - 1;", "var x = 2026277576509488133 * 2;"}) {
        bool threw = false; try { execute(src); } catch (const std::exception&) { threw = true; } CHECK(threw);
    }
    std::mt19937_64 rng(42);
    for (int i = 0; i < 10000; ++i) {
        int64_t a = static_cast<int64_t>(rng() % 2000001) - 1000000;
        int64_t b = static_cast<int64_t>(rng() % 2000001) - 1000000;
        CHECK(contract::add(a,b) == a+b); CHECK(contract::sub(a,b) == a-b); CHECK(contract::mul(a,b) == a*b);
    }
}
static void numerical_boundaries() {
    for (int e = -1074; e <= 1023; ++e) {
        double x = std::ldexp(1.0, e);
        double expected = 1.0 / std::sqrt(x);
        double got = nonlinear::xbt_rsqrt_householder(x);
        CHECK(std::isfinite(got)); CHECK(std::abs(got / expected - 1.0) < 3e-15);
    }
    CHECK(std::isnan(nonlinear::xbt_rsqrt_householder(-1)));
    CHECK(std::isinf(nonlinear::xbt_rsqrt_householder(0)));
    CHECK(nonlinear::xbt_rsqrt_householder(INFINITY) == 0);
    CHECK(std::isnan(nonlinear::xbt_exp_fast(NAN)));
    CHECK(nonlinear::xbt_exp_fast(-740) > 0);
    float input[] = {std::numeric_limits<float>::denorm_min(), 0, -std::numeric_limits<float>::denorm_min()};
    auto encoded = universal::BlockTernaryTensor<8>::from_float_matrix(input, 1, 3);
    float output[3]; encoded.to_float_matrix(output);
    CHECK(output[0] == input[0] && output[2] == input[2]);
    float invalid[] = {INFINITY}; bool rejected = false;
    try { universal::BlockTernaryTensor<8>::from_float_matrix(invalid, 1, 1); } catch (const std::domain_error&) { rejected = true; }
    CHECK(rejected);
}
template <class Exception, class F> static void throws_as(F operation) {
    bool threw = false;
    try { operation(); } catch (const Exception&) { threw = true; }
    CHECK(threw);
}
static void extended_boundaries() {
    rejects("var x:trit = 42;");
    rejects("var x:trit = 1; x = 2;");
    rejects("func narrow(x:int) -> trit { return x; }");
    throws_as<std::invalid_argument>([] { execute("print(zeros(-1, 2));"); });
    throws_as<std::invalid_argument>([] { execute("print(matrix_init(1, 1, 257));"); });
    throws_as<std::invalid_argument>([] { execute("print(matrix_init(1, 1, 1, 1));"); });
    throws_as<std::length_error>([] { DenseMatrixInt64(SIZE_MAX, 2); });
    throws_as<std::length_error>([] { TernaryMatrix(1, SIZE_MAX); });
    throws_as<std::length_error>([] { TCSRMatrix(SIZE_MAX, 1); });
    throws_as<std::invalid_argument>([] { TCSRMatrix::from_dense(1, 1, {2}); });
    throws_as<std::invalid_argument>([] {
        TCSRMatrix::from_triplets(1, 1, {{0,0,Trit::Pos}, {0,0,Trit::Pos}});
    });
    CHECK(execute("var a = matrix_init(1, 1, 1); print(a+a);").find("2") != std::string::npos);
    using F = BTFP<10>;
    const F hi(F::max_mantissa(), INT32_MAX), lo(F::max_mantissa(), INT32_MIN);
    CHECK((hi + lo) == hi);
    CHECK(hi.truncate_fractional(INT32_MAX).is_zero());
    CHECK(fp_detail::round_div_pow3_fast64(INT64_MIN, INT32_MAX) == 0);
    CHECK(fp_detail::round_div_pow3_fast128(-1, INT32_MAX) == 0);
    throws_as<std::overflow_error>([] { fp_detail::pow3_128(81); });
    const F base(F::max_mantissa(), 0), shifted(F::max_mantissa(), 15);
    const auto exact = static_cast<__int128_t>(base.mantissa()) * base.mantissa() +
        static_cast<__int128_t>(shifted.mantissa()) * fp_detail::pow3_128(15);
    CHECK(F::fma(base, base, shifted) == F::normalize_128(exact, 0));
    throws_as<std::overflow_error>([] { F(1, INT32_MIN); });
    throws_as<std::overflow_error>([&] { hi * hi; });
    throws_as<std::overflow_error>([&] { hi / lo; });
    throws_as<std::overflow_error>([&] { F::fma(hi, hi, lo); });
    CHECK(hi.sqrt().is_positive() && lo.sqrt().is_positive());
    throws_as<std::overflow_error>([&] { hi.exp(); });
    CHECK(F::from_double(std::numeric_limits<double>::denorm_min()).to_double() == std::numeric_limits<double>::denorm_min());
    throws_as<std::overflow_error>([] { F::normalize_128(1, INT64_MAX); });
    CHECK(BTFP<1>(1) * BTFP<1>(1) == BTFP<1>(1));
    CHECK(BTFP<40>::from_double(std::numeric_limits<double>::max()).is_positive());
    BTFP<40> wide(BTFP<40>::max_mantissa(), 0);
    CHECK(wide / wide == BTFP<40>(1));
    CHECK(BTFP<1>(5) == BTFP<1>(3));
    CHECK(BTFP<1>(6) == BTFP<1>(9));
    CHECK(wide.sqrt().is_positive()); // W40 is now covered by exact midpoint scaling
    CHECK((BTFP<10, 62>(42)).is_positive());
    throws_as<std::length_error>([] { universal::BlockTernaryTensor<8> bad(SIZE_MAX, 64); });
    universal::BlockTernaryTensor<8> damaged(1, 1);
    damaged.planes.clear(); float dst = 17;
    throws_as<std::invalid_argument>([&] { damaged.to_float_matrix(&dst); }); CHECK(dst == 17);
    throws_as<std::domain_error>([] { universal::TriRadixScalar::from_float_lossless(INFINITY); });
    throws_as<std::overflow_error>([] { universal::TriRadixScalar::from_decimal_exact(INT64_MAX, 0); });
    auto matrix = TCSRMatrix::from_dense(1, 2, {1, 1});
    int64_t input[] = {INT64_MAX, INT64_MAX}, output[] = {123};
    throws_as<std::overflow_error>([&] { matrix.spmv(input, output); }); CHECK(output[0] == 123);
    matrix.col_indices[0] = 2;
    throws_as<std::invalid_argument>([&] { matrix.spmv(input, output); }); CHECK(output[0] == 123);
    auto program = parse("var n = 0; while (n < 10) { var local = n; n = n + 1; } return n;");
    codegen::LLVMCodeGenVisitor generator;
    auto ir = generator.generate(*program);
    auto start = ir.find("define i32 @bt_main");
    auto loop = ir.find("while_cond", start);
    CHECK(start != std::string::npos && loop != std::string::npos);
    CHECK(ir.find(" = alloca ", loop) == std::string::npos);
}
static void abi() {
    uint32_t rows[] = {0,2}, columns[] = {0,1}; uint64_t signs[] = {1};
    bt_csr_t mat{1,2,2,rows,columns,signs}; int64_t x[] = {10,30}, y[] = {99};
    CHECK(bt_csr_spmv_checked(&mat,x,2,y,1,2,2,1) == BT_STATUS_OK && y[0] == -20);
    columns[1] = 2;
    CHECK(bt_csr_spmv_checked(&mat,x,2,y,1,2,2,1) == BT_STATUS_INVALID_ARGUMENT);
    CHECK(y[0] == -20); columns[1] = 1;
    CHECK(bt_csr_spmv_checked(&mat,x,1,y,1,2,2,1) == BT_STATUS_INVALID_ARGUMENT);
    x[0] = INT64_MIN; signs[0] = 0; rows[1] = 1; mat.nnz = 1;
    CHECK(bt_csr_spmv_checked(&mat,x,2,y,1,2,2,1) == BT_STATUS_OVERFLOW);
    bt_fp32_from_double(INFINITY); CHECK(bt_last_error_code() == BT_STATUS_INVALID_ARGUMENT);
    bt_clear_error(); CHECK(bt_last_error_code() == 0);
    std::thread t([] { bt_fp32_from_double(NAN); CHECK(bt_last_error_code() != 0); }); t.join();
    CHECK(bt_last_error_code() == 0);
}
static void processes() {
    host::TemporaryDirectory tmp;
    auto output = tmp.file("arguments");
    auto marker = tmp.file("must-not-exist");
    // Python receives the literal argument; a shell would run the substitution.
    CHECK(host::run_process({"python3", "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])", output, "$(touch " + marker + ")"}) == 0);
    CHECK(!std::filesystem::exists(marker));
    std::ifstream f(output); std::string text((std::istreambuf_iterator<char>(f)), {});
    CHECK(text == "$(touch " + marker + ")");
    CHECK(host::run_process({"python3", "-c", "raise SystemExit(7)"}) == 7);
}
int main() {
    try { semantics(); integers(); numerical_boundaries(); extended_boundaries(); abi(); processes(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    std::cout << "Phase 1 safety, semantics, numeric boundaries and ABI checks passed\n";
}
