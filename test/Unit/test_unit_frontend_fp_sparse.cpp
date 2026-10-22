/**
 * @file test_unit_frontend_fp_sparse.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Frontend BT-FP32 & T-CSR Sparse Matrix
 * @author Antigravity Prime Research & Engineering Engine
 * 
 * Verifies:
 *  1. Frontend Lexer & AST for Float Literals (FloatLiteralExprAST) with BTFP32 precision.
 *  2. Frontend Sparse Matrix Declarations (SparseMatrixDeclAST & TCSRMatrix).
 *  3. TypeChecker / Semantic Analysis for float, trit, tritword, and sparse_matrix.
 *  4. LLVM IR Code Generation (multiplier-free SpMV loop, float arithmetic instructions).
 *  5. Bare-Metal Execution via NativeLLVMJIT.
 */

#include "../../ternary_frontend_ast.hpp"
#include "../../ternary_llvm_codegen.hpp"
#include "../../ternary_jit_engine.hpp"
#include "../../ternary_kokkos_bridge.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <future>
#include <atomic>

using namespace bt;
using namespace bt::frontend;
using namespace bt::codegen;

// ==============================================================================
// 1. Float Lexing & AST Verification
// ==============================================================================

void test_float_lexer_and_ast() {
    std::cout << "  [1/5] Testing Float Literal Lexing and FloatLiteralExprAST...\n";

    std::string src = "var a = 3.5;\n"
                      "var b = 0.75;\n"
                      "var c = 1.2e-3;\n"
                      "var d = -0.5;\n"
                      "var e = 100.0;\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);
    assert(prog->declarations.size() == 5);

    // Verify 'a = 3.5'
    auto var_a = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
    assert(var_a != nullptr);
    auto lit_a = std::dynamic_pointer_cast<FloatLiteralExprAST>(var_a->init_expr);
    assert(lit_a != nullptr);
    assert(std::abs(lit_a->value - 3.5) < 1e-9);
    assert(lit_a->raw_text == "3.5");
    // Verify BTFP32 conversion
    double btfp_val_a = lit_a->btfp_val.to_double();
    assert(std::abs(btfp_val_a - 3.5) < 0.05);

    // Verify 'b = 0.75'
    auto var_b = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[1]);
    assert(var_b != nullptr);
    auto lit_b = std::dynamic_pointer_cast<FloatLiteralExprAST>(var_b->init_expr);
    assert(lit_b != nullptr);
    assert(std::abs(lit_b->value - 0.75) < 1e-9);

    // Verify 'c = 1.2e-3'
    auto var_c = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[2]);
    assert(var_c != nullptr);
    auto lit_c = std::dynamic_pointer_cast<FloatLiteralExprAST>(var_c->init_expr);
    assert(lit_c != nullptr);
    assert(std::abs(lit_c->value - 1.2e-3) < 1e-7);

    // Verify 'd = -0.5'
    auto var_d = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[3]);
    assert(var_d != nullptr);
    auto lit_d = std::dynamic_pointer_cast<FloatLiteralExprAST>(var_d->init_expr);
    assert(lit_d != nullptr);
    assert(std::abs(lit_d->value - (-0.5)) < 1e-9);

    // Verify scientific exponent lookahead guard (Lexer::is_float_ahead)
    Lexer lex_incomplete_e("123e");
    assert(!lex_incomplete_e.is_float_ahead() && "'123e' without digits must not be recognized as float!");
    Lexer lex_incomplete_sign("42e+");
    assert(!lex_incomplete_sign.is_float_ahead() && "'42e+' without digits must not be recognized as float!");
    Lexer lex_valid_sci("42e+3");
    assert(lex_valid_sci.is_float_ahead() && "'42e+3' must be recognized as float!");

    std::cout << "      -> Float Literal Lexing & AST passed.\n";
}

// ==============================================================================
// 2. Sparse Matrix Parsing & AST Verification
// ==============================================================================

void test_sparse_matrix_parsing_and_ast() {
    std::cout << "  [2/5] Testing Sparse Matrix Parsing and SparseMatrixDeclAST...\n";

    // 1. Dense constructor format
    std::string src1 = "sparse_matrix A(2, 2, 1, 0, 0, -1);\n";
    Parser parser1(src1);
    auto prog1 = parser1.parse_program();
    assert(prog1 != nullptr);
    assert(prog1->declarations.size() == 1);

    auto smat1 = std::dynamic_pointer_cast<SparseMatrixDeclAST>(prog1->declarations[0]);
    assert(smat1 != nullptr);
    assert(smat1->name == "A");
    assert(smat1->rows == 2);
    assert(smat1->cols == 2);
    assert(smat1->matrix.nnz == 2);
    assert(smat1->matrix.row_ptr.size() == 3);
    assert(smat1->matrix.row_ptr[0] == 0);
    assert(smat1->matrix.row_ptr[1] == 1);
    assert(smat1->matrix.row_ptr[2] == 2);
    assert(smat1->matrix.col_indices[0] == 0);
    assert(smat1->matrix.col_indices[1] == 1);
    // Sign for +1 is 1, sign for -1 is 0
    assert((smat1->matrix.signs[0] & 1) == 1);
    assert(((smat1->matrix.signs[0] >> 1) & 1) == 0);

    // 2. Expression constructor in var declaration (triplets format: 4x4 with 3 triplets)
    std::string src2 = "var B = sparse_matrix(4, 4, 0, 1, 1, 2, 3, -1, 3, 0, 1);\n";
    Parser parser2(src2);
    auto prog2 = parser2.parse_program();
    assert(prog2 != nullptr);
    assert(prog2->declarations.size() == 1);

    auto var2 = std::dynamic_pointer_cast<VarDeclNode>(prog2->declarations[0]);
    assert(var2 != nullptr);
    auto smat2 = std::dynamic_pointer_cast<SparseMatrixDeclAST>(var2->init_expr);
    assert(smat2 != nullptr);
    assert(smat2->rows == 4);
    assert(smat2->cols == 4);
    assert(smat2->matrix.nnz == 3);

    std::cout << "      -> Sparse Matrix Parsing & AST passed.\n";
}

// ==============================================================================
// 3. TypeChecker Semantic Analysis
// ==============================================================================

void test_typechecker_sema() {
    std::cout << "  [3/5] Testing TypeChecker Semantic Analysis...\n";

    // Valid float expressions
    {
        std::string src = "var x: float = 3.5;\n"
                          "var y: float = 2.0;\n"
                          "var z: float = x + y;\n"
                          "var w: float = x * y - z;\n";
        Parser p(src);
        auto prog = p.parse_program();
        assert(prog != nullptr);
        TypeChecker tc;
        tc.typecheck(*prog);
        assert(tc.errors().empty());
        assert(tc.get_symbol_type("x") == BTType::Float);
        assert(tc.get_symbol_type("z") == BTType::Float);
    }

    // Valid sparse matrix expression
    {
        std::string src = "var A = sparse_matrix(2, 2, 1, 0, 0, -1);\n"
                          "var v = vector_init(10, 20);\n"
                          "var y = A * v;\n";
        Parser p(src);
        auto prog = p.parse_program();
        assert(prog != nullptr);
        TypeChecker tc;
        tc.typecheck(*prog);
        assert(tc.errors().empty());
        assert(tc.get_symbol_type("A") == BTType::SparseMatrix);
    }

    // Negative test 1: sparse_matrix + float is invalid
    {
        std::string src = "var A = sparse_matrix(2, 2, 1, 0, 0, -1);\n"
                          "var x = 3.5;\n"
                          "var err = A + x;\n";
        Parser p(src);
        auto prog = p.parse_program();
        TypeChecker tc;
        bool caught = false;
        try {
            tc.typecheck(*prog);
        } catch (const TypeError& e) {
            caught = true;
        }
        assert(caught && "TypeChecker should reject sparse_matrix + float!");
    }

    // Negative test 2: float + trit is invalid without explicit conversion
    {
        std::string src = "var x: float = 3.5;\n"
                          "var t: trit = +1;\n"
                          "var err = x + t;\n";
        Parser p(src);
        auto prog = p.parse_program();
        TypeChecker tc;
        bool caught = false;
        try {
            tc.typecheck(*prog);
        } catch (const TypeError& e) {
            caught = true;
        }
        assert(caught && "TypeChecker should reject float + trit!");
    }

    std::cout << "      -> TypeChecker Semantic Analysis passed.\n";
}

// ==============================================================================
// 4. LLVM CodeGen Pattern Verification
// ==============================================================================

void test_llvm_codegen_patterns() {
    std::cout << "  [4/5] Testing LLVM IR CodeGen Patterns...\n";

    std::string src = "var x: float = 4.25;\n"
                      "var y: float = 1.75;\n"
                      "var sum: float = x + y;\n"
                      "var prod: float = x * y;\n"
                      "var diff: float = x - y;\n"
                      "print(sum);\n"
                      "var A = sparse_matrix(2, 2, 1, 0, 0, -1);\n"
                      "var v = vector_init(5, 10);\n"
                      "var res = A * v;\n"
                      "print(res);\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    LLVMCodeGenVisitor codegen;
    std::string ir = codegen.generate(*prog);

    // 1. Verify float operations
    assert(ir.find("fadd double") != std::string::npos);
    assert(ir.find("fmul double") != std::string::npos);
    assert(ir.find("fsub double") != std::string::npos);
    assert(ir.find("@bt_print_float") != std::string::npos);
    assert(ir.find("@.str_float") != std::string::npos);

    // 2. Verify sparse matrix memory structures
    assert(ir.find("@smat_1_row_ptr = internal constant") != std::string::npos);
    assert(ir.find("@smat_1_col_ind = internal constant") != std::string::npos);
    assert(ir.find("@smat_1_signs = internal constant") != std::string::npos);

    // 3. Verify multiplier-free SpMV loop
    assert(ir.find("define void @bt_spmv") != std::string::npos);
    assert(ir.find("lshr i64 %k, 6") != std::string::npos);
    assert(ir.find("and i64 %k, 63") != std::string::npos);
    assert(ir.find("select i1 %is_pos, i64 %acc_add, i64 %acc_sub") != std::string::npos);

    // 4. Check multiplier-free property: within @bt_spmv body there is NO 'mul'
    size_t spmv_start = ir.find("define void @bt_spmv");
    size_t spmv_end = ir.find("spmv_done:");
    assert(spmv_start != std::string::npos && spmv_end != std::string::npos);
    std::string spmv_body = ir.substr(spmv_start, spmv_end - spmv_start);
    assert(spmv_body.find("mul i64") == std::string::npos && "SpMV must be 100% multiplier-free!");

    std::cout << "      -> LLVM CodeGen patterns passed.\n";
}

// ==============================================================================
// 5. Bare-Metal Native LLVM JIT Execution
// ==============================================================================

void test_baremetal_native_jit_execution() {
    std::cout << "  [5/5] Testing Bare-Metal Execution via NativeLLVMJIT...\n";

    std::string src = "var x: float = 3.5;\n"
                      "var y: float = 2.0;\n"
                      "var z: float = x * y - 1.5;\n" // 3.5 * 2.0 - 1.5 = 5.5
                      "print(z);\n"
                      "var A = sparse_matrix(2, 2, 1, 0, 0, -1);\n"
                      "var v = vector_init(42, 17);\n"
                      "var res = A * v;\n"            // [1, 0; 0, -1] * [42, 17] = [42, -17]
                      "print(res);\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    TypeChecker tc;
    tc.typecheck(*prog);
    assert(tc.errors().empty());

    int exit_code = NativeLLVMJIT::execute_program(*prog, false);
    assert(exit_code == 0 && "NativeLLVMJIT execution must return 0!");

    std::cout << "      -> Native LLVM JIT execution passed.\n";
}

// ==============================================================================
// 6. Variable Shadowing in Nested If Blocks
// ==============================================================================

void test_variable_shadowing_nested_if() {
    std::cout << "  [6/10] Testing Variable Shadowing in Nested If Blocks...\n";
    std::string src = "var x: int = 10;\n"
                      "var result: int = 0;\n"
                      "if (x == 10) {\n"
                      "    var x: int = 25;\n"
                      "    result = x;\n"
                      "}\n"
                      "result = result + x;\n" // 25 + 10 = 35
                      "return result;\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    LLVMCodeGenVisitor codegen;
    std::string ir = codegen.generate(*prog);
    assert(ir.find("%var_x_1") != std::string::npos && "Outer variable must have unique SSA alloca register %var_x_1");
    assert(ir.find("%var_x_3") != std::string::npos && "Shadowed variable must have unique SSA alloca register %var_x_3");

    int exit_code = NativeLLVMJIT::execute_program(*prog, false);
    assert(exit_code == 35 && "Variable shadowing result must be 35!");
    std::cout << "      -> Variable Shadowing in Nested If passed.\n";
}

// ==============================================================================
// 7. Early Return in If/Else Blocks
// ==============================================================================

void test_early_return_if_else() {
    std::cout << "  [7/10] Testing Early Return in If/Else Blocks...\n";
    // Then-branch return
    {
        std::string src = "var cond: int = 1;\n"
                          "if (cond == 1) {\n"
                          "    return 42;\n"
                          "} else {\n"
                          "    return 99;\n"
                          "}\n";
        Parser parser(src);
        auto prog = parser.parse_program();
        assert(prog != nullptr);

        int exit_code = NativeLLVMJIT::execute_program(*prog, false);
        assert(exit_code == 42 && "Early return in then branch must return 42!");
    }

    // Else-branch return
    {
        std::string src = "var cond: int = -1;\n"
                          "if (cond == 1) {\n"
                          "    return 42;\n"
                          "} else {\n"
                          "    return 99;\n"
                          "}\n";
        Parser parser(src);
        auto prog = parser.parse_program();
        assert(prog != nullptr);

        int exit_code = NativeLLVMJIT::execute_program(*prog, false);
        assert(exit_code == 99 && "Early return in else branch must return 99!");
    }

    // Early return with float expression truncated to i32
    {
        std::string src = "var f: float = 7.8;\n"
                          "return f;\n";
        Parser parser(src);
        auto prog = parser.parse_program();
        assert(prog != nullptr);

        int exit_code = NativeLLVMJIT::execute_program(*prog, false);
        assert(exit_code == 7 && "Float return must truncate to i32 (7)!");
    }
    std::cout << "      -> Early Return in If/Else passed.\n";
}

// ==============================================================================
// 8. Mixed Float + Int Expressions
// ==============================================================================

void test_mixed_float_int_expressions() {
    std::cout << "  [8/10] Testing Mixed Float + Int Expressions...\n";
    std::string src = "var a: float = 3.5;\n"
                      "var b: int = 2;\n"
                      "var sum: float = a + b;\n"       // 5.5
                      "var prod: float = b * a;\n"      // 7.0
                      "var diff: float = a - b;\n"      // 1.5
                      "var cmp1: int = (a < 10);\n"     // 1
                      "var cmp2: int = (a > b);\n"      // 1
                      "var cmp3: int = (2.0 == b);\n"   // 1
                      "var total: int = cmp1 + cmp2 + cmp3;\n" // 3
                      "return total;\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    LLVMCodeGenVisitor codegen;
    std::string ir = codegen.generate(*prog);
    assert(ir.find("sitofp i64") != std::string::npos && "Must emit sitofp for mixed ops!");
    assert(ir.find("fcmp olt double") != std::string::npos && "Must emit fcmp olt!");
    assert(ir.find("fcmp ogt double") != std::string::npos && "Must emit fcmp ogt!");
    assert(ir.find("fcmp oeq double") != std::string::npos && "Must emit fcmp oeq!");

    int exit_code = NativeLLVMJIT::execute_program(*prog, false);
    assert(exit_code == 3 && "Mixed float + int comparisons must evaluate to 3!");
    std::cout << "      -> Mixed Float + Int Expressions passed.\n";
}

// ==============================================================================
// 9. Comparison Precedence (Relational vs Equality)
// ==============================================================================

void test_comparison_precedence() {
    std::cout << "  [9/10] Testing Comparison Precedence (Relational vs Equality)...\n";
    // 1. AST verification: 'a < b == c < d' must parse as '(a < b) == (c < d)'
    std::string src = "var res = a < b == c < d;\n";
    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);
    assert(prog->declarations.size() == 1);

    auto vdecl = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
    assert(vdecl != nullptr);

    auto root_bin = std::dynamic_pointer_cast<BinaryOpNode>(vdecl->init_expr);
    assert(root_bin != nullptr);
    assert(root_bin->op == BinaryOp::Equal && "Root operator of 'a < b == c < d' must be Equality (==)!");

    auto left_bin = std::dynamic_pointer_cast<BinaryOpNode>(root_bin->left);
    assert(left_bin != nullptr);
    assert(left_bin->op == BinaryOp::LessThan && "Left child must be LessThan (<)!");

    auto right_bin = std::dynamic_pointer_cast<BinaryOpNode>(root_bin->right);
    assert(right_bin != nullptr);
    assert(right_bin->op == BinaryOp::LessThan && "Right child must be LessThan (<)!");

    // 2. Bare-metal JIT execution of precedence expression:
    // a = 1, b = 2 (a < b is 1)
    // c = 10, d = 20 (c < d is 1)
    // res = (1 < 2 == 10 < 20) => (1 == 1) => 1
    std::string exec_src = "var a: int = 1;\n"
                           "var b: int = 2;\n"
                           "var c: int = 10;\n"
                           "var d: int = 20;\n"
                           "var res: int = (a < b == c < d);\n"
                           "if (res == 1) {\n"
                           "    return 100;\n"
                           "} else {\n"
                           "    return 50;\n"
                           "}\n";
    Parser exec_parser(exec_src);
    auto exec_prog = exec_parser.parse_program();
    assert(exec_prog != nullptr);

    int exit_code = NativeLLVMJIT::execute_program(*exec_prog, false);
    assert(exit_code == 100 && "Execution of (a < b == c < d) must return 100!");

    std::cout << "      -> Comparison Precedence passed.\n";
}

// ==============================================================================
// 10. Kokkos Bridge Fixes
// ==============================================================================

void test_kokkos_bridge_fixes() {
    std::cout << "  [10/10] Testing Kokkos Bridge Fixes (Host View & Parallel CSA)...\n";
    // 1. make_host_view on allocated matrix
    TernaryMatrix mat(4, 128);
    mat.set(0, 0, Trit::Pos);
    mat.set(1, 65, Trit::Neg);
    auto view = bt::kokkos::make_host_view(mat);
    assert(view.is_allocated() && "Host view of non-empty matrix must be allocated!");
    assert(view.d_pos != nullptr && view.d_neg != nullptr);
    assert((view.d_pos[0] & 1ULL) == 1ULL);
    assert(view.rows == 4 && view.cols == 128);

    // 2. make_host_view on empty matrix
    TernaryMatrix empty_mat(0, 0);
    auto empty_view = bt::kokkos::make_host_view(empty_mat);
    assert(!empty_view.is_allocated() && "Host view of empty matrix must not be allocated!");
    assert(empty_view.d_pos == nullptr && empty_view.d_neg == nullptr);

    // 3. parallel_reduce_csa with num_threads = 0 (guard against division by zero)
    std::vector<TritWord64> words = { TritWord64(1, 0), TritWord64(0, 1), TritWord64(1, 0) };
    auto csa_res = bt::kokkos::parallel_reduce_csa(words, 0);
    assert(csa_res.Sum.isValid());
    assert(csa_res.Carry.isValid());

    std::cout << "      -> Kokkos Bridge Fixes passed.\n";
}

// ==============================================================================
// 10b. Verified JIT Engine Fixes: float comparisons, Kleene-on-float errors,
//      multiplication overflow, env restore, SpMM, full-width printing,
//      and ThreadPool lifecycle guards.
// ==============================================================================

static std::string jit_capture_print(const std::string& src) {
    std::ostringstream out;
    bt::jit::JITConfig cfg;
    cfg.enable_kokkos = false;
    cfg.out = &out;
    bt::jit::TernaryJITEngine engine(cfg);
    engine.execute_source(src);
    return out.str();
}

void test_jit_float_comparison_semantics() {
    std::cout << "  [14/19] Testing JIT float comparisons actually compare (not add)...\n";
    std::string out = jit_capture_print(
        "var a = 9.0;\nvar b = 3.0;\nprint(a<b); print(a>b); print(a==b); print(a<=>b);\n");
    // 9.0 < 3.0 -> -1 ; 9.0 > 3.0 -> +1 ; 9.0 == 3.0 -> -1 ; 9.0 <=> 3.0 -> +1
    assert(out.compare(0, 4, "-1 [") == 0 && "a<b must be -1 (false)");
    assert(out.find("12 [") == std::string::npos &&
           "comparisons must never compute lv + rv (the old float bug printed 12)");
    assert(out.find("1 [trits: +]") != std::string::npos && "a>b must be +1");

    std::string eq_out = jit_capture_print(
        "var a = 3.0;\nvar b = 3.0;\nprint(a<=>b); print(a==b); print(a<b); print(a!=b);\n");
    assert(eq_out.compare(0, 3, "0 [") == 0 && "3.0 <=> 3.0 must be 0");
    assert(eq_out.find("1 [") != std::string::npos && "3.0 <= 3.0 / == 3.0 must be +1");

    std::string mixed = jit_capture_print("var a = 9;\nvar b = 9.0;\nprint(a==b);\n");
    assert(mixed.compare(0, 3, "1 [") == 0 && "int==float must compare numerically");
    std::cout << "      -> Float comparison semantics passed.\n";
}

void test_jit_float_kleene_rejection() {
    std::cout << "  [15/19] Testing Kleene syntax (& | ^) on floats raises a clean error...\n";
    bool threw = false;
    std::string what;
    try {
        (void)jit_capture_print("var a = 1.5;\nvar b = 2.5;\nprint(a & b);\n");
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    assert(threw && "float Kleene op must throw (never crash, never add)");
    assert((what.find("Kleene logic not defined for float") != std::string::npos &&
            ("clean message expected, got: " + what).c_str() != nullptr));
    assert(what.find("Kleene logic not defined for float") != std::string::npos);
    std::cout << "      -> Kleene-on-float rejection passed.\n";
}

void test_jit_multiplication_overflow_clean() {
    std::cout << "  [16/19] Testing JIT multiplication overflow (clean error, exact terms)...\n";
    // 39-trit maximum operand squared: |product| ~ 4.1e36 (far beyond int64/39-trit range)
    bool threw = false;
    std::string what;
    try {
        (void)jit_capture_print(
            "var m = 2026277576509488133;\nvar p = m * m;\nprint(p);\n");
    } catch (const std::runtime_error& e) {
        threw = true;
        what = e.what();
    }
    assert(threw && "39-trit-max squared must raise a clean overflow error");
    assert((what.find("integer overflow in multiplication") != std::string::npos));
    assert(what.find("4105800817065164534531589373667825689") != std::string::npos &&
           "exact __int128 product should be reported");

    // In-range product stays EXACT: 1417211799^2 = 2008489283224816401 (<= (3^39-1)/2)
    std::string out = jit_capture_print("var a = 1417211799;\nvar p = a * a;\nprint(p);\n");
    assert(out.compare(0, 21, "2008489283224816401 [") == 0);
    std::cout << "      -> Multiplication overflow semantics passed.\n";
}

void test_jit_env_restore_on_error() {
    std::cout << "  [17/19] Testing env frame restore when a function body throws...\n";
    bt::jit::JITConfig cfg;
    bt::jit::TernaryJITEngine engine(cfg);
    auto root = engine.get_env();
    bool caught = false;
    try {
        engine.execute_source(
            "var okay = 7;\n"
            "func f() { var x = gtors; return x; }\n" // undefined variable -> throws in body
            "f();\n");
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught && "in-body error must propagate");
    assert(engine.get_env() == root && "RAII guard must restore caller env after throw");
    engine.execute_source("var after_ok = okay + 1;\n");
    assert(engine.get_env() == root && "definitions after the caught error stay in root frame");
    assert(engine.get_env()->get("after_ok").as_int64() == 8 && "after_ok == 8 at root");
    assert(engine.get_env()->get("okay").as_int64() == 7 && "root binding still intact");
    for (int i = 0; i < 2; ++i) {
        try { engine.execute_source("f();\n"); } catch (const std::runtime_error&) {}
        assert(engine.get_env() == root && "no dead-frame nesting across repeated failures");
    }
    engine.execute_source("func g(x) { return x * 2; }\nvar v = g(21);\n");
    assert(engine.get_env()->get("v").as_int64() == 42 && "calls still work after failures");
    std::cout << "      -> Env frame restore passed.\n";
}

void test_jit_sparse_spmm() {
    std::cout << "  [18/19] Testing sparse_matrix * dense matrix performs a TRUE SpMM...\n";
    bt::jit::JITConfig cfg;
    bt::jit::TernaryJITEngine engine(cfg);
    engine.execute_source(
        "sparse_matrix A(3, 3, 1, 0, 0, 1, 0, 1, -1, 0, 0);\n"
        "var B = matrix_init(3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1);\n"
        "var C = A * B;\n");
    DenseMatrixInt64 C = engine.get_env()->get("C").int_matrix;
    assert(C.rows == 3 && C.cols == 3 && "SpMM result shape is M x N (not rows*cols x 1)");
    assert(C(0, 0) == 1 && C(0, 1) == 1 && C(0, 2) == 1 && "C row 0 = [1,1,1]");
    assert(C(1, 0) == 2 && C(1, 1) == 2 && C(1, 2) == 2 && "C row 1 = [2,2,2]");
    assert(C(2, 0) == -1 && C(2, 1) == -1 && C(2, 2) == -1 && "C row 2 = [-1,-1,-1]");

    // Cross-check against the T-CSR SpMM reference of the very same operands.
    std::vector<int64_t> REF = bt::TCSRMatrix::from_dense(3, 3, {1, 0, 0, 1, 0, 1, -1, 0, 0})
                                   .spmm(std::vector<int64_t>(9, 1), 3);
    assert(REF.size() == 9 && "reference SpMM size is 3x3");
    for (size_t i = 0; i < 9; ++i)
        assert(C.data[i] == REF[i] && "JIT SpMM == T-CSR SpMM reference");

    // Non-matching shape must fail cleanly (the old code silently used only row 0).
    bool threw = false;
    std::string what;
    try {
        (void)jit_capture_print(
            "sparse_matrix A(3, 3, 1, 0, 0, 1, 0, 1, -1, 0, 0);\n"
            "var BAD = matrix_init(2, 2, 1, 1, 1, 1);\n"
            "var X = A * BAD;\n");
    } catch (const std::runtime_error& e) { threw = true; what = e.what(); }
    assert(threw && "mismatched right operand must throw");
    assert(what.find("dimension") != std::string::npos);
    (void)what;

    // Vector SpMV (example 05 semantics) must be preserved: S * vector_init(10,20,30)
    std::string out = jit_capture_print(
        "var S = sparse_matrix(3, 3, 1, 0, -1, 0, 1, 0, -1, 0, 1);\n"
        "var v = vector_init(10, 20, 30);\n"
        "var res = S * v;\nprint(res);\n");
    assert(out.find("-20") != std::string::npos && out.find("20") != std::string::npos &&
           "classic SpMV path unchanged");
    std::cout << "      -> Sparse SpMM passed.\n";
}

void test_jit_full_width_print() {
    std::cout << "  [19/19] Testing JIT printing uses the full 64-trit word width...\n";
    // trit 0 = +1, trit 50 = +1 -> exact value 1 + 3^50 (positions > 39 must not be lost)
    std::string out = jit_capture_print(
        "var w = +0000000000000000000000000000000000000000000000000+;\nprint(w);\n");
    assert(out.find("717897987691852588770250 [") == 0);
    // Negative 2-trit word: trit0 = -1, trit49 = -1 -> -(1 + 3^49)
    std::string out2 = jit_capture_print(
        "var w2 = -000000000000000000000000000000000000000000000000-;\nprint(w2);\n");
    assert(out2.find("-239299329230617529590084 [") == 0);
    // Plain 39-trit printing is unchanged.
    std::string out3 = jit_capture_print("var x = 12345;\nprint(x);\n");
    assert(out3.compare(0, 14, "12345 [trits: ") == 0 && "plain int printing unchanged");
    std::cout << "      -> Full-width printing passed.\n";
}

void test_threadpool_guards_and_drain() {
    std::cout << "  [20/19] Testing ThreadPool reinit/no-drop/nested-dispatch guards...\n";
    // a) explicit reinit to a different size must keep queued/latched work running.
    constexpr size_t N = 100000;
    std::atomic<int64_t> hits{0};
    for (int trial = 0; trial < 3; ++trial) {
        hits.store(0);
        auto done = std::async(std::launch::async, [&]() {
            Kokkos::parallel_for(Kokkos::RangePolicy<size_t>(0, N),
                                 [&](size_t) { hits.fetch_add(1, std::memory_order_relaxed); });
        });
        Kokkos::initialize(2 + trial); // churn the pool mid-dispatch
        done.wait();
        assert(hits.load() == static_cast<int64_t>(N) &&
               "shutdown must EXECUTE drained tasks so latch waiters release");
    }

    // b) same-count reinit is a no-op (no teardown churn inside dispatch loops).
    size_t before = Kokkos::num_threads();
    Kokkos::initialize(before);
    assert(Kokkos::num_threads() == before && "same-count reinit must not shrink the pool");

    // c) nested parallel_for from a pool worker runs in place (no latch self-deadlock).
    {
        std::vector<int64_t> flat(32 * 4, -1);
        auto nested = std::async(std::launch::async, [&]() {
            Kokkos::parallel_for(Kokkos::RangePolicy<size_t>(0, 32), [&](size_t i) {
                Kokkos::parallel_for(Kokkos::RangePolicy<size_t>(0, 4), [&](size_t j) {
                    flat[i * 4 + j] = static_cast<int64_t>(i + j);
                });
            });
        });
        constexpr std::chrono::seconds kBudget(15);
        auto status = nested.wait_for(kBudget);
        assert(status != std::future_status::timeout && "nested dispatch must not deadlock");
        bool ok = true;
        for (size_t i = 0; i < 32 && ok; ++i)
            for (size_t j = 0; j < 4 && ok; ++j)
                ok = flat[i * 4 + j] == static_cast<int64_t>(i + j);
        assert(ok && "nested dispatch computed every element");
    }
    std::cout << "      -> ThreadPool guards passed.\n";
}

// ==============================================================================
// 11. DWARF-5 Debug Metadata and Auto-Vectorization Loop Metadata
// ==============================================================================

void test_dwarf_debug_metadata_and_vectorization() {
    std::cout << "  [11/11] Testing DWARF-5 Debug Metadata and Loop Vectorization Metadata...\n";

    std::string src = "sparse_matrix A(2, 2, 1, 0, 0, -1);\n"
                      "var x = 1.0;\n"
                      "var y = 2.0;\n"
                      "var z = x + y;\n"
                      "return 0;\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    LLVMCodeGenVisitor visitor;
    visitor.set_debug_info(true, "test_debug.bt");
    std::string ir = visitor.generate_ir(*prog);

    // Verify DWARF-5 metadata presence
    assert(ir.find("!llvm.dbg.cu") != std::string::npos);
    assert(ir.find("!DICompileUnit") != std::string::npos);
    assert(ir.find("DW_LANG_C99") != std::string::npos);
    assert(ir.find("!\"Dwarf Version\", i32 5") != std::string::npos);
    assert(ir.find("!\"Debug Info Version\", i32 3") != std::string::npos);
    assert(ir.find("!DIFile(filename: \"test_debug.bt\"") != std::string::npos);
    assert(ir.find("!DISubprogram(name: \"bt_main\"") != std::string::npos);
    assert(ir.find("!DILocation(") != std::string::npos);

    // Verify auto-vectorization loop metadata presence
    assert(ir.find("!llvm.loop") != std::string::npos);
    assert(ir.find("!\"llvm.loop.vectorize.enable\", i1 true") != std::string::npos);

    // Strict Clang compilation verification with -Wall -Werror -Wno-override-module -c -x ir
    char tmp_ir_path[] = "/tmp/bt_debug_unit_XXXXXX.ll";
    int fd = mkstemps(tmp_ir_path, 3);
    assert(fd != -1);
    ssize_t written = write(fd, ir.data(), ir.size());
    assert(written == static_cast<ssize_t>(ir.size()));
    close(fd);

    std::string obj_path = std::string(tmp_ir_path) + ".o";
    std::string cmd = "export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk && "
                      "clang -Wall -Werror -Wno-override-module -c -x ir \"" + std::string(tmp_ir_path) + "\" -o \"" + obj_path + "\" 2>&1";

    FILE* pipe = popen(cmd.c_str(), "r");
    assert(pipe != nullptr);
    char buffer[256];
    std::string output = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int status = pclose(pipe);
    unlink(tmp_ir_path);
    unlink(obj_path.c_str());

    assert(status == 0 && "Clang compilation of IR with DWARF-5 & loop metadata must succeed with 0 warnings/errors!");
    assert(output.empty() && "Clang compilation produced unexpected warnings/errors!");

    std::cout << "      -> DWARF-5 Debug Metadata & Auto-Vectorization Loop Metadata passed (0 warnings).\n";
}

// ==============================================================================
// 12. Multi-File Module @import / import Resolution & Circular Guard
// ==============================================================================

void test_module_imports() {
    std::cout << "  [12/13] Testing Multi-File Module @import & import Resolution...\n";

    pid_t pid = getpid();
    std::string mod_a_path = "/tmp/bt_mod_a_" + std::to_string(pid) + ".bt";
    std::string mod_b_path = "/tmp/bt_mod_b_" + std::to_string(pid) + ".bt";

    // Module A imports Module B (and Module B circularly imports Module A to test cycle prevention)
    {
        std::ofstream fa(mod_a_path);
        fa << "@import \"" << mod_b_path << "\";\n"
           << "var mod_a_val: int = 15;\n"
           << "func add_bonus(x: int) -> int {\n"
           << "    return x + 10;\n"
           << "}\n";
    }
    {
        std::ofstream fb(mod_b_path);
        fb << "import \"" << mod_a_path << "\";\n"
           << "var mod_b_val: int = 27;\n";
    }

    std::string main_src = "@import \"" + mod_a_path + "\";\n"
                           "import \"" + mod_b_path + "\";\n"
                           "var combined: int = add_bonus(mod_a_val + mod_b_val);\n"
                           "return combined;\n";

    // Test Preprocessor::resolve_imports directly
    std::string resolved = Preprocessor::resolve_imports(main_src, "/tmp");
    assert(resolved.find("mod_a_val") != std::string::npos);
    assert(resolved.find("mod_b_val") != std::string::npos);

    // Test Parser + TypeChecker + NativeLLVMJIT end-to-end
    Parser parser(main_src, "/tmp");
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    SemanticAnalyzer sema;
    sema.typecheck(*prog);
    assert(sema.errors().empty());

    int exit_code = NativeLLVMJIT::execute_program(*prog, false);
    unlink(mod_a_path.c_str());
    unlink(mod_b_path.c_str());

    // 15 + 27 + 10 = 52
    assert(exit_code == 52 && "Multi-file @import execution must return 52!");
    std::cout << "      -> Multi-File Module @import & Circular Import Guard passed.\n";
}

// ==============================================================================
// 13. Inline Dual-Rail Hardware SSA Intrinsics (planar_dot64, btha_sum_p, btha_carry_p)
// ==============================================================================

void test_dual_rail_ssa_intrinsics() {
    std::cout << "  [13/13] Testing Inline Dual-Rail Hardware SSA Intrinsics (planar_dot64, btha_sum_p, btha_carry_p)...\n";

    // ap = 0b0111 (7) -> trits 0,1,2 are +1
    // an = 0b1000 (8) -> trit 3 is -1
    // bp = 0b0101 (5) -> trits 0,2 are +1
    // bn = 0b1010 (10) -> trits 1,3 are -1
    // dot product:
    //   k=0: (+1)*(+1) = +1
    //   k=1: (+1)*(-1) = -1
    //   k=2: (+1)*(+1) = +1
    //   k=3: (-1)*(-1) = +1
    //   total dot = 1 - 1 + 1 + 1 = 2
    // btha_carry_p(ap, an, bp, bn) = ap & bp = 7 & 5 = 5
    // btha_sum_p(ap, an, bp, bn)   = ((ap ^ bp) & ~(an | bn)) | (an & bn)
    //   ap ^ bp = 7 ^ 5 = 2 (0b0010)
    //   an | bn = 8 | 10 = 10 (0b1010), ~(an | bn) has bit 1 = 0 -> (2 & ~10) = 0
    //   an & bn = 8 & 10 = 8 (0b1000)
    //   btha_sum_p = 0 | 8 = 8
    // Expected return = dot (2) + carry_p (5) + sum_p (8) = 15
    std::string src = "var ap: tritword = 7;\n"
                      "var an: tritword = 8;\n"
                      "var bp: tritword = 5;\n"
                      "var bn: tritword = 10;\n"
                      "var dot: tritword = planar_dot64(ap, an, bp, bn);\n"
                      "var cp: tritword = btha_carry_p(ap, an, bp, bn);\n"
                      "var sp: tritword = btha_sum_p(ap, an, bp, bn);\n"
                      "return dot + cp + sp;\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    SemanticAnalyzer sema;
    sema.typecheck(*prog);
    assert(sema.errors().empty());
    assert(sema.get_symbol_type("dot") == BTType::TritWord);
    assert(sema.get_symbol_type("cp") == BTType::TritWord);
    assert(sema.get_symbol_type("sp") == BTType::TritWord);

    LLVMCodeGenVisitor codegen;
    std::string ir = codegen.generate(*prog);

    // Verify inline SSA emission (no external @planar_dot64 call, uses @llvm.ctpop.i64)
    assert(ir.find("declare i64 @llvm.ctpop.i64(i64)") != std::string::npos);
    assert(ir.find("call i64 @llvm.ctpop.i64(i64") != std::string::npos);
    assert(ir.find("and i64") != std::string::npos);
    assert(ir.find("or i64") != std::string::npos);
    assert(ir.find("xor i64") != std::string::npos);
    assert(ir.find("sub i64") != std::string::npos);
    assert(ir.find("@planar_dot64") == std::string::npos && "planar_dot64 must be inlined as SSA instructions!");

    int exit_code = NativeLLVMJIT::execute_program(*prog, false);
    assert(exit_code == 15 && "NativeLLVMJIT execution of planar_dot64 + btha_carry_p + btha_sum_p must equal 15!");

    std::cout << "      -> Inline Dual-Rail Hardware SSA Intrinsics passed.\n";
}

void test_math_intrinsics_fma_sqrt_abs() {
    std::cout << "  [21/21] Testing Math Intrinsics fma, sqrt, abs across JIT & Native LLVM...\n";

    // 1. JIT execution
    std::string jit_out = jit_capture_print(
        "var a = 2.0;\n"
        "var b = 3.0;\n"
        "var c = 4.0;\n"
        "var f = fma(a, b, c);\n"
        "var s = sqrt(f * 2.0 + 5.0);\n"
        "var a1 = abs(-7.5);\n"
        "var a2 = abs(-42);\n"
        "print(f);\n"
        "print(s);\n"
        "print(a1);\n"
        "print(a2);\n");
    assert(jit_out.find("10 [BTFP32:") != std::string::npos && "fma(2, 3, 4) must be 10.0");
    assert(jit_out.find("5 [BTFP32:") != std::string::npos && "sqrt(25.0) must be 5.0");
    assert(jit_out.find("7.5 [BTFP32:") != std::string::npos && "abs(-7.5) must be 7.5");
    assert(jit_out.find("42 [") != std::string::npos && "abs(-42) must be 42");

    // 2. Typechecker & LLVM IR CodeGen & NativeLLVMJIT
    std::string src = "var a: float = 2.0;\n"
                      "var b: float = 3.0;\n"
                      "var c: float = 4.0;\n"
                      "var f: float = fma(a, b, c);\n"
                      "var s: float = sqrt(f * 2.0 + 5.0);\n"
                      "var a1: float = abs(-7.0);\n"
                      "var a2: int = abs(-42);\n"
                      "var sum_f: float = f + s + a1 + 42.0;\n"
                      "return sum_f;\n";

    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);

    SemanticAnalyzer sema;
    sema.typecheck(*prog);
    assert(sema.errors().empty());
    assert(sema.get_symbol_type("f") == BTType::Float);
    assert(sema.get_symbol_type("s") == BTType::Float);
    assert(sema.get_symbol_type("a1") == BTType::Float);
    assert(sema.get_symbol_type("a2") == BTType::Int || sema.get_symbol_type("a2") == BTType::TritWord);

    LLVMCodeGenVisitor codegen;
    std::string ir = codegen.generate(*prog);

    assert(ir.find("declare double @llvm.fma.f64") != std::string::npos);
    assert(ir.find("declare double @llvm.sqrt.f64") != std::string::npos);
    assert(ir.find("declare double @llvm.fabs.f64") != std::string::npos);
    assert(ir.find("declare i64 @llvm.abs.i64") != std::string::npos);
    assert(ir.find("call double @llvm.fma.f64") != std::string::npos);
    assert(ir.find("call double @llvm.sqrt.f64") != std::string::npos);
    assert(ir.find("call double @llvm.fabs.f64") != std::string::npos);
    assert(ir.find("call i64 @llvm.abs.i64") != std::string::npos);

    int exit_code = NativeLLVMJIT::execute_program(*prog, false);
    assert(exit_code == 64 && "NativeLLVMJIT execution of fma+sqrt+abs must equal 64!");

    std::cout << "      -> Math Intrinsics (fma, sqrt, abs) across JIT & Native LLVM passed.\n";
}

void test_phase3_compiler_fixes() {
    std::cout << "  [22/22] Testing Phase 3 Compiler Fixes (Multi-Trit Literals, @bt_select, Stmt Select & Kleene Ops)...\n";

    // 1. Multi-trit literal (+-0+ = 27 - 9 + 0 + 1 = 19) in Native LLVM JIT
    std::string src1 = "var lit = +-0+;\nreturn lit;\n";
    Parser p1(src1);
    auto prog1 = p1.parse_program();
    int exit1 = NativeLLVMJIT::execute_program(*prog1, false);
    assert(exit1 == 19 && "Multi-trit literal +-0+ must evaluate to 19 in Native LLVM JIT!");

    // 2. Expression select with condition > 1 (select(19, 100, 200, 300) must return 100)
    std::string src2 = "var x = 19;\nvar s = select(x, 100, 200, 300);\nreturn s;\n";
    Parser p2(src2);
    auto prog2 = p2.parse_program();
    int exit2 = NativeLLVMJIT::execute_program(*prog2, false);
    assert(exit2 == 100 && "select(19, ...) must return positive branch (100)!");

    // 3. Statement select branching with basic blocks
    std::string src3 = 
        "var c = 10;\n"
        "var res = 0;\n"
        "select (c) {\n"
        "    pos {\n"
        "        res = 55;\n"
        "    }\n"
        "    zero {\n"
        "        res = 66;\n"
        "    }\n"
        "    neg {\n"
        "        res = 77;\n"
        "    }\n"
        "}\n"
        "return res;\n";
    Parser p3(src3);
    auto prog3 = p3.parse_program();
    int exit3 = NativeLLVMJIT::execute_program(*prog3, false);
    assert(exit3 == 55 && "Statement select must execute only positive block!");

    // 4. Kleene 3-valued logic on integers: 84 & 27 == 0, 84 | 27 == 111, 84 ^ 27 == 0
    std::string src4 = "var a = 84;\nvar b = 27;\nvar res = a | b;\nreturn res;\n";
    Parser p4(src4);
    auto prog4 = p4.parse_program();
    int exit4 = NativeLLVMJIT::execute_program(*prog4, false);
    assert(exit4 == 111 && "84 | 27 must evaluate to 111 via balanced ternary Kleene OR in Native LLVM JIT!");

    std::cout << "      -> Phase 3 Compiler Fixes verified across Native LLVM JIT.\n";
}

// ==============================================================================
// Main Entry Point
// ==============================================================================

int main() {
    std::cout << "======================================================================\n";
    std::cout << "[LLVM Unit Test Layer 1] Frontend BT-FP32 & T-CSR Sparse Matrix Suite\n";
    std::cout << "======================================================================\n";

    test_float_lexer_and_ast();
    test_sparse_matrix_parsing_and_ast();
    test_typechecker_sema();
    test_llvm_codegen_patterns();
    test_baremetal_native_jit_execution();
    test_variable_shadowing_nested_if();
    test_early_return_if_else();
    test_mixed_float_int_expressions();
    test_comparison_precedence();
    test_kokkos_bridge_fixes();
    test_jit_float_comparison_semantics();
    test_jit_float_kleene_rejection();
    test_jit_multiplication_overflow_clean();
    test_jit_env_restore_on_error();
    test_jit_sparse_spmm();
    test_jit_full_width_print();
    test_threadpool_guards_and_drain();
    test_dwarf_debug_metadata_and_vectorization();
    test_module_imports();
    test_dual_rail_ssa_intrinsics();
    test_math_intrinsics_fma_sqrt_abs();
    test_phase3_compiler_fixes();

    std::cout << "\n  -> ALL FRONTEND FP, SPARSE, MODULE IMPORT, INTRINSIC & PHASE 3 COMPILER FIXES PASSED (100%)\n";
    std::cout << "======================================================================\n";
    return 0;
}
