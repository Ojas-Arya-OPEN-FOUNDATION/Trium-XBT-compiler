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

    std::cout << "\n  -> ALL 10 FRONTEND FP & SPARSE MATRIX UNIT TESTS PASSED (100%)\n";
    std::cout << "======================================================================\n";
    return 0;
}
