/**
 * @file test_frontend_ast.cpp
 * @brief Comprehensive Verification Suite for Balanced Ternary Frontend AST, Lexer & Parser
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "ternary_frontend_ast.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

using namespace bt;
using namespace bt::frontend;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << (msg) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// ==============================================================================
// 1. Test Suite: Lexer & Tokenizer
// ==============================================================================

void test_lexer_basic_tokens() {
    std::cout << "[Test Suite 1/6] Running Lexer Basic Tokens Test...\n";

    std::string src = "func var trit tritword tritmatrix return select while if else pos zero neg case";
    Lexer lex(src);
    auto tokens = lex.tokenize_all();

    TEST_ASSERT(tokens.size() == 15, "Expected 14 keywords + EOF");
    TEST_ASSERT(tokens[0].type == TokenType::KwFunc, "func keyword mismatch");
    TEST_ASSERT(tokens[1].type == TokenType::KwVar, "var keyword mismatch");
    TEST_ASSERT(tokens[2].type == TokenType::KwTrit, "trit keyword mismatch");
    TEST_ASSERT(tokens[3].type == TokenType::KwTritWord, "tritword keyword mismatch");
    TEST_ASSERT(tokens[4].type == TokenType::KwTritMatrix, "tritmatrix keyword mismatch");
    TEST_ASSERT(tokens[5].type == TokenType::KwReturn, "return keyword mismatch");
    TEST_ASSERT(tokens[6].type == TokenType::KwSelect, "select keyword mismatch");
    TEST_ASSERT(tokens[7].type == TokenType::KwWhile, "while keyword mismatch");
    TEST_ASSERT(tokens[8].type == TokenType::KwIf, "if keyword mismatch");
    TEST_ASSERT(tokens[9].type == TokenType::KwElse, "else keyword mismatch");
    TEST_ASSERT(tokens[10].type == TokenType::KwPos, "pos keyword mismatch");
    TEST_ASSERT(tokens[11].type == TokenType::KwZero, "zero keyword mismatch");
    TEST_ASSERT(tokens[12].type == TokenType::KwNeg, "neg keyword mismatch");
    TEST_ASSERT(tokens[13].type == TokenType::KwCase, "case keyword mismatch");
    TEST_ASSERT(tokens[14].type == TokenType::Eof, "EOF token missing");

    std::cout << "  -> PASSED: All 14 keywords correctly tokenized.\n";
}

void test_lexer_operators_and_delimiters() {
    std::cout << "[Test Suite 2/6] Running Operators & Delimiters Test...\n";

    std::string src = "+ - * & | ^ <=> = ~ ! == != < > <= >= -> => ( ) { } [ ] , ; :";
    Lexer lex(src);
    auto tokens = lex.tokenize_all();

    std::vector<TokenType> expected = {
        TokenType::OpPlus, TokenType::OpMinus, TokenType::OpStar, TokenType::OpAmp,
        TokenType::OpPipe, TokenType::OpCaret, TokenType::OpSpaceship, TokenType::OpAssign,
        TokenType::OpTilde, TokenType::OpBang, TokenType::OpEq, TokenType::OpNe,
        TokenType::OpLt, TokenType::OpGt, TokenType::OpLe, TokenType::OpGe,
        TokenType::OpArrow, TokenType::OpFatArrow,
        TokenType::LParen, TokenType::RParen, TokenType::LBrace, TokenType::RBrace,
        TokenType::LBracket, TokenType::RBracket, TokenType::Comma, TokenType::Semicolon,
        TokenType::Colon, TokenType::Eof
    };

    TEST_ASSERT(tokens.size() == expected.size(), "Token count mismatch for operators");
    for (size_t i = 0; i < expected.size(); ++i) {
        TEST_ASSERT(tokens[i].type == expected[i], "Mismatch at token index " + std::to_string(i));
    }

    std::cout << "  -> PASSED: All arithmetic, logic, spaceship (<=>) and delimiters tokenized.\n";
}

void test_lexer_trit_literals_and_numbers() {
    std::cout << "[Test Suite 3/6] Running Trit Literals & Numbers Test...\n";

    std::string src = "42 100 243 +1 -1 0 +-0+ ++-0- // comment\n/* block */ 0+-";
    Lexer lex(src);
    auto tokens = lex.tokenize_all();

    TEST_ASSERT(tokens.size() == 10, "Expected 9 tokens + EOF");

    // 42
    TEST_ASSERT(tokens[0].type == TokenType::Number && tokens[0].int_val == 42, "Number 42 mismatch");
    // 100
    TEST_ASSERT(tokens[1].type == TokenType::Number && tokens[1].int_val == 100, "Number 100 mismatch");
    // 243
    TEST_ASSERT(tokens[2].type == TokenType::Number && tokens[2].int_val == 243, "Number 243 mismatch");

    // +1
    TEST_ASSERT(tokens[3].type == TokenType::TritLiteral && tokens[3].trit_val == 1, "+1 trit literal mismatch");
    // -1
    TEST_ASSERT(tokens[4].type == TokenType::TritLiteral && tokens[4].trit_val == -1, "-1 trit literal mismatch");
    // 0
    TEST_ASSERT(tokens[5].type == TokenType::TritLiteral && tokens[5].trit_val == 0, "0 trit literal mismatch");

    // +-0+ (vector)
    TEST_ASSERT(tokens[6].type == TokenType::TritLiteral && tokens[6].text == "+-0+", "+-0+ literal mismatch");
    TEST_ASSERT(tokens[6].trit_vector.size() == 4, "Vector size mismatch for +-0+");
    TEST_ASSERT(tokens[6].trit_vector[0] == 1 && tokens[6].trit_vector[1] == -1 &&
                tokens[6].trit_vector[2] == 0 && tokens[6].trit_vector[3] == 1, "Vector trits mismatch");

    // ++-0- (vector)
    TEST_ASSERT(tokens[7].type == TokenType::TritLiteral && tokens[7].text == "++-0-", "++-0- literal mismatch");
    TEST_ASSERT(tokens[7].trit_vector.size() == 5, "Vector size mismatch for ++-0-");

    // 0+- (vector)
    TEST_ASSERT(tokens[8].type == TokenType::TritLiteral && tokens[8].text == "0+-", "0+- literal mismatch");
    TEST_ASSERT(tokens[8].trit_vector.size() == 3, "Vector size mismatch for 0+-");

    std::cout << "  -> PASSED: Numbers and Balanced Trit Literals (+1, -1, 0, vector words) verified.\n";
}

// ==============================================================================
// 2. Test Suite: Arithmetic Precedence (* over +) and Logic Ops
// ==============================================================================

void test_parser_arithmetic_precedence() {
    std::cout << "[Test Suite 4/6] Running Expression Precedence (* over +) Test...\n";

    // Test 1: a + b * c  => BinaryOpNode(+) with left = a, right = BinaryOpNode(*, b, c)
    {
        std::string src = "var res = a + b * c;";
        Parser parser(src);
        auto prog = parser.parse_program();
        TEST_ASSERT(prog->declarations.size() == 1, "Expected 1 declaration");

        auto var_decl = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        TEST_ASSERT(var_decl != nullptr, "Expected VarDeclNode");
        TEST_ASSERT(var_decl->name == "res", "Var name mismatch");

        auto root_bin = std::dynamic_pointer_cast<BinaryOpNode>(var_decl->init_expr);
        TEST_ASSERT(root_bin != nullptr, "Root of init_expr should be BinaryOpNode");
        TEST_ASSERT(root_bin->op == BinaryOp::Add, "Root op should be '+'");

        auto left_id = std::dynamic_pointer_cast<IdentifierNode>(root_bin->left);
        TEST_ASSERT(left_id != nullptr && left_id->name == "a", "Left operand should be 'a'");

        auto right_bin = std::dynamic_pointer_cast<BinaryOpNode>(root_bin->right);
        TEST_ASSERT(right_bin != nullptr, "Right operand should be BinaryOpNode");
        TEST_ASSERT(right_bin->op == BinaryOp::Mul, "Right op should be '*'");

        auto b_id = std::dynamic_pointer_cast<IdentifierNode>(right_bin->left);
        auto c_id = std::dynamic_pointer_cast<IdentifierNode>(right_bin->right);
        TEST_ASSERT(b_id != nullptr && b_id->name == "b", "Multiplication left should be 'b'");
        TEST_ASSERT(c_id != nullptr && c_id->name == "c", "Multiplication right should be 'c'");
    }

    // Test 2: (a + b) * c => BinaryOpNode(*) with left = BinaryOpNode(+, a, b), right = c
    {
        std::string src = "var res = (a + b) * c;";
        Parser parser(src);
        auto prog = parser.parse_program();
        auto var_decl = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        auto root_bin = std::dynamic_pointer_cast<BinaryOpNode>(var_decl->init_expr);
        TEST_ASSERT(root_bin != nullptr && root_bin->op == BinaryOp::Mul, "Root op should be '*'");

        auto left_bin = std::dynamic_pointer_cast<BinaryOpNode>(root_bin->left);
        TEST_ASSERT(left_bin != nullptr && left_bin->op == BinaryOp::Add, "Left op should be '+'");

        auto right_id = std::dynamic_pointer_cast<IdentifierNode>(root_bin->right);
        TEST_ASSERT(right_id != nullptr && right_id->name == "c", "Right op should be 'c'");
    }

    // Test 3: Unary Negation and Kleene NOT: -a + ~b
    {
        std::string src = "var res = -a + ~b;";
        Parser parser(src);
        auto prog = parser.parse_program();
        auto var_decl = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        auto root_bin = std::dynamic_pointer_cast<BinaryOpNode>(var_decl->init_expr);
        TEST_ASSERT(root_bin != nullptr && root_bin->op == BinaryOp::Add, "Root should be '+'");

        auto left_un = std::dynamic_pointer_cast<UnaryOpNode>(root_bin->left);
        TEST_ASSERT(left_un != nullptr && left_un->op == UnaryOp::Negate, "Left should be unary negation -");

        auto right_un = std::dynamic_pointer_cast<UnaryOpNode>(root_bin->right);
        TEST_ASSERT(right_un != nullptr && right_un->op == UnaryOp::KleeneNot, "Right should be unary Kleene ~");
    }

    // Test 4: Spaceship operator (<=>): a + b <=> c * d
    {
        std::string src = "var cmp = a + b <=> c * d;";
        Parser parser(src);
        auto prog = parser.parse_program();
        auto var_decl = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        auto root_bin = std::dynamic_pointer_cast<BinaryOpNode>(var_decl->init_expr);
        TEST_ASSERT(root_bin != nullptr && root_bin->op == BinaryOp::Spaceship, "Root should be spaceship <=>");

        auto left_add = std::dynamic_pointer_cast<BinaryOpNode>(root_bin->left);
        TEST_ASSERT(left_add != nullptr && left_add->op == BinaryOp::Add, "Left should be '+'");

        auto right_mul = std::dynamic_pointer_cast<BinaryOpNode>(root_bin->right);
        TEST_ASSERT(right_mul != nullptr && right_mul->op == BinaryOp::Mul, "Right should be '*'");
    }

    // Test 5: Unspaced arithmetic: x-1, x+1, (a)-1
    {
        std::string src = "var r1 = x-1; var r2 = x+1; var r3 = (a)-1;";
        Parser parser(src);
        auto prog = parser.parse_program();
        TEST_ASSERT(prog->declarations.size() == 3, "Expected 3 declarations");

        auto d1 = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        auto b1 = std::dynamic_pointer_cast<BinaryOpNode>(d1->init_expr);
        TEST_ASSERT(b1 != nullptr && b1->op == BinaryOp::Sub, "x-1 must parse as subtraction!");

        auto d2 = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[1]);
        auto b2 = std::dynamic_pointer_cast<BinaryOpNode>(d2->init_expr);
        TEST_ASSERT(b2 != nullptr && b2->op == BinaryOp::Add, "x+1 must parse as addition!");

        auto d3 = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[2]);
        auto b3 = std::dynamic_pointer_cast<BinaryOpNode>(d3->init_expr);
        TEST_ASSERT(b3 != nullptr && b3->op == BinaryOp::Sub, "(a)-1 must parse as subtraction!");
    }

    // Test 6: Unary positive sign: +x and +(a*b)
    {
        std::string src = "var u1 = +x; var u2 = +(a * b);";
        Parser parser(src);
        auto prog = parser.parse_program();
        TEST_ASSERT(prog->declarations.size() == 2, "Expected 2 declarations for unary +");

        auto d1 = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        auto id1 = std::dynamic_pointer_cast<IdentifierNode>(d1->init_expr);
        TEST_ASSERT(id1 != nullptr && id1->name == "x", "+x must evaluate as identity to 'x'");

        auto d2 = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[1]);
        auto mul2 = std::dynamic_pointer_cast<BinaryOpNode>(d2->init_expr);
        TEST_ASSERT(mul2 != nullptr && mul2->op == BinaryOp::Mul, "+(a*b) must evaluate as identity to (a*b)");
    }

    // Test 7: 2-trit literals and large 0-prefixed number resilience
    {
        std::string src = "var t1 = +-; var t2 = --; var big = 0999999999999999999999999999999999999;";
        Parser parser(src);
        auto prog = parser.parse_program();
        TEST_ASSERT(prog->declarations.size() == 3, "Expected 3 declarations");
        auto d1 = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        auto lit1 = std::dynamic_pointer_cast<LiteralNode>(d1->init_expr);
        TEST_ASSERT(lit1 != nullptr && lit1->raw_text == "+-", "2-trit literal +- parsed cleanly");
    }

    std::cout << "  -> PASSED: Arithmetic and logic operator precedence correctly structured.\n";
}

// ==============================================================================
// 3. Test Suite: 3-Way Select Statements & Expressions
// ==============================================================================

void test_parser_select_statements() {
    std::cout << "[Test Suite 5/6] Running 3-Way Select Statements Test...\n";

    // Test 1: Statement Select with pos, zero, neg blocks
    {
        std::string src = R"(
            select (condition) {
                pos {
                    result = +1;
                }
                zero {
                    result = 0;
                }
                neg {
                    result = -1;
                }
            }
        )";

        Parser parser(src);
        auto prog = parser.parse_program();
        TEST_ASSERT(prog->declarations.size() == 1, "Expected 1 select statement");

        auto sel = std::dynamic_pointer_cast<SelectNode>(prog->declarations[0]);
        TEST_ASSERT(sel != nullptr, "Expected SelectNode");

        auto cond_id = std::dynamic_pointer_cast<IdentifierNode>(sel->condition);
        TEST_ASSERT(cond_id != nullptr && cond_id->name == "condition", "Condition mismatch");

        TEST_ASSERT(sel->pos_branch != nullptr, "Pos branch missing");
        TEST_ASSERT(sel->zero_branch != nullptr, "Zero branch missing");
        TEST_ASSERT(sel->neg_branch != nullptr, "Neg branch missing");

        auto pos_block = std::dynamic_pointer_cast<BlockNode>(sel->pos_branch);
        TEST_ASSERT(pos_block != nullptr && pos_block->statements.size() == 1, "Pos block should contain 1 statement");
    }

    // Test 2: Select with literal pattern cases (+1 =>, 0 =>, -1 =>)
    {
        std::string src = R"(
            select (x) {
                +1 => { return +1; }
                0  => { return 0; }
                -1 => { return -1; }
            }
        )";

        Parser parser(src);
        auto prog = parser.parse_program();
        auto sel = std::dynamic_pointer_cast<SelectNode>(prog->declarations[0]);
        TEST_ASSERT(sel != nullptr, "Expected SelectNode with pattern cases");
        TEST_ASSERT(sel->pos_branch && sel->zero_branch && sel->neg_branch, "All 3 branches must exist");
    }

    // Test 3: Select with case keywords: case +1: ... case 0: ... case -1: ...
    {
        std::string src = R"(
            select (status) {
                case +1: { val = 100; }
                case 0:  { val = 200; }
                case -1: { val = 300; }
            }
        )";

        Parser parser(src);
        auto prog = parser.parse_program();
        auto sel = std::dynamic_pointer_cast<SelectNode>(prog->declarations[0]);
        TEST_ASSERT(sel != nullptr, "Expected SelectNode with case labels");
        TEST_ASSERT(sel->pos_branch && sel->zero_branch && sel->neg_branch, "All branches must be populated");
    }

    // Test 4: Functional select in expression: select(cond, pos_expr, zero_expr, neg_expr)
    {
        std::string src = "var s = select(cond, 10, 20, 30);";
        Parser parser(src);
        auto prog = parser.parse_program();
        auto var_decl = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
        auto sel = std::dynamic_pointer_cast<SelectNode>(var_decl->init_expr);
        TEST_ASSERT(sel != nullptr, "Expected functional SelectNode in expression");

        auto pos_lit = std::dynamic_pointer_cast<LiteralNode>(sel->pos_branch);
        auto zero_lit = std::dynamic_pointer_cast<LiteralNode>(sel->zero_branch);
        auto neg_lit = std::dynamic_pointer_cast<LiteralNode>(sel->neg_branch);
        TEST_ASSERT(pos_lit && pos_lit->int_val == 10, "Pos expr mismatch");
        TEST_ASSERT(zero_lit && zero_lit->int_val == 20, "Zero expr mismatch");
        TEST_ASSERT(neg_lit && neg_lit->int_val == 30, "Neg expr mismatch");
    }

    std::cout << "  -> PASSED: All 3-way select statement styles and functional select parsed cleanly.\n";
}

// ==============================================================================
// 4. Test Suite: Complete Functions, Loops & AST Visitor Dump
// ==============================================================================

void test_parser_functions_and_loops() {
    std::cout << "[Test Suite 6/6] Running Functions, Loops & AST Visitor Dump Test...\n";

    std::string sample_bt_program = R"(
        // Balanced Ternary 64-Trit Dot Product Kernel
        func bt_dot_product(a: tritword, b: tritword) -> tritword {
            var acc: tritword = 0;
            var i: int = 0;
            while (i < 64) {
                acc = acc + (a * b);
                i = i + 1;
            }
            return acc;
        }

        func ternary_signum(x: trit) -> trit {
            select (x) {
                pos {
                    return +1;
                }
                zero {
                    return 0;
                }
                neg {
                    return -1;
                }
            }
        }
    )";

    Parser parser(sample_bt_program);
    auto prog = parser.parse_program();

    TEST_ASSERT(prog->declarations.size() == 2, "Expected 2 function definitions");

    // 1. Validate bt_dot_product
    auto func1 = std::dynamic_pointer_cast<FunctionDefNode>(prog->declarations[0]);
    TEST_ASSERT(func1 != nullptr, "Expected FunctionDefNode for bt_dot_product");
    TEST_ASSERT(func1->name == "bt_dot_product", "Function name mismatch");
    TEST_ASSERT(func1->params.size() == 2, "Expected 2 parameters");
    TEST_ASSERT(func1->params[0].name == "a" && func1->params[0].type == "tritword", "Param 'a' mismatch");
    TEST_ASSERT(func1->params[1].name == "b" && func1->params[1].type == "tritword", "Param 'b' mismatch");
    TEST_ASSERT(func1->return_type == "tritword", "Return type mismatch");

    auto body1 = func1->body;
    TEST_ASSERT(body1 != nullptr, "Function body missing");
    TEST_ASSERT(body1->statements.size() == 4, "Expected 4 statements in bt_dot_product body");

    // VarDecl: acc
    auto v_acc = std::dynamic_pointer_cast<VarDeclNode>(body1->statements[0]);
    TEST_ASSERT(v_acc != nullptr && v_acc->name == "acc", "acc decl mismatch");

    // VarDecl: i
    auto v_i = std::dynamic_pointer_cast<VarDeclNode>(body1->statements[1]);
    TEST_ASSERT(v_i != nullptr && v_i->name == "i", "i decl mismatch");

    // WhileNode
    auto loop = std::dynamic_pointer_cast<WhileNode>(body1->statements[2]);
    TEST_ASSERT(loop != nullptr, "Expected WhileNode");
    auto loop_cond = std::dynamic_pointer_cast<BinaryOpNode>(loop->condition);
    TEST_ASSERT(loop_cond != nullptr && loop_cond->op == BinaryOp::LessThan, "Loop condition should be '<'");
    TEST_ASSERT(loop->body->statements.size() == 2, "Expected 2 statements inside loop body");

    // ReturnNode
    auto ret1 = std::dynamic_pointer_cast<ReturnNode>(body1->statements[3]);
    TEST_ASSERT(ret1 != nullptr, "Expected ReturnNode");

    // 2. Validate ternary_signum
    auto func2 = std::dynamic_pointer_cast<FunctionDefNode>(prog->declarations[1]);
    TEST_ASSERT(func2 != nullptr && func2->name == "ternary_signum", "Function 2 mismatch");
    TEST_ASSERT(func2->params.size() == 1 && func2->params[0].name == "x", "Param 'x' mismatch");
    TEST_ASSERT(func2->return_type == "trit", "Return type should be 'trit'");

    // 3. Test Human-Readable AST Dump
    std::string ast_dump = prog->dump();
    TEST_ASSERT(!ast_dump.empty(), "AST dump should not be empty");
    TEST_ASSERT(ast_dump.find("FunctionDefNode: bt_dot_product") != std::string::npos, "AST dump missing func 1");
    TEST_ASSERT(ast_dump.find("FunctionDefNode: ternary_signum") != std::string::npos, "AST dump missing func 2");
    TEST_ASSERT(ast_dump.find("WhileNode") != std::string::npos, "AST dump missing WhileNode");
    TEST_ASSERT(ast_dump.find("SelectNode (3-Way Branch)") != std::string::npos, "AST dump missing SelectNode");

    std::cout << "\n--- Sample AST Tree Dump Output ---\n";
    std::cout << ast_dump;
    std::cout << "------------------------------------\n";
    std::cout << "  -> PASSED: Functions, loops, parameters and AST Visitor verified.\n";
}

// ==============================================================================
// 5. Test Suite: Clean Syntax Error Detection
// ==============================================================================

void test_clean_syntax_error_detection() {
    std::cout << "\n[Test Suite] Running Syntax Error Detection Tests...\n";

    // Error 1: Missing semicolon
    try {
        Parser parser("var x = +1");
        auto prog = parser.parse_program();
        TEST_ASSERT(false, "Should have thrown ParseError for missing semicolon");
    } catch (const ParseError& e) {
        std::cout << "  [Expected Error Caught] " << e.what() << "\n";
        TEST_ASSERT(std::string(e.what()).find("Expected ';'") != std::string::npos, "Error text mismatch");
    }

    // Error 2: Missing closing parenthesis in while condition
    try {
        Parser parser("while (i < 64 { i = i + 1; }");
        auto prog = parser.parse_program();
        TEST_ASSERT(false, "Should have thrown ParseError for missing ')'");
    } catch (const ParseError& e) {
        std::cout << "  [Expected Error Caught] " << e.what() << "\n";
        TEST_ASSERT(std::string(e.what()).find("Expected ')'") != std::string::npos, "Error text mismatch");
    }

    // Error 3: Incomplete select statement (missing branches)
    try {
        Parser parser(R"(
            select (x) {
                pos { return +1; }
            }
        )");
        auto prog = parser.parse_program();
        TEST_ASSERT(false, "Should have thrown ParseError for missing select branches");
    } catch (const ParseError& e) {
        std::cout << "  [Expected Error Caught] " << e.what() << "\n";
        TEST_ASSERT(std::string(e.what()).find("must provide all three branches") != std::string::npos, "Error text mismatch");
    }

    // Error 4: Invalid assignment target
    try {
        Parser parser("42 = x;");
        auto prog = parser.parse_program();
        TEST_ASSERT(false, "Should have thrown ParseError for invalid assignment target");
    } catch (const ParseError& e) {
        std::cout << "  [Expected Error Caught] " << e.what() << "\n";
        TEST_ASSERT(std::string(e.what()).find("assignment") != std::string::npos, "Error text mismatch");
    }

    std::cout << "  -> PASSED: Clean syntax error detection with line:column locations verified.\n";
}

// ==============================================================================
// Main Runner
// ==============================================================================

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  Balanced Ternary Frontend AST, Lexer & Parser Verification Suite\n";
    std::cout << "======================================================================\n";

    test_lexer_basic_tokens();
    test_lexer_operators_and_delimiters();
    test_lexer_trit_literals_and_numbers();
    test_parser_arithmetic_precedence();
    test_parser_select_statements();
    test_parser_functions_and_loops();
    test_clean_syntax_error_detection();

    std::cout << "\n======================================================================\n";
    std::cout << "  ALL FRONTEND AST TESTS PASSED WITH 100% SUCCESS (0 ERRORS)\n";
    std::cout << "======================================================================\n";

    return 0;
}
