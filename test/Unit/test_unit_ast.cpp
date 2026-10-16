/**
 * @file test_unit_ast.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Frontend Lexer, Parser & AST
 */

#include "../../ternary_frontend_ast.hpp"
#include <iostream>
#include <cassert>

using namespace bt::frontend;

void test_lexer_tokens() {
    std::string src = "func var trit return + - *";
    Lexer lexer(src);
    auto tokens = lexer.tokenize_all();
    assert(!tokens.empty());
    assert(tokens[0].type == TokenType::KwFunc);
    assert(tokens[1].type == TokenType::KwVar);
    assert(tokens[2].type == TokenType::KwTrit);
}

void test_parser_expression() {
    std::string src = "var res = (a + b) * c;";
    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);
    assert(!prog->declarations.empty());
}

void test_comparison_precedence_parsing() {
    std::string src = "var res = a < b == c < d;";
    Parser parser(src);
    auto prog = parser.parse_program();
    assert(prog != nullptr);
    assert(prog->declarations.size() == 1);

    auto vdecl = std::dynamic_pointer_cast<VarDeclNode>(prog->declarations[0]);
    assert(vdecl != nullptr);
    auto bin = std::dynamic_pointer_cast<BinaryOpNode>(vdecl->init_expr);
    assert(bin != nullptr);
    assert(bin->op == BinaryOp::Equal && "Root operator of 'a < b == c < d' must be Equal");

    auto left = std::dynamic_pointer_cast<BinaryOpNode>(bin->left);
    assert(left != nullptr && left->op == BinaryOp::LessThan);

    auto right = std::dynamic_pointer_cast<BinaryOpNode>(bin->right);
    assert(right != nullptr && right->op == BinaryOp::LessThan);
}

void test_float_lookahead_ast() {
    Lexer lex_incomplete("123e");
    assert(!lex_incomplete.is_float_ahead());

    Lexer lex_incomplete_sign("123e-");
    assert(!lex_incomplete_sign.is_float_ahead());

    Lexer lex_valid("123e-4");
    assert(lex_valid.is_float_ahead());
}

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Frontend AST Unit Tests...\n";
    test_lexer_tokens();
    test_parser_expression();
    test_comparison_precedence_parsing();
    test_float_lookahead_ast();
    std::cout << "  -> PASSED: All AST Unit Tests Verified.\n";
    return 0;
}
