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

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Frontend AST Unit Tests...\n";
    test_lexer_tokens();
    test_parser_expression();
    std::cout << "  -> PASSED: All AST Unit Tests Verified.\n";
    return 0;
}
