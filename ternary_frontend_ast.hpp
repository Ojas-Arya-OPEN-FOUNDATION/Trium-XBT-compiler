/**
 * @file ternary_frontend_ast.hpp
 * @brief High-Performance C++20 Frontend AST, Lexer, and Recursive-Descent Parser for Balanced Ternary (.bt)
 * @author Antigravity Prime Research & Engineering Engine
 * 
 * Part of the Universal Balanced Ternary Compiler Infrastructure.
 * Fully compliant with C++20 standard, targeting hardware-agnostic ternary computing.
 */

#ifndef TERNARY_FRONTEND_AST_HPP
#define TERNARY_FRONTEND_AST_HPP

#include "ternary_dense_algebra.hpp"
#include "ternary_fp.hpp"
#include "ternary_sparse.hpp"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <sstream>
#include <cctype>
#include <stdexcept>
#include <optional>
#include <cstdint>
#include <utility>
#include <initializer_list>
#include <iomanip>

namespace bt::frontend {

// ==============================================================================
// 1. Source Location & Tokens
// ==============================================================================

struct SourceLocation {
    size_t line{1};
    size_t column{1};

    std::string to_string() const {
        return std::to_string(line) + ":" + std::to_string(column);
    }
};

enum class TokenType {
    // Special
    Eof,
    Error,

    // Identifiers & Literals
    Ident,           // variable, function names
    Number,          // integer constants: 0, 42, 100
    FloatLiteral,    // floating point constants: 3.5, 0.75, 1.2e-3, -0.5
    TritLiteral,     // +1, -1, 0, +, -, or multi-trit words (+-0+)

    // Keywords
    KwFunc,          // func
    KwVar,           // var
    KwTrit,          // trit
    KwTritWord,      // tritword
    KwTritMatrix,    // tritmatrix
    KwFloat,         // float
    KwSparseMatrix,  // sparse_matrix
    KwReturn,        // return
    KwSelect,        // select
    KwWhile,         // while
    KwIf,            // if
    KwElse,          // else
    KwPos,           // pos
    KwZero,          // zero
    KwNeg,           // neg
    KwCase,          // case

    // Operators
    OpPlus,          // +
    OpMinus,         // -
    OpStar,          // *
    OpAmp,           // &
    OpPipe,          // |
    OpCaret,         // ^
    OpSpaceship,     // <=>
    OpAssign,        // =
    OpTilde,         // ~ (Kleene NOT)
    OpBang,          // ! (Kleene NOT alternative)
    OpEq,            // ==
    OpNe,            // !=
    OpLt,            // <
    OpGt,            // >
    OpLe,            // <=
    OpGe,            // >=
    OpArrow,         // ->
    OpFatArrow,      // =>

    // Delimiters
    LParen,          // (
    RParen,          // )
    LBrace,          // {
    RBrace,          // }
    LBracket,        // [
    RBracket,        // ]
    Comma,           // ,
    Semicolon,       // ;
    Colon            // :
};

inline std::string token_type_name(TokenType t) {
    switch (t) {
        case TokenType::Eof:          return "EOF";
        case TokenType::Error:        return "ERROR";
        case TokenType::Ident:        return "IDENT";
        case TokenType::Number:       return "NUMBER";
        case TokenType::FloatLiteral: return "FLOAT_LITERAL";
        case TokenType::TritLiteral:  return "TRIT_LITERAL";
        case TokenType::KwFunc:       return "KEYWORD(func)";
        case TokenType::KwVar:        return "KEYWORD(var)";
        case TokenType::KwTrit:       return "KEYWORD(trit)";
        case TokenType::KwTritWord:   return "KEYWORD(tritword)";
        case TokenType::KwTritMatrix: return "KEYWORD(tritmatrix)";
        case TokenType::KwFloat:      return "KEYWORD(float)";
        case TokenType::KwSparseMatrix: return "KEYWORD(sparse_matrix)";
        case TokenType::KwReturn:     return "KEYWORD(return)";
        case TokenType::KwSelect:     return "KEYWORD(select)";
        case TokenType::KwWhile:      return "KEYWORD(while)";
        case TokenType::KwIf:         return "KEYWORD(if)";
        case TokenType::KwElse:       return "KEYWORD(else)";
        case TokenType::KwPos:        return "KEYWORD(pos)";
        case TokenType::KwZero:       return "KEYWORD(zero)";
        case TokenType::KwNeg:        return "KEYWORD(neg)";
        case TokenType::KwCase:       return "KEYWORD(case)";
        case TokenType::OpPlus:       return "OPERATOR(+)";
        case TokenType::OpMinus:      return "OPERATOR(-)";
        case TokenType::OpStar:       return "OPERATOR(*)";
        case TokenType::OpAmp:        return "OPERATOR(&)";
        case TokenType::OpPipe:       return "OPERATOR(|)";
        case TokenType::OpCaret:      return "OPERATOR(^)";
        case TokenType::OpSpaceship:  return "OPERATOR(<=>)";
        case TokenType::OpAssign:     return "OPERATOR(=)";
        case TokenType::OpTilde:      return "OPERATOR(~)";
        case TokenType::OpBang:       return "OPERATOR(!)";
        case TokenType::OpEq:         return "OPERATOR(==)";
        case TokenType::OpNe:         return "OPERATOR(!=)";
        case TokenType::OpLt:         return "OPERATOR(<)";
        case TokenType::OpGt:         return "OPERATOR(>)";
        case TokenType::OpLe:         return "OPERATOR(<=)";
        case TokenType::OpGe:         return "OPERATOR(>=)";
        case TokenType::OpArrow:      return "OPERATOR(->)";
        case TokenType::OpFatArrow:   return "OPERATOR(=>)";
        case TokenType::LParen:       return "DELIMITER('(')";
        case TokenType::RParen:       return "DELIMITER(')')";
        case TokenType::LBrace:       return "DELIMITER('{')";
        case TokenType::RBrace:       return "DELIMITER('}')";
        case TokenType::LBracket:     return "DELIMITER('[')";
        case TokenType::RBracket:     return "DELIMITER(']')";
        case TokenType::Comma:        return "DELIMITER(',')";
        case TokenType::Semicolon:    return "DELIMITER(';')";
        case TokenType::Colon:        return "DELIMITER(':')";
    }
    return "UNKNOWN";
}

struct Token {
    TokenType type{TokenType::Eof};
    std::string text;
    SourceLocation loc;
    int64_t int_val{0};
    double float_val{0.0};
    int8_t trit_val{0}; // -1, 0, +1
    std::vector<int8_t> trit_vector;

    Token() = default;
    Token(TokenType t, std::string txt, SourceLocation l)
        : type(t), text(std::move(txt)), loc(l) {}
    Token(TokenType t, std::string txt, SourceLocation l, int64_t ival, int8_t tval)
        : type(t), text(std::move(txt)), loc(l), int_val(ival), trit_val(tval) {}
    Token(TokenType t, std::string txt, SourceLocation l, int64_t ival, int8_t tval, std::vector<int8_t> tvec)
        : type(t), text(std::move(txt)), loc(l), int_val(ival), trit_val(tval), trit_vector(std::move(tvec)) {}
    Token(TokenType t, std::string txt, SourceLocation l, int64_t ival, double fval, int8_t tval, std::vector<int8_t> tvec)
        : type(t), text(std::move(txt)), loc(l), int_val(ival), float_val(fval), trit_val(tval), trit_vector(std::move(tvec)) {}

    bool is_trit_literal() const {
        return type == TokenType::TritLiteral || (type == TokenType::Number && text == "0");
    }

    bool is_number() const {
        return type == TokenType::Number || (type == TokenType::TritLiteral && (text == "0" || text == "+1" || text == "-1"));
    }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "[" << loc.to_string() << "] " << token_type_name(type) << " ('" << text << "')";
        if (type == TokenType::TritLiteral) {
            ss << " {trit=" << static_cast<int>(trit_val) << "}";
        } else if (type == TokenType::FloatLiteral) {
            ss << " {float=" << float_val << "}";
        } else if (type == TokenType::Number) {
            ss << " {num=" << int_val << "}";
        }
        return ss.str();
    }
};

// ==============================================================================
// 2. C++20 Lexer for Balanced Ternary Language
// ==============================================================================

class Lexer {
private:
    std::string source_;
    size_t pos_{0};
    size_t line_{1};
    size_t col_{1};
    Token prev_token_{TokenType::Eof, "", {0, 0}};

    char peek(size_t offset = 0) const {
        if (pos_ + offset < source_.size()) {
            return source_[pos_ + offset];
        }
        return '\0';
    }

    char advance() {
        if (pos_ < source_.size()) {
            char c = source_[pos_++];
            if (c == '\n') {
                line_++;
                col_ = 1;
            } else {
                col_++;
            }
            return c;
        }
        return '\0';
    }

    bool had_whitespace_{false};

    void skip_whitespace_and_comments() {
        had_whitespace_ = false;
        while (pos_ < source_.size()) {
            char c = peek();
            if (std::isspace(static_cast<unsigned char>(c))) {
                had_whitespace_ = true;
                advance();
            } else if (c == '/' && peek(1) == '/') {
                had_whitespace_ = true;
                // Line comment
                while (pos_ < source_.size() && peek() != '\n') {
                    advance();
                }
            } else if (c == '/' && peek(1) == '*') {
                had_whitespace_ = true;
                // Block comment
                advance(); advance();
                while (pos_ < source_.size() && !(peek() == '*' && peek(1) == '/')) {
                    advance();
                }
                if (pos_ < source_.size()) {
                    advance(); advance(); // consume '*/'
                }
            } else {
                break;
            }
        }
    }

    static bool is_trit_char(char c) {
        return c == '+' || c == '-' || c == '0' || c == 'T' || c == 't';
    }

    static int8_t char_to_trit(char c) {
        if (c == '+' || c == '1') return 1;
        if (c == '-' || c == 'T' || c == 't') return -1;
        return 0;
    }

    bool prev_can_end_expression() const {
        switch (prev_token_.type) {
            case TokenType::Ident:
            case TokenType::RParen:
            case TokenType::RBracket:
                return true;
            case TokenType::Number:
            case TokenType::FloatLiteral:
            case TokenType::TritLiteral:
                return !had_whitespace_;
            default:
                return false;
        }
    }

    bool is_float_ahead(size_t offset = 0) const {
        size_t cur = pos_ + offset;
        if (cur < source_.size() && (source_[cur] == '+' || source_[cur] == '-')) {
            cur++;
        }
        bool has_digits_before = false;
        while (cur < source_.size() && std::isdigit(static_cast<unsigned char>(source_[cur]))) {
            cur++;
            has_digits_before = true;
        }
        if (cur < source_.size() && source_[cur] == '.') {
            if (has_digits_before) {
                return true;
            } else if (cur + 1 < source_.size() && std::isdigit(static_cast<unsigned char>(source_[cur + 1]))) {
                return true;
            }
        }
        if (has_digits_before && cur < source_.size() && (source_[cur] == 'e' || source_[cur] == 'E')) {
            return true;
        }
        return false;
    }

    Token scan_float() {
        SourceLocation loc{line_, col_};
        size_t start = pos_;
        if (peek() == '+' || peek() == '-') {
            advance();
        }
        while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(peek()))) {
            advance();
        }
        if (peek() == '.') {
            advance();
            while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(peek()))) {
                advance();
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            advance();
            if (peek() == '+' || peek() == '-') {
                advance();
            }
            while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(peek()))) {
                advance();
            }
        }
        std::string text = source_.substr(start, pos_ - start);
        double val = 0.0;
        try {
            val = std::stod(text);
        } catch (...) {
            val = 0.0;
        }
        Token tok{TokenType::FloatLiteral, text, loc};
        tok.float_val = val;
        prev_token_ = tok;
        return tok;
    }

public:
    explicit Lexer(std::string source) : source_(std::move(source)) {}

    std::vector<Token> tokenize_all() {
        std::vector<Token> tokens;
        while (true) {
            Token tok = next_token();
            tokens.push_back(tok);
            if (tok.type == TokenType::Eof) {
                break;
            }
        }
        return tokens;
    }

    Token next_token() {
        skip_whitespace_and_comments();

        SourceLocation loc{line_, col_};

        if (pos_ >= source_.size()) {
            Token tok{TokenType::Eof, "", loc};
            prev_token_ = tok;
            return tok;
        }

        char c = peek();

        // 1. Identifiers and Keywords
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = pos_;
            while (pos_ < source_.size() && (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_')) {
                advance();
            }
            std::string text = source_.substr(start, pos_ - start);
            TokenType type = TokenType::Ident;

            if (text == "func")               type = TokenType::KwFunc;
            else if (text == "var")           type = TokenType::KwVar;
            else if (text == "trit")          type = TokenType::KwTrit;
            else if (text == "tritword")      type = TokenType::KwTritWord;
            else if (text == "tritmatrix")    type = TokenType::KwTritMatrix;
            else if (text == "float")         type = TokenType::KwFloat;
            else if (text == "sparse_matrix") type = TokenType::KwSparseMatrix;
            else if (text == "return")        type = TokenType::KwReturn;
            else if (text == "select")        type = TokenType::KwSelect;
            else if (text == "while")         type = TokenType::KwWhile;
            else if (text == "if")            type = TokenType::KwIf;
            else if (text == "else")          type = TokenType::KwElse;
            else if (text == "pos")           type = TokenType::KwPos;
            else if (text == "zero")          type = TokenType::KwZero;
            else if (text == "neg")           type = TokenType::KwNeg;
            else if (text == "case")          type = TokenType::KwCase;

            Token tok{type, text, loc};
            prev_token_ = tok;
            return tok;
        }

        // 1b. Floating point literals (e.g. 3.5, 0.75, 1.2e-3, .5)
        if (is_float_ahead(0) && (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && std::isdigit(static_cast<unsigned char>(peek(1)))))) {
            return scan_float();
        }

        // 2. Numbers (Decimal Integer Constants)
        if (std::isdigit(static_cast<unsigned char>(c)) && c != '0') {
            size_t start = pos_;
            while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(peek()))) {
                advance();
            }
            std::string text = source_.substr(start, pos_ - start);
            int64_t val = 0;
            try {
                val = std::stoll(text);
            } catch (...) {
                val = 0;
            }
            Token tok{TokenType::Number, text, loc, val, (val == 1 ? (int8_t)1 : (val == -1 ? (int8_t)-1 : (int8_t)0))};
            prev_token_ = tok;
            return tok;
        }

        // 3. Multi-Trit Literal (e.g. +-0+, ++-, --0) or single '0'
        if (c == '0') {
            // Check if multi-trit like 0+-, 0-+
            char next = peek(1);
            if (next == '+' || next == '-' || next == 'T' || next == 't') {
                size_t start = pos_;
                std::vector<int8_t> trits;
                while (pos_ < source_.size() && is_trit_char(peek())) {
                    trits.push_back(char_to_trit(peek()));
                    advance();
                }
                std::string text = source_.substr(start, pos_ - start);
                Token tok{TokenType::TritLiteral, text, loc, 0, 0, trits};
                prev_token_ = tok;
                return tok;
            } else if (std::isdigit(static_cast<unsigned char>(next))) {
                // Multi-digit integer starting with 0 (e.g. 0, 0123)
                size_t start = pos_;
                while (pos_ < source_.size() && std::isdigit(static_cast<unsigned char>(peek()))) {
                    advance();
                }
                std::string text = source_.substr(start, pos_ - start);
                int64_t val = 0;
                try {
                    val = std::stoll(text);
                } catch (...) {
                    val = 0;
                }
                Token tok{TokenType::Number, text, loc, val, 0};
                prev_token_ = tok;
                return tok;
            } else {
                // Single '0' - Can function as TRIT_LITERAL (0) and NUMBER (0)
                advance();
                Token tok{TokenType::TritLiteral, "0", loc, 0, 0, {0}};
                prev_token_ = tok;
                return tok;
            }
        }

        // 4. '+' Sign handling (FloatLiteral, TRIT_LITERAL +1, +, multi-trit, or OpPlus)
        if (c == '+') {
            if (!prev_can_end_expression() && is_float_ahead(0)) {
                return scan_float();
            }
            char next = peek(1);
            if (prev_can_end_expression()) {
                // If previous token can end an expression (e.g. 'x', ')', '5'), '+' is a binary operator!
                advance();
                Token tok{TokenType::OpPlus, "+", loc};
                prev_token_ = tok;
                return tok;
            }

            // Case 4a: multi-trit literal (+-0+, ++-, +-, etc.)
            if (is_trit_char(next)) {
                size_t start = pos_;
                std::vector<int8_t> trits;
                while (pos_ < source_.size() && is_trit_char(peek())) {
                    trits.push_back(char_to_trit(peek()));
                    advance();
                }
                std::string text = source_.substr(start, pos_ - start);
                Token tok{TokenType::TritLiteral, text, loc, 0, 0, trits};
                prev_token_ = tok;
                return tok;
            }
            // Case 4b: explicit +1
            if (next == '1' && !std::isdigit(static_cast<unsigned char>(peek(2))) && !std::isalpha(static_cast<unsigned char>(peek(2)))) {
                advance(); advance(); // consume '+' and '1'
                Token tok{TokenType::TritLiteral, "+1", loc, 1, 1, {1}};
                prev_token_ = tok;
                return tok;
            }
            // Case 4c: Arithmetic OpPlus
            advance();
            Token tok{TokenType::OpPlus, "+", loc};
            prev_token_ = tok;
            return tok;
        }

        // 5. '-' Sign handling (FloatLiteral, TRIT_LITERAL -1, ->, multi-trit, or OpMinus)
        if (c == '-') {
            char next = peek(1);
            // Arrow ->
            if (next == '>') {
                advance(); advance();
                Token tok{TokenType::OpArrow, "->", loc};
                prev_token_ = tok;
                return tok;
            }

            if (!prev_can_end_expression() && is_float_ahead(0)) {
                return scan_float();
            }

            if (prev_can_end_expression()) {
                // If previous token can end an expression, '-' is binary subtraction!
                advance();
                Token tok{TokenType::OpMinus, "-", loc};
                prev_token_ = tok;
                return tok;
            }

            // Case 5a: multi-trit literal (--+, -+0, --, etc.)
            if (is_trit_char(next)) {
                size_t start = pos_;
                std::vector<int8_t> trits;
                while (pos_ < source_.size() && is_trit_char(peek())) {
                    trits.push_back(char_to_trit(peek()));
                    advance();
                }
                std::string text = source_.substr(start, pos_ - start);
                Token tok{TokenType::TritLiteral, text, loc, 0, 0, trits};
                prev_token_ = tok;
                return tok;
            }
            // Case 5b: explicit -1
            if (next == '1' && !std::isdigit(static_cast<unsigned char>(peek(2))) && !std::isalpha(static_cast<unsigned char>(peek(2)))) {
                advance(); advance(); // consume '-' and '1'
                Token tok{TokenType::TritLiteral, "-1", loc, -1, -1, {-1}};
                prev_token_ = tok;
                return tok;
            }
            // Case 5c: OpMinus
            advance();
            Token tok{TokenType::OpMinus, "-", loc};
            prev_token_ = tok;
            return tok;
        }

        // 6. Spaceship (<=>) and Comparison Operators (<, <=)
        if (c == '<') {
            advance();
            if (peek() == '=') {
                advance();
                if (peek() == '>') {
                    advance();
                    Token tok{TokenType::OpSpaceship, "<=>", loc};
                    prev_token_ = tok;
                    return tok;
                }
                Token tok{TokenType::OpLe, "<=", loc};
                prev_token_ = tok;
                return tok;
            }
            Token tok{TokenType::OpLt, "<", loc};
            prev_token_ = tok;
            return tok;
        }

        // 7. Assignment (=), Equality (==), Fat Arrow (=>)
        if (c == '=') {
            advance();
            if (peek() == '=') {
                advance();
                Token tok{TokenType::OpEq, "==", loc};
                prev_token_ = tok;
                return tok;
            } else if (peek() == '>') {
                advance();
                Token tok{TokenType::OpFatArrow, "=>", loc};
                prev_token_ = tok;
                return tok;
            }
            Token tok{TokenType::OpAssign, "=", loc};
            prev_token_ = tok;
            return tok;
        }

        // 8. Bang (!) and Not Equal (!=)
        if (c == '!') {
            advance();
            if (peek() == '=') {
                advance();
                Token tok{TokenType::OpNe, "!=", loc};
                prev_token_ = tok;
                return tok;
            }
            Token tok{TokenType::OpBang, "!", loc};
            prev_token_ = tok;
            return tok;
        }

        // 9. Greater Than (>), Greater Equal (>=)
        if (c == '>') {
            advance();
            if (peek() == '=') {
                advance();
                Token tok{TokenType::OpGe, ">=", loc};
                prev_token_ = tok;
                return tok;
            }
            Token tok{TokenType::OpGt, ">", loc};
            prev_token_ = tok;
            return tok;
        }

        // 10. Single character operators & delimiters
        advance();
        TokenType type = TokenType::Error;
        std::string text(1, c);

        switch (c) {
            case '*': type = TokenType::OpStar; break;
            case '&': type = TokenType::OpAmp; break;
            case '|': type = TokenType::OpPipe; break;
            case '^': type = TokenType::OpCaret; break;
            case '~': type = TokenType::OpTilde; break;
            case '(': type = TokenType::LParen; break;
            case ')': type = TokenType::RParen; break;
            case '{': type = TokenType::LBrace; break;
            case '}': type = TokenType::RBrace; break;
            case '[': type = TokenType::LBracket; break;
            case ']': type = TokenType::RBracket; break;
            case ',': type = TokenType::Comma; break;
            case ';': type = TokenType::Semicolon; break;
            case ':': type = TokenType::Colon; break;
            default:
                type = TokenType::Error;
                break;
        }

        Token tok{type, text, loc};
        prev_token_ = tok;
        return tok;
    }
};

// ==============================================================================
// 3. AST Node Hierarchy
// ==============================================================================

class ASTVisitor;

class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void accept(ASTVisitor& visitor) = 0;
    virtual std::string dump() const;
    SourceLocation location;
};

using ASTNodePtr = std::shared_ptr<ASTNode>;

// Forward declarations of AST nodes
class ProgramNode;
class FunctionDefNode;
class VarDeclNode;
class AssignNode;
class BinaryOpNode;
class UnaryOpNode;
class SelectNode;
class WhileNode;
class ReturnNode;
class LiteralNode;
class BlockNode;
class IdentifierNode;
class CallNode;
class IfNode;
class ExpressionStmtNode;
class FloatLiteralExprAST;
class SparseMatrixDeclAST;

// 1. Program Node (Root)
class ProgramNode : public ASTNode {
public:
    std::vector<ASTNodePtr> declarations;

    void accept(ASTVisitor& visitor) override;
};

// 2. Block Node
class BlockNode : public ASTNode {
public:
    std::vector<ASTNodePtr> statements;

    void accept(ASTVisitor& visitor) override;
};

// 3. Function Definition Node
struct Parameter {
    std::string name;
    std::string type;
    SourceLocation loc;
};

class FunctionDefNode : public ASTNode {
public:
    std::string name;
    std::vector<Parameter> params;
    std::string return_type{"void"};
    std::shared_ptr<BlockNode> body;

    void accept(ASTVisitor& visitor) override;
};

// 4. Variable Declaration Node
class VarDeclNode : public ASTNode {
public:
    std::string name;
    std::string type; // e.g. "trit", "tritword", "tritmatrix", "int"
    ASTNodePtr init_expr;

    void accept(ASTVisitor& visitor) override;
};

// 5. Assignment Node
class AssignNode : public ASTNode {
public:
    std::string target;
    ASTNodePtr value;

    void accept(ASTVisitor& visitor) override;
};

// 6. Binary Operation Node (+, -, *, &, |, ^, <=>, ==, !=, <, >, <=, >=)
enum class BinaryOp {
    Add,         // +
    Sub,         // -
    Mul,         // *
    KleeneAnd,   // &
    KleeneOr,    // |
    TernaryXor,  // ^
    Spaceship,   // <=>
    Equal,       // ==
    NotEqual,    // !=
    LessThan,    // <
    GreaterThan, // >
    LessEqual,   // <=
    GreaterEqual // >=
};

inline std::string binary_op_name(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add:         return "+";
        case BinaryOp::Sub:         return "-";
        case BinaryOp::Mul:         return "*";
        case BinaryOp::KleeneAnd:   return "&";
        case BinaryOp::KleeneOr:    return "|";
        case BinaryOp::TernaryXor:  return "^";
        case BinaryOp::Spaceship:   return "<=>";
        case BinaryOp::Equal:       return "==";
        case BinaryOp::NotEqual:    return "!=";
        case BinaryOp::LessThan:    return "<";
        case BinaryOp::GreaterThan: return ">";
        case BinaryOp::LessEqual:   return "<=";
        case BinaryOp::GreaterEqual:return ">=";
    }
    return "?";
}

class BinaryOpNode : public ASTNode {
public:
    BinaryOp op{BinaryOp::Add};
    std::string op_symbol;
    ASTNodePtr left;
    ASTNodePtr right;

    BinaryOpNode(BinaryOp o, std::string sym, ASTNodePtr l, ASTNodePtr r, SourceLocation loc = {})
        : op(o), op_symbol(std::move(sym)), left(std::move(l)), right(std::move(r)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// 7. Unary Operation Node (Negation -, Kleene NOT ~ / !)
enum class UnaryOp {
    Negate,     // -
    KleeneNot   // ~ or !
};

inline std::string unary_op_name(UnaryOp op) {
    switch (op) {
        case UnaryOp::Negate:   return "-";
        case UnaryOp::KleeneNot:return "~";
    }
    return "?";
}

class UnaryOpNode : public ASTNode {
public:
    UnaryOp op{UnaryOp::Negate};
    std::string op_symbol;
    ASTNodePtr operand;

    UnaryOpNode(UnaryOp o, std::string sym, ASTNodePtr opnd, SourceLocation loc = {})
        : op(o), op_symbol(std::move(sym)), operand(std::move(opnd)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// 8. Select Node (3-way Branch: Pos Block/Expr, Zero Block/Expr, Neg Block/Expr)
class SelectNode : public ASTNode {
public:
    ASTNodePtr condition;
    ASTNodePtr pos_branch;  // Selected when condition == +1
    ASTNodePtr zero_branch; // Selected when condition == 0
    ASTNodePtr neg_branch;  // Selected when condition == -1

    SelectNode() = default;
    SelectNode(ASTNodePtr cond, ASTNodePtr pos_b, ASTNodePtr zero_b, ASTNodePtr neg_b, SourceLocation loc = {})
        : condition(std::move(cond)),
          pos_branch(std::move(pos_b)),
          zero_branch(std::move(zero_b)),
          neg_branch(std::move(neg_b)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// 9. While Node / Loop Node
class WhileNode : public ASTNode {
public:
    ASTNodePtr condition;
    std::shared_ptr<BlockNode> body;

    void accept(ASTVisitor& visitor) override;
};

using LoopNode = WhileNode;

// 10. Return Node
class ReturnNode : public ASTNode {
public:
    ASTNodePtr expr; // nullptr if void return

    void accept(ASTVisitor& visitor) override;
};

// 11. Literal Node (Trit, Integer, Vector)
enum class LiteralKind {
    Trit,
    Integer,
    Vector
};

class LiteralNode : public ASTNode {
public:
    LiteralKind kind{LiteralKind::Integer};
    bt::Trit trit_val{bt::Trit::Zero};
    int64_t int_val{0};
    std::vector<bt::Trit> vector_val;
    std::string raw_text;

    static std::shared_ptr<LiteralNode> make_trit(bt::Trit t, SourceLocation loc = {}) {
        auto node = std::make_shared<LiteralNode>();
        node->kind = LiteralKind::Trit;
        node->trit_val = t;
        node->int_val = bt::to_int(t);
        node->location = loc;
        node->raw_text = (t == bt::Trit::Pos ? "+1" : (t == bt::Trit::Neg ? "-1" : "0"));
        return node;
    }

    static std::shared_ptr<LiteralNode> make_int(int64_t v, SourceLocation loc = {}) {
        auto node = std::make_shared<LiteralNode>();
        node->kind = LiteralKind::Integer;
        node->int_val = v;
        node->trit_val = (v == 1 ? bt::Trit::Pos : (v == -1 ? bt::Trit::Neg : bt::Trit::Zero));
        node->location = loc;
        node->raw_text = std::to_string(v);
        return node;
    }

    static std::shared_ptr<LiteralNode> make_vector(const std::vector<bt::Trit>& v, const std::string& text, SourceLocation loc = {}) {
        auto node = std::make_shared<LiteralNode>();
        node->kind = LiteralKind::Vector;
        node->vector_val = v;
        node->raw_text = text;
        node->location = loc;
        return node;
    }

    void accept(ASTVisitor& visitor) override;
};

// 12. Identifier Node
class IdentifierNode : public ASTNode {
public:
    std::string name;

    explicit IdentifierNode(std::string n, SourceLocation loc = {})
        : name(std::move(n)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// 13. Call Node
class CallNode : public ASTNode {
public:
    std::string callee;
    std::vector<ASTNodePtr> arguments;

    CallNode(std::string name, std::vector<ASTNodePtr> args, SourceLocation loc = {})
        : callee(std::move(name)), arguments(std::move(args)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// 14. If Node
class IfNode : public ASTNode {
public:
    ASTNodePtr condition;
    std::shared_ptr<BlockNode> then_block;
    std::shared_ptr<BlockNode> else_block; // nullptr if absent

    void accept(ASTVisitor& visitor) override;
};

// 15. Expression Statement Node
class ExpressionStmtNode : public ASTNode {
public:
    ASTNodePtr expr;

    explicit ExpressionStmtNode(ASTNodePtr e, SourceLocation loc = {})
        : expr(std::move(e)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// 16. Float Literal Node (BT-FP32)
class FloatLiteralExprAST : public ASTNode {
public:
    double value{0.0};
    std::string raw_text;
    bt::BTFP32 btfp_val;

    FloatLiteralExprAST(double val, std::string text, SourceLocation loc = {})
        : value(val), raw_text(std::move(text)), btfp_val(bt::BTFP32::from_double(val)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// 17. Sparse Matrix Declaration Node (T-CSR 1-Bit Matrix)
class SparseMatrixDeclAST : public ASTNode {
public:
    std::string name;
    size_t rows{0};
    size_t cols{0};
    bt::TCSRMatrix matrix;
    std::vector<ASTNodePtr> init_args;

    SparseMatrixDeclAST(std::string n, size_t r, size_t c, bt::TCSRMatrix mat, SourceLocation loc = {})
        : name(std::move(n)), rows(r), cols(c), matrix(std::move(mat)) {
        location = loc;
    }

    void accept(ASTVisitor& visitor) override;
};

// ==============================================================================
// 4. AST Visitor & Printer
// ==============================================================================

class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;
    virtual void visit(ProgramNode& node) = 0;
    virtual void visit(BlockNode& node) = 0;
    virtual void visit(FunctionDefNode& node) = 0;
    virtual void visit(VarDeclNode& node) = 0;
    virtual void visit(AssignNode& node) = 0;
    virtual void visit(BinaryOpNode& node) = 0;
    virtual void visit(UnaryOpNode& node) = 0;
    virtual void visit(SelectNode& node) = 0;
    virtual void visit(WhileNode& node) = 0;
    virtual void visit(ReturnNode& node) = 0;
    virtual void visit(LiteralNode& node) = 0;
    virtual void visit(FloatLiteralExprAST& node) { (void)node; }
    virtual void visit(SparseMatrixDeclAST& node) { (void)node; }
    virtual void visit(IdentifierNode& node) = 0;
    virtual void visit(CallNode& node) = 0;
    virtual void visit(IfNode& node) = 0;
    virtual void visit(ExpressionStmtNode& node) = 0;
};

inline void ProgramNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void BlockNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void FunctionDefNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void VarDeclNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void AssignNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void BinaryOpNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void UnaryOpNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void SelectNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void WhileNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void ReturnNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void LiteralNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void FloatLiteralExprAST::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void SparseMatrixDeclAST::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void IdentifierNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void CallNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void IfNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }
inline void ExpressionStmtNode::accept(ASTVisitor& visitor) { visitor.visit(*this); }

class ASTPrinter : public ASTVisitor {
private:
    std::ostringstream out_;
    int indent_level_{0};

    void indent() {
        for (int i = 0; i < indent_level_; ++i) {
            out_ << "  ";
        }
    }

public:
    std::string print(ASTNode& node) {
        out_.str("");
        out_.clear();
        indent_level_ = 0;
        node.accept(*this);
        return out_.str();
    }

    void visit(ProgramNode& node) override {
        indent();
        out_ << "ProgramNode\n";
        indent_level_++;
        for (const auto& decl : node.declarations) {
            if (decl) decl->accept(*this);
        }
        indent_level_--;
    }

    void visit(BlockNode& node) override {
        indent();
        out_ << "BlockNode (" << node.statements.size() << " stmts)\n";
        indent_level_++;
        for (const auto& stmt : node.statements) {
            if (stmt) stmt->accept(*this);
        }
        indent_level_--;
    }

    void visit(FunctionDefNode& node) override {
        indent();
        out_ << "FunctionDefNode: " << node.name << " (";
        for (size_t i = 0; i < node.params.size(); ++i) {
            out_ << node.params[i].name << ": " << node.params[i].type;
            if (i + 1 < node.params.size()) out_ << ", ";
        }
        out_ << ") -> " << node.return_type << "\n";
        if (node.body) {
            indent_level_++;
            node.body->accept(*this);
            indent_level_--;
        }
    }

    void visit(VarDeclNode& node) override {
        indent();
        out_ << "VarDeclNode: " << node.name;
        if (!node.type.empty()) out_ << ": " << node.type;
        out_ << "\n";
        if (node.init_expr) {
            indent_level_++;
            indent();
            out_ << "Initializer:\n";
            indent_level_++;
            node.init_expr->accept(*this);
            indent_level_ -= 2;
        }
    }

    void visit(AssignNode& node) override {
        indent();
        out_ << "AssignNode: " << node.target << " =\n";
        if (node.value) {
            indent_level_++;
            node.value->accept(*this);
            indent_level_--;
        }
    }

    void visit(BinaryOpNode& node) override {
        indent();
        out_ << "BinaryOpNode (" << node.op_symbol << ")\n";
        indent_level_++;
        if (node.left) node.left->accept(*this);
        if (node.right) node.right->accept(*this);
        indent_level_--;
    }

    void visit(UnaryOpNode& node) override {
        indent();
        out_ << "UnaryOpNode (" << node.op_symbol << ")\n";
        indent_level_++;
        if (node.operand) node.operand->accept(*this);
        indent_level_--;
    }

    void visit(SelectNode& node) override {
        indent();
        out_ << "SelectNode (3-Way Branch)\n";
        indent_level_++;

        indent(); out_ << "Condition:\n";
        indent_level_++;
        if (node.condition) node.condition->accept(*this);
        indent_level_--;

        indent(); out_ << "Pos Branch (+1):\n";
        indent_level_++;
        if (node.pos_branch) node.pos_branch->accept(*this);
        indent_level_--;

        indent(); out_ << "Zero Branch (0):\n";
        indent_level_++;
        if (node.zero_branch) node.zero_branch->accept(*this);
        indent_level_--;

        indent(); out_ << "Neg Branch (-1):\n";
        indent_level_++;
        if (node.neg_branch) node.neg_branch->accept(*this);
        indent_level_--;

        indent_level_--;
    }

    void visit(WhileNode& node) override {
        indent();
        out_ << "WhileNode\n";
        indent_level_++;

        indent(); out_ << "Condition:\n";
        indent_level_++;
        if (node.condition) node.condition->accept(*this);
        indent_level_--;

        indent(); out_ << "Body:\n";
        indent_level_++;
        if (node.body) node.body->accept(*this);
        indent_level_--;

        indent_level_--;
    }

    void visit(ReturnNode& node) override {
        indent();
        out_ << "ReturnNode\n";
        if (node.expr) {
            indent_level_++;
            node.expr->accept(*this);
            indent_level_--;
        }
    }

    void visit(LiteralNode& node) override {
        indent();
        out_ << "LiteralNode: ";
        switch (node.kind) {
            case LiteralKind::Trit:
                out_ << "Trit(" << (node.trit_val == bt::Trit::Pos ? "+1" : (node.trit_val == bt::Trit::Neg ? "-1" : "0")) << ")";
                break;
            case LiteralKind::Integer:
                out_ << "Integer(" << node.int_val << ")";
                break;
            case LiteralKind::Vector:
                out_ << "Vector(" << node.raw_text << ")";
                break;
        }
        out_ << "\n";
    }

    void visit(IdentifierNode& node) override {
        indent();
        out_ << "IdentifierNode: " << node.name << "\n";
    }

    void visit(CallNode& node) override {
        indent();
        out_ << "CallNode: " << node.callee << " (" << node.arguments.size() << " args)\n";
        indent_level_++;
        for (const auto& arg : node.arguments) {
            if (arg) arg->accept(*this);
        }
        indent_level_--;
    }

    void visit(IfNode& node) override {
        indent();
        out_ << "IfNode\n";
        indent_level_++;
        indent(); out_ << "Condition:\n";
        indent_level_++;
        if (node.condition) node.condition->accept(*this);
        indent_level_--;
        indent(); out_ << "Then:\n";
        indent_level_++;
        if (node.then_block) node.then_block->accept(*this);
        indent_level_--;
        if (node.else_block) {
            indent(); out_ << "Else:\n";
            indent_level_++;
            node.else_block->accept(*this);
            indent_level_--;
        }
        indent_level_--;
    }

    void visit(ExpressionStmtNode& node) override {
        indent();
        out_ << "ExpressionStmtNode\n";
        if (node.expr) {
            indent_level_++;
            node.expr->accept(*this);
            indent_level_--;
        }
    }

    void visit(FloatLiteralExprAST& node) override {
        indent();
        out_ << "FloatLiteralExprAST: " << node.value << " (BTFP32: M="
             << node.btfp_val.mantissa() << ", E=" << node.btfp_val.exponent() << ")\n";
    }

    void visit(SparseMatrixDeclAST& node) override {
        indent();
        out_ << "SparseMatrixDeclAST: " << (node.name.empty() ? "<anon>" : node.name)
             << " [" << node.rows << "x" << node.cols << ", nnz=" << node.matrix.nnz << "]\n";
    }
};

inline std::string ASTNode::dump() const {
    ASTPrinter printer;
    return printer.print(const_cast<ASTNode&>(*this));
}

// ==============================================================================
// 5. Recursive-Descent Parser
// ==============================================================================

class ParseError : public std::runtime_error {
public:
    SourceLocation loc;
    ParseError(const std::string& msg, SourceLocation l)
        : std::runtime_error("[Frontend Parse Error] " + l.to_string() + ": " + msg),
          loc(l) {}
};

class Parser {
private:
    std::vector<Token> tokens_;
    size_t current_{0};

    const Token& peek(size_t offset = 0) const {
        if (current_ + offset < tokens_.size()) {
            return tokens_[current_ + offset];
        }
        static Token eof_tok{TokenType::Eof, "", {0, 0}};
        return eof_tok;
    }

    bool is_at_end() const {
        return peek().type == TokenType::Eof;
    }

    const Token& advance() {
        if (!is_at_end()) current_++;
        return previous();
    }

    const Token& previous() const {
        if (current_ > 0) return tokens_[current_ - 1];
        return tokens_[0];
    }

    bool check(TokenType type) const {
        if (is_at_end()) return false;
        return peek().type == type;
    }

    bool match(std::initializer_list<TokenType> types) {
        for (TokenType type : types) {
            if (check(type)) {
                advance();
                return true;
            }
        }
        return false;
    }

    const Token& consume(TokenType type, const std::string& error_msg) {
        if (check(type)) return advance();
        throw ParseError(error_msg + " (found: " + token_type_name(peek().type) + " '" + peek().text + "')", peek().loc);
    }

public:
    explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    explicit Parser(const std::string& source) {
        Lexer lexer(source);
        tokens_ = lexer.tokenize_all();
    }

    std::shared_ptr<ProgramNode> parse_program() {
        auto prog = std::make_shared<ProgramNode>();
        prog->location = peek().loc;

        while (!is_at_end()) {
            prog->declarations.push_back(parse_declaration_or_statement());
        }

        return prog;
    }

private:
    ASTNodePtr parse_declaration_or_statement() {
        if (match({TokenType::KwFunc})) {
            return parse_function_def();
        }
        if (match({TokenType::KwVar})) {
            return parse_var_decl();
        }
        if (match({TokenType::KwSparseMatrix})) {
            if (check(TokenType::Ident)) {
                return parse_sparse_matrix_decl();
            }
        }
        return parse_statement();
    }

    std::shared_ptr<SparseMatrixDeclAST> parse_sparse_matrix_decl() {
        SourceLocation loc = previous().loc;
        Token name_tok = consume(TokenType::Ident, "Expected matrix name after 'sparse_matrix'");
        consume(TokenType::LParen, "Expected '(' after sparse matrix name");

        Token r_tok = consume(TokenType::Number, "Expected number of rows for sparse_matrix");
        consume(TokenType::Comma, "Expected ',' after rows in sparse_matrix");
        Token c_tok = consume(TokenType::Number, "Expected number of cols for sparse_matrix");

        size_t rows = static_cast<size_t>(r_tok.int_val);
        size_t cols = static_cast<size_t>(c_tok.int_val);

        std::vector<int64_t> elements;
        std::vector<ASTNodePtr> args;
        while (match({TokenType::Comma})) {
            ASTNodePtr arg_expr = parse_expression();
            args.push_back(arg_expr);
            if (auto lit = std::dynamic_pointer_cast<LiteralNode>(arg_expr)) {
                elements.push_back(lit->int_val);
            } else if (auto uop = std::dynamic_pointer_cast<UnaryOpNode>(arg_expr)) {
                if (auto num = std::dynamic_pointer_cast<LiteralNode>(uop->operand)) {
                    elements.push_back(-num->int_val);
                } else {
                    elements.push_back(0);
                }
            } else {
                elements.push_back(0);
            }
        }
        consume(TokenType::RParen, "Expected ')' to close sparse_matrix declaration");
        consume(TokenType::Semicolon, "Expected ';' after sparse_matrix declaration");

        bt::TCSRMatrix mat(rows, cols);
        if (elements.size() == rows * cols) {
            std::vector<int8_t> dense_data(elements.begin(), elements.end());
            mat = bt::TCSRMatrix::from_dense(rows, cols, dense_data);
        } else if (elements.size() % 3 == 0) {
            std::vector<std::tuple<size_t, size_t, bt::Trit>> triplets;
            for (size_t i = 0; i < elements.size(); i += 3) {
                triplets.emplace_back(static_cast<size_t>(elements[i]),
                                      static_cast<size_t>(elements[i + 1]),
                                      bt::from_int(static_cast<int8_t>(elements[i + 2])));
            }
            mat = bt::TCSRMatrix::from_triplets(rows, cols, triplets);
        }

        auto node = std::make_shared<SparseMatrixDeclAST>(name_tok.text, rows, cols, std::move(mat), loc);
        node->init_args = std::move(args);
        return node;
    }

    std::shared_ptr<FunctionDefNode> parse_function_def() {
        SourceLocation loc = previous().loc;
        Token name_tok = consume(TokenType::Ident, "Expected function name after 'func'");

        auto func = std::make_shared<FunctionDefNode>();
        func->name = name_tok.text;
        func->location = loc;

        consume(TokenType::LParen, "Expected '(' after function name");

        if (!check(TokenType::RParen)) {
            do {
                Token p_name = consume(TokenType::Ident, "Expected parameter name");
                std::string p_type = "trit";
                if (match({TokenType::Colon})) {
                    if (check(TokenType::KwTrit) || check(TokenType::KwTritWord) ||
                        check(TokenType::KwTritMatrix) || check(TokenType::KwFloat) ||
                        check(TokenType::KwSparseMatrix) || check(TokenType::Ident)) {
                        p_type = advance().text;
                    } else {
                        throw ParseError("Expected type name for parameter '" + p_name.text + "'", peek().loc);
                    }
                }
                func->params.push_back(Parameter{p_name.text, p_type, p_name.loc});
            } while (match({TokenType::Comma}));
        }

        consume(TokenType::RParen, "Expected ')' after parameter list");

        if (match({TokenType::OpArrow})) {
            if (check(TokenType::KwTrit) || check(TokenType::KwTritWord) ||
                check(TokenType::KwTritMatrix) || check(TokenType::KwFloat) ||
                check(TokenType::KwSparseMatrix) || check(TokenType::Ident)) {
                func->return_type = advance().text;
            } else {
                throw ParseError("Expected return type after '->'", peek().loc);
            }
        }

        func->body = parse_block();
        return func;
    }

    std::shared_ptr<VarDeclNode> parse_var_decl() {
        SourceLocation loc = previous().loc;
        Token name_tok = consume(TokenType::Ident, "Expected variable name after 'var'");

        auto var_decl = std::make_shared<VarDeclNode>();
        var_decl->name = name_tok.text;
        var_decl->location = loc;

        if (match({TokenType::Colon})) {
            if (check(TokenType::KwTrit) || check(TokenType::KwTritWord) ||
                check(TokenType::KwTritMatrix) || check(TokenType::KwFloat) ||
                check(TokenType::KwSparseMatrix) || check(TokenType::Ident)) {
                var_decl->type = advance().text;
            } else {
                throw ParseError("Expected type name after ':'", peek().loc);
            }
        }

        if (match({TokenType::OpAssign})) {
            var_decl->init_expr = parse_expression();
            if (var_decl->type == "sparse_matrix") {
                if (auto sm = std::dynamic_pointer_cast<SparseMatrixDeclAST>(var_decl->init_expr)) {
                    if (sm->name.empty()) sm->name = var_decl->name;
                }
            }
        }

        consume(TokenType::Semicolon, "Expected ';' after variable declaration");
        return var_decl;
    }

    ASTNodePtr parse_statement() {
        if (check(TokenType::LBrace)) {
            return parse_block();
        }
        if (match({TokenType::KwSelect})) {
            auto sel = parse_select();
            match({TokenType::Semicolon});
            return sel;
        }
        if (match({TokenType::KwWhile})) {
            return parse_while();
        }
        if (match({TokenType::KwIf})) {
            return parse_if();
        }
        if (match({TokenType::KwReturn})) {
            return parse_return();
        }

        // Assignment or Expression Statement
        ASTNodePtr expr = parse_expression();

        if (match({TokenType::OpAssign})) {
            SourceLocation loc = previous().loc;
            auto id_node = std::dynamic_pointer_cast<IdentifierNode>(expr);
            if (!id_node) {
                throw ParseError("Invalid left-hand side assignment target", loc);
            }
            ASTNodePtr rhs = parse_expression();
            consume(TokenType::Semicolon, "Expected ';' after assignment");
            auto assign = std::make_shared<AssignNode>();
            assign->target = id_node->name;
            assign->value = rhs;
            assign->location = loc;
            return assign;
        }

        consume(TokenType::Semicolon, "Expected ';' after expression statement");
        return std::make_shared<ExpressionStmtNode>(expr, expr->location);
    }

    std::shared_ptr<BlockNode> parse_block() {
        SourceLocation loc = peek().loc;
        consume(TokenType::LBrace, "Expected '{' to start block");

        auto block = std::make_shared<BlockNode>();
        block->location = loc;

        while (!check(TokenType::RBrace) && !is_at_end()) {
            block->statements.push_back(parse_declaration_or_statement());
        }

        consume(TokenType::RBrace, "Expected '}' to end block");
        return block;
    }

    ASTNodePtr parse_block_or_statement() {
        if (check(TokenType::LBrace)) {
            return parse_block();
        }
        return parse_statement();
    }

    // 3-Way Branch Select Parser (Supports statement block syntax & functional select(c, p, z, n))
    std::shared_ptr<SelectNode> parse_select() {
        SourceLocation loc = previous().loc;
        consume(TokenType::LParen, "Expected '(' after 'select'");
        ASTNodePtr cond = parse_expression();

        // Check if this is an expression-style select: select(c, pos, zero, neg)
        if (match({TokenType::Comma})) {
            ASTNodePtr pos_expr = parse_expression();
            consume(TokenType::Comma, "Expected ',' between pos and zero branches in select(...)");
            ASTNodePtr zero_expr = parse_expression();
            consume(TokenType::Comma, "Expected ',' between zero and neg branches in select(...)");
            ASTNodePtr neg_expr = parse_expression();
            consume(TokenType::RParen, "Expected ')' to close select(...)");
            return std::make_shared<SelectNode>(cond, pos_expr, zero_expr, neg_expr, loc);
        }

        consume(TokenType::RParen, "Expected ')' after select condition");
        consume(TokenType::LBrace, "Expected '{' to begin select branch block");

        ASTNodePtr pos_branch = nullptr;
        ASTNodePtr zero_branch = nullptr;
        ASTNodePtr neg_branch = nullptr;

        while (!check(TokenType::RBrace) && !is_at_end()) {
            bool is_pos = false;
            bool is_zero = false;
            bool is_neg = false;

            if (match({TokenType::KwCase})) {
                // case +1: / case 0: / case -1: / case +: / case -:
                if (check(TokenType::TritLiteral) || check(TokenType::OpPlus) || check(TokenType::OpMinus)) {
                    Token t = advance();
                    if (t.text == "+1" || t.text == "+") is_pos = true;
                    else if (t.text == "0") is_zero = true;
                    else if (t.text == "-1" || t.text == "-") is_neg = true;
                } else if (match({TokenType::KwPos})) is_pos = true;
                else if (match({TokenType::KwZero})) is_zero = true;
                else if (match({TokenType::KwNeg})) is_neg = true;
                else throw ParseError("Expected trit literal (+1, 0, -1) in select case", peek().loc);
                consume(TokenType::Colon, "Expected ':' after select case label");
            } else if (match({TokenType::KwPos})) {
                is_pos = true;
                match({TokenType::OpFatArrow, TokenType::Colon});
            } else if (match({TokenType::KwZero})) {
                is_zero = true;
                match({TokenType::OpFatArrow, TokenType::Colon});
            } else if (match({TokenType::KwNeg})) {
                is_neg = true;
                match({TokenType::OpFatArrow, TokenType::Colon});
            } else if (check(TokenType::TritLiteral) || check(TokenType::OpPlus) || check(TokenType::OpMinus)) {
                Token t = advance();
                if (t.text == "+1" || t.text == "+") is_pos = true;
                else if (t.text == "0") is_zero = true;
                else if (t.text == "-1" || t.text == "-") is_neg = true;
                match({TokenType::OpFatArrow, TokenType::Colon});
            } else {
                throw ParseError("Expected select branch label (pos, zero, neg, or +1, 0, -1)", peek().loc);
            }

            ASTNodePtr branch_body = parse_block_or_statement();
            if (is_pos) pos_branch = branch_body;
            else if (is_zero) zero_branch = branch_body;
            else if (is_neg) neg_branch = branch_body;
        }

        consume(TokenType::RBrace, "Expected '}' to terminate select statement");

        if (!pos_branch || !zero_branch || !neg_branch) {
            throw ParseError("Balanced ternary select statement must provide all three branches (pos, zero, neg)", loc);
        }

        return std::make_shared<SelectNode>(cond, pos_branch, zero_branch, neg_branch, loc);
    }

    std::shared_ptr<WhileNode> parse_while() {
        SourceLocation loc = previous().loc;
        consume(TokenType::LParen, "Expected '(' after 'while'");
        ASTNodePtr cond = parse_expression();
        consume(TokenType::RParen, "Expected ')' after while condition");

        auto while_node = std::make_shared<WhileNode>();
        while_node->condition = cond;
        while_node->body = parse_block();
        while_node->location = loc;
        return while_node;
    }

    std::shared_ptr<IfNode> parse_if() {
        SourceLocation loc = previous().loc;
        consume(TokenType::LParen, "Expected '(' after 'if'");
        ASTNodePtr cond = parse_expression();
        consume(TokenType::RParen, "Expected ')' after if condition");

        auto if_node = std::make_shared<IfNode>();
        if_node->condition = cond;
        if_node->then_block = parse_block();
        if_node->location = loc;

        if (match({TokenType::KwElse})) {
            if_node->else_block = parse_block();
        }

        return if_node;
    }

    std::shared_ptr<ReturnNode> parse_return() {
        SourceLocation loc = previous().loc;
        auto ret_node = std::make_shared<ReturnNode>();
        ret_node->location = loc;

        if (!check(TokenType::Semicolon)) {
            ret_node->expr = parse_expression();
        }

        consume(TokenType::Semicolon, "Expected ';' after return");
        return ret_node;
    }

    // ==========================================================================
    // Expression Parsing with Proper Precedence
    // Precedence (lowest to highest):
    // 1. Comparison: <=> (spaceship), ==, !=, <, >, <=, >=
    // 2. Kleene OR: |
    // 3. Ternary XOR: ^
    // 4. Kleene AND: &
    // 5. Add / Sub: +, -
    // 6. Mul: *
    // 7. Unary: -, ~, !
    // 8. Primary: Literals, Identifiers, Grouping, Select expression, Calls
    // ==========================================================================

    ASTNodePtr parse_expression() {
        return parse_comparison();
    }

    ASTNodePtr parse_comparison() {
        ASTNodePtr expr = parse_bitwise_or();

        while (match({TokenType::OpSpaceship, TokenType::OpEq, TokenType::OpNe,
                      TokenType::OpLt, TokenType::OpGt, TokenType::OpLe, TokenType::OpGe})) {
            Token op_tok = previous();
            BinaryOp op = BinaryOp::Spaceship;
            if (op_tok.type == TokenType::OpSpaceship) op = BinaryOp::Spaceship;
            else if (op_tok.type == TokenType::OpEq)  op = BinaryOp::Equal;
            else if (op_tok.type == TokenType::OpNe)  op = BinaryOp::NotEqual;
            else if (op_tok.type == TokenType::OpLt)  op = BinaryOp::LessThan;
            else if (op_tok.type == TokenType::OpGt)  op = BinaryOp::GreaterThan;
            else if (op_tok.type == TokenType::OpLe)  op = BinaryOp::LessEqual;
            else if (op_tok.type == TokenType::OpGe)  op = BinaryOp::GreaterEqual;

            ASTNodePtr right = parse_bitwise_or();
            expr = std::make_shared<BinaryOpNode>(op, op_tok.text, expr, right, op_tok.loc);
        }

        return expr;
    }

    ASTNodePtr parse_bitwise_or() {
        ASTNodePtr expr = parse_bitwise_xor();

        while (match({TokenType::OpPipe})) {
            Token op_tok = previous();
            ASTNodePtr right = parse_bitwise_xor();
            expr = std::make_shared<BinaryOpNode>(BinaryOp::KleeneOr, op_tok.text, expr, right, op_tok.loc);
        }

        return expr;
    }

    ASTNodePtr parse_bitwise_xor() {
        ASTNodePtr expr = parse_bitwise_and();

        while (match({TokenType::OpCaret})) {
            Token op_tok = previous();
            ASTNodePtr right = parse_bitwise_and();
            expr = std::make_shared<BinaryOpNode>(BinaryOp::TernaryXor, op_tok.text, expr, right, op_tok.loc);
        }

        return expr;
    }

    ASTNodePtr parse_bitwise_and() {
        ASTNodePtr expr = parse_term();

        while (match({TokenType::OpAmp})) {
            Token op_tok = previous();
            ASTNodePtr right = parse_term();
            expr = std::make_shared<BinaryOpNode>(BinaryOp::KleeneAnd, op_tok.text, expr, right, op_tok.loc);
        }

        return expr;
    }

    // Add and Subtract (+, -)
    ASTNodePtr parse_term() {
        ASTNodePtr expr = parse_factor();

        while (match({TokenType::OpPlus, TokenType::OpMinus})) {
            Token op_tok = previous();
            BinaryOp op = (op_tok.type == TokenType::OpPlus) ? BinaryOp::Add : BinaryOp::Sub;
            ASTNodePtr right = parse_factor();
            expr = std::make_shared<BinaryOpNode>(op, op_tok.text, expr, right, op_tok.loc);
        }

        return expr;
    }

    // Multiplication (*)
    ASTNodePtr parse_factor() {
        ASTNodePtr expr = parse_unary();

        while (match({TokenType::OpStar})) {
            Token op_tok = previous();
            ASTNodePtr right = parse_unary();
            expr = std::make_shared<BinaryOpNode>(BinaryOp::Mul, op_tok.text, expr, right, op_tok.loc);
        }

        return expr;
    }

    // Unary Operators (+, -, ~, !)
    ASTNodePtr parse_unary() {
        if (match({TokenType::OpPlus})) {
            Token op_tok = previous();
            if (check(TokenType::Semicolon) || check(TokenType::Comma) || check(TokenType::RParen) || check(TokenType::RBrace)) {
                return LiteralNode::make_trit(bt::Trit::Pos, op_tok.loc);
            }
            // Unary positive sign (+x) evaluates to x
            return parse_unary();
        }
        if (match({TokenType::OpMinus, TokenType::OpTilde, TokenType::OpBang})) {
            Token op_tok = previous();
            if (op_tok.type == TokenType::OpMinus && (check(TokenType::Semicolon) || check(TokenType::Comma) || check(TokenType::RParen) || check(TokenType::RBrace))) {
                return LiteralNode::make_trit(bt::Trit::Neg, op_tok.loc);
            }
            UnaryOp op = (op_tok.type == TokenType::OpMinus) ? UnaryOp::Negate : UnaryOp::KleeneNot;
            ASTNodePtr operand = parse_unary();
            return std::make_shared<UnaryOpNode>(op, op_tok.text, operand, op_tok.loc);
        }

        return parse_primary();
    }

    ASTNodePtr parse_primary() {
        // 1a. Float Literals (3.5, 0.75, 1.2e-3, -0.5)
        if (match({TokenType::FloatLiteral})) {
            Token tok = previous();
            return std::make_shared<FloatLiteralExprAST>(tok.float_val, tok.text, tok.loc);
        }

        // 1b. Sparse Matrix Constructor: sparse_matrix(rows, cols, ...)
        if (match({TokenType::KwSparseMatrix}) || (check(TokenType::Ident) && peek().text == "sparse_matrix" && peek(1).type == TokenType::LParen)) {
            if (check(TokenType::Ident)) advance();
            SourceLocation loc = previous().loc;
            consume(TokenType::LParen, "Expected '(' after 'sparse_matrix'");

            Token r_tok = consume(TokenType::Number, "Expected number of rows for sparse_matrix");
            consume(TokenType::Comma, "Expected ',' after rows in sparse_matrix");
            Token c_tok = consume(TokenType::Number, "Expected number of cols for sparse_matrix");

            size_t rows = static_cast<size_t>(r_tok.int_val);
            size_t cols = static_cast<size_t>(c_tok.int_val);

            std::vector<int64_t> elements;
            std::vector<ASTNodePtr> args;
            while (match({TokenType::Comma})) {
                ASTNodePtr arg_expr = parse_expression();
                args.push_back(arg_expr);
                if (auto lit = std::dynamic_pointer_cast<LiteralNode>(arg_expr)) {
                    elements.push_back(lit->int_val);
                } else if (auto uop = std::dynamic_pointer_cast<UnaryOpNode>(arg_expr)) {
                    if (auto num = std::dynamic_pointer_cast<LiteralNode>(uop->operand)) {
                        elements.push_back(-num->int_val);
                    } else {
                        elements.push_back(0);
                    }
                } else {
                    elements.push_back(0);
                }
            }
            consume(TokenType::RParen, "Expected ')' to close sparse_matrix constructor");

            bt::TCSRMatrix mat(rows, cols);
            if (elements.size() == rows * cols) {
                std::vector<int8_t> dense_data(elements.begin(), elements.end());
                mat = bt::TCSRMatrix::from_dense(rows, cols, dense_data);
            } else if (elements.size() % 3 == 0) {
                std::vector<std::tuple<size_t, size_t, bt::Trit>> triplets;
                for (size_t i = 0; i < elements.size(); i += 3) {
                    triplets.emplace_back(static_cast<size_t>(elements[i]),
                                          static_cast<size_t>(elements[i + 1]),
                                          bt::from_int(static_cast<int8_t>(elements[i + 2])));
                }
                mat = bt::TCSRMatrix::from_triplets(rows, cols, triplets);
            }

            auto node = std::make_shared<SparseMatrixDeclAST>("", rows, cols, std::move(mat), loc);
            node->init_args = std::move(args);
            return node;
        }

        // 1. Trit Literals (+1, -1, 0, or +-0+)
        if (match({TokenType::TritLiteral})) {
            Token tok = previous();
            if (tok.trit_vector.size() > 1) {
                std::vector<bt::Trit> vec;
                for (int8_t t : tok.trit_vector) vec.push_back(bt::from_int(t));
                return LiteralNode::make_vector(vec, tok.text, tok.loc);
            }
            return LiteralNode::make_trit(bt::from_int(tok.trit_val), tok.loc);
        }

        // Standalone '+' as TritLiteral (+1)
        if (match({TokenType::OpPlus})) {
            return LiteralNode::make_trit(bt::Trit::Pos, previous().loc);
        }

        // 2. Integer Numbers
        if (match({TokenType::Number})) {
            Token tok = previous();
            return LiteralNode::make_int(tok.int_val, tok.loc);
        }

        // 3. Functional Select in expression: select(cond, pos, zero, neg)
        if (match({TokenType::KwSelect})) {
            return parse_select();
        }

        // 4. Identifiers or Function Calls
        if (match({TokenType::Ident})) {
            Token id_tok = previous();
            if (match({TokenType::LParen})) {
                std::vector<ASTNodePtr> args;
                if (!check(TokenType::RParen)) {
                    do {
                        args.push_back(parse_expression());
                    } while (match({TokenType::Comma}));
                }
                consume(TokenType::RParen, "Expected ')' after argument list in call to " + id_tok.text);
                return std::make_shared<CallNode>(id_tok.text, args, id_tok.loc);
            }
            return std::make_shared<IdentifierNode>(id_tok.text, id_tok.loc);
        }

        // 5. Parenthesized Expressions (a + b)
        if (match({TokenType::LParen})) {
            ASTNodePtr expr = parse_expression();
            consume(TokenType::RParen, "Expected ')' after expression");
            return expr;
        }

        throw ParseError("Expected expression, got " + token_type_name(peek().type) + " ('" + peek().text + "')", peek().loc);
    }
};

// ==============================================================================
// 6. Semantic Analyzer & Type Checker
// ==============================================================================

enum class BTType {
    Unknown,
    Void,
    Trit,
    TritWord,
    TritMatrix,
    Float,
    SparseMatrix,
    Int
};

inline std::string bttype_to_string(BTType t) {
    switch (t) {
        case BTType::Unknown:      return "unknown";
        case BTType::Void:         return "void";
        case BTType::Trit:         return "trit";
        case BTType::TritWord:     return "tritword";
        case BTType::TritMatrix:   return "tritmatrix";
        case BTType::Float:        return "float";
        case BTType::SparseMatrix: return "sparse_matrix";
        case BTType::Int:          return "int";
    }
    return "unknown";
}

inline BTType string_to_bttype(const std::string& s) {
    if (s == "trit") return BTType::Trit;
    if (s == "tritword") return BTType::TritWord;
    if (s == "tritmatrix") return BTType::TritMatrix;
    if (s == "float") return BTType::Float;
    if (s == "sparse_matrix") return BTType::SparseMatrix;
    if (s == "int") return BTType::Int;
    if (s == "void") return BTType::Void;
    return BTType::Unknown;
}

class TypeError : public std::runtime_error {
public:
    SourceLocation loc;
    TypeError(const std::string& msg, SourceLocation l = {})
        : std::runtime_error("[Type Error] " + l.to_string() + ": " + msg), loc(l) {}
};

class TypeChecker : public ASTVisitor {
private:
    std::unordered_map<std::string, BTType> symbol_table_;
    std::unordered_map<std::string, BTType> func_return_types_;
    BTType current_type_{BTType::Unknown};
    std::vector<std::string> errors_;

public:
    const std::vector<std::string>& errors() const { return errors_; }

    BTType get_symbol_type(const std::string& name) const {
        auto it = symbol_table_.find(name);
        return (it != symbol_table_.end()) ? it->second : BTType::Unknown;
    }

    void typecheck(ProgramNode& prog) {
        errors_.clear();
        symbol_table_.clear();
        prog.accept(*this);
        if (!errors_.empty()) {
            throw TypeError(errors_[0]);
        }
    }

    void visit(ProgramNode& node) override {
        for (auto& decl : node.declarations) {
            if (decl) decl->accept(*this);
        }
    }

    void visit(BlockNode& node) override {
        for (auto& stmt : node.statements) {
            if (stmt) stmt->accept(*this);
        }
    }

    void visit(FunctionDefNode& node) override {
        BTType ret_type = string_to_bttype(node.return_type);
        func_return_types_[node.name] = ret_type;
        auto prev_symbols = symbol_table_;
        for (const auto& p : node.params) {
            symbol_table_[p.name] = string_to_bttype(p.type);
        }
        if (node.body) node.body->accept(*this);
        symbol_table_ = std::move(prev_symbols);
    }

    void visit(VarDeclNode& node) override {
        BTType declared = string_to_bttype(node.type);
        if (node.init_expr) {
            node.init_expr->accept(*this);
            BTType init_type = current_type_;
            if (declared != BTType::Unknown && declared != init_type) {
                bool compatible = false;
                if ((declared == BTType::Int || declared == BTType::Trit || declared == BTType::TritWord) &&
                    (init_type == BTType::Int || init_type == BTType::Trit || init_type == BTType::TritWord)) {
                    compatible = true;
                } else if (declared == BTType::Float && init_type == BTType::Float) {
                    compatible = true;
                } else if (declared == BTType::SparseMatrix && init_type == BTType::SparseMatrix) {
                    compatible = true;
                } else if (declared == BTType::TritMatrix && init_type == BTType::TritMatrix) {
                    compatible = true;
                }

                if (!compatible) {
                    errors_.push_back("Incompatible initialization: cannot assign '" +
                                     bttype_to_string(init_type) + "' to variable '" +
                                     node.name + "' of type '" + bttype_to_string(declared) + "'");
                }
            }
            if (declared == BTType::Unknown) declared = init_type;
        }
        symbol_table_[node.name] = (declared != BTType::Unknown) ? declared : BTType::Int;
    }

    void visit(AssignNode& node) override {
        auto it = symbol_table_.find(node.target);
        BTType target_type = (it != symbol_table_.end()) ? it->second : BTType::Unknown;
        if (node.value) {
            node.value->accept(*this);
            BTType val_type = current_type_;
            if (target_type != BTType::Unknown && target_type != val_type) {
                bool compatible = false;
                if ((target_type == BTType::Int || target_type == BTType::Trit || target_type == BTType::TritWord) &&
                    (val_type == BTType::Int || val_type == BTType::Trit || val_type == BTType::TritWord)) {
                    compatible = true;
                } else if (target_type == BTType::Float && val_type == BTType::Float) {
                    compatible = true;
                } else if (target_type == BTType::SparseMatrix && val_type == BTType::SparseMatrix) {
                    compatible = true;
                }
                if (!compatible) {
                    errors_.push_back("Cannot assign type '" + bttype_to_string(val_type) +
                                     "' to target '" + node.target + "' of type '" +
                                     bttype_to_string(target_type) + "'");
                }
            }
        }
    }

    void visit(BinaryOpNode& node) override {
        node.left->accept(*this);
        BTType lt = current_type_;
        node.right->accept(*this);
        BTType rt = current_type_;

        if (lt == BTType::Float && rt == BTType::Float) {
            current_type_ = BTType::Float;
            return;
        }

        if (lt == BTType::Float || rt == BTType::Float) {
            errors_.push_back("Invalid binary operation between 'float' and non-float type '" +
                              bttype_to_string(lt == BTType::Float ? rt : lt) + "'");
            current_type_ = BTType::Unknown;
            return;
        }

        if (lt == BTType::SparseMatrix || rt == BTType::SparseMatrix) {
            if (node.op == BinaryOp::Mul && lt == BTType::SparseMatrix &&
                (rt == BTType::TritMatrix || rt == BTType::TritWord || rt == BTType::Int)) {
                current_type_ = BTType::TritMatrix;
                return;
            }
            errors_.push_back("Invalid operation on 'sparse_matrix' type");
            current_type_ = BTType::Unknown;
            return;
        }

        if (node.op == BinaryOp::Mul && lt == BTType::TritMatrix && rt == BTType::TritMatrix) {
            current_type_ = BTType::TritMatrix;
            return;
        }

        // Int / Trit arithmetic
        if ((lt == BTType::Int || lt == BTType::Trit || lt == BTType::TritWord) &&
            (rt == BTType::Int || rt == BTType::Trit || rt == BTType::TritWord)) {
            current_type_ = (lt == BTType::Trit && rt == BTType::Trit) ? BTType::Trit : BTType::Int;
            return;
        }

        current_type_ = lt;
    }

    void visit(UnaryOpNode& node) override {
        node.operand->accept(*this);
    }

    void visit(SelectNode& node) override {
        node.condition->accept(*this);
        node.pos_branch->accept(*this);
        BTType pt = current_type_;
        node.zero_branch->accept(*this);
        node.neg_branch->accept(*this);
        current_type_ = pt;
    }

    void visit(WhileNode& node) override {
        node.condition->accept(*this);
        if (node.body) node.body->accept(*this);
        current_type_ = BTType::Void;
    }

    void visit(ReturnNode& node) override {
        if (node.expr) {
            node.expr->accept(*this);
        } else {
            current_type_ = BTType::Void;
        }
    }

    void visit(LiteralNode& node) override {
        switch (node.kind) {
            case LiteralKind::Trit:    current_type_ = BTType::Trit; break;
            case LiteralKind::Integer: current_type_ = BTType::Int; break;
            case LiteralKind::Vector:  current_type_ = BTType::TritWord; break;
        }
    }

    void visit(FloatLiteralExprAST& /*node*/) override {
        current_type_ = BTType::Float;
    }

    void visit(SparseMatrixDeclAST& node) override {
        if (!node.name.empty()) {
            symbol_table_[node.name] = BTType::SparseMatrix;
        }
        current_type_ = BTType::SparseMatrix;
    }

    void visit(IdentifierNode& node) override {
        auto it = symbol_table_.find(node.name);
        current_type_ = (it != symbol_table_.end()) ? it->second : BTType::Unknown;
    }

    void visit(CallNode& node) override {
        if (node.callee == "print") {
            for (auto& arg : node.arguments) arg->accept(*this);
            current_type_ = BTType::Void;
            return;
        }
        if (node.callee == "matmul") {
            current_type_ = BTType::TritMatrix;
            return;
        }
        if (node.callee == "sparse_matrix") {
            current_type_ = BTType::SparseMatrix;
            return;
        }
        auto it = func_return_types_.find(node.callee);
        current_type_ = (it != func_return_types_.end()) ? it->second : BTType::Int;
    }

    void visit(IfNode& node) override {
        node.condition->accept(*this);
        if (node.then_block) node.then_block->accept(*this);
        if (node.else_block) node.else_block->accept(*this);
        current_type_ = BTType::Void;
    }

    void visit(ExpressionStmtNode& node) override {
        if (node.expr) node.expr->accept(*this);
    }
};

} // namespace bt::frontend

#endif // TERNARY_FRONTEND_AST_HPP
