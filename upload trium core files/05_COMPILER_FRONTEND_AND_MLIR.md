# Compiler Frontend, Intermediate Representation & MLIR Dialect
## AST Parser, Type Checker, and Lowering to LLVM/IREE via `bt` Dialect

> **AI Context Brief:**
> This document details the compiler pipeline of Trium-XBT:
> 1. **Frontend AST & Type Checker**: Lexer, AST hierarchy, and static type checking for balanced ternary source code (`.bt`).
> 2. **MLIR `bt` Dialect**: Official LLVM MLIR TableGen dialect definition containing custom types (`!bt.trit`, `!bt.dualrail<64>`, `!bt.csa_pair<64>`) and hardware-accelerated operations (`bt.add`, `bt.mul`, `bt.fma`, `bt.btha`, `bt.dot`).
> 3. **Lowering Pipeline**: Converts high-level ternary code into vector-parallel LLVM bitwise SWAR instructions or IREE GPU shaders.

---

## 1. MLIR `bt` Dialect Specification (`TernaryDialect.td`)

```tablegen
//===-- TernaryDialect.td - Balanced Ternary Dialect Definition -*- tablegen -*-===//

#ifndef BT_DIALECT_TD
#define BT_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def BT_Dialect : Dialect {
  let name = "bt";
  let summary = "Balanced Ternary Dialect for Hardware-Agnostic Ternary Computing";
  let description = [{
    The `bt` (Balanced Ternary) dialect introduces first-class language and
    intermediate representation primitives for radix-3 balanced computation
    over the symmetric residue digit set {-1, 0, +1}.
  }];
  let cppNamespace = "::mlir::bt";
}

// -----------------------------------------------------------------------------
// 1. Primitive Trit Type: !bt.trit
// -----------------------------------------------------------------------------
def BT_TritType : TypeDef<BT_Dialect, "Trit"> {
  let mnemonic = "trit";
  let summary = "A single balanced ternary digit in {-1, 0, +1}";
}

// -----------------------------------------------------------------------------
// 2. Dual-Rail Bitplane Register Type: !bt.dualrail<width>
// -----------------------------------------------------------------------------
def BT_DualRailType : TypeDef<BT_Dialect, "DualRail"> {
  let mnemonic = "dualrail";
  let summary = "Bit-sliced dual-rail register representing width trits";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

// -----------------------------------------------------------------------------
// 3. Redundant Carry-Save Accumulator Type: !bt.csa_pair<width>
// -----------------------------------------------------------------------------
def BT_CSAPairType : TypeDef<BT_Dialect, "CSAPair"> {
  let mnemonic = "csa_pair";
  let summary = "Redundant Carry-Save Accumulator (Sum, Carry) pair";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

// -----------------------------------------------------------------------------
// Core Operations
// -----------------------------------------------------------------------------

// Kogge-Stone Parallel Prefix Addition: %res = bt.add %lhs, %rhs : !bt.dualrail<64>
def BT_AddOp : Op<BT_Dialect, "add", [Pure, Commutative]> {
  let summary = "Parallel Kogge-Stone addition of two dual-rail ternary words";
  let arguments = (ins BT_DualRailType:$lhs, BT_DualRailType:$rhs);
  let results = (outs BT_DualRailType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// 0-Cycle Negation via Rail Swap: %res = bt.neg %val : !bt.dualrail<64>
def BT_NegOp : Op<BT_Dialect, "neg", [Pure, Involution]> {
  let summary = "0-cycle negation of dual-rail word via rail swap";
  let arguments = (ins BT_DualRailType:$operand);
  let results = (outs BT_DualRailType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Parallel Half-Adder: %sum, %carry = bt.btha %lhs, %rhs : !bt.dualrail<64>
def BT_BTHAOp : Op<BT_Dialect, "btha", [Pure, Commutative]> {
  let summary = "Balanced ternary half-adder step yielding parallel (Sum, Carry)";
  let arguments = (ins BT_DualRailType:$lhs, BT_DualRailType:$rhs);
  let results = (outs BT_DualRailType:$sum, BT_DualRailType:$carry);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($sum)";
}

// Fused Multiply-Add: %res = bt.fma %a, %b, %c : !bt.dualrail<64>
def BT_FMAOp : Op<BT_Dialect, "fma", [Pure]> {
  let summary = "Fused multiply-add: (a * b) + c";
  let arguments = (ins BT_DualRailType:$a, BT_DualRailType:$b, BT_DualRailType:$c);
  let results = (outs BT_DualRailType:$result);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($result)";
}

// Dense Memory Packing: %byte = bt.pack_base243 %word : !bt.dualrail<64>
def BT_PackBase243Op : Op<BT_Dialect, "pack_base243", [Pure]> {
  let summary = "Compresses 64 trits into 13 Base-243 bytes (95.6% density)";
  let arguments = (ins BT_DualRailType:$input);
  let results = (outs AnyType:$output_bytes);
}

#endif // BT_DIALECT_TD
```

---

## 2. Frontend AST & Type Checker (`ternary_frontend_ast.hpp`)

The frontend lexer and recursive-descent parser support balanced ternary literals (`+0-`, `10T`), dual-rail words, BT-FP32 floating point, spaceship operator (`<=>`), and ternary matrix types.

```cpp
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>

namespace bt::ast {

enum class TokenType {
    Identifier,
    Keyword,
    TritLiteral,     // e.g. + - 0
    WordLiteral,     // e.g. 0t+0-0+T
    NumberLiteral,   // Decimal integers
    FloatLiteral,    // BT-FP32 constants
    Spaceship,       // <=>
    Plus, Minus, Star, Slash,
    LBrace, RBrace, LParen, RParen, LBracket, RBracket,
    Semicolon, Comma, Assign,
    EndOfFile
};

struct Token {
    TokenType type;
    std::string text;
    size_t line{1};
    size_t col{1};
};

// -----------------------------------------------------------------------------
// AST Node Hierarchy
// -----------------------------------------------------------------------------
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void dump(int indent = 0) const = 0;
};

class ExprNode : public ASTNode {};

// Binary Expression: lhs op rhs (e.g. a + b, x <=> y)
class BinaryExprAST : public ExprNode {
public:
    std::string op;
    std::unique_ptr<ExprNode> lhs;
    std::unique_ptr<ExprNode> rhs;

    BinaryExprAST(std::string o, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r)
        : op(std::move(o)), lhs(std::move(l)), rhs(std::move(r)) {}

    void dump(int indent = 0) const override {
        std::cout << std::string(indent, ' ') << "BinaryExpr: " << op << "\n";
        lhs->dump(indent + 2);
        rhs->dump(indent + 2);
    }
};

// 64-Trit Dual-Rail Word Literal
class WordLiteralAST : public ExprNode {
public:
    uint64_t P{0};
    uint64_t N{0};

    WordLiteralAST(uint64_t p, uint64_t n) : P(p), N(n) {}

    void dump(int indent = 0) const override {
        std::cout << std::string(indent, ' ') << "TritWord64 Literal (P: " 
                  << std::hex << P << ", N: " << N << std::dec << ")\n";
    }
};

// Statement Node Hierarchy
class StmtAST : public ASTNode {};

class VarDeclAST : public StmtAST {
public:
    std::string type_name; // "word64", "fp32", "csr_mat"
    std::string var_name;
    std::unique_ptr<ExprNode> init_expr;

    VarDeclAST(std::string t, std::string v, std::unique_ptr<ExprNode> init)
        : type_name(std::move(t)), var_name(std::move(v)), init_expr(std::move(init)) {}

    void dump(int indent = 0) const override {
        std::cout << std::string(indent, ' ') << "VarDecl: " << type_name << " " << var_name << "\n";
        if (init_expr) init_expr->dump(indent + 2);
    }
};

// Function Definition Node
class FunctionAST : public ASTNode {
public:
    std::string name;
    std::string return_type;
    std::vector<std::pair<std::string, std::string>> params;
    std::vector<std::unique_ptr<StmtAST>> body;

    FunctionAST(std::string n, std::string ret) : name(std::move(n)), return_type(std::move(ret)) {}

    void dump(int indent = 0) const override {
        std::cout << std::string(indent, ' ') << "Function: " << name << " -> " << return_type << "\n";
        for (const auto& stmt : body) stmt->dump(indent + 2);
    }
};

} // namespace bt::ast
```
