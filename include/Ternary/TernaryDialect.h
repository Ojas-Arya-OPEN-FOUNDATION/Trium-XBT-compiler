/**
 * @file TernaryDialect.h
 * @brief MLIR Dialect definition for the Balanced Ternary (bt) Dialect
 * @author Antigravity Prime Research & Engineering Engine
 * 
 * Supports both upstream out-of-tree LLVM/MLIR builds and standalone C++20 builds.
 */

#ifndef TERNARY_DIALECT_H
#define TERNARY_DIALECT_H

#include "../../ternary_mlir_lowering.hpp"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <iostream>

#if defined(BT_HAS_MLIR) && __has_include(<mlir/IR/Dialect.h>)
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#endif

namespace mlir {

#if !defined(BT_HAS_MLIR) || !__has_include(<mlir/IR/DialectRegistry.h>)
class DialectRegistry {
public:
    std::vector<std::string> registeredDialects;
    void insert(const std::string& name) {
        if (!has(name)) {
            registeredDialects.push_back(name);
        }
    }
    bool has(const std::string& name) const {
        for (const auto& d : registeredDialects) {
            if (d == name) return true;
        }
        return false;
    }
};

class MLIRContext {
public:
    DialectRegistry registry;
    MLIRContext() = default;
    void appendDialectRegistry(const DialectRegistry& reg) {
        for (const auto& d : reg.registeredDialects) {
            registry.insert(d);
        }
    }
};
#endif

namespace bt {

// Core Types mirroring TableGen definitions
using TritType = ::bt::mlir::MLIRType;
using DualRailType = ::bt::mlir::MLIRType;
using CSAPairType = ::bt::mlir::MLIRType;
using TernaryTensorType = ::bt::mlir::MLIRType;

/**
 * @class TernaryDialect
 * @brief Dialect class for 'bt' (Balanced Ternary)
 */
class TernaryDialect {
public:
    static constexpr const char* kDialectNamespace = "bt";
    static constexpr const char* kSummary = "Balanced Ternary Dialect for Hardware-Agnostic Ternary Computing";

    TernaryDialect() = default;

    static const char* getDialectNamespace() { return kDialectNamespace; }
    static const char* getSummary() { return kSummary; }

    void initialize();

    // Verification helpers
    static bool verifyDualRailDisjointness(uint64_t P, uint64_t N) {
        return (P & N) == 0;
    }

    static bool verifyBTHA(uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn) {
        if (!verifyDualRailDisjointness(ap, an)) return false;
        if (!verifyDualRailDisjointness(bp, bn)) return false;
        auto res = ::bt::mlir::ConvertBTHA::execute_bitwise(ap, an, bp, bn);
        return verifyDualRailDisjointness(res.PS, res.NS) && verifyDualRailDisjointness(res.PC, res.NC);
    }

    static bool verifyBase243Unpack(uint8_t byte) {
        if (byte > 242) return false;
        auto trits = ::bt::mlir::ConvertBase243::execute_unpack(byte);
        for (int i = 0; i < 5; ++i) {
            if (trits[i] < -1 || trits[i] > 1) return false;
        }
        return ::bt::mlir::ConvertBase243::execute_pack(trits) == byte;
    }

    static bool verifyCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, uint64_t& rp, uint64_t& rn);
    static std::pair<uint64_t, uint64_t> evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out = nullptr);
};

// Operations
struct PlanarDotOp {
    static constexpr const char* kOpName = "bt.planar_dot";
    std::string lhs;
    std::string rhs;
    std::string result;
    unsigned width{64};

    static int64_t execute(uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn) {
        return ::bt::mlir::ConvertTernaryToVector::execute_vector_dot_4x64(
            {ap, 0, 0, 0}, {an, 0, 0, 0}, {bp, 0, 0, 0}, {bn, 0, 0, 0}
        );
    }
};

struct MatmulOp {
    static constexpr const char* kOpName = "bt.matmul";
    std::string lhs;
    std::string rhs;
    std::string acc;
    std::string result;
    size_t M{0}, K{0}, N{0};

    static void execute(const ::bt::TernaryMatrix& A, const ::bt::TernaryMatrix& B, ::bt::DenseMatrixInt64& C) {
        ::bt::mlir::ConvertTernaryToLinalg::execute_lowered_linalg(A, B, C);
    }
};

struct BTHAOp {
    static constexpr const char* kOpName = "bt.btha";
    std::string lhs;
    std::string rhs;
    std::string sum;
    std::string carry;

    static ::bt::mlir::ConvertBTHA::BTHABitwiseResult execute(uint64_t ap, uint64_t an, uint64_t bp, uint64_t bn) {
        return ::bt::mlir::ConvertBTHA::execute_bitwise(ap, an, bp, bn);
    }
};

struct Base243UnpackOp {
    static constexpr const char* kOpName = "bt.base243_unpack";
    std::string byteOperand;
    std::vector<std::string> tritResults;

    static std::array<int8_t, 5> execute(uint8_t byte) {
        return ::bt::mlir::ConvertBase243::execute_unpack(byte);
    }
};

struct CSAReduceOp {
    static constexpr const char* kOpName = "bt.csa_reduce";
    std::string acc;
    std::string result;

    static std::pair<uint64_t, uint64_t> execute(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out = nullptr);
    static std::pair<uint64_t, uint64_t> evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out = nullptr);
};

std::pair<uint64_t, uint64_t> evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out = nullptr);

struct CSAAccumOp {
    static constexpr const char* kOpName = "bt.csa_accum";
    std::string acc;
    std::string val;
    std::string result;
};

struct Select3Op {
    static constexpr const char* kOpName = "bt.select3";
    std::string cond;
    std::string pos_val;
    std::string zero_val;
    std::string neg_val;
    std::string result;

    template <typename T>
    static T execute(int cond, T pos, T zero, T neg) {
        if (cond > 0) return pos;
        if (cond < 0) return neg;
        return zero;
    }
};

/**
 * @brief Parses MLIR text into the standalone MLIRModule representation.
 *
 * The standalone (out-of-tree fallback) build has no upstream MLIR parser, so
 * bt-opt must ingest real input text with this loader instead of running the
 * pass pipeline over an empty canned module. The loader:
 *   1. validates structure (balanced braces, well-formed func.func records),
 *   2. verifies every referenced bt.* operation / !bt.* type is registered,
 *   3. extracts the module symbol name, top-level #map definitions, function
 *      signatures and function bodies into `mod` so that emitted output
 *      faithfully reflects the input text.
 *
 * @param mod    Target module (its `name` is overwritten only when the input
 *               text carries an explicit `module @symbol` heading).
 * @param text   Raw MLIR text (comments are tolerated and remain preserved in
 *               function bodies).
 * @param error  On failure, receives a human-readable diagnostic.
 * @return true if the text parsed cleanly; false otherwise.
 */
bool loadMLIRModuleFromString(::bt::mlir::MLIRModule& mod, const std::string& text, std::string& error);

/**
 * @brief True if `opName` (e.g. "bt.matmul") is registered by TernaryDialect.
 */
bool isKnownBtOpName(const std::string& opName);

/**
 * @brief True if `typeName` (e.g. "bt.trit") is registered by TernaryDialect.
 */
bool isKnownBtTypeName(const std::string& typeName);

// Registration functions
void registerTernaryDialect(::mlir::DialectRegistry &registry);
void registerTernaryDialect();
bool isTernaryDialectRegistered();

} // namespace bt
} // namespace mlir

#endif // TERNARY_DIALECT_H
