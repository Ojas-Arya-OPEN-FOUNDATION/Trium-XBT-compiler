/**
 * @file ternary_jit_engine.hpp
 * @brief AST Interpreter and JIT Execution Engine for Balanced Ternary (.bt)
 * @author Antigravity Prime Research & Engineering Engine
 * 
 * Direct execution engine for the Universal Balanced Ternary Compiler:
 * 1. Executes AST nodes from ternary_frontend_ast.hpp using native ternary operations.
 * 2. Evaluates dual-rail expressions (+, -, *, &, |, ^, <=>, ~) via ternary_dense_algebra.hpp.
 * 3. Branchless 3-way select multiplexing: select(cond, pos, zero, neg) and block selects.
 * 4. Planar matrix GEMM and Carry-Save Adder (BT-CSA) reduction trees.
 * 5. High-performance Kokkos multi-threaded parallel dispatch (ternary_kokkos_bridge.hpp).
 * 6. Lexical Environment and Symbol Table for runtime state persistence.
 */

#ifndef TERNARY_JIT_ENGINE_HPP
#define TERNARY_JIT_ENGINE_HPP

#include "ternary_dense_algebra.hpp"
#include "ternary_kokkos_bridge.hpp"
#include "ternary_frontend_ast.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace bt {
namespace jit {

// ==============================================================================
// 1. Runtime Value Representation
// ==============================================================================

/**
 * @brief Chunk-wise exact decode of a full 64-trit word into a signed 128-bit integer.
 * swar_to_int64() throws for num_trits > 39 (3^39 already saturates int64_t), so words
 * with live trits in positions 40..63 cannot go through it. Instead each 16-trit SWAR
 * chunk is converted independently by bct_to_bin_swar32 (value < 3^16 fits in int64_t)
 * and the chunks are combined with exact powers of 3^16 in __int128 arithmetic, giving
 * the exact value of all 64 trits with no truncation and no overflow.
 */
inline __int128 word_to_i128(const TritWord64& w) noexcept {
    constexpr __int128 k3_16 = 43046721; // 3^16
    __int128 total = 0;
    __int128 chunk_scale = 1; // 3^(16 * chunk)
    for (size_t chunk = 0; chunk < 4; ++chunk) {
        const uint32_t p = bct_to_bin_swar32(
            spread_16_to_32(static_cast<uint16_t>((w.P >> (chunk * 16)) & 0xFFFFU)));
        const uint32_t n = bct_to_bin_swar32(
            spread_16_to_32(static_cast<uint16_t>((w.N >> (chunk * 16)) & 0xFFFFU)));
        total += static_cast<__int128>(static_cast<int64_t>(p) - static_cast<int64_t>(n)) * chunk_scale;
        chunk_scale *= k3_16;
    }
    return total;
}

/**
 * @brief Exact decimal rendering of a signed 128-bit integer.
 * std::ostringstream has no __int128 overload in libc++, hence this manual formatter.
 */
inline std::string i128_to_string(__int128 v) {
    if (v == 0) return "0";
    const bool negative = v < 0;
    unsigned __int128 magnitude = static_cast<unsigned __int128>(v);
    if (negative) magnitude = (~magnitude) + 1; // safe even for the most negative i128
    std::string digits;
    while (magnitude > 0) {
        digits.push_back(static_cast<char>('0' + static_cast<int>(magnitude % 10)));
        magnitude /= 10;
    }
    if (negative) digits.push_back('-');
    std::reverse(digits.begin(), digits.end());
    return digits;
}

class JITValue {
public:
    enum class Kind {
        Scalar,
        TernaryMat,
        DenseIntMat,
        Float,
        SparseMat,
        Void
    };

    Kind kind{Kind::Void};
    TritWord64 scalar{0, 0};
    TernaryMatrix matrix;
    DenseMatrixInt64 int_matrix{0, 0};
    double float_val{0.0};
    bt::BTFP32 btfp_val;
    bt::TCSRMatrix sparse_matrix;

    JITValue() : kind(Kind::Void) {}

    static JITValue make_scalar(TritWord64 word) {
        JITValue v;
        v.kind = Kind::Scalar;
        v.scalar = word;
        return v;
    }

    static JITValue make_scalar_int(int64_t val) {
        return make_scalar(Codec::from_int64(val));
    }

    static JITValue make_matrix(TernaryMatrix mat) {
        JITValue v;
        v.kind = Kind::TernaryMat;
        v.matrix = std::move(mat);
        return v;
    }

    static JITValue make_int_matrix(DenseMatrixInt64 mat) {
        JITValue v;
        v.kind = Kind::DenseIntMat;
        v.int_matrix = std::move(mat);
        return v;
    }

    static JITValue make_float(double f) {
        JITValue v;
        v.kind = Kind::Float;
        v.float_val = f;
        v.btfp_val = bt::BTFP32::from_double(f);
        return v;
    }

    static JITValue make_btfp(bt::BTFP32 b) {
        JITValue v;
        v.kind = Kind::Float;
        v.btfp_val = b;
        v.float_val = b.to_double();
        return v;
    }

    static JITValue make_sparse_matrix(bt::TCSRMatrix mat) {
        JITValue v;
        v.kind = Kind::SparseMat;
        v.sparse_matrix = std::move(mat);
        return v;
    }

    static JITValue make_void() {
        return JITValue();
    }

    int64_t as_int64() const {
        if (kind == Kind::Scalar) {
            return Codec::to_int64(scalar);
        } else if (kind == Kind::Float) {
            return static_cast<int64_t>(float_val);
        }
        throw std::runtime_error("Attempted to read non-scalar value as an integer!");
    }

    std::string to_string() const {
        std::ostringstream ss;
        if (kind == Kind::Scalar) {
            // Print with the FULL 64-trit word width (multi-word BigInt values are not
            // held by JITValue, so one word is the whole value). The value is decoded
            // chunk-wise into __int128 via word_to_i128(); the old Codec::to_int64(39)
            // path silently truncated (and therefore mis-rendered) any word whose live
            // trits extend beyond position 39.
            __int128 full_val = word_to_i128(scalar);
            std::string trits;
            size_t width = 0;
            for (size_t i = 0; i < 64; ++i) {
                if (scalar.getTrit(i) != Trit::Zero) width = i + 1;
            }
            if (width == 0) width = 1;
            for (int i = static_cast<int>(width) - 1; i >= 0; --i) {
                trits.push_back(trit_to_char(scalar.getTrit(i)));
            }
            ss << i128_to_string(full_val) << " [trits: " << trits << "]";
        } else if (kind == Kind::Float) {
            ss << float_val << " [BTFP32: M=" << btfp_val.mantissa() << ", E=" << btfp_val.exponent() << "]";
        } else if (kind == Kind::SparseMat) {
            ss << "TCSRMatrix [" << sparse_matrix.rows << "x" << sparse_matrix.cols
               << ", nnz=" << sparse_matrix.nnz << "]:\n";
            for (size_t r = 0; r < sparse_matrix.rows; ++r) {
                ss << "  [ ";
                for (size_t c = 0; c < sparse_matrix.cols; ++c) {
                    ss << trit_to_char(sparse_matrix.get(r, c));
                    if (c + 1 < sparse_matrix.cols) ss << ", ";
                }
                ss << " ]\n";
            }
        } else if (kind == Kind::TernaryMat) {
            ss << "TernaryMatrix [" << matrix.rows << "x" << matrix.cols << "]:\n";
            for (size_t r = 0; r < matrix.rows; ++r) {
                ss << "  [ ";
                for (size_t c = 0; c < matrix.cols; ++c) {
                    ss << trit_to_char(matrix.get(r, c));
                    if (c + 1 < matrix.cols) ss << ", ";
                }
                ss << " ]\n";
            }
        } else if (kind == Kind::DenseIntMat) {
            ss << "DenseMatrixInt64 [" << int_matrix.rows << "x" << int_matrix.cols << "]:\n";
            for (size_t r = 0; r < int_matrix.rows; ++r) {
                ss << "  [ ";
                for (size_t c = 0; c < int_matrix.cols; ++c) {
                    ss << std::setw(4) << int_matrix(r, c);
                    if (c + 1 < int_matrix.cols) ss << ", ";
                }
                ss << " ]\n";
            }
        } else {
            ss << "<void>";
        }
        return ss.str();
    }
};

// ==============================================================================
// 2. Lexical Environment & Function Table
// ==============================================================================

class Environment {
private:
    std::unordered_map<std::string, JITValue> bindings;
    std::unordered_map<std::string, std::shared_ptr<bt::frontend::FunctionDefNode>> functions;
    std::shared_ptr<Environment> parent{nullptr};

public:
    Environment() = default;
    explicit Environment(std::shared_ptr<Environment> p) : parent(std::move(p)) {}

    void define(const std::string& name, JITValue val) {
        bindings[name] = std::move(val);
    }

    void assign(const std::string& name, JITValue val) {
        auto it = bindings.find(name);
        if (it != bindings.end()) {
            it->second = std::move(val);
            return;
        }
        if (parent) {
            parent->assign(name, std::move(val));
            return;
        }
        bindings[name] = std::move(val);
    }

    JITValue get(const std::string& name) const {
        auto it = bindings.find(name);
        if (it != bindings.end()) {
            return it->second;
        }
        if (parent) {
            return parent->get(name);
        }
        throw std::runtime_error("Undefined variable: '" + name + "' in JIT environment!");
    }

    bool exists(const std::string& name) const {
        if (bindings.find(name) != bindings.end()) return true;
        return parent ? parent->exists(name) : false;
    }

    void define_function(const std::string& name, std::shared_ptr<bt::frontend::FunctionDefNode> fn) {
        functions[name] = std::move(fn);
    }

    std::shared_ptr<bt::frontend::FunctionDefNode> get_function(const std::string& name) const {
        auto it = functions.find(name);
        if (it != functions.end()) {
            return it->second;
        }
        if (parent) {
            return parent->get_function(name);
        }
        return nullptr;
    }
};

// ==============================================================================
// 3. JIT Execution Engine
// ==============================================================================

struct JITConfig {
    bool enable_kokkos{false};
    size_t num_threads{0};
    std::ostream* out{&std::cout};
};

class ReturnException : public std::exception {
public:
    JITValue value;
    explicit ReturnException(JITValue val) : value(std::move(val)) {}
    const char* what() const noexcept override {
        return "JIT return statement executed";
    }
};

class TernaryJITEngine : public bt::frontend::ASTVisitor {
private:
    JITConfig config;
    std::shared_ptr<Environment> env;
    JITValue last_eval_result;

    // RAII call-frame scope: pushes a new lexical environment and always restores the
    // previous one when the scope ends. Function bodies that throw exceptions other than
    // ReturnException (user errors, dimension mismatches, ...) unwind through this guard,
    // so this->env can never stay pinned to a dead call frame after an error.
    struct EnvironmentFrameGuard {
        TernaryJITEngine& engine;
        std::shared_ptr<Environment> previous;
        EnvironmentFrameGuard(TernaryJITEngine& eng, std::shared_ptr<Environment> frame)
            : engine(eng), previous(eng.env) {
            engine.env = std::move(frame);
        }
        ~EnvironmentFrameGuard() { engine.env = previous; }
        EnvironmentFrameGuard(const EnvironmentFrameGuard&) = delete;
        EnvironmentFrameGuard& operator=(const EnvironmentFrameGuard&) = delete;
    };

    static inline TritWord64 add_words(const TritWord64& A, const TritWord64& B) noexcept {
        BT_CSATree csa(A, TritWord64(0, 0));
        csa.add(B);
        return csa.resolve();
    }

    static inline TritWord64 sub_words(const TritWord64& A, const TritWord64& B) noexcept {
        return add_words(A, -B);
    }

    // Exact multiplication in __int128 with a clean overflow error; removing noexcept is
    // required because this operation CAN throw (a noexcept wrapper would std::terminate).
    static inline TritWord64 mul_words(const TritWord64& A, const TritWord64& B) {
        int64_t a_val = Codec::to_int64(A);
        int64_t b_val = Codec::to_int64(B);
        // Operands come from the 39-trit domain (|v| <= (3^39 - 1) / 2 ~= 2.03e18), so the
        // true product can reach ~4.1e36 and overflow int64_t. Compute the product exactly
        // in __int128; if it fits within the representable 39-trit range, return the exact
        // word, otherwise throw a clean runtime_error (never silently wrap like the old
        // Codec::from_int64(a_val * b_val) did, which was signed-overflow UB).
        constexpr int64_t kMax39Trit = 2026277576509488133LL; // (3^39 - 1) / 2
        const __int128 product = static_cast<__int128>(a_val) * static_cast<__int128>(b_val);
        if (product > static_cast<__int128>(kMax39Trit) || product < static_cast<__int128>(-kMax39Trit)) {
            throw std::runtime_error("integer overflow in multiplication: " +
                                     std::to_string(a_val) + " * " + std::to_string(b_val) +
                                     " = " + i128_to_string(product) +
                                     " exceeds the representable balanced-ternary range [-" +
                                     std::to_string(kMax39Trit) + ", " + std::to_string(kMax39Trit) + "]");
        }
        return Codec::from_int64(static_cast<int64_t>(product));
    }

public:
    explicit TernaryJITEngine(JITConfig cfg = {})
        : config(cfg), env(std::make_shared<Environment>()) {
        if (config.enable_kokkos) {
            ::Kokkos::initialize(config.num_threads);
        }
    }

    void set_kokkos(bool enabled, size_t threads = 0) {
        config.enable_kokkos = enabled;
        config.num_threads = threads;
        if (enabled) {
            ::Kokkos::initialize(threads);
        }
    }

    std::shared_ptr<Environment> get_env() const { return env; }

    // --------------------------------------------------------------------------
    // Expression / Statement Visitor Implementations
    // --------------------------------------------------------------------------

    void visit(bt::frontend::ProgramNode& node) override {
        for (const auto& decl : node.declarations) {
            if (decl) decl->accept(*this);
        }
    }

    void visit(bt::frontend::BlockNode& node) override {
        for (const auto& stmt : node.statements) {
            if (stmt) stmt->accept(*this);
        }
    }

    void visit(bt::frontend::FunctionDefNode& node) override {
        env->define_function(node.name, std::make_shared<bt::frontend::FunctionDefNode>(node));
        last_eval_result = JITValue::make_void();
    }

    void visit(bt::frontend::VarDeclNode& node) override {
        JITValue val = JITValue::make_scalar_int(0);
        if (node.init_expr) {
            node.init_expr->accept(*this);
            val = last_eval_result;
        }
        env->define(node.name, std::move(val));
        last_eval_result = JITValue::make_void();
    }

    void visit(bt::frontend::AssignNode& node) override {
        if (node.value) {
            node.value->accept(*this);
            env->assign(node.target, last_eval_result);
        }
        last_eval_result = JITValue::make_void();
    }

    void visit(bt::frontend::BinaryOpNode& node) override {
        if (!node.left || !node.right) {
            throw std::runtime_error("Binary operation missing operand!");
        }

        node.left->accept(*this);
        JITValue left = last_eval_result;

        node.right->accept(*this);
        JITValue right = last_eval_result;

        // Case 0a: Float arithmetic & comparison
        if (left.kind == JITValue::Kind::Float || right.kind == JITValue::Kind::Float) {
            double lv = (left.kind == JITValue::Kind::Float) ? left.float_val : static_cast<double>(left.as_int64());
            double rv = (right.kind == JITValue::Kind::Float) ? right.float_val : static_cast<double>(right.as_int64());
            switch (node.op) {
                case bt::frontend::BinaryOp::Add:
                    last_eval_result = JITValue::make_float(lv + rv);
                    return;
                case bt::frontend::BinaryOp::Sub:
                    last_eval_result = JITValue::make_float(lv - rv);
                    return;
                case bt::frontend::BinaryOp::Mul:
                    last_eval_result = JITValue::make_float(lv * rv);
                    return;
                // Comparisons and the 3-way spaceship must actually COMPARE (the old
                // default arm computed lv + rv instead, turning every comparison into
                // an addition). The integer path below returns the ternary convention
                // make_scalar_int(+1 / 0 / -1): +1 = true, -1 = false for predicate ops;
                // <=> returns the balanced-ternary sign of (lv - rv): +1 / 0 / -1.
                case bt::frontend::BinaryOp::Spaceship:
                    last_eval_result = JITValue::make_scalar_int(
                        (lv > rv) ? 1 : ((lv < rv) ? -1 : 0));
                    return;
                case bt::frontend::BinaryOp::Equal:
                    last_eval_result = JITValue::make_scalar_int(lv == rv ? 1 : -1);
                    return;
                case bt::frontend::BinaryOp::NotEqual:
                    last_eval_result = JITValue::make_scalar_int(lv != rv ? 1 : -1);
                    return;
                case bt::frontend::BinaryOp::LessThan:
                    last_eval_result = JITValue::make_scalar_int(lv < rv ? 1 : -1);
                    return;
                case bt::frontend::BinaryOp::GreaterThan:
                    last_eval_result = JITValue::make_scalar_int(lv > rv ? 1 : -1);
                    return;
                case bt::frontend::BinaryOp::LessEqual:
                    last_eval_result = JITValue::make_scalar_int(lv <= rv ? 1 : -1);
                    return;
                case bt::frontend::BinaryOp::GreaterEqual:
                    last_eval_result = JITValue::make_scalar_int(lv >= rv ? 1 : -1);
                    return;
                case bt::frontend::BinaryOp::KleeneAnd:
                case bt::frontend::BinaryOp::KleeneOr:
                case bt::frontend::BinaryOp::TernaryXor:
                    // & | ^ are trit-level dual-rail ops with no IEEE-754 semantics.
                    throw std::runtime_error("Kleene logic not defined for float");
            }
        }

        // Case 0b: T-CSR left operand (A = sparse_matrix).
        //  - A * same-length vector        -> multiplier-free SpMV (unchanged behavior)
        //  - A * matching dense MxN matrix -> TRUE SpMM:  C = A * B with
        //                  C[i][j] = sum_k A[i][k] * B[k][j], yielding a DenseIntMat.
        // The old code flattened the right matrix into rows*cols entries and fed that into
        // spmv(), which only indexes x[col] for col < A.cols — i.e. it silently multiplyed
        // A by just the FIRST ROW (or threw a dimension error for square inputs).
        if (node.op == bt::frontend::BinaryOp::Mul && left.kind == JITValue::Kind::SparseMat &&
            (right.kind == JITValue::Kind::DenseIntMat || right.kind == JITValue::Kind::TernaryMat)) {
            const auto& A = left.sparse_matrix;

            // Materialize the right operand as a row-major int64 buffer.
            std::vector<int64_t> B_data;
            size_t B_rows = 0;
            size_t B_cols = 0;
            if (right.kind == JITValue::Kind::DenseIntMat) {
                B_rows = right.int_matrix.rows;
                B_cols = right.int_matrix.cols;
                B_data = right.int_matrix.data;
            } else {
                B_rows = right.matrix.rows;
                B_cols = right.matrix.cols;
                B_data.assign(B_rows * B_cols, 0);
                for (size_t r = 0; r < B_rows; ++r) {
                    for (size_t c = 0; c < B_cols; ++c) {
                        B_data[r * B_cols + c] = to_int(right.matrix.get(r, c));
                    }
                }
            }

            // Vector SpMV: right operand is a length-A.cols vector stored as an
            // Mx1 or 1xN (row/col vector) dense matrix.
            if ((B_cols == 1 || B_rows == 1) && B_rows * B_cols == A.cols) {
                auto y = A.spmv(B_data);
                DenseMatrixInt64 out_mat(y.size(), 1);
                out_mat.data = std::move(y);
                last_eval_result = JITValue::make_int_matrix(std::move(out_mat));
                return;
            }

            // Matching matrix product: right rows == A.cols -> true SpMM (C is A.rows x B_cols).
            if (B_rows == A.cols && B_rows > 0 && B_cols > 0) {
                auto C_data = A.spmm(B_data, B_cols);
                DenseMatrixInt64 out_mat(A.rows, B_cols);
                out_mat.data = std::move(C_data);
                last_eval_result = JITValue::make_int_matrix(std::move(out_mat));
                return;
            }

            throw std::runtime_error(
                "sparse_matrix dimension mismatch: T-CSR is " + std::to_string(A.rows) +
                "x" + std::to_string(A.cols) + " (nnz=" + std::to_string(A.nnz) +
                "), right operand is " + std::to_string(B_rows) + "x" + std::to_string(B_cols) +
                "; expected right.rows == " + std::to_string(A.cols) +
                " for a matrix product (SpMM), or a length-" + std::to_string(A.cols) +
                " vector for matrix-vector SpMV");
        }

        // Case 1: Both operands are scalars
        if (left.kind == JITValue::Kind::Scalar && right.kind == JITValue::Kind::Scalar) {
            switch (node.op) {
                case bt::frontend::BinaryOp::Add:
                    last_eval_result = JITValue::make_scalar(add_words(left.scalar, right.scalar));
                    return;
                case bt::frontend::BinaryOp::Sub:
                    last_eval_result = JITValue::make_scalar(sub_words(left.scalar, right.scalar));
                    return;
                case bt::frontend::BinaryOp::Mul:
                    last_eval_result = JITValue::make_scalar(mul_words(left.scalar, right.scalar));
                    return;
                case bt::frontend::BinaryOp::KleeneAnd:
                    last_eval_result = JITValue::make_scalar(kleene_and(left.scalar, right.scalar));
                    return;
                case bt::frontend::BinaryOp::KleeneOr:
                    last_eval_result = JITValue::make_scalar(kleene_or(left.scalar, right.scalar));
                    return;
                case bt::frontend::BinaryOp::TernaryXor:
                    last_eval_result = JITValue::make_scalar(kleene_xor(left.scalar, right.scalar));
                    return;
                case bt::frontend::BinaryOp::Spaceship: {
                    int64_t lv = left.as_int64();
                    int64_t rv = right.as_int64();
                    if (lv > rv) last_eval_result = JITValue::make_scalar_int(1);
                    else if (lv < rv) last_eval_result = JITValue::make_scalar_int(-1);
                    else last_eval_result = JITValue::make_scalar_int(0);
                    return;
                }
                case bt::frontend::BinaryOp::Equal: {
                    if (left.kind == JITValue::Kind::Scalar && right.kind == JITValue::Kind::Scalar) {
                        bool eq = (left.scalar.P == right.scalar.P) && (left.scalar.N == right.scalar.N);
                        last_eval_result = JITValue::make_scalar_int(eq ? 1 : -1);
                    } else {
                        int64_t lv = left.as_int64();
                        int64_t rv = right.as_int64();
                        last_eval_result = JITValue::make_scalar_int(lv == rv ? 1 : -1);
                    }
                    return;
                }
                case bt::frontend::BinaryOp::NotEqual: {
                    if (left.kind == JITValue::Kind::Scalar && right.kind == JITValue::Kind::Scalar) {
                        bool eq = (left.scalar.P == right.scalar.P) && (left.scalar.N == right.scalar.N);
                        last_eval_result = JITValue::make_scalar_int(!eq ? 1 : -1);
                    } else {
                        int64_t lv = left.as_int64();
                        int64_t rv = right.as_int64();
                        last_eval_result = JITValue::make_scalar_int(lv != rv ? 1 : -1);
                    }
                    return;
                }
                case bt::frontend::BinaryOp::LessThan: {
                    int64_t lv = left.as_int64();
                    int64_t rv = right.as_int64();
                    last_eval_result = JITValue::make_scalar_int(lv < rv ? 1 : -1);
                    return;
                }
                case bt::frontend::BinaryOp::GreaterThan: {
                    int64_t lv = left.as_int64();
                    int64_t rv = right.as_int64();
                    last_eval_result = JITValue::make_scalar_int(lv > rv ? 1 : -1);
                    return;
                }
                case bt::frontend::BinaryOp::LessEqual: {
                    int64_t lv = left.as_int64();
                    int64_t rv = right.as_int64();
                    last_eval_result = JITValue::make_scalar_int(lv <= rv ? 1 : -1);
                    return;
                }
                case bt::frontend::BinaryOp::GreaterEqual: {
                    int64_t lv = left.as_int64();
                    int64_t rv = right.as_int64();
                    last_eval_result = JITValue::make_scalar_int(lv >= rv ? 1 : -1);
                    return;
                }
            }
        }

        // Case 2: Matrix Multiplication (A * B)
        if (node.op == bt::frontend::BinaryOp::Mul &&
            left.kind == JITValue::Kind::TernaryMat && right.kind == JITValue::Kind::TernaryMat) {
            if (left.matrix.cols != right.matrix.rows) {
                throw std::runtime_error("Matrix multiplication dimension mismatch: left.cols (" +
                                         std::to_string(left.matrix.cols) + ") != right.rows (" +
                                         std::to_string(right.matrix.rows) + ")");
            }
            DenseMatrixInt64 C(left.matrix.rows, right.matrix.cols);
            if (config.enable_kokkos) {
                kokkos::parallel_gemm(left.matrix, right.matrix, C, config.num_threads);
            } else {
                gemm(left.matrix, right.matrix, C);
            }
            last_eval_result = JITValue::make_int_matrix(std::move(C));
            return;
        }

        // Case 3: Element-wise Matrix Addition / Subtraction
        if ((node.op == bt::frontend::BinaryOp::Add || node.op == bt::frontend::BinaryOp::Sub) &&
            left.kind == JITValue::Kind::TernaryMat && right.kind == JITValue::Kind::TernaryMat) {
            if (left.matrix.rows != right.matrix.rows || left.matrix.cols != right.matrix.cols) {
                throw std::runtime_error("Matrix dimensions mismatch in binary operation!");
            }
            TernaryMatrix res(left.matrix.rows, left.matrix.cols);
            for (size_t r = 0; r < left.matrix.rows; ++r) {
                for (size_t c = 0; c < left.matrix.cols; ++c) {
                    int8_t l = to_int(left.matrix.get(r, c));
                    int8_t r_val = to_int(right.matrix.get(r, c));
                    int8_t s = (node.op == bt::frontend::BinaryOp::Add) ? (l + r_val) : (l - r_val);
                    res.set(r, c, from_int(s));
                }
            }
            last_eval_result = JITValue::make_matrix(std::move(res));
            return;
        }

        throw std::runtime_error("Unsupported operands for binary operator " + node.op_symbol);
    }

    void visit(bt::frontend::UnaryOpNode& node) override {
        if (!node.operand) throw std::runtime_error("Unary operation missing operand!");
        node.operand->accept(*this);
        JITValue val = last_eval_result;

        if (val.kind == JITValue::Kind::Float) {
            if (node.op == bt::frontend::UnaryOp::Negate) {
                last_eval_result = JITValue::make_float(-val.float_val);
                return;
            }
        }

        if (node.op == bt::frontend::UnaryOp::Negate) {
            if (val.kind == JITValue::Kind::Scalar) {
                last_eval_result = JITValue::make_scalar(-val.scalar);
                return;
            } else if (val.kind == JITValue::Kind::TernaryMat) {
                TernaryMatrix negated(val.matrix.rows, val.matrix.cols);
                for (size_t r = 0; r < val.matrix.rows; ++r) {
                    for (size_t c = 0; c < val.matrix.cols; ++c) {
                        negated.set(r, c, neg(val.matrix.get(r, c)));
                    }
                }
                last_eval_result = JITValue::make_matrix(std::move(negated));
                return;
            } else if (val.kind == JITValue::Kind::DenseIntMat) {
                DenseMatrixInt64 negated(val.int_matrix.rows, val.int_matrix.cols);
                for (size_t i = 0; i < val.int_matrix.data.size(); ++i) {
                    negated.data[i] = -val.int_matrix.data[i];
                }
                last_eval_result = JITValue::make_int_matrix(std::move(negated));
                return;
            }
        } else if (node.op == bt::frontend::UnaryOp::KleeneNot) {
            if (val.kind == JITValue::Kind::Scalar) {
                last_eval_result = JITValue::make_scalar(kleene_not(val.scalar));
                return;
            }
        }

        throw std::runtime_error("Unsupported unary operation!");
    }

    void visit(bt::frontend::SelectNode& node) override {
        if (!node.condition) throw std::runtime_error("Select statement/expression missing condition!");
        node.condition->accept(*this);
        JITValue cond_v = last_eval_result;

        // Functional expression select: select(c, pos_val, zero_val, neg_val)
        // If all branches are pure scalar expressions:
        if (cond_v.kind == JITValue::Kind::Scalar) {
            int64_t c_int = cond_v.as_int64();
            if (c_int > 0) {
                if (node.pos_branch) node.pos_branch->accept(*this);
                else last_eval_result = JITValue::make_void();
            } else if (c_int < 0) {
                if (node.neg_branch) node.neg_branch->accept(*this);
                else last_eval_result = JITValue::make_void();
            } else {
                if (node.zero_branch) node.zero_branch->accept(*this);
                else last_eval_result = JITValue::make_void();
            }
            return;
        }

        throw std::runtime_error("Select condition must evaluate to a balanced ternary scalar!");
    }

    void visit(bt::frontend::WhileNode& node) override {
        while (true) {
            node.condition->accept(*this);
            int64_t c = last_eval_result.as_int64();
            if (c <= 0) break;
            if (node.body) node.body->accept(*this);
        }
        last_eval_result = JITValue::make_void();
    }

    void visit(bt::frontend::IfNode& node) override {
        node.condition->accept(*this);
        int64_t c = last_eval_result.as_int64();
        if (c > 0) {
            if (node.then_block) node.then_block->accept(*this);
        } else {
            if (node.else_block) node.else_block->accept(*this);
        }
    }

    void visit(bt::frontend::ReturnNode& node) override {
        JITValue ret_val = JITValue::make_void();
        if (node.expr) {
            node.expr->accept(*this);
            ret_val = last_eval_result;
        }
        throw ReturnException(ret_val);
    }

    void visit(bt::frontend::LiteralNode& node) override {
        if (node.kind == bt::frontend::LiteralKind::Integer) {
            last_eval_result = JITValue::make_scalar_int(node.int_val);
        } else if (node.kind == bt::frontend::LiteralKind::Trit) {
            last_eval_result = JITValue::make_scalar_int(bt::to_int(node.trit_val));
        } else if (node.kind == bt::frontend::LiteralKind::Vector) {
            TritWord64 word(0, 0);
            for (size_t i = 0; i < node.vector_val.size() && i < 64; ++i) {
                word.setTrit(i, node.vector_val[i]);
            }
            last_eval_result = JITValue::make_scalar(word);
        }
    }

    void visit(bt::frontend::FloatLiteralExprAST& node) override {
        last_eval_result = JITValue::make_float(node.value);
    }

    void visit(bt::frontend::SparseMatrixDeclAST& node) override {
        last_eval_result = JITValue::make_sparse_matrix(node.matrix);
        if (!node.name.empty()) {
            env->define(node.name, last_eval_result);
        }
    }

    void visit(bt::frontend::IdentifierNode& node) override {
        last_eval_result = env->get(node.name);
    }

    void visit(bt::frontend::ExpressionStmtNode& node) override {
        if (node.expr) node.expr->accept(*this);
    }

    void visit(bt::frontend::CallNode& node) override {
        const std::string& fn = node.callee;

        // 1. Builtin: print(...)
        if (fn == "print" || fn == "print_matrix") {
            std::ostream& out = *(config.out);
            for (size_t i = 0; i < node.arguments.size(); ++i) {
                node.arguments[i]->accept(*this);
                out << last_eval_result.to_string();
                if (i + 1 < node.arguments.size()) out << " ";
            }
            out << "\n";
            last_eval_result = JITValue::make_void();
            return;
        }

        // 2. Builtin: matmul(A, B)
        if (fn == "matmul") {
            if (node.arguments.size() != 2) throw std::runtime_error("matmul requires exactly 2 matrix arguments!");
            node.arguments[0]->accept(*this);
            JITValue A = last_eval_result;
            node.arguments[1]->accept(*this);
            JITValue B = last_eval_result;

            if (A.kind != JITValue::Kind::TernaryMat || B.kind != JITValue::Kind::TernaryMat) {
                throw std::runtime_error("matmul arguments must be ternary matrices!");
            }
            if (A.matrix.cols != B.matrix.rows) {
                throw std::runtime_error("matmul dimension mismatch: A.cols (" + std::to_string(A.matrix.cols) +
                                         ") != B.rows (" + std::to_string(B.matrix.rows) + ")");
            }
            DenseMatrixInt64 C(A.matrix.rows, B.matrix.cols);
            if (config.enable_kokkos) {
                kokkos::parallel_gemm(A.matrix, B.matrix, C, config.num_threads);
            } else {
                gemm(A.matrix, B.matrix, C);
            }
            last_eval_result = JITValue::make_int_matrix(std::move(C));
            return;
        }

        // 3. Builtin: csa_reduce(mat1, mat2, mat3, ...)
        if (fn == "csa_reduce") {
            if (node.arguments.empty()) throw std::runtime_error("csa_reduce requires at least 1 matrix argument!");
            std::vector<TernaryMatrix> mats;
            mats.reserve(node.arguments.size());
            for (const auto& arg : node.arguments) {
                arg->accept(*this);
                if (last_eval_result.kind != JITValue::Kind::TernaryMat) {
                    throw std::runtime_error("csa_reduce arguments must be ternary matrices!");
                }
                mats.push_back(std::move(last_eval_result.matrix));
            }
            for (size_t k = 1; k < mats.size(); ++k) {
                if (mats[k].rows != mats[0].rows || mats[k].cols != mats[0].cols) {
                    throw std::runtime_error("csa_reduce dimension mismatch across matrices: matrix[0] is " +
                                             std::to_string(mats[0].rows) + "x" + std::to_string(mats[0].cols) +
                                             ", but matrix[" + std::to_string(k) + "] is " +
                                             std::to_string(mats[k].rows) + "x" + std::to_string(mats[k].cols));
                }
            }
            if (config.enable_kokkos) {
                TernaryMatrix resolved = kokkos::parallel_reduce_csa(mats, config.num_threads);
                last_eval_result = JITValue::make_matrix(std::move(resolved));
            } else {
                size_t M = mats[0].rows;
                size_t N = mats[0].cols;
                size_t total_words = M * mats[0].words_per_row;
                TernaryMatrix resolved(M, N);
                for (size_t idx = 0; idx < total_words; ++idx) {
                    BT_CSATree csa;
                    for (size_t k = 0; k < mats.size(); ++k) {
                        csa.add(mats[k].data[idx]);
                    }
                    resolved.data[idx] = csa.resolve();
                }
                last_eval_result = JITValue::make_matrix(std::move(resolved));
            }
            return;
        }

        // 4. Builtin: transpose(A)
        if (fn == "transpose") {
            if (node.arguments.size() != 1) throw std::runtime_error("transpose requires 1 matrix argument!");
            node.arguments[0]->accept(*this);
            if (last_eval_result.kind == JITValue::Kind::TernaryMat) {
                last_eval_result = JITValue::make_matrix(last_eval_result.matrix.transpose());
                return;
            }
            throw std::runtime_error("transpose requires a ternary matrix!");
        }

        // 5. Builtin: zeros(rows, cols)
        if (fn == "zeros") {
            if (node.arguments.size() != 2) throw std::runtime_error("zeros requires (rows, cols)!");
            node.arguments[0]->accept(*this);
            size_t r = static_cast<size_t>(last_eval_result.as_int64());
            node.arguments[1]->accept(*this);
            size_t c = static_cast<size_t>(last_eval_result.as_int64());
            last_eval_result = JITValue::make_matrix(TernaryMatrix(r, c));
            return;
        }

        // 6. Builtin: ones(rows, cols)
        if (fn == "ones") {
            if (node.arguments.size() != 2) throw std::runtime_error("ones requires (rows, cols)!");
            node.arguments[0]->accept(*this);
            size_t r = static_cast<size_t>(last_eval_result.as_int64());
            node.arguments[1]->accept(*this);
            size_t c = static_cast<size_t>(last_eval_result.as_int64());
            TernaryMatrix mat(r, c);
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    mat.set(i, j, Trit::Pos);
                }
            }
            last_eval_result = JITValue::make_matrix(std::move(mat));
            return;
        }

        // 7. Builtin: rand_matrix(rows, cols)
        if (fn == "rand_matrix") {
            if (node.arguments.size() != 2) throw std::runtime_error("rand_matrix requires (rows, cols)!");
            node.arguments[0]->accept(*this);
            size_t r = static_cast<size_t>(last_eval_result.as_int64());
            node.arguments[1]->accept(*this);
            size_t c = static_cast<size_t>(last_eval_result.as_int64());
            TernaryMatrix mat(r, c);
            std::mt19937_64 rng(1337);
            std::uniform_int_distribution<int> dist(-1, 1);
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    mat.set(i, j, from_int(static_cast<int8_t>(dist(rng))));
                }
            }
            last_eval_result = JITValue::make_matrix(std::move(mat));
            return;
        }

        // 8. Builtin: matrix_init(rows, cols, t0, t1, ...)
        if (fn == "matrix_init") {
            if (node.arguments.size() < 2) throw std::runtime_error("matrix_init requires at least (rows, cols)!");
            node.arguments[0]->accept(*this);
            size_t r = static_cast<size_t>(last_eval_result.as_int64());
            node.arguments[1]->accept(*this);
            size_t c = static_cast<size_t>(last_eval_result.as_int64());
            TernaryMatrix mat(r, c);
            size_t arg_idx = 2;
            for (size_t i = 0; i < r; ++i) {
                for (size_t j = 0; j < c; ++j) {
                    if (arg_idx < node.arguments.size()) {
                        node.arguments[arg_idx++]->accept(*this);
                        int8_t t = static_cast<int8_t>(last_eval_result.as_int64());
                        mat.set(i, j, from_int(t));
                    }
                }
            }
            last_eval_result = JITValue::make_matrix(std::move(mat));
            return;
        }

        // 8b. Builtin: vector_init(v0, v1, ...)
        if (fn == "vector_init") {
            size_t count = node.arguments.size();
            DenseMatrixInt64 vec(count, 1);
            for (size_t i = 0; i < count; ++i) {
                node.arguments[i]->accept(*this);
                vec.data[i] = last_eval_result.as_int64();
            }
            last_eval_result = JITValue::make_int_matrix(std::move(vec));
            return;
        }

        // 9. Builtin: to_int(x)
        if (fn == "to_int") {
            if (node.arguments.size() != 1) throw std::runtime_error("to_int requires 1 argument!");
            node.arguments[0]->accept(*this);
            last_eval_result = JITValue::make_scalar_int(last_eval_result.as_int64());
            return;
        }

        // 9b. Builtin: fma(a, b, c)
        if (fn == "fma") {
            if (node.arguments.size() != 3) throw std::runtime_error("fma requires 3 arguments!");
            node.arguments[0]->accept(*this);
            JITValue a = last_eval_result;
            node.arguments[1]->accept(*this);
            JITValue b = last_eval_result;
            node.arguments[2]->accept(*this);
            JITValue c = last_eval_result;
            bt::BTFP32 af = (a.kind == JITValue::Kind::Float) ? a.btfp_val : bt::BTFP32::from_double(static_cast<double>(a.as_int64()));
            bt::BTFP32 bf = (b.kind == JITValue::Kind::Float) ? b.btfp_val : bt::BTFP32::from_double(static_cast<double>(b.as_int64()));
            bt::BTFP32 cf = (c.kind == JITValue::Kind::Float) ? c.btfp_val : bt::BTFP32::from_double(static_cast<double>(c.as_int64()));
            bt::BTFP32 res = bt::BTFP32::fma(af, bf, cf);
            last_eval_result = JITValue::make_btfp(res);
            return;
        }

        // 9c. Builtin: sqrt(x)
        if (fn == "sqrt") {
            if (node.arguments.size() != 1) throw std::runtime_error("sqrt requires 1 argument!");
            node.arguments[0]->accept(*this);
            JITValue x = last_eval_result;
            bt::BTFP32 xf = (x.kind == JITValue::Kind::Float) ? x.btfp_val : bt::BTFP32::from_double(static_cast<double>(x.as_int64()));
            bt::BTFP32 res = xf.sqrt();
            last_eval_result = JITValue::make_btfp(res);
            return;
        }

        // 9d. Builtin: abs(x)
        if (fn == "abs") {
            if (node.arguments.size() != 1) throw std::runtime_error("abs requires 1 argument!");
            node.arguments[0]->accept(*this);
            JITValue x = last_eval_result;
            if (x.kind == JITValue::Kind::Float) {
                bt::BTFP32 res = x.btfp_val.abs();
                last_eval_result = JITValue::make_btfp(res);
            } else {
                int64_t val = x.as_int64();
                last_eval_result = JITValue::make_scalar_int(val < 0 ? -val : val);
            }
            return;
        }

        // 10. User-defined function call
        auto func_def = env->get_function(fn);
        if (func_def) {
            if (node.arguments.size() != func_def->params.size()) {
                throw std::runtime_error("Function '" + fn + "' expects " +
                                         std::to_string(func_def->params.size()) +
                                         " arguments, got " + std::to_string(node.arguments.size()));
            }
            std::vector<JITValue> evaluated_args;
            evaluated_args.reserve(node.arguments.size());
            for (const auto& arg : node.arguments) {
                arg->accept(*this);
                evaluated_args.push_back(last_eval_result);
            }

            auto fn_env = std::make_shared<Environment>(env);
            for (size_t i = 0; i < func_def->params.size(); ++i) {
                fn_env->define(func_def->params[i].name, std::move(evaluated_args[i]));
            }

            last_eval_result = JITValue::make_void();
            JITValue return_val = JITValue::make_void();
            // The guard restores this->env on EVERY exit path: normal return, caught
            // ReturnException, and any exception propagating out of the body.
            EnvironmentFrameGuard frame_guard(*this, fn_env);
            try {
                if (func_def->body) func_def->body->accept(*this);
            } catch (const ReturnException& ret) {
                return_val = ret.value;
            }
            last_eval_result = return_val;
            return;
        }

        throw std::runtime_error("Call to undefined function: '" + fn + "'");
    }

    // --------------------------------------------------------------------------
    // Entry Point Execution APIs
    // --------------------------------------------------------------------------

    void execute(bt::frontend::ProgramNode& program) {
        try {
            visit(program);
        } catch (const ReturnException& ret) {
            last_eval_result = ret.value;
        }
    }

    void execute_source(const std::string& source) {
        bt::frontend::Parser parser(source);
        auto prog = parser.parse_program();
        execute(*prog);
    }

    void execute_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open source file: " + filepath);
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        execute_source(ss.str());
    }
};

} // namespace jit
} // namespace bt

#endif // TERNARY_JIT_ENGINE_HPP
