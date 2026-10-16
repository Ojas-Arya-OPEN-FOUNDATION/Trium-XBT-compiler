/**
 * @file ternary_llvm_codegen.hpp
 * @brief AST-to-LLVM IR Lowering Visitor & Native JIT Engine
 * @author Universal Balanced Ternary Compiler Engineering Team
 */

#ifndef TERNARY_LLVM_CODEGEN_HPP
#define TERNARY_LLVM_CODEGEN_HPP

#include "ternary_frontend_ast.hpp"
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>

namespace bt::codegen {

class LLVMCodeGenVisitor : public frontend::ASTVisitor {
private:
    std::ostringstream ir_;
    std::ostringstream fn_body_;
    std::unordered_map<std::string, std::string> symbols_; // var name -> LLVM pointer
    std::unordered_map<std::string, std::string> types_;   // var name -> "i64" or "double" or "sparse_matrix" or "i64*"
    std::unordered_map<std::string, size_t> vector_dims_;  // var name -> vector dimension
    uint64_t reg_counter_{0};
    uint64_t label_counter_{0};
    uint64_t sparse_counter_{0};
    std::string current_val_;
    std::string current_type_{"i64"};
    size_t current_dim_{0};

    struct SparseMatrixInfo {
        std::string name;
        size_t rows{0};
        size_t cols{0};
        size_t nnz{0};
        std::string g_row_ptr;
        std::string g_col_ind;
        std::string g_signs;
    };
    std::unordered_map<std::string, SparseMatrixInfo> sparse_matrices_;

    std::string new_reg() {
        return "%t" + std::to_string(++reg_counter_);
    }

    std::string new_label(const std::string& prefix = "bb") {
        return prefix + "_" + std::to_string(++label_counter_);
    }

public:
    std::string generate(frontend::ProgramNode& program,
                          const std::string& target_triple = "arm64-apple-macosx",
                          const std::string& datalayout = "e-m:o-i64:64-i128:128-n32:64-S128") {
        ir_.str(""); ir_.clear();
        fn_body_.str(""); fn_body_.clear();
        symbols_.clear();
        types_.clear();
        vector_dims_.clear();
        sparse_matrices_.clear();
        reg_counter_ = 0;
        label_counter_ = 0;
        sparse_counter_ = 0;
        current_val_ = "0";
        current_type_ = "i64";
        current_dim_ = 0;

        // 1. Module Header
        ir_ << "; ModuleID = 'bt_native_module'\n";
        ir_ << "source_filename = \"bt_program.bt\"\n";
        ir_ << "target datalayout = \"" << datalayout << "\"\n";
        ir_ << "target triple = \"" << target_triple << "\"\n\n";

        // 2. Standard Intrinsics & Builtins
        ir_ << "declare i64 @llvm.ctpop.i64(i64) nounwind readnone speculatable willreturn\n";
        ir_ << "declare i32 @printf(i8*, ...) nounwind\n\n";

        ir_ << "@.str_int = private unnamed_addr constant [14 x i8] c\"%lld [trits]\\0A\\00\", align 1\n";
        ir_ << "@.str_fmt = private unnamed_addr constant [6 x i8] c\"%lld\\0A\\00\", align 1\n";
        ir_ << "@.str_float = private unnamed_addr constant [6 x i8] c\"%.6f\\0A\\00\", align 1\n\n";

        // Helper: Spaceship operator (<=>)
        ir_ << "define i64 @bt_spaceship(i64 %a, i64 %b) nounwind readnone {\n";
        ir_ << "entry:\n";
        ir_ << "  %cmp_gt = icmp sgt i64 %a, %b\n";
        ir_ << "  %cmp_lt = icmp slt i64 %a, %b\n";
        ir_ << "  %res_gt = select i1 %cmp_gt, i64 1, i64 0\n";
        ir_ << "  %res = select i1 %cmp_lt, i64 -1, i64 %res_gt\n";
        ir_ << "  ret i64 %res\n";
        ir_ << "}\n\n";

        // Helper: 3-way branchless select
        ir_ << "define i64 @bt_select(i64 %cond, i64 %p_val, i64 %z_val, i64 %n_val) nounwind readnone {\n";
        ir_ << "entry:\n";
        ir_ << "  %is_pos = icmp eq i64 %cond, 1\n";
        ir_ << "  %is_neg = icmp eq i64 %cond, -1\n";
        ir_ << "  %sel1 = select i1 %is_pos, i64 %p_val, i64 %z_val\n";
        ir_ << "  %res = select i1 %is_neg, i64 %n_val, i64 %sel1\n";
        ir_ << "  ret i64 %res\n";
        ir_ << "}\n\n";

        // Helper: Print integer
        ir_ << "define void @bt_print(i64 %val) nounwind {\n";
        ir_ << "entry:\n";
        ir_ << "  %fmt = getelementptr inbounds [6 x i8], [6 x i8]* @.str_fmt, i64 0, i64 0\n";
        ir_ << "  %call = call i32 (i8*, ...) @printf(i8* %fmt, i64 %val)\n";
        ir_ << "  ret void\n";
        ir_ << "}\n\n";

        // Helper: Print float
        ir_ << "define void @bt_print_float(double %val) nounwind {\n";
        ir_ << "entry:\n";
        ir_ << "  %fmt = getelementptr inbounds [6 x i8], [6 x i8]* @.str_float, i64 0, i64 0\n";
        ir_ << "  %call = call i32 (i8*, ...) @printf(i8* %fmt, double %val)\n";
        ir_ << "  ret void\n";
        ir_ << "}\n\n";

        // Helper: Multiplier-free T-CSR Sparse Matrix-Vector Multiply (SpMV)
        ir_ << "define void @bt_spmv(i64 %rows, i32* %row_ptr, i32* %col_indices, i64* %signs, i64* %vec_in, i64* %vec_out) nounwind {\n";
        ir_ << "entry:\n";
        ir_ << "  br label %row_loop_cond\n\n";
        ir_ << "row_loop_cond:\n";
        ir_ << "  %r = phi i64 [ 0, %entry ], [ %next_r, %row_loop_inc ]\n";
        ir_ << "  %cmp_row = icmp ult i64 %r, %rows\n";
        ir_ << "  br i1 %cmp_row, label %row_loop_body, label %spmv_done\n\n";
        ir_ << "row_loop_body:\n";
        ir_ << "  %p_start = getelementptr inbounds i32, i32* %row_ptr, i64 %r\n";
        ir_ << "  %raw_start = load i32, i32* %p_start, align 4\n";
        ir_ << "  %start = zext i32 %raw_start to i64\n";
        ir_ << "  %r_plus_1 = add i64 %r, 1\n";
        ir_ << "  %p_end = getelementptr inbounds i32, i32* %row_ptr, i64 %r_plus_1\n";
        ir_ << "  %raw_end = load i32, i32* %p_end, align 4\n";
        ir_ << "  %end = zext i32 %raw_end to i64\n";
        ir_ << "  br label %nz_loop_cond\n\n";
        ir_ << "nz_loop_cond:\n";
        ir_ << "  %k = phi i64 [ %start, %row_loop_body ], [ %next_k, %nz_loop_inc ]\n";
        ir_ << "  %acc = phi i64 [ 0, %row_loop_body ], [ %new_acc, %nz_loop_inc ]\n";
        ir_ << "  %cmp_k = icmp ult i64 %k, %end\n";
        ir_ << "  br i1 %cmp_k, label %nz_loop_body, label %row_loop_inc\n\n";
        ir_ << "nz_loop_body:\n";
        ir_ << "  %p_col = getelementptr inbounds i32, i32* %col_indices, i64 %k\n";
        ir_ << "  %raw_col = load i32, i32* %p_col, align 4\n";
        ir_ << "  %col = zext i32 %raw_col to i64\n";
        ir_ << "  %word_idx = lshr i64 %k, 6\n";
        ir_ << "  %bit_idx = and i64 %k, 63\n";
        ir_ << "  %p_sign_word = getelementptr inbounds i64, i64* %signs, i64 %word_idx\n";
        ir_ << "  %sign_word = load i64, i64* %p_sign_word, align 8\n";
        ir_ << "  %shifted_sign = lshr i64 %sign_word, %bit_idx\n";
        ir_ << "  %sign_bit = and i64 %shifted_sign, 1\n";
        ir_ << "  %is_pos = icmp eq i64 %sign_bit, 1\n";
        ir_ << "  %p_in = getelementptr inbounds i64, i64* %vec_in, i64 %col\n";
        ir_ << "  %x_val = load i64, i64* %p_in, align 8\n";
        ir_ << "  %acc_add = add i64 %acc, %x_val\n";
        ir_ << "  %acc_sub = sub i64 %acc, %x_val\n";
        ir_ << "  %new_acc = select i1 %is_pos, i64 %acc_add, i64 %acc_sub\n";
        ir_ << "  br label %nz_loop_inc\n\n";
        ir_ << "nz_loop_inc:\n";
        ir_ << "  %next_k = add i64 %k, 1\n";
        ir_ << "  br label %nz_loop_cond\n\n";
        ir_ << "row_loop_inc:\n";
        ir_ << "  %final_acc = phi i64 [ %acc, %nz_loop_cond ]\n";
        ir_ << "  %p_out = getelementptr inbounds i64, i64* %vec_out, i64 %r\n";
        ir_ << "  store i64 %final_acc, i64* %p_out, align 8\n";
        ir_ << "  %next_r = add i64 %r, 1\n";
        ir_ << "  br label %row_loop_cond\n\n";
        ir_ << "spmv_done:\n";
        ir_ << "  ret void\n";
        ir_ << "}\n\n";

        // 3. Generate main entry function containing top-level statements
        fn_body_ << "define i32 @bt_main() nounwind {\n";
        fn_body_ << "entry:\n";

        // Visit top-level AST declarations
        for (auto& decl : program.declarations) {
            if (decl) decl->accept(*this);
        }

        fn_body_ << "  ret i32 0\n";
        fn_body_ << "}\n";

        // Combine module header + generated functions
        ir_ << fn_body_.str();
        return ir_.str();
    }

    void visit(frontend::ProgramNode& node) override {
        for (auto& d : node.declarations) {
            if (d) d->accept(*this);
        }
    }

    void visit(frontend::BlockNode& node) override {
        for (auto& s : node.statements) {
            if (s) s->accept(*this);
        }
    }

    void visit(frontend::FunctionDefNode& /*node*/) override {
        // Function definitions can be extended for multi-function modules
    }

    void visit(frontend::VarDeclNode& node) override {
        std::string ptr_reg = "%var_" + node.name;
        symbols_[node.name] = ptr_reg;

        std::string v_type = "i64";
        if (node.type == "float") {
            v_type = "double";
        } else if (node.type == "sparse_matrix") {
            v_type = "sparse_matrix";
        }

        if (node.init_expr) {
            node.init_expr->accept(*this);
            if (current_type_ == "double") v_type = "double";
            else if (current_type_ == "sparse_matrix") v_type = "sparse_matrix";
            else if (current_type_ == "i64*") v_type = "i64*";
        }

        types_[node.name] = v_type;

        if (v_type == "double") {
            fn_body_ << "  " << ptr_reg << " = alloca double, align 8\n";
            if (node.init_expr) {
                fn_body_ << "  store double " << current_val_ << ", double* " << ptr_reg << ", align 8\n";
            } else {
                fn_body_ << "  store double 0.0, double* " << ptr_reg << ", align 8\n";
            }
        } else if (v_type == "sparse_matrix") {
            if (current_type_ == "sparse_matrix") {
                sparse_matrices_[node.name] = sparse_matrices_[current_val_];
            }
        } else if (v_type == "i64*") {
            symbols_[node.name] = current_val_;
            vector_dims_[node.name] = current_dim_;
        } else {
            fn_body_ << "  " << ptr_reg << " = alloca i64, align 8\n";
            if (node.init_expr) {
                fn_body_ << "  store i64 " << current_val_ << ", i64* " << ptr_reg << ", align 8\n";
            } else {
                fn_body_ << "  store i64 0, i64* " << ptr_reg << ", align 8\n";
            }
        }
    }

    void visit(frontend::AssignNode& node) override {
        if (node.value) {
            node.value->accept(*this);
            auto it = symbols_.find(node.target);
            if (it != symbols_.end()) {
                auto it_type = types_.find(node.target);
                std::string type = (it_type != types_.end()) ? it_type->second : "i64";
                if (type == "double") {
                    fn_body_ << "  store double " << current_val_ << ", double* " << it->second << ", align 8\n";
                } else if (type == "i64*") {
                    symbols_[node.target] = current_val_;
                    vector_dims_[node.target] = current_dim_;
                } else {
                    fn_body_ << "  store i64 " << current_val_ << ", i64* " << it->second << ", align 8\n";
                }
            }
        }
    }

    void visit(frontend::BinaryOpNode& node) override {
        node.left->accept(*this);
        std::string left_val = current_val_;
        std::string left_type = current_type_;

        node.right->accept(*this);
        std::string right_val = current_val_;
        std::string right_type = current_type_;

        // 1. SpMV: sparse_matrix * vector
        if (node.op == frontend::BinaryOp::Mul && left_type == "sparse_matrix") {
            auto it = sparse_matrices_.find(left_val);
            if (it != sparse_matrices_.end()) {
                const auto& info = it->second;
                std::string out_vec = new_reg();
                fn_body_ << "  " << out_vec << " = alloca [" << info.rows << " x i64], align 8\n";
                std::string p_out = new_reg();
                fn_body_ << "  " << p_out << " = getelementptr inbounds [" << info.rows
                         << " x i64], [" << info.rows << " x i64]* " << out_vec << ", i64 0, i64 0\n";

                std::string p_r = new_reg();
                fn_body_ << "  " << p_r << " = getelementptr inbounds [" << (info.rows + 1)
                         << " x i32], [" << (info.rows + 1) << " x i32]* " << info.g_row_ptr << ", i64 0, i64 0\n";

                std::string p_c = new_reg();
                size_t c_sz = std::max<size_t>(info.nnz, 1);
                fn_body_ << "  " << p_c << " = getelementptr inbounds [" << c_sz
                         << " x i32], [" << c_sz << " x i32]* " << info.g_col_ind << ", i64 0, i64 0\n";

                std::string p_s = new_reg();
                size_t s_sz = std::max<size_t>((info.nnz + 63) / 64, 1);
                fn_body_ << "  " << p_s << " = getelementptr inbounds [" << s_sz
                         << " x i64], [" << s_sz << " x i64]* " << info.g_signs << ", i64 0, i64 0\n";

                fn_body_ << "  call void @bt_spmv(i64 " << info.rows << ", i32* " << p_r
                         << ", i32* " << p_c << ", i64* " << p_s << ", i64* " << right_val
                         << ", i64* " << p_out << ")\n";

                current_val_ = p_out;
                current_type_ = "i64*";
                current_dim_ = info.rows;
                return;
            }
        }

        // 2. Float arithmetic
        if (left_type == "double" || right_type == "double") {
            std::string res = new_reg();
            switch (node.op) {
                case frontend::BinaryOp::Add:
                    fn_body_ << "  " << res << " = fadd double " << left_val << ", " << right_val << "\n";
                    break;
                case frontend::BinaryOp::Sub:
                    fn_body_ << "  " << res << " = fsub double " << left_val << ", " << right_val << "\n";
                    break;
                case frontend::BinaryOp::Mul:
                    fn_body_ << "  " << res << " = fmul double " << left_val << ", " << right_val << "\n";
                    break;
                default:
                    fn_body_ << "  " << res << " = fadd double " << left_val << ", " << right_val << "\n";
                    break;
            }
            current_val_ = res;
            current_type_ = "double";
            return;
        }

        // 3. Integer & Trit arithmetic
        std::string res = new_reg();
        switch (node.op) {
            case frontend::BinaryOp::Add:
                fn_body_ << "  " << res << " = add i64 " << left_val << ", " << right_val << "\n";
                break;
            case frontend::BinaryOp::Sub:
                fn_body_ << "  " << res << " = sub i64 " << left_val << ", " << right_val << "\n";
                break;
            case frontend::BinaryOp::Mul:
                fn_body_ << "  " << res << " = mul i64 " << left_val << ", " << right_val << "\n";
                break;
            case frontend::BinaryOp::KleeneAnd:
                fn_body_ << "  " << res << " = and i64 " << left_val << ", " << right_val << "\n";
                break;
            case frontend::BinaryOp::KleeneOr:
                fn_body_ << "  " << res << " = or i64 " << left_val << ", " << right_val << "\n";
                break;
            case frontend::BinaryOp::TernaryXor:
                fn_body_ << "  " << res << " = xor i64 " << left_val << ", " << right_val << "\n";
                break;
            case frontend::BinaryOp::Spaceship:
                fn_body_ << "  " << res << " = call i64 @bt_spaceship(i64 " << left_val << ", i64 " << right_val << ")\n";
                break;
            case frontend::BinaryOp::Equal: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp eq i64 " << left_val << ", " << right_val << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1\n";
                break;
            }
            case frontend::BinaryOp::NotEqual: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp ne i64 " << left_val << ", " << right_val << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1\n";
                break;
            }
            case frontend::BinaryOp::LessThan: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp slt i64 " << left_val << ", " << right_val << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1\n";
                break;
            }
            case frontend::BinaryOp::GreaterThan: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp sgt i64 " << left_val << ", " << right_val << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1\n";
                break;
            }
            default:
                fn_body_ << "  " << res << " = add i64 " << left_val << ", " << right_val << "\n";
                break;
        }
        current_val_ = res;
        current_type_ = "i64";
    }

    void visit(frontend::UnaryOpNode& node) override {
        node.operand->accept(*this);
        std::string res = new_reg();
        if (current_type_ == "double") {
            fn_body_ << "  " << res << " = fsub double -0.0, " << current_val_ << "\n";
        } else {
            fn_body_ << "  " << res << " = sub i64 0, " << current_val_ << "\n";
        }
        current_val_ = res;
    }

    void visit(frontend::SelectNode& node) override {
        node.condition->accept(*this);
        std::string cond_val = current_val_;

        node.pos_branch->accept(*this);
        std::string p_val = current_val_;

        node.zero_branch->accept(*this);
        std::string z_val = current_val_;

        node.neg_branch->accept(*this);
        std::string n_val = current_val_;

        std::string res = new_reg();
        fn_body_ << "  " << res << " = call i64 @bt_select(i64 " << cond_val << ", i64 "
                 << p_val << ", i64 " << z_val << ", i64 " << n_val << ")\n";
        current_val_ = res;
        current_type_ = "i64";
    }

    void visit(frontend::WhileNode& node) override {
        std::string cond_lbl = new_label("while_cond");
        std::string body_lbl = new_label("while_body");
        std::string exit_lbl = new_label("while_exit");

        fn_body_ << "  br label %" << cond_lbl << "\n";
        fn_body_ << cond_lbl << ":\n";

        node.condition->accept(*this);
        std::string cmp = new_reg();
        fn_body_ << "  " << cmp << " = icmp ne i64 " << current_val_ << ", 0\n";
        fn_body_ << "  br i1 " << cmp << ", label %" << body_lbl << ", label %" << exit_lbl << "\n";

        fn_body_ << body_lbl << ":\n";
        if (node.body) node.body->accept(*this);
        fn_body_ << "  br label %" << cond_lbl << "\n";

        fn_body_ << exit_lbl << ":\n";
    }

    void visit(frontend::ReturnNode& node) override {
        if (node.expr) {
            node.expr->accept(*this);
            if (current_type_ == "double") {
                fn_body_ << "  ret double " << current_val_ << "\n";
            } else {
                fn_body_ << "  ret i64 " << current_val_ << "\n";
            }
        } else {
            fn_body_ << "  ret void\n";
        }
    }

    void visit(frontend::LiteralNode& node) override {
        current_val_ = std::to_string(node.int_val);
        current_type_ = "i64";
    }

    void visit(frontend::FloatLiteralExprAST& node) override {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(10) << node.value;
        current_val_ = ss.str();
        current_type_ = "double";
    }

    void visit(frontend::SparseMatrixDeclAST& node) override {
        std::string s_id = node.name.empty() ? ("smat_" + std::to_string(++sparse_counter_)) : node.name;
        std::string g_row_ptr = "@" + s_id + "_row_ptr";
        std::string g_col_ind = "@" + s_id + "_col_ind";
        std::string g_signs   = "@" + s_id + "_signs";

        size_t r_count = node.rows + 1;
        size_t c_count = std::max<size_t>(node.matrix.col_indices.size(), 1);
        size_t s_count = std::max<size_t>(node.matrix.signs.size(), 1);

        ir_ << g_row_ptr << " = internal constant [" << r_count << " x i32] [";
        for (size_t i = 0; i < r_count; ++i) {
            uint32_t val = (i < node.matrix.row_ptr.size()) ? node.matrix.row_ptr[i] : 0;
            ir_ << "i32 " << val;
            if (i + 1 < r_count) ir_ << ", ";
        }
        ir_ << "], align 4\n";

        ir_ << g_col_ind << " = internal constant [" << c_count << " x i32] [";
        for (size_t i = 0; i < c_count; ++i) {
            uint32_t val = (i < node.matrix.col_indices.size()) ? node.matrix.col_indices[i] : 0;
            ir_ << "i32 " << val;
            if (i + 1 < c_count) ir_ << ", ";
        }
        ir_ << "], align 4\n";

        ir_ << g_signs << " = internal constant [" << s_count << " x i64] [";
        for (size_t i = 0; i < s_count; ++i) {
            uint64_t val = (i < node.matrix.signs.size()) ? node.matrix.signs[i] : 0;
            ir_ << "i64 " << static_cast<int64_t>(val);
            if (i + 1 < s_count) ir_ << ", ";
        }
        ir_ << "], align 8\n\n";

        SparseMatrixInfo info{s_id, node.rows, node.cols, node.matrix.nnz, g_row_ptr, g_col_ind, g_signs};
        sparse_matrices_[s_id] = info;
        if (!node.name.empty()) {
            sparse_matrices_[node.name] = info;
            types_[node.name] = "sparse_matrix";
        }
        current_val_ = s_id;
        current_type_ = "sparse_matrix";
    }

    void visit(frontend::IdentifierNode& node) override {
        auto it_type = types_.find(node.name);
        std::string type = (it_type != types_.end()) ? it_type->second : "i64";

        if (type == "sparse_matrix") {
            current_val_ = node.name;
            current_type_ = "sparse_matrix";
            return;
        }

        if (type == "i64*") {
            auto it = symbols_.find(node.name);
            current_val_ = (it != symbols_.end()) ? it->second : "null";
            current_type_ = "i64*";
            auto d_it = vector_dims_.find(node.name);
            current_dim_ = (d_it != vector_dims_.end()) ? d_it->second : 0;
            return;
        }

        auto it = symbols_.find(node.name);
        if (it != symbols_.end()) {
            std::string res = new_reg();
            if (type == "double") {
                fn_body_ << "  " << res << " = load double, double* " << it->second << ", align 8\n";
                current_type_ = "double";
            } else {
                fn_body_ << "  " << res << " = load i64, i64* " << it->second << ", align 8\n";
                current_type_ = "i64";
            }
            current_val_ = res;
        } else {
            current_val_ = "0";
            current_type_ = "i64";
        }
    }

    void visit(frontend::CallNode& node) override {
        if (node.callee == "print") {
            if (!node.arguments.empty()) {
                node.arguments[0]->accept(*this);
                if (current_type_ == "double") {
                    fn_body_ << "  call void @bt_print_float(double " << current_val_ << ")\n";
                } else if (current_type_ == "i64*") {
                    for (size_t i = 0; i < current_dim_; ++i) {
                        std::string p_elem = new_reg();
                        fn_body_ << "  " << p_elem << " = getelementptr inbounds i64, i64* "
                                 << current_val_ << ", i64 " << i << "\n";
                        std::string val = new_reg();
                        fn_body_ << "  " << val << " = load i64, i64* " << p_elem << ", align 8\n";
                        fn_body_ << "  call void @bt_print(i64 " << val << ")\n";
                    }
                } else {
                    fn_body_ << "  call void @bt_print(i64 " << current_val_ << ")\n";
                }
            }
            current_val_ = "0";
            current_type_ = "i64";
        } else if (node.callee == "vector_init" || (node.callee == "matrix_init" && node.arguments.size() >= 2)) {
            size_t count = 0;
            size_t start_idx = 0;
            if (node.callee == "vector_init") {
                count = node.arguments.size();
                start_idx = 0;
            } else {
                count = node.arguments.size() - 2;
                start_idx = 2;
            }

            std::string arr = new_reg();
            fn_body_ << "  " << arr << " = alloca [" << count << " x i64], align 8\n";
            for (size_t i = 0; i < count; ++i) {
                node.arguments[start_idx + i]->accept(*this);
                std::string p_i = new_reg();
                fn_body_ << "  " << p_i << " = getelementptr inbounds [" << count << " x i64], ["
                         << count << " x i64]* " << arr << ", i64 0, i64 " << i << "\n";
                fn_body_ << "  store i64 " << current_val_ << ", i64* " << p_i << ", align 8\n";
            }
            std::string p_ret = new_reg();
            fn_body_ << "  " << p_ret << " = getelementptr inbounds [" << count << " x i64], ["
                     << count << " x i64]* " << arr << ", i64 0, i64 0\n";
            current_val_ = p_ret;
            current_type_ = "i64*";
            current_dim_ = count;
        } else if (node.callee == "select" && node.arguments.size() == 4) {
            node.arguments[0]->accept(*this);
            std::string c_val = current_val_;
            node.arguments[1]->accept(*this);
            std::string p_val = current_val_;
            node.arguments[2]->accept(*this);
            std::string z_val = current_val_;
            node.arguments[3]->accept(*this);
            std::string n_val = current_val_;

            std::string res = new_reg();
            fn_body_ << "  " << res << " = call i64 @bt_select(i64 " << c_val << ", i64 "
                     << p_val << ", i64 " << z_val << ", i64 " << n_val << ")\n";
            current_val_ = res;
            current_type_ = "i64";
        } else {
            current_val_ = "0";
            current_type_ = "i64";
        }
    }

    void visit(frontend::IfNode& node) override {
        std::string then_lbl = new_label("if_then");
        std::string else_lbl = new_label("if_else");
        std::string exit_lbl = new_label("if_exit");

        node.condition->accept(*this);
        std::string cmp = new_reg();
        fn_body_ << "  " << cmp << " = icmp ne i64 " << current_val_ << ", 0\n";
        fn_body_ << "  br i1 " << cmp << ", label %" << then_lbl << ", label %"
                 << (node.else_block ? else_lbl : exit_lbl) << "\n";

        fn_body_ << then_lbl << ":\n";
        if (node.then_block) node.then_block->accept(*this);
        fn_body_ << "  br label %" << exit_lbl << "\n";

        if (node.else_block) {
            fn_body_ << else_lbl << ":\n";
            node.else_block->accept(*this);
            fn_body_ << "  br label %" << exit_lbl << "\n";
        }

        fn_body_ << exit_lbl << ":\n";
    }

    void visit(frontend::ExpressionStmtNode& node) override {
        if (node.expr) node.expr->accept(*this);
    }
};

/**
 * @brief Native JIT Execution Engine
 * Compiles emitted LLVM IR to a native shared library (.dylib) on the fly via Clang,
 * dynamically loads it using dlopen/dlsym, and executes @bt_main at 100% bare-metal speed!
 */
class NativeLLVMJIT {
public:
    static int execute_program(frontend::ProgramNode& program, bool verbose = false) {
        LLVMCodeGenVisitor codegen;
        std::string llvm_ir = codegen.generate(program);

        char tmp_ll_path[] = "/tmp/bt_jit_XXXXXX.ll";
        int fd_ll = mkstemps(tmp_ll_path, 3);
        if (fd_ll == -1) {
            std::cerr << "[-] JIT Error: Failed to create temporary .ll file!\n";
            return 1;
        }
        write(fd_ll, llvm_ir.data(), llvm_ir.size());
        close(fd_ll);

        std::string tmp_dylib_path = std::string(tmp_ll_path) + ".dylib";

        // Build native shared library via Clang
        std::string cmd = "export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk && "
                          "clang -O3 -shared -fPIC " + std::string(tmp_ll_path) +
                          " -o " + tmp_dylib_path + " 2>/dev/null";

        if (verbose) {
            std::cout << "[Native JIT] Compiling LLVM IR to native shared library: " << tmp_dylib_path << "\n";
        }

        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "[-] JIT Error: Clang native compilation failed!\n";
            unlink(tmp_ll_path);
            return 1;
        }

        // Dynamically load compiled dylib
        void* handle = dlopen(tmp_dylib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            std::cerr << "[-] JIT Error: dlopen failed: " << dlerror() << "\n";
            unlink(tmp_ll_path);
            unlink(tmp_dylib_path.c_str());
            return 1;
        }

        // Resolve entry point
        using MainFn = int (*)();
        MainFn fn = reinterpret_cast<MainFn>(dlsym(handle, "bt_main"));
        if (!fn) {
            std::cerr << "[-] JIT Error: Symbol 'bt_main' not found: " << dlerror() << "\n";
            dlclose(handle);
            unlink(tmp_ll_path);
            unlink(tmp_dylib_path.c_str());
            return 1;
        }

        // Execute at native CPU speed!
        int exit_code = fn();

        dlclose(handle);
        unlink(tmp_ll_path);
        unlink(tmp_dylib_path.c_str());

        return exit_code;
    }
};

} // namespace bt::codegen

#endif // TERNARY_LLVM_CODEGEN_HPP
