/**
 * @file ternary_llvm_codegen.hpp
 * @brief AST-to-LLVM IR Lowering Visitor & Native JIT Engine
 * @author Trium-XBT Foundational Compiler Engineering Team
 */

#ifndef TERNARY_LLVM_CODEGEN_HPP
#define TERNARY_LLVM_CODEGEN_HPP

#include "ternary_frontend_ast.hpp"
#include <map>
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
#include <atomic>

namespace bt::codegen {

class LLVMCodeGenVisitor : public frontend::ASTVisitor {
private:
    std::ostringstream ir_;
    std::ostringstream fn_body_;
    std::ostringstream user_fns_;
    std::unordered_map<std::string, std::string> user_func_ret_types_;

    struct VarInfo {
        std::string ptr_reg; // e.g. %var_x_1
        std::string type;    // "i64", "double", "sparse_matrix", "i64*"
        size_t dim{0};
    };

    std::vector<std::unordered_map<std::string, VarInfo>> scope_stack_;
    uint64_t alloca_counter_{0};
    uint64_t reg_counter_{0};
    uint64_t label_counter_{0};
    uint64_t sparse_counter_{0};
    std::string current_val_;
    std::string current_type_{"i64"};
    size_t current_dim_{0};
    bool block_terminated_{false};
    bool is_in_bt_main_{false};

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

    bool emit_debug_{false};
    std::string source_filename_{"program.bt"};
    frontend::SourceLocation current_loc_{1, 1};
    uint64_t main_subprog_id_{4};
    uint64_t current_subprog_id_{4};
    uint64_t loop_meta_id_{6};
    uint64_t loop_enable_id_{7};
    uint64_t next_meta_id_{8};

    struct LocEntry {
        uint64_t id;
        size_t line;
        size_t col;
        uint64_t scope;
    };
    std::map<std::pair<size_t, size_t>, uint64_t> loc_map_;
    std::vector<LocEntry> loc_entries_;

    uint64_t get_or_create_loc(size_t line, size_t col, uint64_t scope = 4) {
        if (line == 0) line = 1;
        if (col == 0) col = 1;
        auto key = std::make_pair(line, col);
        auto it = loc_map_.find(key);
        if (it != loc_map_.end()) {
            return it->second;
        }
        uint64_t id = next_meta_id_++;
        loc_map_[key] = id;
        loc_entries_.push_back({id, line, col, scope});
        return id;
    }

    std::string dbg_tag() {
        if (!emit_debug_) return "";
        uint64_t loc_id = get_or_create_loc(current_loc_.line, current_loc_.column, current_subprog_id_);
        return ", !dbg !" + std::to_string(loc_id);
    }

    std::string new_reg() {
        return "%t" + std::to_string(++reg_counter_);
    }

    std::string new_label(const std::string& prefix = "bb") {
        return prefix + "_" + std::to_string(++label_counter_);
    }

    void enter_scope() {
        scope_stack_.emplace_back();
    }

    void exit_scope() {
        if (!scope_stack_.empty()) {
            scope_stack_.pop_back();
        }
    }

    const VarInfo* lookup_var(const std::string& name) const {
        for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
            auto fit = it->find(name);
            if (fit != it->end()) {
                return &fit->second;
            }
        }
        return nullptr;
    }

    VarInfo* lookup_var_mut(const std::string& name) {
        for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
            auto fit = it->find(name);
            if (fit != it->end()) {
                return &fit->second;
            }
        }
        return nullptr;
    }

public:
    LLVMCodeGenVisitor() = default;
    explicit LLVMCodeGenVisitor(bool emit_debug, const std::string& filename = "program.bt")
        : emit_debug_(emit_debug), source_filename_(filename) {}

    void set_debug_info(bool enable, const std::string& filename = "program.bt") {
        emit_debug_ = enable;
        source_filename_ = filename;
    }

    std::string generate(frontend::ProgramNode& program,
                          const std::string& target_triple = "arm64-apple-macosx",
                          const std::string& datalayout = "e-m:o-i64:64-i128:128-n32:64-S128") {
        ir_.str(""); ir_.clear();
        fn_body_.str(""); fn_body_.clear();
        user_fns_.str(""); user_fns_.clear();
        user_func_ret_types_.clear();
        scope_stack_.clear();
        enter_scope();
        sparse_matrices_.clear();
        loc_map_.clear();
        loc_entries_.clear();
        alloca_counter_ = 0;
        reg_counter_ = 0;
        label_counter_ = 0;
        sparse_counter_ = 0;
        current_val_ = "0";
        current_type_ = "i64";
        current_dim_ = 0;
        block_terminated_ = false;
        is_in_bt_main_ = false;
        current_loc_ = {1, 1};

        if (emit_debug_) {
            main_subprog_id_ = 4;
            current_subprog_id_ = 4;
            loop_meta_id_ = 6;
            loop_enable_id_ = 7;
            next_meta_id_ = 8;
        } else {
            loop_meta_id_ = 0;
            loop_enable_id_ = 1;
            next_meta_id_ = 2;
        }

        // 1. Module Header
        ir_ << "; ModuleID = 'bt_native_module'\n";
        ir_ << "source_filename = \"" << source_filename_ << "\"\n";
        ir_ << "target datalayout = \"" << datalayout << "\"\n";
        ir_ << "target triple = \"" << target_triple << "\"\n\n";

        // 2. Standard Intrinsics & Builtins
        ir_ << "declare i64 @llvm.ctpop.i64(i64) nounwind readnone speculatable willreturn\n";
        ir_ << "declare double @llvm.fma.f64(double, double, double) nounwind readnone speculatable willreturn\n";
        ir_ << "declare double @llvm.sqrt.f64(double) nounwind readnone speculatable willreturn\n";
        ir_ << "declare double @llvm.fabs.f64(double) nounwind readnone speculatable willreturn\n";
        ir_ << "declare i64 @llvm.abs.i64(i64, i1) nounwind readnone speculatable willreturn\n";
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
        ir_ << "  %is_pos = icmp sgt i64 %cond, 0\n";
        ir_ << "  %is_neg = icmp slt i64 %cond, 0\n";
        ir_ << "  %sel1 = select i1 %is_pos, i64 %p_val, i64 %z_val\n";
        ir_ << "  %res = select i1 %is_neg, i64 %n_val, i64 %sel1\n";
        ir_ << "  ret i64 %res\n";
        ir_ << "}\n\n";

        // Helper: Kleene 3-valued AND: min(a, b) for trits, digit-wise min for multi-trit integers
        ir_ << "define i64 @bt_kleene_and(i64 %a, i64 %b) nounwind readnone {\n";
        ir_ << "entry:\n";
        ir_ << "  %is_trit_a_lo = icmp sge i64 %a, -1\n";
        ir_ << "  %is_trit_a_hi = icmp sle i64 %a, 1\n";
        ir_ << "  %is_trit_a = and i1 %is_trit_a_lo, %is_trit_a_hi\n";
        ir_ << "  %is_trit_b_lo = icmp sge i64 %b, -1\n";
        ir_ << "  %is_trit_b_hi = icmp sle i64 %b, 1\n";
        ir_ << "  %is_trit_b = and i1 %is_trit_b_lo, %is_trit_b_hi\n";
        ir_ << "  %both_trits = and i1 %is_trit_a, %is_trit_b\n";
        ir_ << "  br i1 %both_trits, label %trit_and, label %loop_and_init\n";
        ir_ << "trit_and:\n";
        ir_ << "  %c_and = icmp slt i64 %a, %b\n";
        ir_ << "  %r_trit_and = select i1 %c_and, i64 %a, i64 %b\n";
        ir_ << "  ret i64 %r_trit_and\n";
        ir_ << "loop_and_init:\n";
        ir_ << "  br label %loop_and\n";
        ir_ << "loop_and:\n";
        ir_ << "  %cur_a = phi i64 [ %a, %loop_and_init ], [ %next_a, %loop_and_body ]\n";
        ir_ << "  %cur_b = phi i64 [ %b, %loop_and_init ], [ %next_b, %loop_and_body ]\n";
        ir_ << "  %cur_p3 = phi i64 [ 1, %loop_and_init ], [ %next_p3, %loop_and_body ]\n";
        ir_ << "  %cur_res = phi i64 [ 0, %loop_and_init ], [ %next_res, %loop_and_body ]\n";
        ir_ << "  %done_a = icmp eq i64 %cur_a, 0\n";
        ir_ << "  %done_b = icmp eq i64 %cur_b, 0\n";
        ir_ << "  %both_done = and i1 %done_a, %done_b\n";
        ir_ << "  br i1 %both_done, label %loop_and_exit, label %loop_and_body\n";
        ir_ << "loop_and_body:\n";
        ir_ << "  %rem_a = srem i64 %cur_a, 3\n";
        ir_ << "  %div_a = sdiv i64 %cur_a, 3\n";
        ir_ << "  %is_a2 = icmp eq i64 %rem_a, 2\n";
        ir_ << "  %is_am2 = icmp eq i64 %rem_a, -2\n";
        ir_ << "  %adj_div_a1 = select i1 %is_a2, i64 1, i64 0\n";
        ir_ << "  %adj_div_a2 = select i1 %is_am2, i64 -1, i64 %adj_div_a1\n";
        ir_ << "  %next_a = add i64 %div_a, %adj_div_a2\n";
        ir_ << "  %adj_rem_a1 = select i1 %is_a2, i64 -1, i64 %rem_a\n";
        ir_ << "  %trit_a_val = select i1 %is_am2, i64 1, i64 %adj_rem_a1\n";
        ir_ << "  %rem_b = srem i64 %cur_b, 3\n";
        ir_ << "  %div_b = sdiv i64 %cur_b, 3\n";
        ir_ << "  %is_b2 = icmp eq i64 %rem_b, 2\n";
        ir_ << "  %is_bm2 = icmp eq i64 %rem_b, -2\n";
        ir_ << "  %adj_div_b1 = select i1 %is_b2, i64 1, i64 0\n";
        ir_ << "  %adj_div_b2 = select i1 %is_bm2, i64 -1, i64 %adj_div_b1\n";
        ir_ << "  %next_b = add i64 %div_b, %adj_div_b2\n";
        ir_ << "  %adj_rem_b1 = select i1 %is_b2, i64 -1, i64 %rem_b\n";
        ir_ << "  %trit_b_val = select i1 %is_bm2, i64 1, i64 %adj_rem_b1\n";
        ir_ << "  %cmp_ab = icmp slt i64 %trit_a_val, %trit_b_val\n";
        ir_ << "  %min_trit = select i1 %cmp_ab, i64 %trit_a_val, i64 %trit_b_val\n";
        ir_ << "  %term = mul i64 %min_trit, %cur_p3\n";
        ir_ << "  %next_res = add i64 %cur_res, %term\n";
        ir_ << "  %next_p3 = mul i64 %cur_p3, 3\n";
        ir_ << "  br label %loop_and\n";
        ir_ << "loop_and_exit:\n";
        ir_ << "  ret i64 %cur_res\n";
        ir_ << "}\n\n";

        // Helper: Kleene 3-valued OR: max(a, b) for trits, digit-wise max for multi-trit integers
        ir_ << "define i64 @bt_kleene_or(i64 %a, i64 %b) nounwind readnone {\n";
        ir_ << "entry:\n";
        ir_ << "  %is_trit_a_lo = icmp sge i64 %a, -1\n";
        ir_ << "  %is_trit_a_hi = icmp sle i64 %a, 1\n";
        ir_ << "  %is_trit_a = and i1 %is_trit_a_lo, %is_trit_a_hi\n";
        ir_ << "  %is_trit_b_lo = icmp sge i64 %b, -1\n";
        ir_ << "  %is_trit_b_hi = icmp sle i64 %b, 1\n";
        ir_ << "  %is_trit_b = and i1 %is_trit_b_lo, %is_trit_b_hi\n";
        ir_ << "  %both_trits = and i1 %is_trit_a, %is_trit_b\n";
        ir_ << "  br i1 %both_trits, label %trit_or, label %loop_or_init\n";
        ir_ << "trit_or:\n";
        ir_ << "  %c_or = icmp sgt i64 %a, %b\n";
        ir_ << "  %r_trit_or = select i1 %c_or, i64 %a, i64 %b\n";
        ir_ << "  ret i64 %r_trit_or\n";
        ir_ << "loop_or_init:\n";
        ir_ << "  br label %loop_or\n";
        ir_ << "loop_or:\n";
        ir_ << "  %cur_a = phi i64 [ %a, %loop_or_init ], [ %next_a, %loop_or_body ]\n";
        ir_ << "  %cur_b = phi i64 [ %b, %loop_or_init ], [ %next_b, %loop_or_body ]\n";
        ir_ << "  %cur_p3 = phi i64 [ 1, %loop_or_init ], [ %next_p3, %loop_or_body ]\n";
        ir_ << "  %cur_res = phi i64 [ 0, %loop_or_init ], [ %next_res, %loop_or_body ]\n";
        ir_ << "  %done_a = icmp eq i64 %cur_a, 0\n";
        ir_ << "  %done_b = icmp eq i64 %cur_b, 0\n";
        ir_ << "  %both_done = and i1 %done_a, %done_b\n";
        ir_ << "  br i1 %both_done, label %loop_or_exit, label %loop_or_body\n";
        ir_ << "loop_or_body:\n";
        ir_ << "  %rem_a = srem i64 %cur_a, 3\n";
        ir_ << "  %div_a = sdiv i64 %cur_a, 3\n";
        ir_ << "  %is_a2 = icmp eq i64 %rem_a, 2\n";
        ir_ << "  %is_am2 = icmp eq i64 %rem_a, -2\n";
        ir_ << "  %adj_div_a1 = select i1 %is_a2, i64 1, i64 0\n";
        ir_ << "  %adj_div_a2 = select i1 %is_am2, i64 -1, i64 %adj_div_a1\n";
        ir_ << "  %next_a = add i64 %div_a, %adj_div_a2\n";
        ir_ << "  %adj_rem_a1 = select i1 %is_a2, i64 -1, i64 %rem_a\n";
        ir_ << "  %trit_a_val = select i1 %is_am2, i64 1, i64 %adj_rem_a1\n";
        ir_ << "  %rem_b = srem i64 %cur_b, 3\n";
        ir_ << "  %div_b = sdiv i64 %cur_b, 3\n";
        ir_ << "  %is_b2 = icmp eq i64 %rem_b, 2\n";
        ir_ << "  %is_bm2 = icmp eq i64 %rem_b, -2\n";
        ir_ << "  %adj_div_b1 = select i1 %is_b2, i64 1, i64 0\n";
        ir_ << "  %adj_div_b2 = select i1 %is_bm2, i64 -1, i64 %adj_div_b1\n";
        ir_ << "  %next_b = add i64 %div_b, %adj_div_b2\n";
        ir_ << "  %adj_rem_b1 = select i1 %is_b2, i64 -1, i64 %rem_b\n";
        ir_ << "  %trit_b_val = select i1 %is_bm2, i64 1, i64 %adj_rem_b1\n";
        ir_ << "  %cmp_ab = icmp sgt i64 %trit_a_val, %trit_b_val\n";
        ir_ << "  %max_trit = select i1 %cmp_ab, i64 %trit_a_val, i64 %trit_b_val\n";
        ir_ << "  %term = mul i64 %max_trit, %cur_p3\n";
        ir_ << "  %next_res = add i64 %cur_res, %term\n";
        ir_ << "  %next_p3 = mul i64 %cur_p3, 3\n";
        ir_ << "  br label %loop_or\n";
        ir_ << "loop_or_exit:\n";
        ir_ << "  ret i64 %cur_res\n";
        ir_ << "}\n\n";

        // Helper: Ternary XOR: a * b for trits, digit-wise GF(3) product for multi-trit integers
        ir_ << "define i64 @bt_kleene_xor(i64 %a, i64 %b) nounwind readnone {\n";
        ir_ << "entry:\n";
        ir_ << "  %is_trit_a_lo = icmp sge i64 %a, -1\n";
        ir_ << "  %is_trit_a_hi = icmp sle i64 %a, 1\n";
        ir_ << "  %is_trit_a = and i1 %is_trit_a_lo, %is_trit_a_hi\n";
        ir_ << "  %is_trit_b_lo = icmp sge i64 %b, -1\n";
        ir_ << "  %is_trit_b_hi = icmp sle i64 %b, 1\n";
        ir_ << "  %is_trit_b = and i1 %is_trit_b_lo, %is_trit_b_hi\n";
        ir_ << "  %both_trits = and i1 %is_trit_a, %is_trit_b\n";
        ir_ << "  br i1 %both_trits, label %trit_xor, label %loop_xor_init\n";
        ir_ << "trit_xor:\n";
        ir_ << "  %r_trit_xor = mul i64 %a, %b\n";
        ir_ << "  ret i64 %r_trit_xor\n";
        ir_ << "loop_xor_init:\n";
        ir_ << "  br label %loop_xor\n";
        ir_ << "loop_xor:\n";
        ir_ << "  %cur_a = phi i64 [ %a, %loop_xor_init ], [ %next_a, %loop_xor_body ]\n";
        ir_ << "  %cur_b = phi i64 [ %b, %loop_xor_init ], [ %next_b, %loop_xor_body ]\n";
        ir_ << "  %cur_p3 = phi i64 [ 1, %loop_xor_init ], [ %next_p3, %loop_xor_body ]\n";
        ir_ << "  %cur_res = phi i64 [ 0, %loop_xor_init ], [ %next_res, %loop_xor_body ]\n";
        ir_ << "  %done_a = icmp eq i64 %cur_a, 0\n";
        ir_ << "  %done_b = icmp eq i64 %cur_b, 0\n";
        ir_ << "  %both_done = and i1 %done_a, %done_b\n";
        ir_ << "  br i1 %both_done, label %loop_xor_exit, label %loop_xor_body\n";
        ir_ << "loop_xor_body:\n";
        ir_ << "  %rem_a = srem i64 %cur_a, 3\n";
        ir_ << "  %div_a = sdiv i64 %cur_a, 3\n";
        ir_ << "  %is_a2 = icmp eq i64 %rem_a, 2\n";
        ir_ << "  %is_am2 = icmp eq i64 %rem_a, -2\n";
        ir_ << "  %adj_div_a1 = select i1 %is_a2, i64 1, i64 0\n";
        ir_ << "  %adj_div_a2 = select i1 %is_am2, i64 -1, i64 %adj_div_a1\n";
        ir_ << "  %next_a = add i64 %div_a, %adj_div_a2\n";
        ir_ << "  %adj_rem_a1 = select i1 %is_a2, i64 -1, i64 %rem_a\n";
        ir_ << "  %trit_a_val = select i1 %is_am2, i64 1, i64 %adj_rem_a1\n";
        ir_ << "  %rem_b = srem i64 %cur_b, 3\n";
        ir_ << "  %div_b = sdiv i64 %cur_b, 3\n";
        ir_ << "  %is_b2 = icmp eq i64 %rem_b, 2\n";
        ir_ << "  %is_bm2 = icmp eq i64 %rem_b, -2\n";
        ir_ << "  %adj_div_b1 = select i1 %is_b2, i64 1, i64 0\n";
        ir_ << "  %adj_div_b2 = select i1 %is_bm2, i64 -1, i64 %adj_div_b1\n";
        ir_ << "  %next_b = add i64 %div_b, %adj_div_b2\n";
        ir_ << "  %adj_rem_b1 = select i1 %is_b2, i64 -1, i64 %rem_b\n";
        ir_ << "  %trit_b_val = select i1 %is_bm2, i64 1, i64 %adj_rem_b1\n";
        ir_ << "  %xor_trit = mul i64 %trit_a_val, %trit_b_val\n";
        ir_ << "  %term = mul i64 %xor_trit, %cur_p3\n";
        ir_ << "  %next_res = add i64 %cur_res, %term\n";
        ir_ << "  %next_p3 = mul i64 %cur_p3, 3\n";
        ir_ << "  br label %loop_xor\n";
        ir_ << "loop_xor_exit:\n";
        ir_ << "  ret i64 %cur_res\n";
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
        ir_ << "  %r = phi i64 [ 0, %entry ], [ %next_r, %spmv_inner_exit ]\n";
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
        ir_ << "  br i1 %cmp_k, label %spmv_inner_body, label %spmv_inner_exit, !llvm.loop !" << loop_meta_id_ << "\n\n";
        ir_ << "spmv_inner_body:\n";
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
        ir_ << "spmv_inner_exit:\n";
        ir_ << "  %final_acc = phi i64 [ %acc, %nz_loop_cond ]\n";
        ir_ << "  %p_out = getelementptr inbounds i64, i64* %vec_out, i64 %r\n";
        ir_ << "  store i64 %final_acc, i64* %p_out, align 8\n";
        ir_ << "  %next_r = add i64 %r, 1\n";
        ir_ << "  br label %row_loop_cond\n\n";
        ir_ << "spmv_done:\n";
        ir_ << "  ret void\n";
        ir_ << "}\n\n";

        // 3. Generate main entry function containing top-level statements
        is_in_bt_main_ = true;
        current_subprog_id_ = main_subprog_id_;
        if (program.location.line > 0) current_loc_ = program.location;
        if (emit_debug_) {
            fn_body_ << "define i32 @bt_main() !dbg !" << main_subprog_id_ << " {\n";
        } else {
            fn_body_ << "define i32 @bt_main() nounwind {\n";
        }
        fn_body_ << "entry:\n";
        block_terminated_ = false;

        // Visit top-level AST declarations
        for (auto& decl : program.declarations) {
            if (block_terminated_) break;
            if (decl) {
                if (decl->location.line > 0) current_loc_ = decl->location;
                decl->accept(*this);
            }
        }

        if (!block_terminated_) {
            fn_body_ << "  ret i32 0" << dbg_tag() << "\n";
        }
        fn_body_ << "}\n";
        is_in_bt_main_ = false;

        // Combine module header + user functions + generated main function
        ir_ << user_fns_.str();
        ir_ << fn_body_.str();

        if (emit_debug_) {
            ir_ << "\n!llvm.dbg.cu = !{!0}\n";
            ir_ << "!llvm.module.flags = !{!1, !2}\n\n";
            ir_ << "!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: \"bt-compiler\", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)\n";
            ir_ << "!1 = !{i32 7, !\"Dwarf Version\", i32 5}\n";
            ir_ << "!2 = !{i32 2, !\"Debug Info Version\", i32 3}\n";
            ir_ << "!3 = !DIFile(filename: \"" << source_filename_ << "\", directory: \".\")\n";
            ir_ << "!4 = distinct !DISubprogram(name: \"bt_main\", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)\n";
            ir_ << "!5 = !DISubroutineType(types: !{})\n";
            for (const auto& entry : loc_entries_) {
                ir_ << "!" << entry.id << " = !DILocation(line: " << entry.line
                    << ", column: " << entry.col << ", scope: !" << entry.scope << ")\n";
            }
            ir_ << "!" << loop_meta_id_ << " = distinct !{!" << loop_meta_id_ << ", !" << loop_enable_id_ << "}\n";
            ir_ << "!" << loop_enable_id_ << " = !{!\"llvm.loop.vectorize.enable\", i1 true}\n";
        } else {
            ir_ << "\n!" << loop_meta_id_ << " = distinct !{!" << loop_meta_id_ << ", !" << loop_enable_id_ << "}\n";
            ir_ << "!" << loop_enable_id_ << " = !{!\"llvm.loop.vectorize.enable\", i1 true}\n";
        }

        return ir_.str();
    }

    std::string generate_ir(frontend::ProgramNode& program,
                             const std::string& target_triple = "arm64-apple-macosx",
                             const std::string& datalayout = "e-m:o-i64:64-i128:128-n32:64-S128") {
        return generate(program, target_triple, datalayout);
    }

    void visit(frontend::ProgramNode& node) override {
        for (auto& d : node.declarations) {
            if (block_terminated_) break;
            if (d) d->accept(*this);
        }
    }

    void visit(frontend::BlockNode& node) override {
        enter_scope();
        for (auto& s : node.statements) {
            if (block_terminated_) break;
            if (s) s->accept(*this);
        }
        exit_scope();
    }

    void visit(frontend::FunctionDefNode& node) override {
        std::string saved_body = fn_body_.str();
        fn_body_.str("");
        fn_body_.clear();
        bool saved_in_main = is_in_bt_main_;
        bool saved_terminated = block_terminated_;
        is_in_bt_main_ = false;
        block_terminated_ = false;

        std::string ret_llvm_type = "i64";
        if (node.return_type == "float") ret_llvm_type = "double";
        else if (node.return_type == "void") ret_llvm_type = "void";
        user_func_ret_types_[node.name] = ret_llvm_type;

        fn_body_ << "define " << ret_llvm_type << " @" << node.name << "(";
        for (size_t i = 0; i < node.params.size(); ++i) {
            std::string p_type = (node.params[i].type == "float") ? "double" : "i64";
            fn_body_ << p_type << " %arg_" << node.params[i].name;
            if (i + 1 < node.params.size()) fn_body_ << ", ";
        }
        fn_body_ << ") nounwind {\nentry:\n";

        enter_scope();
        for (const auto& p : node.params) {
            std::string p_type = (p.type == "float") ? "double" : "i64";
            std::string ptr_reg = "%var_" + p.name + "_" + std::to_string(++alloca_counter_);
            fn_body_ << "  " << ptr_reg << " = alloca " << p_type << ", align 8\n";
            fn_body_ << "  store " << p_type << " %arg_" << p.name << ", " << p_type << "* " << ptr_reg << ", align 8\n";
            scope_stack_.back()[p.name] = VarInfo{ptr_reg, p_type, 0};
        }

        if (node.body) {
            for (auto& s : node.body->statements) {
                if (block_terminated_) break;
                if (s) s->accept(*this);
            }
        }

        if (!block_terminated_) {
            if (ret_llvm_type == "void") fn_body_ << "  ret void\n";
            else if (ret_llvm_type == "double") fn_body_ << "  ret double 0.0\n";
            else fn_body_ << "  ret i64 0\n";
        }
        fn_body_ << "}\n\n";
        exit_scope();

        user_fns_ << fn_body_.str();
        fn_body_.str(saved_body);
        fn_body_.seekp(0, std::ios_base::end);
        is_in_bt_main_ = saved_in_main;
        block_terminated_ = saved_terminated;
    }

    void visit(frontend::VarDeclNode& node) override {
        if (block_terminated_) return;
        if (node.location.line > 0) current_loc_ = node.location;
        if (scope_stack_.empty()) enter_scope();

        std::string ptr_reg = "%var_" + node.name + "_" + std::to_string(++alloca_counter_);

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
        if (node.location.line > 0) current_loc_ = node.location;

        VarInfo info;
        info.type = v_type;
        info.dim = current_dim_;
        if (v_type == "sparse_matrix") {
            // No alloca is emitted for sparse_matrix values (they live in the
            // global T-CSR constants referenced through sparse_matrices_); the
            // variable resolves by name, so point the descriptor at the name
            // instead of an undefined SSA register.
            info.ptr_reg = node.name;
        } else {
            info.ptr_reg = ptr_reg;
        }

        if (v_type == "double") {
            fn_body_ << "  " << ptr_reg << " = alloca double, align 8" << dbg_tag() << "\n";
            if (node.init_expr) {
                if (current_type_ == "i64") {
                    std::string cast = new_reg();
                    fn_body_ << "  " << cast << " = sitofp i64 " << current_val_ << " to double" << dbg_tag() << "\n";
                    current_val_ = cast;
                }
                fn_body_ << "  store double " << current_val_ << ", double* " << ptr_reg << ", align 8" << dbg_tag() << "\n";
            } else {
                fn_body_ << "  store double 0.0, double* " << ptr_reg << ", align 8" << dbg_tag() << "\n";
            }
        } else if (v_type == "sparse_matrix") {
            if (current_type_ == "sparse_matrix") {
                sparse_matrices_[node.name] = sparse_matrices_[current_val_];
            } else if (node.init_expr) {
                // Only a sparse_matrix constructor may initialize a sparse_matrix
                // variable; scalar/vector initializers are a codegen error.
                throw std::runtime_error("[Codegen Error] " + node.location.to_string() +
                                         ": cannot initialize sparse_matrix variable '" + node.name +
                                         "' from a '" + current_type_ + "' value");
            }
        } else if (v_type == "i64*") {
            info.ptr_reg = current_val_;
            info.dim = current_dim_;
        } else {
            fn_body_ << "  " << ptr_reg << " = alloca i64, align 8" << dbg_tag() << "\n";
            if (node.init_expr) {
                if (current_type_ == "double") {
                    std::string cast = new_reg();
                    fn_body_ << "  " << cast << " = fptosi double " << current_val_ << " to i64" << dbg_tag() << "\n";
                    current_val_ = cast;
                }
                fn_body_ << "  store i64 " << current_val_ << ", i64* " << ptr_reg << ", align 8" << dbg_tag() << "\n";
            } else {
                fn_body_ << "  store i64 0, i64* " << ptr_reg << ", align 8" << dbg_tag() << "\n";
            }
        }

        scope_stack_.back()[node.name] = info;
    }

    void visit(frontend::AssignNode& node) override {
        if (block_terminated_) return;
        if (node.location.line > 0) current_loc_ = node.location;
        if (node.value) {
            node.value->accept(*this);
            if (node.location.line > 0) current_loc_ = node.location;
            VarInfo* var = lookup_var_mut(node.target);
            if (var) {
                if (var->type == "sparse_matrix") {
                    // sparse_matrix assignment routes through the T-CSR map
                    // (reference copy), like the declaration does. There is no
                    // alloca behind a sparse_matrix variable, so storing an
                    // arbitrary i64 through its name would be invalid IR.
                    if (current_type_ == "sparse_matrix") {
                        sparse_matrices_[node.target] = sparse_matrices_[current_val_];
                    } else {
                        throw std::runtime_error("[Codegen Error] " + node.location.to_string() +
                                                 ": cannot assign a '" + current_type_ +
                                                 "' value to sparse_matrix variable '" + node.target +
                                                 "' (only another sparse_matrix may be assigned)");
                    }
                } else if (var->type == "double") {
                    if (current_type_ == "i64") {
                        std::string cast = new_reg();
                        fn_body_ << "  " << cast << " = sitofp i64 " << current_val_ << " to double" << dbg_tag() << "\n";
                        current_val_ = cast;
                    }
                    fn_body_ << "  store double " << current_val_ << ", double* " << var->ptr_reg << ", align 8" << dbg_tag() << "\n";
                } else if (var->type == "i64*") {
                    var->ptr_reg = current_val_;
                    var->dim = current_dim_;
                } else {
                    if (current_type_ == "double") {
                        std::string cast = new_reg();
                        fn_body_ << "  " << cast << " = fptosi double " << current_val_ << " to i64" << dbg_tag() << "\n";
                        current_val_ = cast;
                    }
                    fn_body_ << "  store i64 " << current_val_ << ", i64* " << var->ptr_reg << ", align 8" << dbg_tag() << "\n";
                }
            } else {
                std::string ptr_reg = "%var_" + node.target + "_" + std::to_string(++alloca_counter_);
                VarInfo info;
                info.ptr_reg = ptr_reg;
                info.type = (current_type_ == "double") ? "double" : (current_type_ == "i64*" ? "i64*" : "i64");
                info.dim = current_dim_;
                if (info.type == "double") {
                    fn_body_ << "  " << ptr_reg << " = alloca double, align 8" << dbg_tag() << "\n";
                    fn_body_ << "  store double " << current_val_ << ", double* " << ptr_reg << ", align 8" << dbg_tag() << "\n";
                } else if (info.type == "i64*") {
                    info.ptr_reg = current_val_;
                } else {
                    fn_body_ << "  " << ptr_reg << " = alloca i64, align 8" << dbg_tag() << "\n";
                    fn_body_ << "  store i64 " << current_val_ << ", i64* " << ptr_reg << ", align 8" << dbg_tag() << "\n";
                }
                if (scope_stack_.empty()) enter_scope();
                scope_stack_.back()[node.target] = info;
            }
        }
    }

    void visit(frontend::BinaryOpNode& node) override {
        if (node.location.line > 0) current_loc_ = node.location;
        node.left->accept(*this);
        std::string left_val = current_val_;
        std::string left_type = current_type_;

        node.right->accept(*this);
        std::string right_val = current_val_;
        std::string right_type = current_type_;
        if (node.location.line > 0) current_loc_ = node.location;

        // 1. SpMV: sparse_matrix * vector
        if (node.op == frontend::BinaryOp::Mul && left_type == "sparse_matrix") {
            auto it = sparse_matrices_.find(left_val);
            if (it != sparse_matrices_.end()) {
                const auto& info = it->second;
                std::string out_vec = new_reg();
                fn_body_ << "  " << out_vec << " = alloca [" << info.rows << " x i64], align 8" << dbg_tag() << "\n";
                std::string p_out = new_reg();
                fn_body_ << "  " << p_out << " = getelementptr inbounds [" << info.rows
                         << " x i64], [" << info.rows << " x i64]* " << out_vec << ", i64 0, i64 0" << dbg_tag() << "\n";

                std::string p_r = new_reg();
                fn_body_ << "  " << p_r << " = getelementptr inbounds [" << (info.rows + 1)
                         << " x i32], [" << (info.rows + 1) << " x i32]* " << info.g_row_ptr << ", i64 0, i64 0" << dbg_tag() << "\n";

                std::string p_c = new_reg();
                size_t c_sz = std::max<size_t>(info.nnz, 1);
                fn_body_ << "  " << p_c << " = getelementptr inbounds [" << c_sz
                         << " x i32], [" << c_sz << " x i32]* " << info.g_col_ind << ", i64 0, i64 0" << dbg_tag() << "\n";

                std::string p_s = new_reg();
                size_t s_sz = std::max<size_t>((info.nnz + 63) / 64, 1);
                fn_body_ << "  " << p_s << " = getelementptr inbounds [" << s_sz
                         << " x i64], [" << s_sz << " x i64]* " << info.g_signs << ", i64 0, i64 0" << dbg_tag() << "\n";

                fn_body_ << "  call void @bt_spmv(i64 " << info.rows << ", i32* " << p_r
                         << ", i32* " << p_c << ", i64* " << p_s << ", i64* " << right_val
                         << ", i64* " << p_out << ")" << dbg_tag() << "\n";

                current_val_ = p_out;
                current_type_ = "i64*";
                current_dim_ = info.rows;
                return;
            }
        }

        // 2. Float arithmetic and comparisons: promote mixed operands
        if (left_type == "double" || right_type == "double") {
            if (left_type == "i64") {
                std::string cast = new_reg();
                fn_body_ << "  " << cast << " = sitofp i64 " << left_val << " to double" << dbg_tag() << "\n";
                left_val = cast;
                left_type = "double";
            }
            if (right_type == "i64") {
                std::string cast = new_reg();
                fn_body_ << "  " << cast << " = sitofp i64 " << right_val << " to double" << dbg_tag() << "\n";
                right_val = cast;
                right_type = "double";
            }

            switch (node.op) {
                case frontend::BinaryOp::Add: {
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = fadd double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "double";
                    return;
                }
                case frontend::BinaryOp::Sub: {
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = fsub double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "double";
                    return;
                }
                case frontend::BinaryOp::Mul: {
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = fmul double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "double";
                    return;
                }
                case frontend::BinaryOp::Equal: {
                    std::string cmp = new_reg();
                    fn_body_ << "  " << cmp << " = fcmp oeq double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "i64";
                    return;
                }
                case frontend::BinaryOp::NotEqual: {
                    std::string cmp = new_reg();
                    fn_body_ << "  " << cmp << " = fcmp one double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "i64";
                    return;
                }
                case frontend::BinaryOp::LessThan: {
                    std::string cmp = new_reg();
                    fn_body_ << "  " << cmp << " = fcmp olt double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "i64";
                    return;
                }
                case frontend::BinaryOp::GreaterThan: {
                    std::string cmp = new_reg();
                    fn_body_ << "  " << cmp << " = fcmp ogt double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "i64";
                    return;
                }
                case frontend::BinaryOp::LessEqual: {
                    std::string cmp = new_reg();
                    fn_body_ << "  " << cmp << " = fcmp ole double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "i64";
                    return;
                }
                case frontend::BinaryOp::GreaterEqual: {
                    std::string cmp = new_reg();
                    fn_body_ << "  " << cmp << " = fcmp oge double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "i64";
                    return;
                }
                case frontend::BinaryOp::Spaceship: {
                    std::string cmp_gt = new_reg();
                    fn_body_ << "  " << cmp_gt << " = fcmp ogt double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string cmp_lt = new_reg();
                    fn_body_ << "  " << cmp_lt << " = fcmp olt double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    std::string sel1 = new_reg();
                    fn_body_ << "  " << sel1 << " = select i1 " << cmp_gt << ", i64 1, i64 0" << dbg_tag() << "\n";
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = select i1 " << cmp_lt << ", i64 -1, i64 " << sel1 << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "i64";
                    return;
                }
                default: {
                    std::string res = new_reg();
                    fn_body_ << "  " << res << " = fadd double " << left_val << ", " << right_val << dbg_tag() << "\n";
                    current_val_ = res;
                    current_type_ = "double";
                    return;
                }
            }
        }

        // 3. Integer & Trit arithmetic
        std::string res = new_reg();
        switch (node.op) {
            case frontend::BinaryOp::Add:
                fn_body_ << "  " << res << " = add i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                break;
            case frontend::BinaryOp::Sub:
                fn_body_ << "  " << res << " = sub i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                break;
            case frontend::BinaryOp::Mul:
                fn_body_ << "  " << res << " = mul i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                break;
            case frontend::BinaryOp::KleeneAnd:
                fn_body_ << "  " << res << " = call i64 @bt_kleene_and(i64 " << left_val << ", i64 " << right_val << ")" << dbg_tag() << "\n";
                break;
            case frontend::BinaryOp::KleeneOr:
                fn_body_ << "  " << res << " = call i64 @bt_kleene_or(i64 " << left_val << ", i64 " << right_val << ")" << dbg_tag() << "\n";
                break;
            case frontend::BinaryOp::TernaryXor:
                fn_body_ << "  " << res << " = call i64 @bt_kleene_xor(i64 " << left_val << ", i64 " << right_val << ")" << dbg_tag() << "\n";
                break;
            case frontend::BinaryOp::Spaceship:
                fn_body_ << "  " << res << " = call i64 @bt_spaceship(i64 " << left_val << ", i64 " << right_val << ")" << dbg_tag() << "\n";
                break;
            case frontend::BinaryOp::Equal: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp eq i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                break;
            }
            case frontend::BinaryOp::NotEqual: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp ne i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                break;
            }
            case frontend::BinaryOp::LessThan: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp slt i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                break;
            }
            case frontend::BinaryOp::GreaterThan: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp sgt i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                break;
            }
            case frontend::BinaryOp::LessEqual: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp sle i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                break;
            }
            case frontend::BinaryOp::GreaterEqual: {
                std::string cmp = new_reg();
                fn_body_ << "  " << cmp << " = icmp sge i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                fn_body_ << "  " << res << " = select i1 " << cmp << ", i64 1, i64 -1" << dbg_tag() << "\n";
                break;
            }
            default:
                fn_body_ << "  " << res << " = add i64 " << left_val << ", " << right_val << dbg_tag() << "\n";
                break;
        }
        current_val_ = res;
        current_type_ = "i64";
    }

    void visit(frontend::UnaryOpNode& node) override {
        if (node.location.line > 0) current_loc_ = node.location;
        node.operand->accept(*this);
        if (node.location.line > 0) current_loc_ = node.location;
        std::string res = new_reg();
        if (current_type_ == "double") {
            fn_body_ << "  " << res << " = fsub double -0.0, " << current_val_ << dbg_tag() << "\n";
        } else {
            fn_body_ << "  " << res << " = sub i64 0, " << current_val_ << dbg_tag() << "\n";
        }
        current_val_ = res;
    }

    void visit(frontend::SelectNode& node) override {
        if (block_terminated_) return;
        if (node.location.line > 0) current_loc_ = node.location;
        node.condition->accept(*this);
        std::string cond_val = current_val_;

        bool is_pure_expr = (!dynamic_cast<frontend::BlockNode*>(node.pos_branch.get()) &&
                             !dynamic_cast<frontend::BlockNode*>(node.zero_branch.get()) &&
                             !dynamic_cast<frontend::BlockNode*>(node.neg_branch.get()));

        if (is_pure_expr) {
            bool any_terminated = false;
            std::string p_val = "0";
            std::string z_val = "0";
            std::string n_val = "0";

            node.pos_branch->accept(*this);
            p_val = current_val_;
            any_terminated = block_terminated_;

            if (!any_terminated) {
                node.zero_branch->accept(*this);
                z_val = current_val_;
                any_terminated = block_terminated_;
            }

            if (!any_terminated) {
                node.neg_branch->accept(*this);
                n_val = current_val_;
                any_terminated = block_terminated_;
            }

            if (any_terminated) {
                current_val_ = "0";
                current_type_ = "i64";
                return;
            }

            if (node.location.line > 0) current_loc_ = node.location;
            std::string res = new_reg();
            fn_body_ << "  " << res << " = call i64 @bt_select(i64 " << cond_val << ", i64 "
                     << p_val << ", i64 " << z_val << ", i64 " << n_val << ")" << dbg_tag() << "\n";
            current_val_ = res;
            current_type_ = "i64";
            return;
        }

        // Block statement form: branch to pos, zero, neg basic blocks
        std::string is_pos_reg = new_reg();
        std::string is_neg_reg = new_reg();
        fn_body_ << "  " << is_pos_reg << " = icmp sgt i64 " << cond_val << ", 0" << dbg_tag() << "\n";
        fn_body_ << "  " << is_neg_reg << " = icmp slt i64 " << cond_val << ", 0" << dbg_tag() << "\n";

        std::string pos_lbl = new_label("sel_pos");
        std::string not_pos_lbl = new_label("sel_not_pos");
        std::string zero_lbl = new_label("sel_zero");
        std::string neg_lbl = new_label("sel_neg");
        std::string merge_lbl = new_label("sel_merge");

        fn_body_ << "  br i1 " << is_pos_reg << ", label %" << pos_lbl << ", label %" << not_pos_lbl << dbg_tag() << "\n";

        // Pos Branch
        fn_body_ << pos_lbl << ":\n";
        block_terminated_ = false;
        node.pos_branch->accept(*this);
        bool pos_term = block_terminated_;
        if (!pos_term) {
            fn_body_ << "  br label %" << merge_lbl << dbg_tag() << "\n";
        }

        // Not Pos Branch -> branch to neg or zero
        fn_body_ << not_pos_lbl << ":\n";
        fn_body_ << "  br i1 " << is_neg_reg << ", label %" << neg_lbl << ", label %" << zero_lbl << dbg_tag() << "\n";

        // Zero Branch
        fn_body_ << zero_lbl << ":\n";
        block_terminated_ = false;
        node.zero_branch->accept(*this);
        bool zero_term = block_terminated_;
        if (!zero_term) {
            fn_body_ << "  br label %" << merge_lbl << dbg_tag() << "\n";
        }

        // Neg Branch
        fn_body_ << neg_lbl << ":\n";
        block_terminated_ = false;
        node.neg_branch->accept(*this);
        bool neg_term = block_terminated_;
        if (!neg_term) {
            fn_body_ << "  br label %" << merge_lbl << dbg_tag() << "\n";
        }

        if (pos_term && zero_term && neg_term) {
            block_terminated_ = true;
            return;
        }

        fn_body_ << merge_lbl << ":\n";
        block_terminated_ = false;
        current_val_ = "0";
        current_type_ = "i64";
    }

    void visit(frontend::WhileNode& node) override {
        if (block_terminated_) return;
        if (node.location.line > 0) current_loc_ = node.location;

        std::string cond_lbl = new_label("while_cond");
        std::string body_lbl = new_label("while_body");
        std::string exit_lbl = new_label("while_exit");

        fn_body_ << "  br label %" << cond_lbl << dbg_tag() << "\n";
        fn_body_ << cond_lbl << ":\n";
        block_terminated_ = false;

        node.condition->accept(*this);
        std::string cond_val = current_val_;
        if (node.location.line > 0) current_loc_ = node.location;
        if (current_type_ == "double") {
            std::string fcmp = new_reg();
            fn_body_ << "  " << fcmp << " = fcmp ogt double " << cond_val << ", 0.0" << dbg_tag() << "\n";
            fn_body_ << "  br i1 " << fcmp << ", label %" << body_lbl << ", label %" << exit_lbl << dbg_tag() << "\n";
        } else {
            std::string cmp = new_reg();
            fn_body_ << "  " << cmp << " = icmp sgt i64 " << cond_val << ", 0" << dbg_tag() << "\n";
            fn_body_ << "  br i1 " << cmp << ", label %" << body_lbl << ", label %" << exit_lbl << dbg_tag() << "\n";
        }

        fn_body_ << body_lbl << ":\n";
        block_terminated_ = false;
        if (node.body) node.body->accept(*this);
        if (!block_terminated_) {
            if (node.location.line > 0) current_loc_ = node.location;
            fn_body_ << "  br label %" << cond_lbl << dbg_tag() << "\n";
        }

        fn_body_ << exit_lbl << ":\n";
        block_terminated_ = false;
    }

    void visit(frontend::ReturnNode& node) override {
        if (block_terminated_) return;
        if (node.location.line > 0) current_loc_ = node.location;
        if (is_in_bt_main_) {
            if (node.expr) {
                node.expr->accept(*this);
                if (node.location.line > 0) current_loc_ = node.location;
                if (current_type_ == "double") {
                    std::string conv = new_reg();
                    fn_body_ << "  " << conv << " = fptosi double " << current_val_ << " to i32" << dbg_tag() << "\n";
                    fn_body_ << "  ret i32 " << conv << dbg_tag() << "\n";
                } else {
                    std::string ret_i32 = new_reg();
                    fn_body_ << "  " << ret_i32 << " = trunc i64 " << current_val_ << " to i32" << dbg_tag() << "\n";
                    fn_body_ << "  ret i32 " << ret_i32 << dbg_tag() << "\n";
                }
            } else {
                fn_body_ << "  ret i32 0" << dbg_tag() << "\n";
            }
        } else {
            if (node.expr) {
                node.expr->accept(*this);
                if (node.location.line > 0) current_loc_ = node.location;
                if (current_type_ == "double") {
                    fn_body_ << "  ret double " << current_val_ << dbg_tag() << "\n";
                } else {
                    fn_body_ << "  ret i64 " << current_val_ << dbg_tag() << "\n";
                }
            } else {
                fn_body_ << "  ret void" << dbg_tag() << "\n";
            }
        }
        block_terminated_ = true;
    }

    void visit(frontend::LiteralNode& node) override {
        current_val_ = std::to_string(node.int_val);
        current_type_ = "i64";
    }

    void visit(frontend::FloatLiteralExprAST& node) override {
        std::ostringstream ss;
        // 17 significant decimal digits guarantee an exact IEEE-754 double
        // round-trip when the LLVM IR parser re-reads the literal (precision 10
        // can round DBL_MAX-sized literals past the maximum and clamp to inf).
        ss << std::scientific << std::setprecision(17) << node.value;
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
            VarInfo vinfo;
            vinfo.ptr_reg = node.name;
            vinfo.type = "sparse_matrix";
            if (scope_stack_.empty()) enter_scope();
            scope_stack_.back()[node.name] = vinfo;
        }
        current_val_ = s_id;
        current_type_ = "sparse_matrix";
    }

    void visit(frontend::IdentifierNode& node) override {
        if (node.location.line > 0) current_loc_ = node.location;
        const VarInfo* var = lookup_var(node.name);
        if (var) {
            if (var->type == "sparse_matrix") {
                current_val_ = node.name;
                current_type_ = "sparse_matrix";
                return;
            }
            if (var->type == "i64*") {
                current_val_ = var->ptr_reg;
                current_type_ = "i64*";
                current_dim_ = var->dim;
                return;
            }
            std::string res = new_reg();
            if (var->type == "double") {
                fn_body_ << "  " << res << " = load double, double* " << var->ptr_reg << ", align 8" << dbg_tag() << "\n";
                current_type_ = "double";
            } else {
                fn_body_ << "  " << res << " = load i64, i64* " << var->ptr_reg << ", align 8" << dbg_tag() << "\n";
                current_type_ = "i64";
            }
            current_val_ = res;
            return;
        }

        auto s_it = sparse_matrices_.find(node.name);
        if (s_it != sparse_matrices_.end()) {
            current_val_ = node.name;
            current_type_ = "sparse_matrix";
            return;
        }

        current_val_ = "0";
        current_type_ = "i64";
    }

    void visit(frontend::CallNode& node) override {
        if (node.location.line > 0) current_loc_ = node.location;
        if (node.callee == "print") {
            if (!node.arguments.empty()) {
                node.arguments[0]->accept(*this);
                if (node.location.line > 0) current_loc_ = node.location;
                if (current_type_ == "double") {
                    fn_body_ << "  call void @bt_print_float(double " << current_val_ << ")" << dbg_tag() << "\n";
                } else if (current_type_ == "i64*") {
                    for (size_t i = 0; i < current_dim_; ++i) {
                        std::string p_elem = new_reg();
                        fn_body_ << "  " << p_elem << " = getelementptr inbounds i64, i64* "
                                 << current_val_ << ", i64 " << i << dbg_tag() << "\n";
                        std::string val = new_reg();
                        fn_body_ << "  " << val << " = load i64, i64* " << p_elem << ", align 8" << dbg_tag() << "\n";
                        fn_body_ << "  call void @bt_print(i64 " << val << ")" << dbg_tag() << "\n";
                    }
                } else {
                    fn_body_ << "  call void @bt_print(i64 " << current_val_ << ")" << dbg_tag() << "\n";
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
            fn_body_ << "  " << arr << " = alloca [" << count << " x i64], align 8" << dbg_tag() << "\n";
            for (size_t i = 0; i < count; ++i) {
                node.arguments[start_idx + i]->accept(*this);
                if (node.location.line > 0) current_loc_ = node.location;
                std::string p_i = new_reg();
                fn_body_ << "  " << p_i << " = getelementptr inbounds [" << count << " x i64], ["
                         << count << " x i64]* " << arr << ", i64 0, i64 " << i << dbg_tag() << "\n";
                fn_body_ << "  store i64 " << current_val_ << ", i64* " << p_i << ", align 8" << dbg_tag() << "\n";
            }
            std::string p_ret = new_reg();
            fn_body_ << "  " << p_ret << " = getelementptr inbounds [" << count << " x i64], ["
                     << count << " x i64]* " << arr << ", i64 0, i64 0" << dbg_tag() << "\n";
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
            if (node.location.line > 0) current_loc_ = node.location;

            std::string res = new_reg();
            fn_body_ << "  " << res << " = call i64 @bt_select(i64 " << c_val << ", i64 "
                     << p_val << ", i64 " << z_val << ", i64 " << n_val << ")" << dbg_tag() << "\n";
            current_val_ = res;
            current_type_ = "i64";
        } else if ((node.callee == "planar_dot64" || node.callee == "btha_sum_p" ||
                    node.callee == "btha_carry_p" || node.callee == "btha_sum_n" ||
                    node.callee == "btha_carry_n") && node.arguments.size() == 4) {
            node.arguments[0]->accept(*this);
            std::string ap = current_val_;
            node.arguments[1]->accept(*this);
            std::string an = current_val_;
            node.arguments[2]->accept(*this);
            std::string bp = current_val_;
            node.arguments[3]->accept(*this);
            std::string bn = current_val_;
            if (node.location.line > 0) current_loc_ = node.location;

            if (node.callee == "planar_dot64") {
                std::string pp1 = new_reg();
                fn_body_ << "  " << pp1 << " = and i64 " << ap << ", " << bp << dbg_tag() << "\n";
                std::string pp2 = new_reg();
                fn_body_ << "  " << pp2 << " = and i64 " << an << ", " << bn << dbg_tag() << "\n";
                std::string p_prod = new_reg();
                fn_body_ << "  " << p_prod << " = or i64 " << pp1 << ", " << pp2 << dbg_tag() << "\n";

                std::string np1 = new_reg();
                fn_body_ << "  " << np1 << " = and i64 " << ap << ", " << bn << dbg_tag() << "\n";
                std::string np2 = new_reg();
                fn_body_ << "  " << np2 << " = and i64 " << an << ", " << bp << dbg_tag() << "\n";
                std::string n_prod = new_reg();
                fn_body_ << "  " << n_prod << " = or i64 " << np1 << ", " << np2 << dbg_tag() << "\n";

                std::string pop_p = new_reg();
                fn_body_ << "  " << pop_p << " = call i64 @llvm.ctpop.i64(i64 " << p_prod << ")" << dbg_tag() << "\n";
                std::string pop_n = new_reg();
                fn_body_ << "  " << pop_n << " = call i64 @llvm.ctpop.i64(i64 " << n_prod << ")" << dbg_tag() << "\n";
                std::string res = new_reg();
                fn_body_ << "  " << res << " = sub i64 " << pop_p << ", " << pop_n << dbg_tag() << "\n";
                current_val_ = res;
                current_type_ = "i64";
            } else if (node.callee == "btha_sum_p") {
                // PS = ((ap ^ bp) & ~(an | bn)) | (an & bn)
                std::string xor_p = new_reg();
                fn_body_ << "  " << xor_p << " = xor i64 " << ap << ", " << bp << dbg_tag() << "\n";
                std::string or_n = new_reg();
                fn_body_ << "  " << or_n << " = or i64 " << an << ", " << bn << dbg_tag() << "\n";
                std::string not_or_n = new_reg();
                fn_body_ << "  " << not_or_n << " = xor i64 " << or_n << ", -1" << dbg_tag() << "\n";
                std::string half_ps = new_reg();
                fn_body_ << "  " << half_ps << " = and i64 " << xor_p << ", " << not_or_n << dbg_tag() << "\n";
                std::string nc = new_reg();
                fn_body_ << "  " << nc << " = and i64 " << an << ", " << bn << dbg_tag() << "\n";
                std::string ps = new_reg();
                fn_body_ << "  " << ps << " = or i64 " << half_ps << ", " << nc << dbg_tag() << "\n";
                current_val_ = ps;
                current_type_ = "i64";
            } else if (node.callee == "btha_carry_p") {
                // PC = ap & bp
                std::string pc = new_reg();
                fn_body_ << "  " << pc << " = and i64 " << ap << ", " << bp << dbg_tag() << "\n";
                current_val_ = pc;
                current_type_ = "i64";
            } else if (node.callee == "btha_sum_n") {
                // NS = ((an ^ bn) & ~(ap | bp)) | (ap & bp)
                std::string xor_n = new_reg();
                fn_body_ << "  " << xor_n << " = xor i64 " << an << ", " << bn << dbg_tag() << "\n";
                std::string or_p = new_reg();
                fn_body_ << "  " << or_p << " = or i64 " << ap << ", " << bp << dbg_tag() << "\n";
                std::string not_or_p = new_reg();
                fn_body_ << "  " << not_or_p << " = xor i64 " << or_p << ", -1" << dbg_tag() << "\n";
                std::string half_ns = new_reg();
                fn_body_ << "  " << half_ns << " = and i64 " << xor_n << ", " << not_or_p << dbg_tag() << "\n";
                std::string pc = new_reg();
                fn_body_ << "  " << pc << " = and i64 " << ap << ", " << bp << dbg_tag() << "\n";
                std::string ns = new_reg();
                fn_body_ << "  " << ns << " = or i64 " << half_ns << ", " << pc << dbg_tag() << "\n";
                current_val_ = ns;
                current_type_ = "i64";
            } else if (node.callee == "btha_carry_n") {
                // NC = an & bn
                std::string nc = new_reg();
                fn_body_ << "  " << nc << " = and i64 " << an << ", " << bn << dbg_tag() << "\n";
                current_val_ = nc;
                current_type_ = "i64";
            }
        } else if (node.callee == "fma" && node.arguments.size() == 3) {
            node.arguments[0]->accept(*this);
            std::string a_val = current_val_;
            std::string a_type = current_type_;
            node.arguments[1]->accept(*this);
            std::string b_val = current_val_;
            std::string b_type = current_type_;
            node.arguments[2]->accept(*this);
            std::string c_val = current_val_;
            std::string c_type = current_type_;
            if (node.location.line > 0) current_loc_ = node.location;

            auto to_double_val = [&](const std::string& val, const std::string& type) -> std::string {
                if (type == "double") return val;
                std::string r = new_reg();
                fn_body_ << "  " << r << " = sitofp i64 " << val << " to double" << dbg_tag() << "\n";
                return r;
            };
            std::string ad = to_double_val(a_val, a_type);
            std::string bd = to_double_val(b_val, b_type);
            std::string cd = to_double_val(c_val, c_type);

            std::string res = new_reg();
            fn_body_ << "  " << res << " = call double @llvm.fma.f64(double " << ad
                     << ", double " << bd << ", double " << cd << ")" << dbg_tag() << "\n";
            current_val_ = res;
            current_type_ = "double";
        } else if (node.callee == "sqrt" && node.arguments.size() == 1) {
            node.arguments[0]->accept(*this);
            std::string val = current_val_;
            std::string type = current_type_;
            if (node.location.line > 0) current_loc_ = node.location;

            if (type != "double") {
                std::string conv = new_reg();
                fn_body_ << "  " << conv << " = sitofp i64 " << val << " to double" << dbg_tag() << "\n";
                val = conv;
            }
            std::string res = new_reg();
            fn_body_ << "  " << res << " = call double @llvm.sqrt.f64(double " << val << ")" << dbg_tag() << "\n";
            current_val_ = res;
            current_type_ = "double";
        } else if (node.callee == "abs" && node.arguments.size() == 1) {
            node.arguments[0]->accept(*this);
            std::string val = current_val_;
            std::string type = current_type_;
            if (node.location.line > 0) current_loc_ = node.location;

            if (type == "double") {
                std::string res = new_reg();
                fn_body_ << "  " << res << " = call double @llvm.fabs.f64(double " << val << ")" << dbg_tag() << "\n";
                current_val_ = res;
                current_type_ = "double";
            } else {
                std::string res = new_reg();
                fn_body_ << "  " << res << " = call i64 @llvm.abs.i64(i64 " << val << ", i1 false)" << dbg_tag() << "\n";
                current_val_ = res;
                current_type_ = "i64";
            }
        } else if (user_func_ret_types_.find(node.callee) != user_func_ret_types_.end()) {
            std::string ret_type = user_func_ret_types_[node.callee];
            std::vector<std::pair<std::string, std::string>> arg_vals;
            for (auto& arg : node.arguments) {
                arg->accept(*this);
                arg_vals.emplace_back(current_type_ == "double" ? "double" : "i64", current_val_);
            }
            if (node.location.line > 0) current_loc_ = node.location;
            if (ret_type == "void") {
                fn_body_ << "  call void @" << node.callee << "(";
                for (size_t i = 0; i < arg_vals.size(); ++i) {
                    fn_body_ << arg_vals[i].first << " " << arg_vals[i].second;
                    if (i + 1 < arg_vals.size()) fn_body_ << ", ";
                }
                fn_body_ << ")" << dbg_tag() << "\n";
                current_val_ = "0";
                current_type_ = "i64";
            } else {
                std::string res = new_reg();
                fn_body_ << "  " << res << " = call " << ret_type << " @" << node.callee << "(";
                for (size_t i = 0; i < arg_vals.size(); ++i) {
                    fn_body_ << arg_vals[i].first << " " << arg_vals[i].second;
                    if (i + 1 < arg_vals.size()) fn_body_ << ", ";
                }
                fn_body_ << ")" << dbg_tag() << "\n";
                current_val_ = res;
                current_type_ = ret_type;
            }
        } else {
            current_val_ = "0";
            current_type_ = "i64";
        }
    }

    void visit(frontend::IfNode& node) override {
        if (block_terminated_) return;
        if (node.location.line > 0) current_loc_ = node.location;

        std::string then_lbl = new_label("if_then");
        std::string else_lbl = new_label("if_else");
        std::string exit_lbl = new_label("if_exit");

        node.condition->accept(*this);
        std::string cond_val = current_val_;
        if (node.location.line > 0) current_loc_ = node.location;
        if (current_type_ == "double") {
            std::string fcmp = new_reg();
            fn_body_ << "  " << fcmp << " = fcmp ogt double " << cond_val << ", 0.0" << dbg_tag() << "\n";
            fn_body_ << "  br i1 " << fcmp << ", label %" << then_lbl << ", label %"
                     << (node.else_block ? else_lbl : exit_lbl) << dbg_tag() << "\n";
        } else {
            std::string cmp = new_reg();
            fn_body_ << "  " << cmp << " = icmp sgt i64 " << cond_val << ", 0" << dbg_tag() << "\n";
            fn_body_ << "  br i1 " << cmp << ", label %" << then_lbl << ", label %"
                     << (node.else_block ? else_lbl : exit_lbl) << dbg_tag() << "\n";
        }

        // Then block
        fn_body_ << then_lbl << ":\n";
        block_terminated_ = false;
        if (node.then_block) node.then_block->accept(*this);
        bool then_terminated = block_terminated_;
        if (!then_terminated) {
            if (node.location.line > 0) current_loc_ = node.location;
            fn_body_ << "  br label %" << exit_lbl << dbg_tag() << "\n";
        }

        // Else block
        bool else_terminated = false;
        if (node.else_block) {
            fn_body_ << else_lbl << ":\n";
            block_terminated_ = false;
            node.else_block->accept(*this);
            else_terminated = block_terminated_;
            if (!else_terminated) {
                if (node.location.line > 0) current_loc_ = node.location;
                fn_body_ << "  br label %" << exit_lbl << dbg_tag() << "\n";
            }
        }

        // Only emit exit block if at least one branch does not terminate
        if (!then_terminated || (node.else_block && !else_terminated) || !node.else_block) {
            fn_body_ << exit_lbl << ":\n";
            block_terminated_ = false;
        } else {
            block_terminated_ = true;
        }
    }

    void visit(frontend::ExpressionStmtNode& node) override {
        if (node.location.line > 0) current_loc_ = node.location;
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
    static int execute_program(frontend::ProgramNode& program, bool verbose = false,
                               bool enable_debug = false, bool enable_lto = false,
                               const std::string& filename = "program.bt") {
        LLVMCodeGenVisitor codegen(enable_debug, filename);
        std::string llvm_ir = codegen.generate(program);

        static std::atomic<uint64_t> jit_counter{0};
        uint64_t count = ++jit_counter;
        pid_t pid = getpid();
        std::string base_path = "/tmp/bt_jit_" + std::to_string(pid) + "_" + std::to_string(count);
        std::string tmp_ll_path = base_path + ".ll";
        std::string tmp_dylib_path = base_path + ".dylib";

        std::ofstream ll_file(tmp_ll_path);
        if (!ll_file.is_open()) {
            std::cerr << "[-] JIT Error: Failed to create temporary .ll file: " << tmp_ll_path << "\n";
            return 1;
        }
        ll_file << llvm_ir;
        ll_file.close();

        // Build native shared library via Clang
        std::string cmd = "export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk && "
                          "clang -O3 -shared -fPIC -Wno-override-module ";
        if (enable_debug) cmd += "-g ";
        if (enable_lto) cmd += "-flto=thin ";
        cmd += "\"" + tmp_ll_path + "\" -o \"" + tmp_dylib_path + "\"";
        if (!verbose) {
            cmd += " 2>/dev/null";
        }

        if (verbose) {
            std::cout << "[Native JIT] Compiling LLVM IR to native shared library: " << tmp_dylib_path << "\n";
        }

        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "[-] JIT Error: Clang native compilation failed!\n";
            unlink(tmp_ll_path.c_str());
            return 1;
        }

        // Dynamically load compiled dylib
        void* handle = dlopen(tmp_dylib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            std::cerr << "[-] JIT Error: dlopen failed: " << dlerror() << "\n";
            unlink(tmp_ll_path.c_str());
            unlink(tmp_dylib_path.c_str());
            return 1;
        }

        // Resolve entry point
        using MainFn = int (*)();
        MainFn fn = reinterpret_cast<MainFn>(dlsym(handle, "bt_main"));
        if (!fn) {
            std::cerr << "[-] JIT Error: Symbol 'bt_main' not found: " << dlerror() << "\n";
            dlclose(handle);
            unlink(tmp_ll_path.c_str());
            unlink(tmp_dylib_path.c_str());
            return 1;
        }

        // Execute at native CPU speed!
        int exit_code = fn();

        dlclose(handle);
        unlink(tmp_ll_path.c_str());
        unlink(tmp_dylib_path.c_str());

        return exit_code;
    }
};

} // namespace bt::codegen

#endif // TERNARY_LLVM_CODEGEN_HPP
