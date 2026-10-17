// RUN: %bt-opt %s --convert-all | FileCheck %s
//
// This file is the real input module. bt-opt must parse it and lower the bt.*
// sites it contains; the CHECKs below verify BOTH that the original functions
// survive ingestion (annotated with the lowering that was applied) and that
// the canonical lowering templates were appended.

module @bt_opt_pipeline {

  func.func @gemm_kernel(%A_P: !bt.tensor<4x4x64>, %A_N: !bt.tensor<4x4x64>, %B_P: !bt.tensor<4x4x64>, %B_N: !bt.tensor<4x4x64>) -> !bt.tensor<4x4x64> {
    %r = bt.matmul %A_P, %A_N, %B_P, %B_N : (!bt.tensor<4x4x64>, !bt.tensor<4x4x64>, !bt.tensor<4x4x64>, !bt.tensor<4x4x64>) -> !bt.tensor<4x4x64>
    return %r : !bt.tensor<4x4x64>
  }

  func.func @dot_kernel(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> i64 {
    %d = bt.planar_dot %ap, %an, %bp, %bn : (i64, i64, i64, i64) -> i64
    return %d : i64
  }

  func.func @adder_kernel(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> (i64, i64, i64, i64) {
    %s, %c = bt.btha %ap, %an, %bp, %bn : (i64, i64, i64, i64) -> (i64, i64, i64, i64)
    return %s, %c, %ap, %an : i64, i64, i64, i64
  }

  func.func @unpack_kernel(%b: i32) -> i32 {
    %t = bt.base243_unpack %b : (i32) -> i32
    return %t : i32
  }

  func.func @csa_kernel(%sp: i64, %sn: i64, %cp: i64, %cn: i64) -> i64 {
    %w = bt.csa_reduce %sp, %sn, %cp, %cn : (i64, i64, i64, i64) -> i64
    return %w : i64
  }
}

// RUN-scoped CHECK group for --convert-all:
// CHECK: Processed by bt-opt

// Input module preserved through ingestion (maps are emitted before functions):
// CHECK-LABEL: module @bt_opt_pipeline {
// CHECK: #map_A = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #map_B = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #map_C = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: func.func @gemm_kernel
// CHECK: bt.matmul /* lowered-by: TernaryToLinalgPass -> @bt_gemm_planar_linalg */
// CHECK: func.func @dot_kernel
// CHECK: bt.planar_dot /* lowered-by: TernaryToVectorPass -> @bt_planar_dot_vector_simd */
// CHECK: func.func @adder_kernel
// CHECK: bt.btha /* lowered-by: ConvertBTHAPass -> @bt_btha_lowering */
// CHECK: func.func @unpack_kernel
// CHECK: bt.base243_unpack /* lowered-by: ConvertBase243Pass -> @bt_unpack_base243_fast */
// CHECK: func.func @csa_kernel
// CHECK: bt.csa_reduce /* lowered-by: ConvertCSAReducePass -> @bt_csa_reduce_canonical */

// Canonical lowerings appended (exactly one per recognized op site):
// CHECK-LABEL: func.func @bt_gemm_planar_linalg
// CHECK: linalg.generic
// CHECK: "llvm.intr.ctpop"(%p_prod) : (i64) -> i64
// CHECK-LABEL: func.func @bt_planar_dot_vector_simd
// CHECK: vector<4xi64>
// CHECK: vector.reduction <add>
// CHECK-LABEL: func.func @bt_btha_lowering
// CHECK: arith.xori %ap, %bp : i64
// CHECK-LABEL: func.func @bt_unpack_base243_fast
// CHECK: arith.constant 171 : i32
// CHECK-LABEL: func.func @bt_csa_reduce_canonical
// CHECK: scf.while (%curr_sp = %sp

// RUN: %bt-opt %s --emit-llvm-ir | FileCheck %s

// RUN-scoped CHECK group for --emit-llvm-ir (module name comes from the input):
// CHECK: ; ModuleID = 'bt_opt_pipeline'
// CHECK: declare i64 @llvm.ctpop.i64(i64)
// CHECK-LABEL: define i64 @bt_planar_dot_64(i64 %ap, i64 %an, i64 %bp, i64 %bn) {
// CHECK: %pop_p = call i64 @llvm.ctpop.i64(i64 %p_prod)
// CHECK: ret i64 %res
