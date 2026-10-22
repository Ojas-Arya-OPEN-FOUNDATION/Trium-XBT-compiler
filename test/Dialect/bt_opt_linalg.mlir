// RUN: %bt-opt %s --convert-linalg | FileCheck %s
//
// Verifies the --convert-linalg pass runs in isolation: only the linalg
// lowering may be emitted, and it must be applied to the bt.gemm site below.

module @linalg_only_pipeline {

  func.func @mm(%amp: !bt.tensor<8x8x64>, %amn: !bt.tensor<8x8x64>, %bmp: !bt.tensor<8x8x64>, %bmn: !bt.tensor<8x8x64>) -> !bt.tensor<8x8x64> {
    %acc = bt.gemm %amp, %amn, %bmp, %bmn : (!bt.tensor<8x8x64>, !bt.tensor<8x8x64>, !bt.tensor<8x8x64>, !bt.tensor<8x8x64>) -> !bt.tensor<8x8x64>
    return %acc : !bt.tensor<8x8x64>
  }
}

// CHECK: Processed by bt-opt
// CHECK-LABEL: module @linalg_only_pipeline
// CHECK: #map_A = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #map_B = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: func.func @mm
// CHECK-LABEL: func.func @bt_gemm_planar_linalg
// CHECK-NOT: func.func @bt_csa_reduce_canonical
// CHECK-NOT: func.func @bt_planar_dot_vector_simd
// CHECK-NOT: func.func @bt_btha_lowering
