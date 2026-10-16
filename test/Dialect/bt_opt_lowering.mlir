// RUN: %bt-opt %s --convert-all | FileCheck %s

// CHECK: // Processed by bt-opt
// CHECK: module @bt_opt_pipeline
// CHECK: #map_A = affine_map
// CHECK: func.func @bt_gemm_planar_linalg
// CHECK: linalg.generic
// CHECK: func.func @bt_planar_dot_vector_simd
// CHECK: vector<4xi64>
// CHECK: func.func @bt_btha_lowering
// CHECK: arith.andi
// CHECK: func.func @bt_unpack_base243_fast
// CHECK: 171
// CHECK: func.func @bt_csa_reduce_canonical
