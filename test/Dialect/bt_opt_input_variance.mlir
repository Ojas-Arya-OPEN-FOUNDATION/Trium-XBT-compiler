// Input-variance test: two different input modules must produce different
// bt-opt output. The tool output below is checked against real tool invocations
// (printf builds the temporary input; %t is the runner sandbox path).
//
// RUN: printf 'module @variance_a {\n  func.func @dot(%%ap: i64, %%an: i64, %%bp: i64, %%bn: i64) -> i64 {\n    %%r = bt.planar_dot %%ap, %%an, %%bp, %%bn : (i64, i64, i64, i64) -> i64\n    return %%r : i64\n  }\n}\n' > %t && %bt-opt %t --convert-all | FileCheck %s

// CHECK: Processed by bt-opt
// CHECK-LABEL: module @variance_a
// CHECK: bt.planar_dot /* lowered-by: TernaryToVectorPass -> @bt_planar_dot_vector_simd */
// CHECK-LABEL: func.func @bt_planar_dot_vector_simd

// RUN: printf 'module @variance_b {\n  func.func @no_ternary_ops(%%x: i64, %%y: i64) -> i64 {\n    %%s = arith.addi %%x, %%y : i64\n    return %%s : i64\n  }\n}\n' > %t && %bt-opt %t --convert-all | FileCheck %s

// CHECK: Processed by bt-opt
// CHECK-LABEL: module @variance_b
// CHECK: func.func @no_ternary_ops
// CHECK: arith.addi %x, %y : i64
// CHECK-NOT: func.func @bt_planar_dot_vector_simd
// CHECK-NOT: func.func @bt_gemm_planar_linalg
// CHECK-NOT: func.func @bt_btha_lowering
// CHECK-NOT: func.func @bt_unpack_base243_fast
// CHECK-NOT: func.func @bt_csa_reduce_canonical
