// RUN: %bt-opt %s | FileCheck %s

// This module is ingested by bt-opt and re-emitted through the real module
// representation. The CHECKs below must match the *tool output*, which cannot
// contain this file's own CHECK comment lines (CHECK-NOT enforces that), so
// this test can no longer pass by echoing itself.

// CHECK: Processed by bt-opt
// CHECK-NOT: // CHECK
// CHECK-LABEL: module @test_ternary_dialect

//===----------------------------------------------------------------------===//
// Layer 6: MLIR Dialect Operations & Lowering Pattern Tests
//===----------------------------------------------------------------------===//

module @test_ternary_dialect {

  // CHECK-LABEL: func.func @test_btha
  func.func @test_btha(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> (i64, i64, i64, i64) {
    // CHECK: arith.andi
    // CHECK: arith.xori
    // CHECK: arith.ori
    // CHECK: arith.constant -1
    %pc = arith.andi %ap, %bp : i64
    %nc = arith.andi %an, %bn : i64

    %p_xor = arith.xori %ap, %bp : i64
    %n_or = arith.ori %an, %bn : i64
    %c_all1 = arith.constant -1 : i64
    %n_nor = arith.xori %n_or, %c_all1 : i64
    %ps_term = arith.andi %p_xor, %n_nor : i64
    %ps = arith.ori %ps_term, %nc : i64

    %n_xor = arith.xori %an, %bn : i64
    %p_or = arith.ori %ap, %bp : i64
    %p_nor = arith.xori %p_or, %c_all1 : i64
    %ns_term = arith.andi %n_xor, %p_nor : i64
    %ns = arith.ori %ns_term, %pc : i64

    return %ps, %ns, %pc, %nc : i64, i64, i64, i64
  }

  // CHECK-LABEL: func.func @test_dot_product
  func.func @test_dot_product(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> i64 {
    // CHECK: arith.andi
    // CHECK: arith.ori
    // CHECK: "llvm.intr.ctpop"
    // CHECK: arith.subi
    // CHECK: return %res : i64
    %pp1 = arith.andi %ap, %bp : i64
    %pp2 = arith.andi %an, %bn : i64
    %p_prod = arith.ori %pp1, %pp2 : i64

    %np1 = arith.andi %ap, %bn : i64
    %np2 = arith.andi %an, %bp : i64
    %n_prod = arith.ori %np1, %np2 : i64

    %pop_p = "llvm.intr.ctpop"(%p_prod) : (i64) -> i64
    %pop_n = "llvm.intr.ctpop"(%n_prod) : (i64) -> i64
    %res = arith.subi %pop_p, %pop_n : i64
    return %res : i64
  }
}
