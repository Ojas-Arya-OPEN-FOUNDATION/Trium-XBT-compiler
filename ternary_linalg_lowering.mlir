//===-- ternary_linalg_lowering.mlir - Lowering bt Dialect to linalg/arith/llvm -*- MLIR -*-===//
//
// Part of the Universal Balanced Ternary Compiler Infrastructure.
// Grounded in Lean 4 formal proofs and Z3 SMT bit-level verification.
//
// This file demonstrates the concrete lowering of high-level Balanced Ternary
// operations (`bt.planar_dot`, `bt.gemm`, `bt.btha`, `bt.unpack_base243`) into
// standard MLIR dialects:
//   - `linalg.generic` (structured loop contraction over tensor tiles)
//   - `arith` (bitwise logic: andi, ori, xori, subi, addi, shrui, muli)
//   - `llvm.intr.ctpop` (hardware population count intrinsic)
//
//===----------------------------------------------------------------------===//

module @ternary_compilation_pipeline {

  // ============================================================================
  // 1. Scalar Lowering: bt.planar_dot -> arith + llvm.intr.ctpop
  // ============================================================================
  // Computes the inner product between two 64-trit dual-rail words:
  //   u = (ap, an), v = (bp, bn)
  //   P_prod = (ap & bp) | (an & bn)   // Positions where product is +1
  //   N_prod = (ap & bn) | (an & bp)   // Positions where product is -1
  //   dot = popcount(P_prod) - popcount(N_prod)
  //
  // Latency on modern ALUs: ~3 clock cycles (Zero floating-point multipliers!)
  // ============================================================================
  func.func @bt_planar_dot_scalar_i64(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> i64 {
    // 1. Positive product rail: (+1 * +1 = +1) and (-1 * -1 = +1)
    %pp1 = arith.andi %ap, %bp : i64
    %pp2 = arith.andi %an, %bn : i64
    %p_prod = arith.ori %pp1, %pp2 : i64

    // 2. Negative product rail: (+1 * -1 = -1) and (-1 * +1 = -1)
    %np1 = arith.andi %ap, %bn : i64
    %np2 = arith.andi %an, %bp : i64
    %n_prod = arith.ori %np1, %np2 : i64

    // 3. Hardware Population Count (maps directly to POPCNT/CNT/OpBitCount)
    %pop_p = "llvm.intr.ctpop"(%p_prod) : (i64) -> i64
    %pop_n = "llvm.intr.ctpop"(%n_prod) : (i64) -> i64

    // 4. Balanced reduction: (+1 count) - (-1 count)
    %res = arith.subi %pop_p, %pop_n : i64
    return %res : i64
  }

  // ============================================================================
  // 2. SIMD Vector Lowering: bt.planar_dot -> vector<4xi64> (256-bit AVX2/NEON)
  // ============================================================================
  // Executes 4 simultaneous 64-trit dot products (256 trits total) in parallel
  // across SIMD vector lanes.
  // ============================================================================
  func.func @bt_planar_dot_vector_4x64(
      %vap: vector<4xi64>, %van: vector<4xi64>,
      %vbp: vector<4xi64>, %vbn: vector<4xi64>) -> vector<4xi64> {

    // Vector positive products
    %vpp1 = arith.andi %vap, %vbp : vector<4xi64>
    %vpp2 = arith.andi %van, %vbn : vector<4xi64>
    %vp_prod = arith.ori %vpp1, %vpp2 : vector<4xi64>

    // Vector negative products
    %vnp1 = arith.andi %vap, %vbn : vector<4xi64>
    %vnp2 = arith.andi %van, %vbp : vector<4xi64>
    %vn_prod = arith.ori %vnp1, %vnp2 : vector<4xi64>

    // SIMD Vector Hardware Population Count
    %vpop_p = "llvm.intr.ctpop"(%vp_prod) : (vector<4xi64>) -> vector<4xi64>
    %vpop_n = "llvm.intr.ctpop"(%vn_prod) : (vector<4xi64>) -> vector<4xi64>

    // Vector subtract
    %vres = arith.subi %vpop_p, %vpop_n : vector<4xi64>
    return %vres : vector<4xi64>
  }

  // ============================================================================
  // 3. Matrix Multiplication Lowering: bt.gemm -> linalg.generic
  // ============================================================================
  // Matrix Multiplication Contraction: C = A x B + C
  // - Matrix A: M x K bitplane tiles (A_P, A_N)
  // - Matrix B: K x N bitplane tiles (B_P, B_N)
  // - Matrix C: M x N integer accumulator
  //
  // Every reduction iteration contracts 64 trits along dimension K using
  // bitwise operations and single-cycle hardware popcount.
  // ============================================================================

  #map_A = affine_map<(d0, d1, d2) -> (d0, d2)>
  #map_B = affine_map<(d0, d1, d2) -> (d2, d1)>
  #map_C = affine_map<(d0, d1, d2) -> (d0, d1)>

  func.func @bt_gemm_planar_linalg(
      %A_P: tensor<?x?xi64>, %A_N: tensor<?x?xi64>,
      %B_P: tensor<?x?xi64>, %B_N: tensor<?x?xi64>,
      %C_init: tensor<?x?xi64>) -> tensor<?x?xi64> {

    %result = linalg.generic {
      indexing_maps = [#map_A, #map_A, #map_B, #map_B, #map_C],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A_P, %A_N, %B_P, %B_N : tensor<?x?xi64>, tensor<?x?xi64>, tensor<?x?xi64>, tensor<?x?xi64>)
      outs(%C_init : tensor<?x?xi64>) {
    ^bb0(%ap: i64, %an: i64, %bp: i64, %bn: i64, %acc_in: i64):
      // --- Bit-Sliced Planar Multiplication (Contraction Core) ---
      // P_prod = (ap & bp) | (an & bn)
      %pp1 = arith.andi %ap, %bp : i64
      %pp2 = arith.andi %an, %bn : i64
      %p_prod = arith.ori %pp1, %pp2 : i64

      // N_prod = (ap & bn) | (an & bp)
      %np1 = arith.andi %ap, %bn : i64
      %np2 = arith.andi %an, %bp : i64
      %n_prod = arith.ori %np1, %np2 : i64

      // Hardware Popcount Intrinsic (1 cycle on AVX-512 VPOPCNT, ARM NEON, SPIR-V OpBitCount)
      %pop_p = "llvm.intr.ctpop"(%p_prod) : (i64) -> i64
      %pop_n = "llvm.intr.ctpop"(%n_prod) : (i64) -> i64

      // Partial inner product for this 64-trit chunk
      %partial_sum = arith.subi %pop_p, %pop_n : i64

      // Accumulate into running sum
      %acc_out = arith.addi %acc_in, %partial_sum : i64

      linalg.yield %acc_out : i64
    } -> tensor<?x?xi64>

    return %result : tensor<?x?xi64>
  }

  // ============================================================================
  // 4. Arithmetic Core Lowering: bt.btha -> arith bitwise ops
  // ============================================================================
  // Exact Balanced Ternary Half-Adder across 64 trits in parallel:
  // Carry plane:
  //   PC = AP & BP
  //   NC = AN & BN
  // Sum plane:
  //   PS = ((AP ^ BP) & ~(AN | BN)) | NC
  //   NS = ((AN ^ BN) & ~(AP | BP)) | PC
  // ============================================================================
  func.func @bt_btha_lowering(
      %ap: i64, %an: i64, %bp: i64, %bn: i64) -> (i64, i64, i64, i64) {
    %c_all_ones = arith.constant -1 : i64

    // 1. Carry generation
    %pc = arith.andi %ap, %bp : i64
    %nc = arith.andi %an, %bn : i64

    // 2. Sum generation
    // Inverted masks: ~(AN | BN) and ~(AP | BP)
    %or_n = arith.ori %an, %bn : i64
    %not_or_n = arith.xori %or_n, %c_all_ones : i64

    %or_p = arith.ori %ap, %bp : i64
    %not_or_p = arith.xori %or_p, %c_all_ones : i64

    // XOR rails
    %xor_p = arith.xori %ap, %bp : i64
    %xor_n = arith.xori %an, %bn : i64

    // Form PS and NS
    %ps_cand = arith.andi %xor_p, %not_or_n : i64
    %ps = arith.ori %ps_cand, %nc : i64

    %ns_cand = arith.andi %xor_n, %not_or_p : i64
    %ns = arith.ori %ns_cand, %pc : i64

    // Return (PS, NS, PC, NC)
    return %ps, %ns, %pc, %nc : i64, i64, i64, i64
  }

  // ============================================================================
  // 5. Memory Decompression Lowering: bt.unpack_base243 -> division-free fixed point
  // ============================================================================
  // Decompresses 1 byte [0..242] containing 5 trits using Z3-verified
  // reciprocal magic constants: 171, 57, 19, 203.
  // Zero hardware division cycles required!
  // ============================================================================
  func.func @bt_unpack_base243_fast(%byte: i32) -> (i32, i32, i32, i32, i32) {
    %c3   = arith.constant 3 : i32
    %c1   = arith.constant 1 : i32
    %c171 = arith.constant 171 : i32
    %c57  = arith.constant 57 : i32
    %c19  = arith.constant 19 : i32
    %c203 = arith.constant 203 : i32
    %c9   = arith.constant 9 : i32
    %c14  = arith.constant 14 : i32

    // Reciprocal fixed-point quotients: q_k = floor(B / 3^k)
    %m1 = arith.muli %byte, %c171 : i32
    %q1 = arith.shrui %m1, %c9 : i32

    %m2 = arith.muli %byte, %c57 : i32
    %q2 = arith.shrui %m2, %c9 : i32

    %m3 = arith.muli %byte, %c19 : i32
    %q3 = arith.shrui %m3, %c9 : i32

    %m4 = arith.muli %byte, %c203 : i32
    %q4 = arith.shrui %m4, %c14 : i32

    // Extract balanced digits: t_k = q_k - 3*q_{k+1} - 1 in {-1, 0, 1}
    // t0 = byte - 3*q1 - 1
    %p1 = arith.muli %q1, %c3 : i32
    %sub0 = arith.subi %byte, %p1 : i32
    %t0 = arith.subi %sub0, %c1 : i32

    // t1 = q1 - 3*q2 - 1
    %p2 = arith.muli %q2, %c3 : i32
    %sub1 = arith.subi %q1, %p2 : i32
    %t1 = arith.subi %sub1, %c1 : i32

    // t2 = q2 - 3*q3 - 1
    %p3 = arith.muli %q3, %c3 : i32
    %sub2 = arith.subi %q2, %p3 : i32
    %t2 = arith.subi %sub2, %c1 : i32

    // t3 = q3 - 3*q4 - 1
    %p4 = arith.muli %q4, %c3 : i32
    %sub3 = arith.subi %q3, %p4 : i32
    %t3 = arith.subi %sub3, %c1 : i32

    // t4 = q4 - 1
    %t4 = arith.subi %q4, %c1 : i32

    return %t0, %t1, %t2, %t3, %t4 : i32, i32, i32, i32, i32
  }

}
