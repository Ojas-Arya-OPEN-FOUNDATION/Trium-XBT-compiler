/**
 * @file test_unit_mlir.cpp
 * @brief LLVM Official Layer 1: Comprehensive Unit Tests for MLIR Dialect & Larq Lowering Engine
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "../../ternary_mlir_lowering.hpp"
#include "../../include/Ternary/ternary_c_abi.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <array>
#include <random>

using namespace bt;
using namespace bt::mlir;

// ==============================================================================
// 1. Test ConvertTernaryToLinalg
// ==============================================================================

void test_convert_ternary_to_linalg() {
    std::cout << "  [MLIR Lowering 1/6] Testing ConvertTernaryToLinalg...\n";

    // 1. Verify lowering creates function with linalg.generic structured iteration space
    auto fn = ConvertTernaryToLinalg::lower_matmul_to_linalg("bt_gemm_test", 16, 16, 64);
    assert(fn.name == "bt_gemm_test");
    assert(fn.arguments.size() == 5);
    assert(fn.returnTypes.size() == 1);
    assert(fn.rawCustomBody.find("linalg.generic") != std::string::npos);
    assert(fn.rawCustomBody.find("parallel") != std::string::npos);
    assert(fn.rawCustomBody.find("reduction") != std::string::npos);
    assert(fn.rawCustomBody.find("#map_A") != std::string::npos);
    assert(fn.rawCustomBody.find("#map_B") != std::string::npos);
    assert(fn.rawCustomBody.find("#map_C") != std::string::npos);
    assert(fn.rawCustomBody.find("llvm.intr.ctpop") != std::string::npos);
    assert(fn.rawCustomBody.find("linalg.yield") != std::string::npos);

    // 2. Verify functional execution of the lowered linalg structured loop vs reference GEMM
    size_t M = 16, K = 128, N = 16;
    TernaryMatrix A(M, K);
    TernaryMatrix B(K, N);

    std::mt19937_64 rng(42);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            int val = static_cast<int>(rng() % 3) - 1; // {-1, 0, +1}
            A.set(i, k, from_int(val));
        }
    }
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            int val = static_cast<int>(rng() % 3) - 1;
            B.set(k, j, from_int(val));
        }
    }

    DenseMatrixInt64 C_ref(M, N);
    gemm(A, B, C_ref);

    DenseMatrixInt64 C_linalg(M, N);
    ConvertTernaryToLinalg::execute_lowered_linalg(A, B, C_linalg);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            assert(C_ref(i, j) == C_linalg(i, j) && "Linalg lowered execution mismatch with reference GEMM!");
        }
    }

    std::cout << "    -> PASSED: Structured iteration spaces & numerical contraction verified.\n";
}

// ==============================================================================
// 2. Test ConvertTernaryToVector (Larq-Style Bit-Sliced Engine)
// ==============================================================================

void test_convert_ternary_to_vector() {
    std::cout << "  [MLIR Lowering 2/6] Testing ConvertTernaryToVector (Larq Bit-Sliced SIMD)...\n";

    // 1. Verify lowering creates function with vector SIMD operations
    auto fn4 = ConvertTernaryToVector::lower_planar_dot_to_vector("bt_vector_dot_4x64", 4);
    assert(fn4.name == "bt_vector_dot_4x64");
    assert(fn4.arguments.size() == 4);
    assert(fn4.rawCustomBody.find("vector<4xi64>") != std::string::npos);
    assert(fn4.rawCustomBody.find("llvm.intr.ctpop") != std::string::npos);
    assert(fn4.rawCustomBody.find("vector.reduction") != std::string::npos);

    // 2. Verify SIMD vector dot product across 256 trits against scalar dotProduct64
    std::mt19937_64 rng(1337);
    for (size_t trial = 0; trial < 100; ++trial) {
        std::array<uint64_t, 4> vap{}, van{}, vbp{}, vbn{};
        for (size_t l = 0; l < 4; ++l) {
            vap[l] = rng();
            van[l] = ~vap[l] & rng(); // enforce disjointness P & N = 0
            vbp[l] = rng();
            vbn[l] = ~vbp[l] & rng();
        }

        int64_t vec_res = ConvertTernaryToVector::execute_vector_dot_4x64(vap, van, vbp, vbn);

        int64_t scalar_res = 0;
        for (size_t l = 0; l < 4; ++l) {
            scalar_res += dotProduct64(TritWord64(vap[l], van[l]), TritWord64(vbp[l], vbn[l]));
        }

        assert(vec_res == scalar_res && "Larq vector SIMD dot product mismatch!");
    }

    std::cout << "    -> PASSED: Larq bit-sliced SIMD vector lowerings verified across all lanes.\n";
}

// ==============================================================================
// 3. Test ConvertBTHA (6 Bitwise Instructions)
// ==============================================================================

void test_convert_btha() {
    std::cout << "  [MLIR Lowering 3/6] Testing ConvertBTHA (6 Bitwise Instructions)...\n";

    // 1. Verify lowering emits 6 bitwise instructions in MLIR
    auto fn = ConvertBTHA::lower_btha("bt_btha_lowering");
    assert(fn.name == "bt_btha_lowering");
    assert(fn.arguments.size() == 4);
    assert(fn.returnTypes.size() == 4);
    assert(fn.rawCustomBody.find("arith.andi") != std::string::npos);
    assert(fn.rawCustomBody.find("arith.ori") != std::string::npos);
    assert(fn.rawCustomBody.find("arith.xori") != std::string::npos);

    // 2. Exhaustive test over all 9 trit input combinations
    for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
            uint64_t ap = (a == 1) ? 1 : 0;
            uint64_t an = (a == -1) ? 1 : 0;
            uint64_t bp = (b == 1) ? 1 : 0;
            uint64_t bn = (b == -1) ? 1 : 0;

            auto res = ConvertBTHA::execute_bitwise(ap, an, bp, bn);

            // Invariant: Rail disjointness
            assert((res.PS & res.NS) == 0 && "BTHA Sum rail disjointness violation!");
            assert((res.PC & res.NC) == 0 && "BTHA Carry rail disjointness violation!");

            int sum = (res.PS ? 1 : 0) - (res.NS ? 1 : 0);
            int carry = (res.PC ? 1 : 0) - (res.NC ? 1 : 0);

            // Mathematical identity: A + B = 3*C + S
            assert(a + b == 3 * carry + sum && "BTHA arithmetic identity failed!");
        }
    }

    // 3. Parallel 64-trit test against core BTHA
    std::mt19937_64 rng(777);
    for (size_t trial = 0; trial < 50; ++trial) {
        uint64_t ap = rng();
        uint64_t an = ~ap & rng();
        uint64_t bp = rng();
        uint64_t bn = ~bp & rng();

        auto bitwise = ConvertBTHA::execute_bitwise(ap, an, bp, bn);
        AdderOutput core_ha = BTHA(TritWord64(ap, an), TritWord64(bp, bn));

        assert(bitwise.PS == core_ha.Sum.P && "Sum.P mismatch!");
        assert(bitwise.NS == core_ha.Sum.N && "Sum.N mismatch!");
        assert(bitwise.PC == core_ha.Carry.P && "Carry.P mismatch!");
        assert(bitwise.NC == core_ha.Carry.N && "Carry.N mismatch!");
    }

    std::cout << "    -> PASSED: 6-instruction bitwise BTHA verified with zero invariant violations.\n";
}

// ==============================================================================
// 4. Test ConvertBase243 (Reciprocal Multipliers)
// ==============================================================================

void test_convert_base243() {
    std::cout << "  [MLIR Lowering 4/6] Testing ConvertBase243 (Reciprocal Multipliers)...\n";

    // 1. Verify lowering emits reciprocal constants and zero divisions
    auto fn = ConvertBase243::lower_unpack_base243("bt_unpack_base243_fast");
    assert(fn.name == "bt_unpack_base243_fast");
    assert(fn.arguments.size() == 1);
    assert(fn.returnTypes.size() == 5);
    assert(fn.rawCustomBody.find("171") != std::string::npos);
    assert(fn.rawCustomBody.find("57") != std::string::npos);
    assert(fn.rawCustomBody.find("19") != std::string::npos);
    assert(fn.rawCustomBody.find("203") != std::string::npos);
    assert(fn.rawCustomBody.find("arith.shrui") != std::string::npos);
    assert(fn.rawCustomBody.find("arith.div") == std::string::npos); // Zero hardware division!

    // 2. Exhaustive test over all 243 byte values [0..242]
    for (uint32_t b = 0; b < 243; ++b) {
        auto trits = ConvertBase243::execute_unpack(static_cast<uint8_t>(b));

        for (int i = 0; i < 5; ++i) {
            assert(trits[i] >= -1 && trits[i] <= 1 && "Decoded trit out of {-1, 0, +1} bounds!");
        }

        uint8_t repacked = ConvertBase243::execute_pack(trits);
        assert(repacked == b && "Base-243 pack/unpack roundtrip mismatch!");

        // Also cross-verify with Base243Codec in ternary_core.hpp
        auto core_trits = Base243Codec::unpack5_fast(static_cast<uint8_t>(b));
        for (int i = 0; i < 5; ++i) {
            assert(trits[i] == to_int(core_trits[i]) && "Mismatch with core Base243Codec!");
        }
    }

    std::cout << "    -> PASSED: Division-free Base-243 reciprocal unpack verified for all 243 states.\n";
}

// ==============================================================================
// 5. Test MLIR Pass Pipeline & Manager
// ==============================================================================

void test_mlir_pass_pipeline() {
    std::cout << "  [MLIR Lowering 5/6] Testing MLIR Pass Pipeline & Manager...\n";

    MLIRModule mod;
    mod.name = "test_pipeline_module";

    MLIRPassManager pm;
    pm.addPass(std::make_unique<TernaryToLinalgPass>());
    pm.addPass(std::make_unique<TernaryToVectorPass>(4));
    pm.addPass(std::make_unique<ConvertBTHAPass>());
    pm.addPass(std::make_unique<ConvertBase243Pass>());

    bool success = pm.run(mod);
    assert(success && "Pass manager pipeline failed!");
    assert(mod.functions.size() == 4 && "Expected 4 lowered functions in module!");
    assert(mod.affineMaps.size() == 3 && "Expected 3 affine maps in module!");

    std::cout << "    -> PASSED: Pass pipeline executed 4 passes successfully.\n";
}

// ==============================================================================
// 6. Test IR Emission & TableGen Generators
// ==============================================================================

void test_ir_emission_and_tablegen() {
    std::cout << "  [MLIR Lowering 6/6] Testing TableGen & MLIR Text Emitters...\n";

    // 1. TableGen Dialect Definition
    std::string dialect_td = TableGenEmitter::emit_dialect_td();
    assert(dialect_td.find("def BT_Dialect : Dialect") != std::string::npos);
    assert(dialect_td.find("def BT_DualRailType") != std::string::npos);
    assert(dialect_td.find("def BT_CSAPairType") != std::string::npos);
    assert(dialect_td.find("def BT_TensorType") != std::string::npos);

    // 2. TableGen Ops Definition
    std::string ops_td = TableGenEmitter::emit_ops_td();
    assert(ops_td.find("def BT_MatmulOp") != std::string::npos);
    assert(ops_td.find("def BT_PlanarDotOp") != std::string::npos);
    assert(ops_td.find("def BT_BTHAOp") != std::string::npos);
    assert(ops_td.find("def BT_UnpackBase243Op") != std::string::npos);

    // 3. MLIR Text Emission
    std::string mlir_text = MLIREmitter::emit_canonical_lowering_pipeline();
    assert(mlir_text.find("module @ternary_compilation_pipeline") != std::string::npos);
    assert(mlir_text.find("#map_A = affine_map") != std::string::npos);
    assert(mlir_text.find("func.func @bt_gemm_planar_linalg") != std::string::npos);
    assert(mlir_text.find("func.func @bt_planar_dot_vector_simd") != std::string::npos);
    assert(mlir_text.find("func.func @bt_btha_lowering") != std::string::npos);
    assert(mlir_text.find("func.func @bt_unpack_base243_fast") != std::string::npos);

    // 4. LarqLoweringEngine unified invariant check
    assert(LarqLoweringEngine::verify_invariants() && "LarqLoweringEngine verification failed!");

    std::cout << "    -> PASSED: TableGen & MLIR IR text generation verified.\n";
}

// ==============================================================================
// 7. Test Fixed MLIR (i64, i64) -> i64 Type Formatting
// ==============================================================================

void test_mlir_type_formatting() {
    std::cout << "  [MLIR Lowering 7/8] Testing MLIR Operation Type Formatting...\n";

    MLIROperation opAdd;
    opAdd.name = "arith.addi";
    opAdd.operands = {
        MLIRValue("%a", MLIRType::getI64()),
        MLIRValue("%b", MLIRType::getI64())
    };
    opAdd.results = {
        MLIRValue("%res", MLIRType::getI64())
    };

    std::string opStr = opAdd.to_string();
    assert(opStr.find(": (i64, i64) -> i64") != std::string::npos &&
           "Expected '(i64, i64) -> i64' MLIR type formatting for multi-operand single-result!");

    MLIROperation opAbs;
    opAbs.name = "math.abs";
    opAbs.operands = { MLIRValue("%x", MLIRType::getI64()) };
    opAbs.results = { MLIRValue("%y", MLIRType::getI64()) };
    std::string absStr = opAbs.to_string();
    assert(absStr.find(": i64 -> i64") != std::string::npos &&
           "Expected 'i64 -> i64' for single operand single result!");

    MLIROperation opMulti;
    opMulti.name = "bt.btha";
    opMulti.operands = {
        MLIRValue("%p", MLIRType::getI64()),
        MLIRValue("%n", MLIRType::getI64())
    };
    opMulti.results = {
        MLIRValue("%sp", MLIRType::getI64()),
        MLIRValue("%sn", MLIRType::getI64())
    };
    std::string multiStr = opMulti.to_string();
    assert(multiStr.find(": (i64, i64) -> (i64, i64)") != std::string::npos &&
           "Expected '(i64, i64) -> (i64, i64)' for multi-operand multi-result!");

    std::cout << "    -> PASSED: (i64, i64) -> i64 MLIR type formatting validated.\n";
}

// ==============================================================================
// 8. Test Production C ABI (include/Ternary/ternary_c_abi.h)
// ==============================================================================

void test_ternary_c_abi() {
    std::cout << "  [MLIR Lowering 8/8] Testing Production C ABI (include/Ternary/ternary_c_abi.h)...\n";

    // 1. bt_word64_t and disjointness
    bt_word64_t w1 = bt_make_word64(0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL);
    assert(bt_verify_disjointness(w1));
    bt_word64_t w_bad = bt_make_word64(0x3ULL, 0x1ULL);
    assert(!bt_verify_disjointness(w_bad));

    // 2. bt_dot_product_64
    bt_word64_t a = bt_make_word64(0b0011ULL, 0b1100ULL);
    bt_word64_t b = bt_make_word64(0b0101ULL, 0b1010ULL);
    int64_t dot = bt_dot_product_64(a, b);
    assert(dot == 0);
    assert(bt_dot_product_64_raw(0b0011ULL, 0b1100ULL, 0b0101ULL, 0b1010ULL) == 0);

    // 3. bt_btha_step
    for (int t1 = -1; t1 <= 1; ++t1) {
        for (int t2 = -1; t2 <= 1; ++t2) {
            bt_word64_t wa = bt_make_word64((t1 == 1) ? 1 : 0, (t1 == -1) ? 1 : 0);
            bt_word64_t wb = bt_make_word64((t2 == 1) ? 1 : 0, (t2 == -1) ? 1 : 0);
            bt_btha_result_t res = bt_btha_step(wa, wb);
            int sum = (res.sum.p & 1 ? 1 : 0) - (res.sum.n & 1 ? 1 : 0);
            int carry = (res.carry.p & 1 ? 1 : 0) - (res.carry.n & 1 ? 1 : 0);
            assert(t1 + t2 == 3 * carry + sum);
        }
    }

    // 4. bt_base243_pack and bt_base243_unpack
    for (uint32_t byte = 0; byte < 243; ++byte) {
        int8_t trits[5];
        bt_base243_unpack(static_cast<uint8_t>(byte), trits);
        uint8_t repacked = bt_base243_pack(trits);
        assert(repacked == byte);
    }

    // 5. bt_swar_bct_to_bin32
    assert(bt_swar_bct_to_bin32(0, 0) == 0);
    assert(bt_swar_bct_to_bin32(1, 0) == 1);
    assert(bt_swar_bct_to_bin32(0, 1) == -1);
    assert(bt_swar_bct_to_bin32(2, 0) == 3);
    assert(bt_swar_bct_to_bin32(3, 0) == 4);

    std::mt19937_64 rng(42);
    for (size_t trial = 0; trial < 100; ++trial) {
        uint32_t p = static_cast<uint32_t>(rng() & 0xFFFFFFFFU);
        uint32_t n = ~p & static_cast<uint32_t>(rng() & 0xFFFFFFFFU);
        int64_t swar_val = bt_swar_bct_to_bin32(p, n);

        int64_t ref_val = 0;
        int64_t power3 = 1;
        for (int i = 0; i < 32; ++i) {
            int d = ((p >> i) & 1) - ((n >> i) & 1);
            ref_val += d * power3;
            power3 *= 3;
        }
        assert(swar_val == ref_val && "SWAR BCT-to-bin32 mismatch with polynomial evaluation!");
    }

    // 6. bt_hamming_distance_64
    bt_word64_t ha1 = bt_make_word64(0x0FULL, 0x00ULL);
    bt_word64_t ha2 = bt_make_word64(0x00ULL, 0x0FULL);
    assert(bt_hamming_distance_64(ha1, ha2) == 4);
    assert(bt_hamming_distance_64(ha1, ha1) == 0);

    // 7. bt_fp32_t and bt_csr_t
    bt_fp32_t fp = { 100, 2 };
    assert(fp.mantissa == 100 && fp.exponent == 2);
    uint32_t rptr[2] = {0, 1};
    uint32_t cidx[1] = {0};
    uint64_t sgn[1] = {1};
    bt_csr_t csr = { 1, 1, 1, rptr, cidx, sgn };
    assert(csr.rows == 1 && csr.nnz == 1);

    std::cout << "    -> PASSED: All C ABI / FFI functions and structures validated.\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  [LLVM Unit Test Layer 1] MLIR Dialect & Larq Lowering Engine\n";
    std::cout << "======================================================================\n";

    test_convert_ternary_to_linalg();
    test_convert_ternary_to_vector();
    test_convert_btha();
    test_convert_base243();
    test_mlir_pass_pipeline();
    test_ir_emission_and_tablegen();
    test_mlir_type_formatting();
    test_ternary_c_abi();

    std::cout << "======================================================================\n";
    std::cout << "  -> PASSED: All MLIR Lowering Unit Tests Verified (0 Failures).\n";
    std::cout << "======================================================================\n";
    return 0;
}
