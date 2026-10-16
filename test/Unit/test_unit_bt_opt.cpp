/**
 * @file test_unit_bt_opt.cpp
 * @brief LLVM Unit Test: bt-opt Standalone Optimizer & IREE Compiler Plugin
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "../../include/Ternary/TernaryDialect.h"
#include "../../include/Ternary/TernaryPasses.h"
#include "../../iree/TernaryIREEPlugin.h"
#include "../../ternary_mlir_lowering.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <array>
#include <random>

using namespace mlir::bt;
using namespace mlir::iree_compiler;

// ==============================================================================
// 1. Test Dialect Registration & Invariant Verifiers
// ==============================================================================

void test_dialect_registration() {
    std::cout << "  [bt-opt 1/6] Testing Dialect Registration & Invariant Verifiers...\n";

    mlir::DialectRegistry reg;
    registerTernaryDialect(reg);
    assert(reg.has("bt") && "Ternary dialect 'bt' registration failed!");
    assert(isTernaryDialectRegistered() && "Global dialect registration flag not set!");

    assert(std::string(TernaryDialect::getDialectNamespace()) == "bt");
    assert(std::string(TernaryDialect::getSummary()).find("Balanced Ternary") != std::string::npos);

    // Test dual-rail disjointness verifier: P & N must be 0
    assert(TernaryDialect::verifyDualRailDisjointness(0xAAAAAAAAULL, 0x55555555ULL));
    assert(!TernaryDialect::verifyDualRailDisjointness(0x3ULL, 0x1ULL) && "Failed to detect rail overlap!");

    // Test BTHA verifier across all 9 trit input combinations
    for (int a = -1; a <= 1; ++a) {
        for (int b = -1; b <= 1; ++b) {
            uint64_t ap = (a == 1) ? 1 : 0;
            uint64_t an = (a == -1) ? 1 : 0;
            uint64_t bp = (b == 1) ? 1 : 0;
            uint64_t bn = (b == -1) ? 1 : 0;
            assert(TernaryDialect::verifyBTHA(ap, an, bp, bn) && "BTHA invariant verifier failed!");
        }
    }

    // Test Base-243 reciprocal unpack verifier
    for (uint32_t byte = 0; byte < 243; ++byte) {
        assert(TernaryDialect::verifyBase243Unpack(static_cast<uint8_t>(byte)) && "Base-243 unpack verifier failed!");
    }
    assert(!TernaryDialect::verifyBase243Unpack(243) && "Expected failure on out-of-bounds byte 243!");

    // Test CSA reduction verifier
    uint64_t rp = 0, rn = 0;
    assert(TernaryDialect::verifyCSAReduce(0x0FULL, 0xF0ULL, 0x00ULL, 0x00ULL, rp, rn));
    assert((rp & rn) == 0 && "CSA reduce produced overlapping rails!");

    std::cout << "    -> PASSED: Dialect registration, types, and invariant verifiers.\n";
}

// ==============================================================================
// 2. Test TableGen Definitions
// ==============================================================================

void test_tablegen_definitions() {
    std::cout << "  [bt-opt 2/6] Testing TableGen Specification Files...\n";

    // Check include/Ternary/TernaryDialect.td
    std::ifstream dialectFile("include/Ternary/TernaryDialect.td");
    assert(dialectFile.is_open() && "Could not open include/Ternary/TernaryDialect.td!");
    std::stringstream dialectBuf;
    dialectBuf << dialectFile.rdbuf();
    std::string dialectContent = dialectBuf.str();

    assert(dialectContent.find("def BT_Dialect : Dialect") != std::string::npos);
    assert(dialectContent.find("let cppNamespace = \"::mlir::bt\";") != std::string::npos);
    assert(dialectContent.find("def BT_DualRailType") != std::string::npos);
    assert(dialectContent.find("def BT_CSAPairType") != std::string::npos);
    assert(dialectContent.find("def BT_TensorType") != std::string::npos);

    // Check include/Ternary/TernaryOps.td
    std::ifstream opsFile("include/Ternary/TernaryOps.td");
    assert(opsFile.is_open() && "Could not open include/Ternary/TernaryOps.td!");
    std::stringstream opsBuf;
    opsBuf << opsFile.rdbuf();
    std::string opsContent = opsBuf.str();

    assert(opsContent.find("def BT_PlanarDotOp") != std::string::npos);
    assert(opsContent.find("def BT_MatmulOp") != std::string::npos);
    assert(opsContent.find("def BT_BTHAOp") != std::string::npos);
    assert(opsContent.find("def BT_Base243UnpackOp") != std::string::npos);
    assert(opsContent.find("def BT_CSAReduceOp") != std::string::npos);
    assert(opsContent.find("def BT_CSAAccumOp") != std::string::npos);
    assert(opsContent.find("def BT_Select3Op") != std::string::npos);

    std::cout << "    -> PASSED: TableGen dialect and operation schemas validated.\n";
}

// ==============================================================================
// 3. Test Pass Pipelining
// ==============================================================================

void test_pass_pipelining() {
    std::cout << "  [bt-opt 3/6] Testing Pass Pipelining & Transformations...\n";

    ::bt::mlir::MLIRModule mod;
    mod.name = "bt_opt_pipeline_test";

    ::bt::mlir::MLIRPassManager pm;
    buildTernaryCanonicalPipeline(pm, 4);

    assert(pm.passes.size() == 5 && "Expected 5 passes in canonical pipeline!");

    bool success = pm.run(mod);
    assert(success && "Canonical pass pipeline failed to execute!");
    assert(mod.functions.size() == 5 && "Expected 5 lowered functions!");
    assert(mod.affineMaps.size() == 3 && "Expected 3 affine maps!");

    // Verify presence of all lowering functions
    std::vector<std::string> expectedFns = {
        "bt_gemm_planar_linalg",
        "bt_planar_dot_vector_simd",
        "bt_btha_lowering",
        "bt_unpack_base243_fast",
        "bt_csa_reduce_canonical"
    };

    for (const auto& name : expectedFns) {
        bool found = false;
        for (const auto& fn : mod.functions) {
            if (fn.name == name) {
                found = true;
                break;
            }
        }
        assert(found && "Expected lowered function not found in module!");
    }

    std::cout << "    -> PASSED: All 5 dialect lowerings executed into MLIR module.\n";
}

// ==============================================================================
// 4. Test Bit-Sliced Execution Correctness
// ==============================================================================

void test_bit_sliced_execution() {
    std::cout << "  [bt-opt 4/6] Testing Bit-Sliced Numerical Execution Invariants...\n";

    // 1. Test PlanarDotOp execute
    uint64_t ap = 0b0011ULL; // trits: +1, +1,  0,  0
    uint64_t an = 0b1100ULL; // trits: -1, -1, +1, +1 -> (indices 2,3 are -1)
    uint64_t bp = 0b0101ULL; // trits: +1,  0, +1,  0
    uint64_t bn = 0b1010ULL; // trits:  0, -1,  0, -1
    // products:
    // k=0: A=+1, B=+1 -> +1
    // k=1: A=+1, B=-1 -> -1
    // k=2: A=-1, B=+1 -> -1
    // k=3: A=-1, B=-1 -> +1
    // sum = 1 - 1 - 1 + 1 = 0
    int64_t dot = PlanarDotOp::execute(ap, an, bp, bn);
    assert(dot == 0 && "PlanarDotOp execution mismatch!");

    // 2. Test BTHAOp execute
    auto bthaRes = BTHAOp::execute(1, 0, 1, 0); // 1 + 1 = 2 = 3*(+1) + (-1) -> C=+1, S=-1
    assert(bthaRes.PC == 1 && bthaRes.NC == 0 && "Carry positive rail mismatch!");
    assert(bthaRes.PS == 0 && bthaRes.NS == 1 && "Sum negative rail mismatch!");

    // 3. Test CSAReduceOp execute
    // S = +1 (sp=1, sn=0), C = +1 (cp=1, cn=0)
    // S + 3*C = 1 + 3 = 4 = 3*(+1) + 1 = 9*0 + 3*1 + 1 -> trit word [1, 1] -> 1 + 3 = 4!
    auto csaRes = CSAReduceOp::execute(1, 0, 1, 0);
    // Position 0 = +1 (bit 0 of P), Position 1 = +1 (bit 1 of P)
    assert(csaRes.first == 0b11ULL && csaRes.second == 0 && "CSA reduce result mismatch!");

    // 4. Test Base243UnpackOp execute
    // 0 = (-1) + 3*(-1) + 9*(-1) + 27*(-1) + 81*(-1) + 121 = 0 in bias -> [-1, -1, -1, -1, -1]
    auto trits0 = Base243UnpackOp::execute(0);
    for (int i = 0; i < 5; ++i) {
        assert(trits0[i] == -1 && "Byte 0 should unpack to all -1 trits!");
    }
    // 121 represents all 0 trits: sum = 121
    auto trits121 = Base243UnpackOp::execute(121);
    for (int i = 0; i < 5; ++i) {
        assert(trits121[i] == 0 && "Byte 121 should unpack to all 0 trits!");
    }
    // 242 represents all +1 trits
    auto trits242 = Base243UnpackOp::execute(242);
    for (int i = 0; i < 5; ++i) {
        assert(trits242[i] == 1 && "Byte 242 should unpack to all +1 trits!");
    }

    // 5. Test Select3Op execute
    assert(Select3Op::execute(1, 100, 200, 300) == 100);
    assert(Select3Op::execute(0, 100, 200, 300) == 200);
    assert(Select3Op::execute(-1, 100, 200, 300) == 300);

    std::cout << "    -> PASSED: Bit-sliced arithmetic execution verified with zero errors.\n";
}

// ==============================================================================
// 5. Test IREE Compiler Plugin & Backend Dispatch
// ==============================================================================

void test_iree_plugin_initialization() {
    std::cout << "  [bt-opt 5/6] Testing IREE Compiler Plugin & Target Backends...\n";

    TernaryIREEPlugin plugin;
    assert(std::string(plugin.getPluginId()) == "ternary_iree");
    assert(std::string(plugin.getVersion()) == "1.0.0");
    assert(std::string(plugin.getDescription()).find("IREE Compiler Plugin") != std::string::npos);

    // Verify supported backends
    auto backends = plugin.getSupportedTargetBackends();
    assert(backends.size() == 4 && "Expected 4 target backends in IREE plugin!");
    assert(backends[0] == "vulkan-spirv");
    assert(backends[1] == "metal-msl");
    assert(backends[2] == "cuda-nvptx");
    assert(backends[3] == "llvm-cpu");

    // Test Vulkan backend emission
    auto vulkan = plugin.getBackend(TargetBackendKind::VulkanSPIRV);
    assert(vulkan != nullptr);
    std::string vulkanShader = vulkan->emitShaderKernel("bt.matmul", 32, 32, 128);
    assert(vulkanShader.find("OpBitCount") != std::string::npos || vulkanShader.find("bitCount") != std::string::npos);
    assert(vulkanShader.find("gl_GlobalInvocationID") != std::string::npos);

    // Test Metal backend emission
    auto metal = plugin.getBackend(TargetBackendKind::MetalMSL);
    assert(metal != nullptr);
    std::string metalShader = metal->emitShaderKernel("bt.matmul", 32, 32, 128);
    assert(metalShader.find("popcount") != std::string::npos);
    assert(metalShader.find("kernel void bt_iree_kernel") != std::string::npos);

    // Test CUDA backend emission
    auto cuda = plugin.getBackend(TargetBackendKind::CUDAPTX);
    assert(cuda != nullptr);
    std::string cudaShader = cuda->emitShaderKernel("bt.matmul", 32, 32, 128);
    assert(cudaShader.find("__popcll") != std::string::npos);

    // Test CPU backend emission
    auto cpu = plugin.getBackend(TargetBackendKind::CPULLVM);
    assert(cpu != nullptr);
    std::string cpuShader = cpu->emitShaderKernel("bt.matmul", 32, 32, 128);
    assert(cpuShader.find("@llvm.intr.ctpop.i64") != std::string::npos);

    // Test Flow dispatch bridging
    std::string flowModule = plugin.buildFlowDispatchModule("bt_gemm_tile", 64, 64, 256);
    assert(flowModule.find("flow.executable @dispatch_kernel") != std::string::npos);
    assert(flowModule.find("flow.dispatch.tensor") != std::string::npos);

    // Test Stream dispatch bridging
    std::string streamModule = plugin.buildStreamDispatchModule("bt_gemm_tile", 64, 64, 256);
    assert(streamModule.find("stream.cmd.execute") != std::string::npos);
    assert(streamModule.find("stream.cmd.dispatch") != std::string::npos);

    // Test C entrypoint
    assert(iree_compiler_register_plugin_ternary() == 0 && "C entrypoint registration failed!");

    std::cout << "    -> PASSED: IREE plugin initialized, Flow/Stream bridges and 4 backends verified.\n";
}

// ==============================================================================
// 6. Test bt-opt IR Emission & CLI Pipeline Emulation
// ==============================================================================

void test_bt_opt_emission() {
    std::cout << "  [bt-opt 6/6] Testing bt-opt IR Emission & CLI Pipeline Emulation...\n";

    // Test emission of canonical pipeline
    std::string fullIR = ::bt::mlir::MLIREmitter::emit_canonical_lowering_pipeline();
    assert(fullIR.find("module @ternary_compilation_pipeline") != std::string::npos);
    assert(fullIR.find("func.func @bt_gemm_planar_linalg") != std::string::npos);
    assert(fullIR.find("func.func @bt_planar_dot_vector_simd") != std::string::npos);
    assert(fullIR.find("func.func @bt_btha_lowering") != std::string::npos);
    assert(fullIR.find("func.func @bt_unpack_base243_fast") != std::string::npos);

    std::cout << "    -> PASSED: Valid MLIR IR generation and canonical pipeline verified.\n";
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  [LLVM Unit Test Layer 1] bt-opt MLIR Optimizer & IREE Plugin Suite\n";
    std::cout << "======================================================================\n";

    test_dialect_registration();
    test_tablegen_definitions();
    test_pass_pipelining();
    test_bit_sliced_execution();
    test_iree_plugin_initialization();
    test_bt_opt_emission();

    std::cout << "======================================================================\n";
    std::cout << "  -> PASSED: 100% of bt-opt & IREE Plugin Unit Tests Succeeded (0 Errors)!\n";
    std::cout << "======================================================================\n";
    return 0;
}
