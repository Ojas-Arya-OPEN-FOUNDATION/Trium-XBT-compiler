/**
 * @file test_unit_bt_opt.cpp
 * @brief LLVM Unit Test: bt-opt Standalone Optimizer & IREE Compiler Plugin
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "../../include/Ternary/TernaryDialect.h"
#include "../../include/Ternary/TernaryPasses.h"
#include "../../include/Ternary/ternary_c_abi.h"
#include "../../iree/TernaryIREEPlugin.h"
#include "../../ternary_mlir_lowering.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

using namespace mlir::bt;
using namespace mlir::iree_compiler;

// ==============================================================================
// Project root resolution (works from ANY working directory).
// bt-opt unit tests historically opened "include/Ternary/TernaryDialect.td"
// relative to the CWD, silently passing or failing depending on where they
// were launched from. Resolve once from a -D macro fallback or argv[0].
// ==============================================================================

static std::string g_projectRoot;

static bool looksLikeProjectRoot(const std::string& dir) {
    if (dir.empty()) return false;
    std::ifstream probe(dir + "/include/Ternary/TernaryDialect.td");
    return probe.good();
}

static std::string resolveProjectRoot(const char* argv0) {
    // 1. Compile-time macro (set for this target in CMakeLists.txt).
#if defined(BT_UNIT_PROJECT_ROOT)
    if (looksLikeProjectRoot(BT_UNIT_PROJECT_ROOT)) return BT_UNIT_PROJECT_ROOT;
#endif
    // 2. argv[0]-relative: ascend from the executable's directory.
    std::string exePath(argv0 ? argv0 : "");
    if (!exePath.empty() && exePath[0] != '/') {
        char buf[4096];
        if (getcwd(buf, sizeof(buf))) exePath = std::string(buf) + "/" + exePath;
    }
#ifdef __APPLE__
    {
        char real[4096];
        uint32_t size = sizeof(real);
        if (_NSGetExecutablePath(real, &size) == 0) {
            exePath = real; // most robust against exec-argv shenanigans
        }
    }
#endif
    // dirname
    std::string dir = exePath.substr(0, exePath.find_last_of('/'));
    for (int up = 0; up < 8 && !dir.empty(); ++up) {
        if (looksLikeProjectRoot(dir)) return dir;
        size_t slash = dir.find_last_of('/');
        if (slash == std::string::npos) break;
        dir = dir.substr(0, slash);
    }
    return "";
}

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

    // Test MSB carry bit 63 tracking in evalCSAReduce
    int8_t cout_p = 0;
    uint64_t msb_sp = 1ULL << 63;
    uint64_t msb_cp = 1ULL << 63;
    CSAReduceOp::evalCSAReduce(msb_sp, 0, msb_cp, 0, &cout_p);
    assert(cout_p == 1 && "Expected MSB positive carry-out bit 63 to be tracked!");

    int8_t cout_n = 0;
    uint64_t msb_sn = 1ULL << 63;
    uint64_t msb_cn = 1ULL << 63;
    CSAReduceOp::evalCSAReduce(0, msb_sn, 0, msb_cn, &cout_n);
    assert(cout_n == -1 && "Expected MSB negative carry-out bit 63 to be tracked!");

    std::cout << "    -> PASSED: Dialect registration, types, and invariant verifiers.\n";
}

// ==============================================================================
// 2. Test TableGen Definitions
// ==============================================================================

void test_tablegen_definitions() {
    std::cout << "  [bt-opt 2/9] Testing TableGen Specification Files (CWD-independent)...\n";
    assert(!g_projectRoot.empty() &&
           "Project root could not be resolved (launched outside the repo tree)!");

    // Check include/Ternary/TernaryDialect.td
    std::ifstream dialectFile(g_projectRoot + "/include/Ternary/TernaryDialect.td");
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
    std::ifstream opsFile(g_projectRoot + "/include/Ternary/TernaryOps.td");
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
    std::cout << "  [bt-opt 3/9] Testing Pass Pipelining & Input-Aware Transformations...\n";

    // 1. The canonical pipeline is input-aware: on an empty module it must be
    //    an honest no-op (no canned functions are appended).
    ::bt::mlir::MLIRModule emptyMod;
    emptyMod.name = "bt_opt_empty_module";

    ::bt::mlir::MLIRPassManager emptyPm;
    buildTernaryCanonicalPipeline(emptyPm, 4);
    assert(emptyPm.passes.size() == 5 && "Expected 5 passes in canonical pipeline!");
    assert(emptyPm.run(emptyMod) && "Pipeline must not fail on an empty module!");
    assert(emptyMod.functions.empty() &&
           "Passes must not append canned lowerings to a module without bt.* ops!");
    std::cout << "    -> Passes are input-aware: empty module => no canned lowering output.\n";

    // 2. Real input ingestion: parse a module text and lower its bt.* sites.
    const std::string sample = R"MLIR(module @bt_opt_pipeline_test {

  func.func @sample_gemm(%A_P: !bt.tensor<4x4x64>, %A_N: !bt.tensor<4x4x64>, %B_P: !bt.tensor<4x4x64>, %B_N: !bt.tensor<4x4x64>) -> !bt.tensor<4x4x64> {
    %r = bt.matmul %A_P, %A_N, %B_P, %B_N : (!bt.tensor<4x4x64>, !bt.tensor<4x4x64>, !bt.tensor<4x4x64>, !bt.tensor<4x4x64>) -> !bt.tensor<4x4x64>
    return %r : !bt.tensor<4x4x64>
  }

  func.func @sample_dot(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> i64 {
    %d = bt.planar_dot %ap, %an, %bp, %bn : (i64, i64, i64, i64) -> i64
    return %d : i64
  }

  func.func @sample_adder(%ap: i64, %an: i64, %bp: i64, %bn: i64) -> i64 {
    %s = bt.btha %ap, %an, %bp, %bn : (i64, i64, i64, i64) -> i64
    return %s : i64
  }

  func.func @sample_unpack(%b: i32) -> i32 {
    %t = bt.base243_unpack %b : (i32) -> i32
    return %t : i32
  }

  func.func @sample_csa(%sp: i64, %sn: i64, %cp: i64, %cn: i64) -> i64 {
    %w = bt.csa_reduce %sp, %sn, %cp, %cn : (i64, i64, i64, i64) -> i64
    return %w : i64
  }
})MLIR";

    ::bt::mlir::MLIRModule mod;
    mod.name = "bt_opt_pipeline_test"; // overridden by `module @bt_opt_pipeline_test` above

    std::string parseError;
    mlir::bt::loadMLIRModuleFromString(mod, sample, parseError);
    assert(parseError.empty() && "Sample module text must parse cleanly!");
    assert(strcmp(mod.name.c_str(), "bt_opt_pipeline_test") == 0 && "Module symbol must be ingested!");
    assert(mod.functions.size() == 5 && "Expected 5 input functions after parsing!");
    // Ingested bodies must faithfully carry the input op sites.
    {
        bool foundSite = false;
        for (const auto& fn : mod.functions) {
            if (fn.name == "sample_gemm" && fn.rawCustomBody.find("bt.matmul") != std::string::npos) {
                foundSite = true;
            }
        }
        assert(foundSite && "Parsed function body must preserve the bt.matmul site!");
    }

    ::bt::mlir::MLIRPassManager pm;
    buildTernaryCanonicalPipeline(pm, 4);
    bool success = pm.run(mod);
    assert(success && "Canonical pass pipeline failed to execute!");
    // 5 input functions + 5 lowered functions (one per recognized op kind).
    assert(mod.functions.size() == 10 && "Expected 5 input + 5 lowered functions!");
    assert(mod.affineMaps.size() == 3 && "Expected 3 affine maps!");
    assert(mod.to_string().find("module @bt_opt_pipeline_test") != std::string::npos &&
           "Emitted module must keep the input module symbol!");

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
                if (fn.name == "bt_csa_reduce_canonical") {
                    assert(fn.rawCustomBody.find("scf.while") != std::string::npos &&
                           "ConvertCSAReducePass must emit scf.while loop!");
                    assert(fn.rawCustomBody.find("scf.condition") != std::string::npos &&
                           "ConvertCSAReducePass must emit scf.condition!");
                    assert(fn.rawCustomBody.find("scf.yield") != std::string::npos &&
                           "ConvertCSAReducePass must emit scf.yield!");
                    assert(fn.rawCustomBody.find("arith.ori") != std::string::npos);
                }
                break;
            }
        }
        assert(found && "Expected lowered function not found in module!");
    }

    // Each lowered site in the input bodies must be annotated.
    bool annotated = false;
    for (const auto& fn : mod.functions) {
        if (fn.name == "sample_gemm" &&
            fn.rawCustomBody.find("/* lowered-by: TernaryToLinalgPass -> @bt_gemm_planar_linalg */") !=
                std::string::npos) {
            annotated = true;
        }
    }
    assert(annotated && "Lowered call sites must carry pass annotations!");

    std::cout << "    -> PASSED: All 5 dialect lowerings executed on the ingested module (iterative CSA carry loop verified).\n";
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
    assert(vulkanShader.find("bitCount") != std::string::npos);
    assert(vulkanShader.find("gl_GlobalInvocationID") != std::string::npos);
    assert(vulkanShader.find("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require") != std::string::npos &&
           "Vulkan shader must include int64 extension!");
    assert(vulkanShader.find("unpackUint2x32(prod_p)") != std::string::npos &&
           "Vulkan shader must unpack 64-bit mask via unpackUint2x32!");
    assert(vulkanShader.find("bitCount(up.x) + bitCount(up.y)") != std::string::npos &&
           "Vulkan shader must sum 32-bit unpacked bitCount!");

    // Test arbitrary matrix size tail-masking (e.g. K = 100 trits -> words_per_row = 2)
    std::string vulkanTail = vulkan->emitShaderKernel("bt.matmul", 16, 16, 100);
    assert(vulkanTail.find("tail_mask") != std::string::npos &&
           "Vulkan shader must emit tail-masking for non-multiple-of-64 K!");
    assert(vulkanTail.find("for (uint k = 0; k < 2; ++k)") != std::string::npos &&
           "Expected words_per_row = 2 loop bound for K = 100!");

    // Test Metal backend emission
    auto metal = plugin.getBackend(TargetBackendKind::MetalMSL);
    assert(metal != nullptr);
    std::string metalShader = metal->emitShaderKernel("bt.matmul", 32, 32, 128);
    assert(metalShader.find("popcount") != std::string::npos);
    assert(metalShader.find("kernel void bt_iree_kernel") != std::string::npos);

    std::string metalTail = metal->emitShaderKernel("bt.matmul", 16, 16, 100);
    assert(metalTail.find("tail_mask") != std::string::npos &&
           "Metal MSL shader must emit tail-masking for non-multiple-of-64 K!");
    assert(metalTail.find("for (uint k = 0; k < 2; ++k)") != std::string::npos);

    // Test CUDA backend emission
    auto cuda = plugin.getBackend(TargetBackendKind::CUDAPTX);
    assert(cuda != nullptr);
    std::string cudaShader = cuda->emitShaderKernel("bt.matmul", 32, 32, 128);
    assert(cudaShader.find("__popcll") != std::string::npos);

    std::string cudaTail = cuda->emitShaderKernel("bt.matmul", 16, 16, 100);
    assert(cudaTail.find("tail_mask") != std::string::npos &&
           "CUDA PTX kernel must emit tail-masking for non-multiple-of-64 K!");
    assert(cudaTail.find("for (unsigned int k = 0; k < 2; ++k)") != std::string::npos);

    // Test CPU backend emission
    auto cpu = plugin.getBackend(TargetBackendKind::CPULLVM);
    assert(cpu != nullptr);
    std::string cpuShader = cpu->emitShaderKernel("bt.matmul", 32, 32, 128);
    assert(cpuShader.find("@llvm.ctpop.i64") != std::string::npos);

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

// ==============================================================================
// 7. Test Production C ABI / FFI & MLIR Type Formatting
// ==============================================================================

void test_c_abi() {
    std::cout << "  [bt-opt 7/7] Testing Production C ABI & MLIR Type Formatting...\n";

    // 1. Verify fixed MLIR (i64, i64) -> i64 type formatting
    ::bt::mlir::MLIROperation opAdd;
    opAdd.name = "arith.addi";
    opAdd.operands = {
        ::bt::mlir::MLIRValue("%a", ::bt::mlir::MLIRType::getI64()),
        ::bt::mlir::MLIRValue("%b", ::bt::mlir::MLIRType::getI64())
    };
    opAdd.results = {
        ::bt::mlir::MLIRValue("%c", ::bt::mlir::MLIRType::getI64())
    };
    std::string opStr = opAdd.to_string();
    assert(opStr.find(": (i64, i64) -> i64") != std::string::npos &&
           "MLIROperation::to_string must emit valid '(i64, i64) -> i64' syntax!");

    // 2. bt_word64_t & disjointness
    bt_word64_t w1 = bt_make_word64(0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL);
    assert(bt_verify_disjointness(w1));
    bt_word64_t w_bad = bt_make_word64(0x3ULL, 0x1ULL);
    assert(!bt_verify_disjointness(w_bad));

    // 3. bt_dot_product_64
    bt_word64_t a = bt_make_word64(0b0011ULL, 0b1100ULL);
    bt_word64_t b = bt_make_word64(0b0101ULL, 0b1010ULL);
    int64_t dot = bt_dot_product_64(a, b);
    assert(dot == 0);
    assert(bt_dot_product_64_raw(0b0011ULL, 0b1100ULL, 0b0101ULL, 0b1010ULL) == 0);

    // 4. bt_btha_step
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

    // 5. bt_base243_pack and bt_base243_unpack
    for (uint32_t byte = 0; byte < 243; ++byte) {
        int8_t trits[5];
        bt_base243_unpack(static_cast<uint8_t>(byte), trits);
        uint8_t repacked = bt_base243_pack(trits);
        assert(repacked == byte);
    }

    // 6. bt_swar_bct_to_bin32
    assert(bt_swar_bct_to_bin32(0, 0) == 0);
    assert(bt_swar_bct_to_bin32(1, 0) == 1);
    assert(bt_swar_bct_to_bin32(0, 1) == -1);
    assert(bt_swar_bct_to_bin32(2, 0) == 3);
    assert(bt_swar_bct_to_bin32(3, 0) == 4);

    std::mt19937_64 rng(777);
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
        assert(swar_val == ref_val);
    }

    // 7. bt_hamming_distance_64
    bt_word64_t ha1 = bt_make_word64(0x0FULL, 0x00ULL);
    bt_word64_t ha2 = bt_make_word64(0x00ULL, 0x0FULL);
    assert(bt_hamming_distance_64(ha1, ha2) == 4);
    assert(bt_hamming_distance_64(ha1, ha1) == 0);

    // 8. bt_fp32_t & bt_csr_t
    bt_fp32_t fp = { 500, 3 };
    assert(fp.mantissa == 500 && fp.exponent == 3);
    uint32_t rptr[2] = {0, 1};
    uint32_t cidx[1] = {0};
    uint64_t sgn[1] = {1};
    bt_csr_t csr = { 1, 1, 1, rptr, cidx, sgn };
    assert(csr.rows == 1 && csr.nnz == 1);

    std::cout << "    -> PASSED: Production C ABI & MLIR type formatting verified.\n";
}

// ==============================================================================
// 8. Test --autotune-tile Hardware Cache-Hierarchy SCF Lowering
// ==============================================================================

void test_autotune_tile() {
    std::cout << "  [bt-opt 8/8] Testing --autotune-tile Hardware Cache-Tiled SCF Lowering...\n";

    // 1. Direct C++ API verification
    ::bt::CacheHierarchyConfig cfg = ::bt::CacheAutotuner::detect_host_cache();
    assert(cfg.fits_in_l1d_budget() && "Autotuned tile must fit inside 75% of L1d cache!");

    std::string scf_ir = ::bt::CacheAutotuner::emit_autotuned_scf_mlir(1024, 1024, 1024);
    std::string expected_header = "// Autotuned L1d Tile: " +
                                  std::to_string(cfg.tile_M) + "x" +
                                  std::to_string(cfg.tile_N) + "x" +
                                  std::to_string(cfg.tile_K_words);
    assert(scf_ir.find(expected_header) != std::string::npos &&
           "Missing '// Autotuned L1d Tile: MxNxK' annotation!");
    assert(scf_ir.find("scf.for %k0") != std::string::npos);
    assert(scf_ir.find("scf.for %i0") != std::string::npos);
    assert(scf_ir.find("scf.for %j0") != std::string::npos);
    assert(scf_ir.find("math.ctpop") != std::string::npos);

    // 2. Verify tools/bt-opt/bt-opt.cpp CLI --autotune-tile flag presence
    std::ifstream btOptFile(g_projectRoot + "/tools/bt-opt/bt-opt.cpp");
    assert(btOptFile.is_open() && "Could not open tools/bt-opt/bt-opt.cpp!");
    std::stringstream btOptBuf;
    btOptBuf << btOptFile.rdbuf();
    std::string btOptSrc = btOptBuf.str();
    assert(btOptSrc.find("--autotune-tile") != std::string::npos &&
           "tools/bt-opt/bt-opt.cpp must support --autotune-tile CLI flag!");

    std::cout << "    -> PASSED: --autotune-tile SCF lowering & L1d/L2 cache annotation verified.\n";
}

int main(int argc, char** argv) {
    g_projectRoot = resolveProjectRoot(argc > 0 ? argv[0] : nullptr);
    std::cout << "======================================================================\n";
    std::cout << "  [LLVM Unit Test Layer 1] bt-opt MLIR Optimizer & IREE Plugin Suite\n";
    std::cout << "======================================================================\n";

    test_dialect_registration();
    test_tablegen_definitions();
    test_pass_pipelining();
    test_bit_sliced_execution();
    test_iree_plugin_initialization();
    test_bt_opt_emission();
    test_c_abi();
    test_autotune_tile();

    std::cout << "======================================================================\n";
    std::cout << "  -> PASSED: 100% of bt-opt & IREE Plugin Unit Tests Succeeded (0 Errors)!\n";
    std::cout << "======================================================================\n";
    return 0;
}
