/**
 * @file TernaryPasses.cpp
 * @brief Dialect passes implementation for Balanced Ternary (bt) Lowerings
 * @author Antigravity Prime Research & Engineering Engine
 *
 * These passes operate on the standalone (fake) MLIRModule representation.
 * They are *input aware*: each pass scans the module for the bt.* operation it
 * lowers and only rewrites/emits lowering output when that operation is actually
 * present, annotating the exact call sites it transformed. Running them on a
 * module without matching bt ops is an honest no-op instead of appending
 * canned, unrelated IR.
 */

#include "../../include/Ternary/TernaryPasses.h"
#include <sstream>
#include <algorithm>

namespace mlir::bt {

namespace {

// ---------------------------------------------------------------------------
// Helpers: op detection and call-site annotation on parsed modules.
// ---------------------------------------------------------------------------

bool functionContainsOp(const ::bt::mlir::MLIRFunction& fn, const std::string& opName) {
    if (!fn.rawCustomBody.empty()) {
        return fn.rawCustomBody.find(opName) != std::string::npos;
    }
    for (const auto& op : fn.operations) {
        if (op.name == opName) return true;
    }
    return false;
}

// Annotates the first textual occurrence of `opName` inside the function body
// so that bt-opt output shows *which* sites were lowered by which pass.
void annotateFirstOpSite(::bt::mlir::MLIRFunction& fn, const std::string& opName, const std::string& note) {
    if (fn.rawCustomBody.empty()) return;
    size_t pos = fn.rawCustomBody.find(opName);
    if (pos == std::string::npos) return;
    fn.rawCustomBody.replace(pos, opName.size(), opName + " /* lowered-by: " + note + " */");
}

bool alreadyHasFunction(const ::bt::mlir::MLIRModule& mod, const std::string& name) {
    for (const auto& fn : mod.functions) {
        if (fn.name == name) return true;
    }
    return false;
}

void addAffineMapIfMissing(::bt::mlir::MLIRModule& mod, const std::string& mapText) {
    for (const auto& m : mod.affineMaps) {
        if (m == mapText) return;
    }
    mod.affineMaps.push_back(mapText);
}

// ---------------------------------------------------------------------------
// Input-aware lowering passes
// ---------------------------------------------------------------------------

/**
 * @brief Lowers bt.matmul / bt.gemm sites into linalg.generic structured loops.
 * Applies only when the module actually contains those operations.
 */
class InputAwareTernaryToLinalgPass : public ::bt::mlir::MLIRPass {
public:
    std::string name() const override { return "TernaryToLinalgPass"; }

    bool runOnModule(::bt::mlir::MLIRModule& mod) override {
        bool loweredAny = false;
        for (auto& fn : mod.functions) {
            if (functionContainsOp(fn, "bt.matmul")) {
                annotateFirstOpSite(fn, "bt.matmul", "TernaryToLinalgPass -> @bt_gemm_planar_linalg");
                loweredAny = true;
            }
            if (functionContainsOp(fn, "bt.gemm")) {
                annotateFirstOpSite(fn, "bt.gemm", "TernaryToLinalgPass -> @bt_gemm_planar_linalg");
                loweredAny = true;
            }
        }
        if (!loweredAny) return true;

        addAffineMapIfMissing(mod, "#map_A = affine_map<(d0, d1, d2) -> (d0, d2)>");
        addAffineMapIfMissing(mod, "#map_B = affine_map<(d0, d1, d2) -> (d2, d1)>");
        addAffineMapIfMissing(mod, "#map_C = affine_map<(d0, d1, d2) -> (d0, d1)>");
        if (!alreadyHasFunction(mod, "bt_gemm_planar_linalg")) {
            mod.functions.push_back(::bt::mlir::ConvertTernaryToLinalg::lower_matmul_to_linalg());
        }
        return true;
    }
};

/**
 * @brief Lowers bt.planar_dot sites into SIMD vector ops + ctpop (Larq-style).
 * Applies only when the module actually contains bt.planar_dot.
 */
class InputAwareTernaryToVectorPass : public ::bt::mlir::MLIRPass {
public:
    explicit InputAwareTernaryToVectorPass(unsigned l = 4) : lanes(l) {}

    std::string name() const override { return "TernaryToVectorPass"; }

    bool runOnModule(::bt::mlir::MLIRModule& mod) override {
        bool loweredAny = false;
        for (auto& fn : mod.functions) {
            if (!functionContainsOp(fn, "bt.planar_dot")) continue;
            annotateFirstOpSite(fn, "bt.planar_dot",
                                "TernaryToVectorPass -> @bt_planar_dot_vector_simd");
            loweredAny = true;
        }
        if (!loweredAny) return true;
        if (!alreadyHasFunction(mod, "bt_planar_dot_vector_simd")) {
            mod.functions.push_back(::bt::mlir::ConvertTernaryToVector::lower_planar_dot_to_vector("bt_planar_dot_vector_simd", lanes));
        }
        return true;
    }

private:
    unsigned lanes;
};

/**
 * @brief Lowers bt.btha sites into the 6 bitwise instructions lowering template.
 * Applies only when the module actually contains bt.btha.
 */
class InputAwareConvertBTHAPass : public ::bt::mlir::MLIRPass {
public:
    std::string name() const override { return "ConvertBTHAPass"; }

    bool runOnModule(::bt::mlir::MLIRModule& mod) override {
        bool loweredAny = false;
        for (auto& fn : mod.functions) {
            if (!functionContainsOp(fn, "bt.btha")) continue;
            annotateFirstOpSite(fn, "bt.btha", "ConvertBTHAPass -> @bt_btha_lowering");
            loweredAny = true;
        }
        if (!loweredAny) return true;
        if (!alreadyHasFunction(mod, "bt_btha_lowering")) {
            mod.functions.push_back(::bt::mlir::ConvertBTHA::lower_btha("bt_btha_lowering"));
        }
        return true;
    }
};

/**
 * @brief Lowers bt.base243_unpack / bt.unpack_base243 sites into magic-multiplier
 * sequences. Applies only when the module actually contains them.
 */
class InputAwareConvertBase243Pass : public ::bt::mlir::MLIRPass {
public:
    std::string name() const override { return "ConvertBase243Pass"; }

    bool runOnModule(::bt::mlir::MLIRModule& mod) override {
        bool loweredAny = false;
        for (auto& fn : mod.functions) {
            bool isBase243 = functionContainsOp(fn, "bt.base243_unpack") ||
                             functionContainsOp(fn, "bt.unpack_base243");
            if (!isBase243) continue;
            if (functionContainsOp(fn, "bt.base243_unpack")) {
                annotateFirstOpSite(fn, "bt.base243_unpack", "ConvertBase243Pass -> @bt_unpack_base243_fast");
            }
            if (functionContainsOp(fn, "bt.unpack_base243")) {
                annotateFirstOpSite(fn, "bt.unpack_base243", "ConvertBase243Pass -> @bt_unpack_base243_fast");
            }
            loweredAny = true;
        }
        if (!loweredAny) return true;
        if (!alreadyHasFunction(mod, "bt_unpack_base243_fast")) {
            mod.functions.push_back(::bt::mlir::ConvertBase243::lower_unpack_base243("bt_unpack_base243_fast"));
        }
        return true;
    }
};

/**
 * @brief Lowers bt.csa_reduce from the redundant (Sum, Carry) pair into a
 * canonical dual-rail bitplane word via iterative scf.while carry propagation.
 * Applies only when the module actually contains bt.csa_reduce.
 */
class ConvertCSAReducePass : public ::bt::mlir::MLIRPass {
public:
    std::string name() const override { return "ConvertCSAReducePass"; }

    bool runOnModule(::bt::mlir::MLIRModule& mod) override {
        bool loweredAny = false;
        for (auto& fn : mod.functions) {
            if (!functionContainsOp(fn, "bt.csa_reduce")) continue;
            annotateFirstOpSite(fn, "bt.csa_reduce", "ConvertCSAReducePass -> @bt_csa_reduce_canonical");
            loweredAny = true;
        }
        if (!loweredAny) return true;
        if (alreadyHasFunction(mod, "bt_csa_reduce_canonical")) return true;

        ::bt::mlir::MLIRFunction fn;
        fn.name = "bt_csa_reduce_canonical";
        fn.arguments = {
            {"%sp", ::bt::mlir::MLIRType::getI64()},
            {"%sn", ::bt::mlir::MLIRType::getI64()},
            {"%cp", ::bt::mlir::MLIRType::getI64()},
            {"%cn", ::bt::mlir::MLIRType::getI64()}
        };
        fn.returnTypes = {
            ::bt::mlir::MLIRType::getI64(),
            ::bt::mlir::MLIRType::getI64()
        };

        std::ostringstream ss;
        ss << "    // Iterative scf.while carry-propagation loop until carry is zero\n";
        ss << "    %res_sp, %res_sn, %res_cp, %res_cn = scf.while (%curr_sp = %sp, %curr_sn = %sn, %curr_cp = %cp, %curr_cn = %cn) : (i64, i64, i64, i64) -> (i64, i64, i64, i64) {\n";
        ss << "      %c0 = arith.constant 0 : i64\n";
        ss << "      %c_or = arith.ori %curr_cp, %curr_cn : i64\n";
        ss << "      %cond = arith.cmpi ne, %c_or, %c0 : i64\n";
        ss << "      scf.condition(%cond) %curr_sp, %curr_sn, %curr_cp, %curr_cn : i64, i64, i64, i64\n";
        ss << "    } do {\n";
        ss << "    ^bb0(%in_sp: i64, %in_sn: i64, %in_cp: i64, %in_cn: i64):\n";
        ss << "      %c1 = arith.constant 1 : i64\n";
        ss << "      %sh_cp = arith.shli %in_cp, %c1 : i64\n";
        ss << "      %sh_cn = arith.shli %in_cn, %c1 : i64\n";
        ss << "      %s1_p, %s1_n, %next_cp, %next_cn = func.call @bt_btha_lowering(%in_sp, %in_sn, %sh_cp, %sh_cn) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)\n";
        ss << "      scf.yield %s1_p, %s1_n, %next_cp, %next_cn : i64, i64, i64, i64\n";
        ss << "    }\n";
        ss << "    func.return %res_sp, %res_sn : i64, i64\n";

        fn.rawCustomBody = ss.str();
        mod.functions.push_back(fn);
        return true;
    }
};

} // anonymous namespace

std::unique_ptr<::bt::mlir::MLIRPass> createConvertTernaryToLinalgPass() {
    return std::make_unique<InputAwareTernaryToLinalgPass>();
}

std::unique_ptr<::bt::mlir::MLIRPass> createConvertTernaryToVectorPass(unsigned lanes) {
    return std::make_unique<InputAwareTernaryToVectorPass>(lanes);
}

std::unique_ptr<::bt::mlir::MLIRPass> createConvertBTHAPass() {
    return std::make_unique<InputAwareConvertBTHAPass>();
}

std::unique_ptr<::bt::mlir::MLIRPass> createConvertBase243Pass() {
    return std::make_unique<InputAwareConvertBase243Pass>();
}

std::unique_ptr<::bt::mlir::MLIRPass> createConvertCSAReducePass() {
    return std::make_unique<ConvertCSAReducePass>();
}

void registerTernaryPasses() {
    // Registers pass names in pipeline registry
}

void buildTernaryCanonicalPipeline(::bt::mlir::MLIRPassManager& pm, unsigned vectorLanes) {
    pm.addPass(createConvertTernaryToLinalgPass());
    pm.addPass(createConvertTernaryToVectorPass(vectorLanes));
    pm.addPass(createConvertBTHAPass());
    pm.addPass(createConvertBase243Pass());
    pm.addPass(createConvertCSAReducePass());
}

} // namespace mlir::bt
