/**
 * @file TernaryPasses.cpp
 * @brief Dialect passes implementation for Balanced Ternary (bt) Lowerings
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "../../include/Ternary/TernaryPasses.h"
#include <sstream>

namespace mlir::bt {

class ConvertCSAReducePass : public ::bt::mlir::MLIRPass {
public:
    std::string name() const override { return "ConvertCSAReducePass"; }

    bool runOnModule(::bt::mlir::MLIRModule& mod) override {
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
        ss << "    // Shift carry plane by 1 position (multiplication by 3)\n";
        ss << "    %c1 = arith.constant 1 : i64\n";
        ss << "    %sh_cp = arith.shli %cp, %c1 : i64\n";
        ss << "    %sh_cn = arith.shli %cn, %c1 : i64\n";
        ss << "    // BTHA addition of Sum and Shifted Carry\n";
        ss << "    %s1, %s1_n, %c1_p, %c1_n = func.call @bt_btha_lowering(%sp, %sn, %sh_cp, %sh_cn) : (i64, i64, i64, i64) -> (i64, i64, i64, i64)\n";
        ss << "    func.return %s1, %s1_n : i64, i64\n";

        fn.rawCustomBody = ss.str();
        mod.functions.push_back(fn);
        return true;
    }
};

std::unique_ptr<::bt::mlir::MLIRPass> createConvertTernaryToLinalgPass() {
    return std::make_unique<::bt::mlir::TernaryToLinalgPass>();
}

std::unique_ptr<::bt::mlir::MLIRPass> createConvertTernaryToVectorPass(unsigned lanes) {
    return std::make_unique<::bt::mlir::TernaryToVectorPass>(lanes);
}

std::unique_ptr<::bt::mlir::MLIRPass> createConvertBTHAPass() {
    return std::make_unique<::bt::mlir::ConvertBTHAPass>();
}

std::unique_ptr<::bt::mlir::MLIRPass> createConvertBase243Pass() {
    return std::make_unique<::bt::mlir::ConvertBase243Pass>();
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
