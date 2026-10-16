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
