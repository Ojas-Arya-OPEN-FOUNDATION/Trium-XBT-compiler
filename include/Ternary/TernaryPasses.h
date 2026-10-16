/**
 * @file TernaryPasses.h
 * @brief Dialect passes for Balanced Ternary (bt) Lowerings
 * @author Antigravity Prime Research & Engineering Engine
 */

#ifndef TERNARY_PASSES_H
#define TERNARY_PASSES_H

#include "../../ternary_mlir_lowering.hpp"
#include <memory>
#include <string>

namespace mlir::bt {

/**
 * @brief Creates a pass to lower high-level bt tensor operations (bt.matmul / bt.gemm)
 * into linalg.generic structured iteration spaces.
 */
std::unique_ptr<::bt::mlir::MLIRPass> createConvertTernaryToLinalgPass();

/**
 * @brief Creates a pass to lower bt.planar_dot into SIMD vector operations
 * and llvm.intr.ctpop intrinsics (Larq-style bit-sliced engine).
 */
std::unique_ptr<::bt::mlir::MLIRPass> createConvertTernaryToVectorPass(unsigned lanes = 4);

/**
 * @brief Creates a pass to lower bt.btha into 6 bitwise instructions (andi, ori, xori)
 * with zero carry rippling.
 */
std::unique_ptr<::bt::mlir::MLIRPass> createConvertBTHAPass();

/**
 * @brief Creates a pass to lower bt.base243_unpack into division-free reciprocal
 * magic multiplier sequences.
 */
std::unique_ptr<::bt::mlir::MLIRPass> createConvertBase243Pass();

/**
 * @brief Creates a pass to lower bt.csa_reduce from redundant (Sum, Carry) pair
 * into a canonical dual-rail bitplane word.
 */
std::unique_ptr<::bt::mlir::MLIRPass> createConvertCSAReducePass();

/**
 * @brief Registers all Ternary dialect passes into the global registry.
 */
void registerTernaryPasses();

/**
 * @brief Constructs the standard canonical balanced ternary optimization pipeline.
 */
void buildTernaryCanonicalPipeline(::bt::mlir::MLIRPassManager& pm, unsigned vectorLanes = 4);

} // namespace mlir::bt

#endif // TERNARY_PASSES_H
