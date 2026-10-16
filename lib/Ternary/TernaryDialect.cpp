/**
 * @file TernaryDialect.cpp
 * @brief MLIR Dialect implementation for the Balanced Ternary (bt) Dialect
 * @author Antigravity Prime Research & Engineering Engine
 */

#include "../../include/Ternary/TernaryDialect.h"
#include <iostream>
#include <unordered_set>

namespace mlir::bt {

static bool g_ternary_dialect_registered = false;
static std::unordered_set<std::string> g_registered_ops;
static std::unordered_set<std::string> g_registered_types;

void TernaryDialect::initialize() {
    // 1. Register Dialect Types
    g_registered_types.insert("!bt.trit");
    g_registered_types.insert("!bt.dualrail");
    g_registered_types.insert("!bt.csa_pair");
    g_registered_types.insert("!bt.tensor");

    // 2. Register Dialect Operations
    g_registered_ops.insert(PlanarDotOp::kOpName);
    g_registered_ops.insert(MatmulOp::kOpName);
    g_registered_ops.insert("bt.gemm");
    g_registered_ops.insert(BTHAOp::kOpName);
    g_registered_ops.insert("bt.btfa");
    g_registered_ops.insert(Base243UnpackOp::kOpName);
    g_registered_ops.insert("bt.unpack_base243");
    g_registered_ops.insert(CSAReduceOp::kOpName);
    g_registered_ops.insert(CSAAccumOp::kOpName);
    g_registered_ops.insert(Select3Op::kOpName);
    g_registered_ops.insert("bt.pack_base243");

    g_ternary_dialect_registered = true;
}

std::pair<uint64_t, uint64_t> CSAReduceOp::evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    ::bt::TritWord64 curr_S(sp, sn);
    ::bt::TritWord64 curr_C(cp, cn);
    int8_t accumulated_cout = 0;
    size_t iters = 0;
    while ((curr_C.P != 0 || curr_C.N != 0) && iters < 64) {
        // Track MSB carry bit 63 before shifting left
        if ((curr_C.P >> 63) & 1ULL) {
            accumulated_cout += 1;
        } else if ((curr_C.N >> 63) & 1ULL) {
            accumulated_cout -= 1;
        }
        uint64_t sh_P = curr_C.P << 1;
        uint64_t sh_N = curr_C.N << 1;
        auto ha = ::bt::BTHA(curr_S, ::bt::TritWord64(sh_P, sh_N));
        curr_S = ha.Sum;
        curr_C = ha.Carry;
        iters++;
    }
    if (carry_out) {
        *carry_out = accumulated_cout;
    }
    return {curr_S.P, curr_S.N};
}

std::pair<uint64_t, uint64_t> CSAReduceOp::execute(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    return evalCSAReduce(sp, sn, cp, cn, carry_out);
}

std::pair<uint64_t, uint64_t> evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    return CSAReduceOp::evalCSAReduce(sp, sn, cp, cn, carry_out);
}

std::pair<uint64_t, uint64_t> TernaryDialect::evalCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, int8_t* carry_out) {
    return CSAReduceOp::evalCSAReduce(sp, sn, cp, cn, carry_out);
}

bool TernaryDialect::verifyCSAReduce(uint64_t sp, uint64_t sn, uint64_t cp, uint64_t cn, uint64_t& rp, uint64_t& rn) {
    if (!verifyDualRailDisjointness(sp, sn) || !verifyDualRailDisjointness(cp, cn)) return false;
    auto res = CSAReduceOp::execute(sp, sn, cp, cn);
    rp = res.first;
    rn = res.second;
    return verifyDualRailDisjointness(rp, rn);
}

void registerTernaryDialect(::mlir::DialectRegistry &registry) {
    registry.insert("bt");
    TernaryDialect dialect;
    dialect.initialize();
}

void registerTernaryDialect() {
    TernaryDialect dialect;
    dialect.initialize();
}

bool isTernaryDialectRegistered() {
    return g_ternary_dialect_registered;
}

} // namespace mlir::bt
