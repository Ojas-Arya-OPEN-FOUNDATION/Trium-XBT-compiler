/**
 * @file test_unit_memory.cpp
 * @brief LLVM Official Layer 1: Unit Tests for Balanced Ternary Memory Arena & Page Reclaim
 */

#include "../../ternary_memory_arena.hpp"
#include <iostream>
#include <cassert>

using namespace bt;

void test_dual_rail_planes() {
    DualRailPlanes planes(16);
    assert(planes.num_words() == 16);
    assert(planes.P() != nullptr);
    assert(planes.N() != nullptr);
    planes.P()[0] = 0xAA;
    planes.N()[0] = 0x55;
    assert((planes.P()[0] & planes.N()[0]) == 0);
}

void test_hierarchical_zero_matrix() {
    HierarchicalZeroMatrix mat(512, 512);
    auto initial_stats = mat.stats();
    assert(initial_stats.materialized_pages == 0);

    // Write a trit
    mat.set(10, 20, Trit::Pos);
    auto write_stats = mat.stats();
    assert(write_stats.materialized_pages == 1);
    assert(mat.get(10, 20) == Trit::Pos);
    assert(mat.get(10, 21) == Trit::Zero);

    // Reclaim should retain non-zero page
    size_t reclaimed = mat.reclaim_empty_pages();
    assert(reclaimed == 0);

    // Clear and reclaim
    mat.set(10, 20, Trit::Zero);
    reclaimed = mat.reclaim_empty_pages();
    assert(reclaimed == 1);
    assert(mat.stats().materialized_pages == 0);
}

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Memory Arena & Zero-Page Tests...\n";
    test_dual_rail_planes();
    test_hierarchical_zero_matrix();
    std::cout << "  -> PASSED: All Memory Arena Unit Tests Verified.\n";
    return 0;
}
