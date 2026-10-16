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

void test_hierarchical_zero_page_bounds_and_atomic() {
    std::cout << "  [Memory] Testing HierarchicalZeroPage Bounds & Atomic Read Count...\n";

    HierarchicalZeroPage page(1024);
    assert(page.read_count() == 0);

    // Reads within capacity increment atomic read_count
    uint8_t b0 = page.get_byte(0);
    uint8_t b1 = page.get_byte(100);
    assert(b0 == 0 && b1 == 0);
    assert(page.read_count() == 2);

    // Out of bounds get_byte throws std::out_of_range
    bool caught_read_oob = false;
    try {
        (void)page.get_byte(1024);
    } catch (const std::out_of_range&) {
        caught_read_oob = true;
    }
    assert(caught_read_oob && "get_byte(offset >= virtual_capacity) must throw std::out_of_range");

    // Out of bounds set_byte throws std::out_of_range
    bool caught_write_oob = false;
    try {
        page.set_byte(1024, 1);
    } catch (const std::out_of_range&) {
        caught_write_oob = true;
    }
    assert(caught_write_oob && "set_byte(offset >= virtual_capacity) must throw std::out_of_range");

    // HierarchicalZeroMatrix bounds checks
    HierarchicalZeroMatrix mat(10, 10);
    bool caught_mat_get_oob = false;
    try {
        (void)mat.get(10, 0);
    } catch (const std::out_of_range&) {
        caught_mat_get_oob = true;
    }
    assert(caught_mat_get_oob && "mat.get(r >= rows) must throw std::out_of_range");

    bool caught_mat_set_oob = false;
    try {
        mat.set(0, 10, Trit::Pos);
    } catch (const std::out_of_range&) {
        caught_mat_set_oob = true;
    }
    assert(caught_mat_set_oob && "mat.set(c >= cols) must throw std::out_of_range");
}

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Memory Arena & Zero-Page Tests...\n";
    test_dual_rail_planes();
    test_hierarchical_zero_matrix();
    test_hierarchical_zero_page_bounds_and_atomic();
    std::cout << "  -> PASSED: All Memory Arena Unit Tests Verified.\n";
    return 0;
}
