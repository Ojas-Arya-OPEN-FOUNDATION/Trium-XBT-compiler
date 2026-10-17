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
    std::cout << "      -> PASSED\n";
}

// ==============================================================================
// Fix 5: HierarchicalZeroPage word API operates on the byte-per-trit substrate.
// word w covers trits [64*w, 64*w + 64); one byte per trit (1.6 bits/trit belongs
// to Base243Arena only). set_word/get_word must agree with set_trit/get_trit.
// ==============================================================================
void test_hzp_word_trit_consistency() {
    std::cout << "  [Memory] Testing HierarchicalZeroPage word/trit consistency...\n";
    constexpr size_t CAP = 4 * 1024; // 64 words on the byte-per-trit substrate
    HierarchicalZeroPage page(CAP);

    // a) set_word(w=X) then get_trit spot checks match the encoded trits
    TritWord64 w = Codec::from_int64(12345, 39);
    page.set_word(0, w);
    TritWord64 back = page.get_word(0);
    assert(back.P == w.P && back.N == w.N && "set_word/get_word must roundtrip");
    for (size_t i = 0; i < 64; ++i) {
        assert(back.getTrit(i) == w.getTrit(i) && "every trit must survive the roundtrip");
        uint8_t raw = page.get_byte(i);
        // byte-per-trit substrate: 0x01 = Pos, 0xFF = Neg, 0x00 = Zero
        assert(raw == static_cast<uint8_t>(static_cast<int8_t>(w.getTrit(i))) &&
               "word entries must be stored as one signed byte per trit");
    }
    assert(page.get_trit(0) == w.getTrit(0) && "get_trit agrees with set_word");
    assert(page.get_trit(63) == w.getTrit(63) && "get_trit(63) agrees with set_word");

    // b) set_trit(i, Pos); get_word(0) bit i is Pos
    page.set_word(1, TritWord64(0, 0));
    page.set_trit(64 * 1 + 5, Trit::Pos);
    page.set_trit(64 * 1 + 9, Trit::Neg);
    TritWord64 w1 = page.get_word(1);
    assert(w1.getTrit(5) == Trit::Pos && "set_trit(Pos) must be visible as word bit 5");
    assert(w1.getTrit(9) == Trit::Neg && "set_trit(Neg) must be visible as word bit 9");
    assert(w1.getTrit(4) == Trit::Zero && "untouched trits stay zero");
    assert(Codec::to_int64(page.get_word(0), 39) == 12345 && "word 0 unaffected by word 1 writes");

    // c) multi-word independence + numeric roundtrip
    TritWord64 w2 = Codec::from_int64(-98765, 39);
    page.set_word(2, w2);
    TritWord64 back2 = page.get_word(2);
    assert((back2.P == w2.P && back2.N == w2.N) && "word 2 roundtrip");
    assert(Codec::to_int64(back2, 39) == -98765 && "word 2 exact numeric roundtrip");

    // d) word spanning the 64 KB page boundary (word 1026 starts at trit 65664)
    {
        HierarchicalZeroPage big_page(65664 + 128);
        TritWord64 wb = Codec::from_int64(999999, 39);
        big_page.set_word(1026, wb);
        TritWord64 got = big_page.get_word(1026);
        assert(got.P == wb.P && got.N == wb.N && "page-crossing word roundtrip");
        assert(got.getTrit(15) == wb.getTrit(15) && "last trit of page 0");
        assert(got.getTrit(16) == wb.getTrit(16) && "first trit of page 1");
    }

    // e) bounds: 4096 trits = 64 words (indices 0..63)
    bool threw = false;
    try { (void)page.get_word(64); } catch (const std::out_of_range&) { threw = true; }
    assert(threw && "get_word beyond the last full word must throw");
    threw = false;
    try { page.set_word(64, TritWord64(0, 0)); } catch (const std::out_of_range&) { threw = true; }
    assert(threw && "set_word beyond the last full word must throw");
    threw = false;
    try { (void)page.get_word(63); } catch (const std::out_of_range&) { threw = true; }
    assert(!threw && "get_word(63) on 64 words must be legal");

    // f) CoW: one word write materializes at most one 64 KB page
    HierarchicalZeroPage fresh(8192);
    fresh.set_word(10, Codec::from_int64(7, 39));
    assert(fresh.stats().materialized_pages == 1 && "single word must materialize 1 page");
    assert(Codec::to_int64(fresh.get_word(10), 39) == 7 && "word visible after CoW write");
    std::cout << "      -> PASSED\n";
}

// ==============================================================================
// Fix 6: Base243Arena move ctor/assign must leave the moved-from arena with invalid
// bump state (nullptr pointers / 0 chunk index), never aliasing freed chunk storage.
// ==============================================================================
void test_base243_arena_move_safety() {
    std::cout << "  [Memory] Testing Base243Arena move safety...\n";
    void* first_block = nullptr;
    {
        Base243Arena arena(512 * 1024);
        first_block = arena.allocate_bytes(128);
        void* second = arena.allocate_bytes(64);
        (void)second;
        assert(arena.allocation_count() == 2 && "two bump allocations");

        Base243Arena moved_to(std::move(arena)); // explicit move ctor
        assert(moved_to.allocation_count() == 2 && "stats travel with the chunks");
        // Moved-from state is invalid: empty stats, no bump pointers, no chunk index.
        assert(arena.allocation_count() == 0 && "moved-from allocation_count zeroed");
        assert(arena.total_allocated_bytes() == 0 && "moved-from bytes zeroed");
        void* dest = moved_to.allocate_bytes(256); // must not touch source internals
        assert(dest != nullptr && dest >= first_block && "dest keeps bumping monotonically");

        Base243Arena assigned(64 * 1024);
        Base243Block blk = assigned.allocate_trits(25);
        (void)blk;
        assert(assigned.allocation_count() == 1 && "trit block counted");
        assigned = std::move(moved_to); // explicit move assignment
        assert(assigned.allocation_count() == 3 && "move-assignment carries 128+64+256 stats");
        assert(moved_to.allocation_count() == 0 && "source of assignment invalidated");
        assert(assigned.allocate_bytes(32) != nullptr && "assigned pool still allocates");

        // Memory written before the move is untouched through the destination alias.
        std::memset(first_block, 0xAB, 128);
    }
    {
        // Self-move and sub-arena interplay must stay well-defined.
        Base243Arena a(256 * 1024);
        (void)a.allocate_trits(10);
        Base243Arena b(std::move(a));
        auto sub = b.create_sub_arena(1024);
        (void)sub;
        assert(b.allocate_trits(10).num_trits == 10 && "destination arena usable after move");
    }
    std::cout << "      -> PASSED\n";
}

// ==============================================================================
// Fix 8: HierarchicalZeroMatrix rows*cols overflow is detected (__int128 product,
// throws std::overflow_error when exceeding SIZE_MAX/2) instead of wrapping.
// ==============================================================================
void test_hzm_rows_cols_overflow() {
    std::cout << "  [Memory] Testing HierarchicalZeroMatrix rows*cols overflow...\n";
    bool threw = false;
    try {
        HierarchicalZeroMatrix mat(static_cast<size_t>(1) << 62, static_cast<size_t>(1) << 3);
        (void)mat;
    } catch (const std::overflow_error&) {
        threw = true;
    }
    assert(threw && "(2^62 x 2^3) must throw std::overflow_error (> SIZE_MAX/2)");

    threw = false;
    try {
        HierarchicalZeroMatrix mat(static_cast<size_t>(1) << 62, static_cast<size_t>(1) << 4);
        (void)mat;
    } catch (const std::overflow_error&) {
        threw = true;
    }
    assert(threw && "(2^62 x 2^4) must throw; checked multiply must not wrap");

    // Legal large matrices still construct with < 1 KB physical RAM.
    HierarchicalZeroMatrix mat(32768, 32768);
    assert(mat.total_elements() == 32768ULL * 32768ULL && "legal 1 GB trit matrix");
    assert(mat.get(100, 200) == Trit::Zero && "consistent with the byte-per-trit substrate");
    std::cout << "      -> PASSED\n";
}

// ==============================================================================
// Fix 7: allocation paths CAN throw (std::bad_alloc) and must not be marked noexcept
// (noexcept + throwing OOM path == std::terminate).
// ==============================================================================
void test_throwing_allocators_not_noexcept() {
    std::cout << "  [Memory] Testing allocator noexcept correctness...\n";
    assert(!noexcept(std::declval<Base243Arena&>().allocate_bytes(16)) &&
           "Base243Arena::allocate_bytes must not be noexcept (slow path can bad_alloc)");
    assert(!noexcept(std::declval<Base243Arena&>().allocate_trits(64)) &&
           "Base243Arena::allocate_trits must not be noexcept");
    // Non-throwing members keep their guarantees:
    assert(noexcept(std::declval<Base243Arena&>().reset()) && "reset stays noexcept");
    assert(noexcept(std::declval<Base243Arena&>().release()) && "release stays noexcept");
    std::cout << "      -> PASSED\n";
}

int main() {
    std::cout << "[LLVM Unit Test Layer 1] Running Memory Arena & Zero-Page Tests...\n";
    test_dual_rail_planes();
    test_hierarchical_zero_matrix();
    test_hierarchical_zero_page_bounds_and_atomic();
    test_hzp_word_trit_consistency();
    test_base243_arena_move_safety();
    test_hzm_rows_cols_overflow();
    test_throwing_allocators_not_noexcept();
    std::cout << "  -> PASSED: All Memory Arena Unit Tests Verified.\n";
    return 0;
}
