/**
 * @file test_unit_memory_model.cpp
 * @brief Conformance: the C++ memory model against the verified model.
 *
 * verification/verify_memory.py discharges 12 obligations. This file pins
 * include/Ternary/Memory/Memory.hpp to them.
 *
 * The one that matters most is M1, the view extent formula. It uses the SUM
 * over dimensions of (shape-1)*stride, not the max and not the element count.
 * The max-over-dimensions version UNDER-reports the span, which makes the
 * aliasing check miss real overlaps -- a silent wrong result rather than a
 * crash. That bug was in this header and was found by the Python model, not
 * by inspection. This test is the regression guard.
 */

#include "Ternary/Memory/Memory.hpp"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace bt::mem;

namespace {

int g_checks = 0;
int g_fails = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

/** The exact set of byte addresses a view touches (brute force). */
std::set<size_t> touched(const View& v, size_t elem) {
    std::set<size_t> out;
    if (v.shape.empty()) { out.insert(v.offset_bytes); return out; }
    std::vector<int64_t> idx(v.shape.size(), 0);
    while (true) {
        size_t off = 0;
        for (size_t i = 0; i < idx.size(); ++i)
            off += static_cast<size_t>(idx[i]) *
                   static_cast<size_t>(v.strides[i]) * elem;
        out.insert(v.offset_bytes + off);
        // increment odometer
        size_t d = 0;
        for (; d < idx.size(); ++d) {
            if (++idx[d] < v.shape[d]) break;
            idx[d] = 0;
        }
        if (d == idx.size()) break;
    }
    return out;
}

}  // namespace

// ---- Arena ----------------------------------------------------------------

static void test_arena() {
    std::printf("  arena\n");
    Arena a(4096);
    check(a.bytes_outstanding() == 0, "starts empty");

    void* p = a.allocate(100);
    check(p != nullptr, "allocation succeeds");
    check(a.bytes_outstanding() >= 100, "outstanding grows");

    void* q = a.allocate(8, 64);
    check(reinterpret_cast<uintptr_t>(q) % 64 == 0, "honours 64-byte alignment");

    const size_t high = a.high_water_mark();
    a.reset();
    check(a.bytes_outstanding() == 0, "reset() reclaims to zero (O(1))");
    check(a.high_water_mark() == high, "high-water mark survives reset");

    // Bad alignment is refused, not silently ignored.
    bool threw = false;
    try { a.allocate(8, 3); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "non-power-of-two alignment is refused");

    // Exhaustion is a clean failure, not corruption.
    bool oom = false;
    try { a.allocate(1u << 30); } catch (const std::bad_alloc&) { oom = true; }
    check(oom, "exhaustion throws bad_alloc");

    // Zero capacity is refused at construction.
    bool zero = false;
    try { Arena bad(0); } catch (const std::invalid_argument&) { zero = true; }
    check(zero, "zero capacity is refused");
}

// ---- Pool -----------------------------------------------------------------

static void test_pool() {
    std::printf("  pool\n");
    Pool p(32, 4);
    check(p.live() == 0, "starts empty");

    void* a = p.acquire();
    void* b = p.acquire();
    check(p.live() == 2, "two live");
    check(a != b, "distinct blocks");

    p.release(a);
    check(p.live() == 1, "release returns the block");
    p.release(b);
    check(p.live() == 0, "back to empty");
    check(p.high_water_mark() == 2, "high-water mark recorded");

    // Releasing twice must not corrupt the freelist.
    p.release(a);
    check(p.live() == 0, "double release is a no-op, not corruption");

    // Exhaustion is clean.
    Pool small(8, 1);
    void* s = small.acquire();
    bool oom = false;
    try { small.acquire(); } catch (const std::bad_alloc&) { oom = true; }
    check(oom, "pool exhaustion throws bad_alloc");
    small.release(s);
    check(small.live() == 0, "released after exhaustion");
}

// ---- BufferTable: leak and double-free ------------------------------------

static void test_buffer_table() {
    std::printf("  buffer table (leak / double-free)\n");
    using bt::ir::TensorType;
    using bt::ir::Type;
    const TensorType ty{Type::Trit, {4, 8}, bt::target::Layout::DualRail};

    BufferTable t;
    check(t.count() == 0 && t.bytes() == 0, "starts empty");

    const uint64_t h1 = t.alloc(ty, host_device(), 64);
    const uint64_t h2 = t.alloc(ty, Device{DeviceKind::Gpu, 0}, 128);
    check(t.count() == 2, "two live");
    check(t.bytes() == 192, "bytes tracked");
    check(t.bytes_on(host_device()) == 64, "per-device accounting");
    check(t.bytes_on(Device{DeviceKind::Gpu, 0}) == 128, "gpu accounting");

    t.free(h1);
    check(t.count() == 1 && t.bytes() == 128, "free reclaims");

    bool threw = false;
    try { t.free(h1); } catch (const std::runtime_error&) { threw = true; }
    check(threw, "double free is reported, not silent");

    t.free(h2);
    check(t.count() == 0 && t.bytes() == 0, "returns to empty (no leak)");
    check(t.peak_count() == 2, "peak recorded");
}

// ---- M1: the extent formula -----------------------------------------------

static void test_extent() {
    std::printf("  M1 view extent (sum over dims, not max, not count)\n");

    const std::vector<std::vector<int64_t>> shapes = {
        {}, {1}, {2}, {4}, {3, 2}, {2, 3}, {2, 2, 2}, {3, 1, 2}};
    const std::vector<std::vector<int64_t>> stridesets = {
        {}, {1}, {2}, {3}, {1, 2}, {2, 1}, {2, 3}, {1, 1, 1}, {2, 1, 3}};

    int bad = 0, tested = 0;
    for (const auto& sh : shapes)
        for (const auto& st : stridesets) {
            if (sh.size() != st.size()) continue;
            for (size_t elem : {1u, 2u}) {
                View v{0, 0, sh, st};
                const auto t = touched(v, elem);
                const size_t span = (*t.rbegin() - *t.begin()) + elem;
                const size_t got = v.extent_bytes(elem);
                ++tested;
                if (got != span) ++bad;
            }
        }
    check(bad == 0, "extent == true touched span (" + std::to_string(tested) +
                        " (shape,strides,elem) combos)");

    // The specific case the max-over-dimensions formula got wrong.
    View v{0, 0, {3, 2}, {1, 2}};
    check(v.extent_bytes(1) == 5,
          "shape {3,2} strides {1,2}: extent is 5 (sum), not 3 (max) -- " +
              std::to_string(v.extent_bytes(1)));

    // And the case the element-count formula gets wrong.
    View s{0, 0, {4}, {2}};
    check(s.extent_bytes(1) == 7,
          "strided {4} stride {2}: extent is 7, not 4 (element count)");
}

// ---- M2: aliasing ----------------------------------------------------------

static void test_aliasing() {
    std::printf("  M2/M3/M4 aliasing\n");

    std::vector<View> views;
    for (uint64_t buf : {0u, 1u})
        for (size_t off : {0u, 1u, 3u, 4u, 8u})
            for (auto sh_st : std::vector<std::pair<std::vector<int64_t>,
                                                    std::vector<int64_t>>>{
                     {{4}, {1}}, {{4}, {2}}, {{2, 2}, {1, 1}},
                     {{3}, {1}}, {{2, 2}, {2, 1}}, {{2}, {3}}})
                views.push_back(View{buf, off, sh_st.first, sh_st.second});

    int false_neg = 0, inexact_contig = 0, over = 0, tested = 0;
    for (const auto& a : views)
        for (const auto& b : views) {
            const size_t elem = 1;
            const auto ta = touched(a, elem), tb = touched(b, elem);
            bool truth = (a.buffer == b.buffer);
            if (truth) {
                std::set<size_t> inter;
                for (size_t x : ta) if (tb.count(x)) { inter.insert(x); break; }
                truth = !inter.empty();
            }
            const bool got = aliases(a, b, elem);
            ++tested;

            if (truth && !got) ++false_neg;          // the dangerous direction
            if (!truth && got) ++over;               // conservative, safe

            const bool ac = [&]{ for (int64_t s : a.strides) if (s != 1) return false; return true; }();
            const bool bc = [&]{ for (int64_t s : b.strides) if (s != 1) return false; return true; }();
            if (ac && bc && got != truth) ++inexact_contig;
        }

    check(false_neg == 0, "no false negatives (" + std::to_string(tested) +
                              " view pairs)");
    check(inexact_contig == 0, "exact for contiguous views");
    check(over > 0, "strided views over-report conservatively (" +
                        std::to_string(over) + " cases) -- safe direction");

    // Named cases from the Python model.
    check(!aliases(View{0, 0, {4}, {1}}, View{0, 4, {4}, {1}}, 1),
          "adjacent-but-disjoint do not alias");
    check(aliases(View{0, 0, {4}, {1}}, View{0, 2, {4}, {1}}, 1),
          "overlapping do alias");
    check(!aliases(View{0, 0, {4}, {1}}, View{1, 0, {4}, {1}}, 1),
          "different buffers never alias");
}

// ---- Layouts and devices ---------------------------------------------------

static void test_devices() {
    std::printf("  devices and layouts\n");
    check(host_device().str() == "host", "host device names itself");
    check(Device{DeviceKind::Gpu, 2}.str() == "gpu:2", "gpu device names itself");
    check(host_device() != Device{DeviceKind::Gpu, 0}, "devices compare");

    Lifetime a{0, 5}, b{3, 8}, c{9, 12};
    check(a.overlaps(b), "overlapping lifetimes detected");
    check(!a.overlaps(c), "disjoint lifetimes detected");
}

int main() {
    std::printf("memory model\n");
    test_arena();
    test_pool();
    test_buffer_table();
    test_extent();
    test_aliasing();
    test_devices();
    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
