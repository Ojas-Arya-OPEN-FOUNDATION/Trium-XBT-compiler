/**
 * @file test/Unit/test_unit_audit_regressions.cpp
 * @brief Regression tests for the 2026-11-02 deep-audit findings.
 *
 * Every check here fails against the audited code and passes with the repair:
 *   M-04/M-05/M-06  allocator size arithmetic wrap-around
 *   M-07/M-08       assert-only bounds/shape checks (compile with NDEBUG too)
 *   M-01/M-02/M-03  thread-pool exception and lifecycle safety
 *   S-03            child processes must not inherit unrelated descriptors
 *   N-01..N-06      GEMM underflow/cancellation, RMSNorm, activations, BTFP32 exp,
 *                   TriRadixScalar::to_double
 *   A-01/A-02       C ABI null buffers and TCSR builder visibility
 */

#include "../../ternary_memory_arena.hpp"
#include "../../ternary_sparse.hpp"
#include "../../ternary_kokkos_bridge.hpp"
#include "../../ternary_xbt_blas.hpp"
#include "../../ternary_nonlinear_engine.hpp"
#include "../../bt_process.hpp"

#include <atomic>
#include <cassert>
#include <cmath>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "    FAIL: " << what << '\n';
        ++failures;
    }
}

template <typename Exception, typename Fn>
void expect_throws(Fn&& fn, const char* what) {
    try {
        fn();
        std::cerr << "    FAIL: " << what << " did not throw\n";
        ++failures;
    } catch (const Exception&) {
        // expected
    } catch (const std::exception& e) {
        std::cerr << "    FAIL: " << what << " threw " << e.what() << '\n';
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// M-04/M-05/M-06: allocation size arithmetic must reject overflow
// ---------------------------------------------------------------------------
void test_allocator_overflow() {
    using bt::mem::Base243Arena;
    using bt::mem::DualRailPlanes;

    expect_throws<std::length_error>([] {
        DualRailPlanes planes;
        planes.allocate(std::numeric_limits<size_t>::max());
    }, "DualRailPlanes::allocate(SIZE_MAX)");

    expect_throws<std::length_error>([] {
        DualRailPlanes planes;
        planes.allocate(std::numeric_limits<size_t>::max() / 8 + 2);
    }, "DualRailPlanes::allocate(SIZE_MAX/8 + 2)");

    expect_throws<std::length_error>([] {
        (void)bt::mem::allocate_aligned(std::numeric_limits<size_t>::max() - 31);
    }, "allocate_aligned(SIZE_MAX - 31)");

    expect_throws<std::invalid_argument>([] {
        (void)bt::mem::allocate_aligned(64, 24); // not a power of two
    }, "allocate_aligned with non-power-of-two alignment");

    expect_throws<std::length_error>([] {
        Base243Arena arena;
        (void)arena.allocate<uint64_t>(std::numeric_limits<size_t>::max() / 4);
    }, "Base243Arena::allocate<uint64_t> overflow");

    expect_throws<std::length_error>([] {
        Base243Arena arena;
        (void)arena.allocate_trits(std::numeric_limits<size_t>::max());
    }, "Base243Arena::allocate_trits overflow");

    // A valid request must still work and stay aligned.
    DualRailPlanes planes;
    planes.allocate(70);
    check(planes.num_words() == 70, "DualRailPlanes keeps requested word count");
    check(planes.capacity_words() == 72, "DualRailPlanes rounds capacity to 8 words");
    check(planes.is_aligned_64(), "DualRailPlanes rails are 64-byte aligned");
}

// ---------------------------------------------------------------------------
// M-07/M-08/A-02: bounds and shape checks must survive NDEBUG
// ---------------------------------------------------------------------------
void test_bounds_checks() {
    bt::mem::DualRailPlanes planes;
    planes.allocate(1);
    expect_throws<std::out_of_range>([&] { (void)planes.get_word(1); }, "DualRailPlanes::get_word OOB");
    expect_throws<std::out_of_range>([&] { (void)planes.get_trit(64); }, "DualRailPlanes::get_trit OOB");
    expect_throws<std::out_of_range>([&] {
        planes.set_word(1, bt::TritWord64(1, 0));
    }, "DualRailPlanes::set_word OOB");

    bt::TernaryMatrix matrix(2, 64);
    expect_throws<std::out_of_range>([&] { (void)matrix.get(0, 64); }, "TernaryMatrix::get OOB");
    expect_throws<std::out_of_range>([&] { matrix.set(2, 0, bt::Trit::Pos); }, "TernaryMatrix::set OOB");
    expect_throws<std::out_of_range>([&] { (void)matrix.word(0, 1); }, "TernaryMatrix::word OOB");

    bt::DenseMatrixInt64 dense(2, 2);
    expect_throws<std::out_of_range>([&] { (void)dense(2, 0); }, "DenseMatrixInt64 OOB");

    bt::TernaryMatrix A(2, 3), B(4, 3);
    bt::DenseMatrixInt64 C(2, 3);
    expect_throws<std::invalid_argument>([&] { bt::gemm(A, B, C); }, "gemm shape mismatch");
    expect_throws<std::invalid_argument>([&] {
        bt::kokkos::parallel_gemm(A, B, C);
    }, "parallel_gemm shape mismatch");

    // A-02: elements appended to the still-open last row must be visible before
    // finalize() (they used to read back as Zero).
    bt::TCSRMatrix sparse(2, 4);
    sparse.push_element(0, 1, bt::Trit::Pos);
    check(sparse.get(0, 1) == bt::Trit::Pos, "TCSR open row 0 visible");
    sparse.push_element(1, 3, bt::Trit::Neg);
    check(sparse.get(0, 1) == bt::Trit::Pos, "TCSR closed row 0 still visible");
    check(sparse.get(1, 3) == bt::Trit::Neg, "TCSR open row 1 visible before finalize");
    sparse.finalize();
    check(sparse.get(1, 3) == bt::Trit::Neg, "TCSR open row 1 visible after finalize");
    expect_throws<std::out_of_range>([&] { (void)sparse.get_sign(sparse.nnz); }, "TCSR get_sign OOB");
}

// ---------------------------------------------------------------------------
// M-01/M-02/M-03: pool exception propagation and concurrent initialization
// ---------------------------------------------------------------------------
void test_thread_pool_exceptions() {
    ::Kokkos::initialize(4);

    // Caller chunk (last chunk) throws: the latch must outlive queued tasks.
    for (int repeat = 0; repeat < 64; ++repeat) {
        std::atomic<int> ran{0};
        bool caught = false;
        try {
            ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, 64), [&](size_t i) {
                ran.fetch_add(1, std::memory_order_relaxed);
                if (i == 63) throw std::runtime_error("caller chunk");
            });
        } catch (const std::runtime_error&) {
            caught = true;
        }
        check(caught, "parallel_for propagates caller-chunk exception");
        check(ran.load() == 64, "parallel_for ran every chunk before propagating");
    }

    // Worker chunk throws: previously std::terminate, now reported to the caller.
    bool worker_caught = false;
    try {
        ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, 64), [&](size_t i) {
            if (i == 0) throw std::runtime_error("worker chunk");
        });
    } catch (const std::runtime_error&) {
        worker_caught = true;
    }
    check(worker_caught, "parallel_for propagates worker exception instead of terminating");

    bool reduce_caught = false;
    int64_t total = 0;
    try {
        ::Kokkos::parallel_reduce(::Kokkos::RangePolicy<size_t>(0, 64),
                                  [&](size_t i, int64_t& acc) {
                                      if (i == 0) throw std::runtime_error("reduce worker");
                                      acc += 1;
                                  }, total);
    } catch (const std::runtime_error&) {
        reduce_caught = true;
    }
    check(reduce_caught, "parallel_reduce propagates worker exception");

    // M-03: concurrent initialize/reinit must be serialized, not SIGSEGV.
    std::vector<std::thread> racers;
    for (int t = 0; t < 8; ++t) {
        racers.emplace_back([t] {
            for (int k = 0; k < 12; ++k) ::Kokkos::initialize(static_cast<size_t>(2 + ((t + k) % 4)));
        });
    }
    for (auto& thread : racers) thread.join();
    ::Kokkos::initialize(4);
    std::atomic<long long> sum{0};
    ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, 1000), [&](size_t) {
        sum.fetch_add(1, std::memory_order_relaxed);
    });
    check(sum.load() == 1000, "pool still usable after concurrent initialize");
}

// ---------------------------------------------------------------------------
// S-03: spawned tools must not inherit unrelated descriptors
// ---------------------------------------------------------------------------
void test_child_descriptor_hygiene() {
    int pipe_fds[2] = {-1, -1};
    if (pipe(pipe_fds) != 0) {
        std::cerr << "    SKIP: pipe() unavailable\n";
        return;
    }
    const std::string probe = "test -e /proc/self/fd/" + std::to_string(pipe_fds[0]) +
                              " && exit 7 || exit 0";
    int status = -1;
    try {
        status = bt::host::run_process({"/bin/sh", "-c", probe});
    } catch (const std::exception& e) {
        std::cerr << "    SKIP: cannot spawn /bin/sh: " << e.what() << '\n';
    }
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    check(status == 0, "child process did not inherit the parent pipe descriptor");
}

// ---------------------------------------------------------------------------
// N-01..N-06: numerical edge cases
// ---------------------------------------------------------------------------
void test_numerics() {
    // N-01: tiny products must not collapse to zero.
    {
        const float a[1] = {1e-22f};
        const float b[1] = {1e-22f};
        float c = -1.0f;
        bt::blas::xbt_sgemm_unmodified(a, b, &c, 1, 1, 1);
        check(c > 0.0f && std::isfinite(c), "1x1 tiny product is positive and finite");
        const double relative = std::fabs(static_cast<double>(c) - 1e-44) / 1e-44;
        check(relative < 0.05, "1x1 tiny product is accurate to 5%");
    }

    // N-02: exact cancellation of large terms must not produce NaN.
    {
        const float a[2] = {1e38f, 1e38f};
        const float b[2] = {1e38f, -1e38f};
        float c = -1.0f;
        bt::blas::xbt_sgemm_unmodified(a, b, &c, 1, 2, 1);
        check(!std::isnan(c), "large exact cancellation is not NaN");
        check(c == 0.0f, "large exact cancellation yields zero");
    }

    // N-03: subnormal RMSNorm input with eps = 0 stays finite.
    {
        const float x[4] = {1e-45f, 1e-45f, 1e-45f, 1e-45f};
        float out[4] = {0, 0, 0, 0};
        bt::nonlinear::xbt_rmsnorm(x, nullptr, out, 4, 0.0f);
        for (float value : out) {
            check(std::isfinite(value), "RMSNorm of subnormal input is finite");
            check(std::fabs(value - 1.0f) < 1e-3f, "RMSNorm of equal subnormals is ~1");
        }
        const float zeros[4] = {0, 0, 0, 0};
        bt::nonlinear::xbt_rmsnorm(zeros, nullptr, out, 4, 0.0f);
        for (float value : out) check(value == 0.0f, "RMSNorm of all-zero input with eps=0 is zero");
    }

    // N-05: activations at infinities follow their limits.
    {
        const double inf = std::numeric_limits<double>::infinity();
        check(bt::nonlinear::xbt_silu_fast(-inf) == 0.0, "SiLU(-inf) == 0");
        check(bt::nonlinear::xbt_silu_fast(inf) == inf, "SiLU(+inf) == +inf");
        check(bt::nonlinear::xbt_gelu_fast(-inf) == 0.0, "GELU(-inf) == 0");
        check(bt::nonlinear::xbt_gelu_fast(inf) == inf, "GELU(+inf) == +inf");
        check(std::isnan(bt::nonlinear::xbt_silu_fast(std::nan(""))), "SiLU(NaN) stays NaN");
    }

    // N-06: xbt_exp_btfp32 must report overflow, not std::terminate.
    {
        const bt::BTFP32 thousand = bt::BTFP32::from_double(1000.0);
        expect_throws<std::domain_error>([&] {
            (void)bt::nonlinear::xbt_exp_btfp32(thousand);
        }, "xbt_exp_btfp32 overflow reports domain_error");
        const bt::BTFP32 one = bt::BTFP32::from_double(1.0);
        const double value = bt::nonlinear::xbt_exp_btfp32(one).to_double();
        check(std::fabs(value - std::exp(1.0)) < 1e-4, "xbt_exp_btfp32(1) is e");
    }

    // N-04: exact large-factor cancellation in the tri-radix converter.
    {
        const bt::universal::TriRadixScalar value(bt::Codec::from_int64(1, 16), -2000, 0, 800);
        const double converted = value.to_double();
        check(std::isfinite(converted), "TriRadix 2^-2000 * 5^800 is finite");
        check(converted > 1.0e-43 && converted < 2.0e-43, "TriRadix 2^-2000 * 5^800 ~= 1.3e-43");
    }
}

} // namespace

int main() {
    std::cout << "=== Audit regression tests (2026-11-02 deep audit) ===\n";
    test_allocator_overflow();
    test_bounds_checks();
    test_thread_pool_exceptions();
    test_child_descriptor_hygiene();
    test_numerics();
    if (failures != 0) {
        std::cerr << failures << " regression check(s) failed\n";
        return 1;
    }
    std::cout << "ALL AUDIT REGRESSION TESTS PASSED\n";
    return 0;
}
