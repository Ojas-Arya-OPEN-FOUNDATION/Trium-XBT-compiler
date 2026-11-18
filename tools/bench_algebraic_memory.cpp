/**
 * @file bench_algebraic_memory.cpp
 * @brief Reproducible benchmark suite for Algebraic Virtual Memory & Circulant
 *        Weighing operators.
 *
 * Enforces bench_harness discipline:
 *   1. Bit-exact / exact-value verification BEFORE any timing.
 *   2. Best / median / worst ms across repetitions.
 *   3. Full-output sink outside timed regions (no dead-code elimination).
 *   4. Machine fingerprint via btbench::probe_machine().
 *   5. PhysicalMemoryMonitor RSS & zero-swap enforcement (ru_nswap == 0).
 */

#include "Ternary/Algebra/CirculantWeighing.hpp"
#include "Ternary/Algebra/AlgebraicVirtualMemory.hpp"
#include "bench_harness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = btbench::Clock;

struct TimingStats {
    double best_ms{0.0};
    double median_ms{0.0};
    double worst_ms{0.0};
};

template <typename Fn>
TimingStats measure_reps(int reps, Fn&& fn) {
    std::vector<double> ts;
    ts.reserve(static_cast<size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    TimingStats s;
    s.best_ms = ts.front();
    s.median_ms = ts[ts.size() / 2];
    s.worst_ms = ts.back();
    return s;
}

void print_machine_header() {
    const btbench::MachineInfo mi = btbench::probe_machine();
    std::printf("machine   : %s\n", mi.cpu.c_str());
    std::printf("isa       : %s\n", mi.isa_summary().c_str());
    std::printf("threads   : %d   build: %s\n", mi.threads,
#if defined(__OPTIMIZE__)
                "optimised"
#else
                "UNOPTIMISED (numbers are meaningless)"
#endif
    );
}

// ============================================================================
// 1. CW(120, 36) MatVec + Exact Inverse
// ============================================================================
bool bench_cw_120_36(int reps, bool check_mode) {
    using bt::algebra::CirculantTernaryOperator;

    const auto cw = CirculantTernaryOperator::cw_120_36();
    const size_t n = cw.dim();
    if (!cw.verify_orthogonality()) {
        std::fprintf(stderr, "ERROR: CW(120,36) failed verify_orthogonality()\n");
        return false;
    }

    std::mt19937_64 rng(0xC012036ULL);
    std::uniform_int_distribution<int64_t> dist(-10000, 10000);
    std::vector<int64_t> x_init(n);
    for (size_t i = 0; i < n; ++i) {
        x_init[i] = dist(rng);
    }

    // Verify bit-exact forward + inverse BEFORE timing
    std::vector<int64_t> y(n, 0), x_rec(n, 0);
    cw.apply(x_init, y);
    if (!cw.apply_exact_inverse(y, x_rec) || x_rec != x_init) {
        std::fprintf(stderr, "ERROR: CW(120,36) failed exact inverse verification\n");
        return false;
    }

    const int batch = check_mode ? 2000 : 10000;
    std::vector<int64_t> x_cur = x_init;

    const TimingStats ts = measure_reps(reps, [&]() {
        for (int b = 0; b < batch; ++b) {
            cw.apply(x_cur, y);
            cw.apply_exact_inverse(y, x_cur);
        }
    });

    // Post-timing full-array verification sink (identity roundtrip must stay exact)
    if (x_cur != x_init) {
        std::fprintf(stderr, "ERROR: CW(120,36) post-timing state mismatch\n");
        return false;
    }

    // Operations per (apply + exact_inverse): 2 * (120 * 36) add/sub ops + 120 divs
    const double ops_per_rep = static_cast<double>(batch) * (2.0 * 120.0 * 36.0 + 120.0);
    const double gops = ops_per_rep / (ts.best_ms * 1e-3) / 1e9;
    const size_t seed_bytes = cw.memory_bytes();
    const size_t dense_fp32_bytes = n * n * sizeof(float);

    std::printf("\n[1] CW(120, 36) MatVec + Exact Inverse (batch=%d)\n", batch);
    std::printf("  Verified        : PASS (bit-exact roundtrip)\n");
    std::printf("  Seed RAM        : %zu B   vs Dense FP32 RAM: %zu B (%.1fx smaller)\n",
                seed_bytes, dense_fp32_bytes,
                static_cast<double>(dense_fp32_bytes) / static_cast<double>(seed_bytes));
    std::printf("  Timing (ms)     : best=%.3f  median=%.3f  worst=%.3f  (%.2f GOp/s)\n",
                ts.best_ms, ts.median_ms, ts.worst_ms, gops);
    return true;
}

// ============================================================================
// 2. Kronecker CW(120,36)^3  (D = 1,728,000, ~11.94 TB Virtual FP32)
// ============================================================================
bool bench_kronecker_cw120_cubed(int reps) {
    using bt::algebra::CirculantTernaryOperator;
    using bt::algebra::KroneckerTernaryOperator;
    using bt::vmem::PhysicalMemoryMonitor;

    KroneckerTernaryOperator kron({
        CirculantTernaryOperator::cw_120_36(),
        CirculantTernaryOperator::cw_120_36(),
        CirculantTernaryOperator::cw_120_36()
    });

    const size_t D = kron.dim(); // 1,728,000
    if (D != 1728000ULL) {
        std::fprintf(stderr, "ERROR: unexpected Kronecker dim %zu\n", D);
        return false;
    }

    // Deterministic input vector
    std::vector<int64_t> x(D);
    for (size_t i = 0; i < D; ++i) {
        x[i] = static_cast<int64_t>((i * 1315423911ULL) % 2001ULL) - 1000LL;
    }
    std::vector<int64_t> y(D, 0);
    std::vector<int64_t> x_rec(D, 0);

    // 1. Verify exact apply -> apply_exact_inverse roundtrip BEFORE timing
    kron.apply(x, y);
    if (!kron.apply_exact_inverse(y, x_rec) || x_rec != x) {
        std::fprintf(stderr, "ERROR: Kronecker CW(120,36)^3 failed exact inverse verification\n");
        return false;
    }

    // 2. Spot-check eval_entry on sampled rows (first & middle row)
    //    Checking full 1.728M-entry dot product for 2 rows takes ~20ms and proves
    //    factorized Kronecker apply matches element-wise W(r,c) = prod W_l(r_l, c_l).
    const size_t spot_rows[] = {0, D / 2 + 17};
    for (size_t r : spot_rows) {
        int64_t dot = 0;
        for (size_t c = 0; c < D; ++c) {
            dot += static_cast<int64_t>(kron.eval_entry(r, c)) * x[c];
        }
        if (dot != y[r]) {
            std::fprintf(stderr, "ERROR: Kronecker eval_entry spot-check failed at row %zu\n", r);
            return false;
        }
    }

    // 3. Time apply
    const TimingStats ts = measure_reps(reps, [&]() {
        kron.apply(x, y);
    });

    // Post-timing verification sink
    if (!kron.apply_exact_inverse(y, x_rec) || x_rec != x) {
        std::fprintf(stderr, "ERROR: Kronecker post-timing verification failed\n");
        return false;
    }

    const auto mem = PhysicalMemoryMonitor::sample();
    const size_t seed_bytes = kron.memory_bytes();
    const double virtual_fp32_tb = kron.virtual_fp32_bytes() / 1e12;
    const double amp_factor = kron.virtual_fp32_bytes() / static_cast<double>(seed_bytes);
    // 3 factors * D * 36 ops per factor
    const double gops = (3.0 * static_cast<double>(D) * 36.0) / (ts.best_ms * 1e-3) / 1e9;

    std::printf("\n[2] Kronecker CW(120,36)^3 (D=%zu, Virtual Elements=2.986e12)\n", D);
    std::printf("  Verified        : PASS (exact inverse + eval_entry spot-check)\n");
    std::printf("  Seed Bytes      : %zu B\n", seed_bytes);
    std::printf("  Virtual FP32    : %.2f TB\n", virtual_fp32_tb);
    std::printf("  Amplification   : %.3e x (Virtual / Seed)\n", amp_factor);
    std::printf("  Peak RSS        : %.2f MB   ru_nswap: %llu\n",
                static_cast<double>(mem.peak_rss_bytes) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(mem.swaps));
    std::printf("  Timing (ms)     : best=%.3f  median=%.3f  worst=%.3f  (%.2f GOp/s)\n",
                ts.best_ms, ts.median_ms, ts.worst_ms, gops);

    if (mem.swaps > 0) {
        std::fprintf(stderr, "ERROR: OS swap detected (ru_nswap=%llu)\n",
                     static_cast<unsigned long long>(mem.swaps));
        return false;
    }
    return true;
}

// ============================================================================
// 3. VirtualStencil3D Out-of-Core (128^3) with ZeroSwapStreamStore (2 MB Slab)
// ============================================================================
bool bench_virtual_stencil_3d(int reps) {
    using bt::vmem::PhysicalMemoryMonitor;
    using bt::vmem::VirtualStencil3D;
    using bt::vmem::ZeroSwapStreamStore;

    constexpr size_t NX = 128, NY = 128, NZ = 128;
    constexpr size_t TOTAL = NX * NY * NZ; // 2,097,152 nodes
    constexpr size_t SLAB_BYTES = 2 * 1024 * 1024; // 2 MB cap

    VirtualStencil3D<double> stencil(NX, NY, NZ);
    ZeroSwapStreamStore<double> x_store(TOTAL, SLAB_BYTES);
    ZeroSwapStreamStore<double> y_store(TOTAL, SLAB_BYTES);

    // Populate u(x, y, z) = x^2 + y^2 + z^2
    x_store.generate([=](size_t idx) -> double {
        const size_t ix = idx % NX;
        const size_t iy = (idx / NX) % NY;
        const size_t iz = idx / (NX * NY);
        const double dx = static_cast<double>(ix);
        const double dy = static_cast<double>(iy);
        const double dz = static_cast<double>(iz);
        return dx * dx + dy * dy + dz * dz;
    });

    // Verify exact Laplacian BEFORE timing (interior points must equal 6.0 exactly)
    stencil.apply_laplacian_3d(x_store, y_store);

    const bool ok = y_store.stream_reduce(true, [&](bool acc, size_t idx, double val) {
        if (!acc) return false;
        const size_t ix = idx % NX;
        const size_t iy = (idx / NX) % NY;
        const size_t iz = idx / (NX * NY);
        if (ix > 0 && ix + 1 < NX &&
            iy > 0 && iy + 1 < NY &&
            iz > 0 && iz + 1 < NZ) {
            if (std::abs(val - 6.0) > 1e-12) {
                return false;
            }
        }
        return true;
    });
    if (!ok) {
        std::fprintf(stderr, "ERROR: VirtualStencil3D interior Laplacian != 6.0\n");
        return false;
    }

    const TimingStats ts = measure_reps(reps, [&]() {
        stencil.apply_laplacian_3d(x_store, y_store);
    });

    // Post-timing full-store reduction sink
    const double sum_y = y_store.stream_reduce(0.0, [](double acc, double v) {
        return acc + v;
    });
    if (!std::isfinite(sum_y)) {
        std::fprintf(stderr, "ERROR: VirtualStencil3D non-finite output\n");
        return false;
    }

    const auto mem = PhysicalMemoryMonitor::sample();
    const size_t explicit_csr_bytes = stencil.explicit_csr_bytes_estimate();
    const size_t explicit_vec_bytes = 2 * TOTAL * sizeof(double);
    const double explicit_total_mb =
        static_cast<double>(explicit_csr_bytes + explicit_vec_bytes) / (1024.0 * 1024.0);
    const double slab_cap_mb = static_cast<double>(SLAB_BYTES) / (1024.0 * 1024.0);
    // 7-point stencil: ~7 flops/node (or 8 ops: 1 mul + 6 adds)
    const double gflops = (8.0 * static_cast<double>(TOTAL)) / (ts.best_ms * 1e-3) / 1e9;

    std::printf("\n[3] VirtualStencil3D Out-of-Core (128x128x128 = %zu nodes, 14.7M edges)\n",
                TOTAL);
    std::printf("  Verified        : PASS (interior Laplacian u=x^2+y^2+z^2 == 6.0 exact)\n");
    std::printf("  Slab Cap        : %.2f MB   vs Explicit CSR+Vecs: %.2f MB (0 B matrix RAM)\n",
                slab_cap_mb, explicit_total_mb);
    std::printf("  Peak RSS        : %.2f MB   ru_nswap: %llu\n",
                static_cast<double>(mem.peak_rss_bytes) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(mem.swaps));
    std::printf("  Timing (ms)     : best=%.3f  median=%.3f  worst=%.3f  (%.2f GFLOP/s)\n",
                ts.best_ms, ts.median_ms, ts.worst_ms, gflops);

    if (mem.swaps > 0) {
        std::fprintf(stderr, "ERROR: OS swap detected (ru_nswap=%llu)\n",
                     static_cast<unsigned long long>(mem.swaps));
        return false;
    }
    return true;
}

// ============================================================================
// 4. ReversibleSymplecticTape (100,000 Steps)
// ============================================================================
bool bench_reversible_symplectic_tape(int reps) {
    using bt::vmem::ReversibleSymplecticTape;
    using PhaseState = ReversibleSymplecticTape::PhaseState;

    const bt::TritWord64 q0(0x0F0F0F0F00FF00FFULL, 0xF0F00000FF000000ULL);
    const bt::TritWord64 p0(0x1234000056780000ULL, 0x0000ABCD00001234ULL);
    const bt::TritWord64 force(0x00AA005500AA0055ULL, 0x5500AA005500AA00ULL);
    const PhaseState init_state(q0, p0);

    constexpr uint64_t STEPS = 100000;

    // Verify exact reverse recovery BEFORE timing
    ReversibleSymplecticTape tape(init_state, 0);
    tape.step_forward(STEPS, force);
    const size_t hist_bytes = tape.history_ram_bytes();
    const uint64_t virt_bytes = tape.virtual_history_bytes();
    tape.step_backward(STEPS, force);
    const PhaseState rec = tape.current_state();
    if (rec.Q.P != init_state.Q.P || rec.Q.N != init_state.Q.N ||
        rec.P.P != init_state.P.P || rec.P.N != init_state.P.N) {
        std::fprintf(stderr, "ERROR: ReversibleSymplecticTape failed exact reverse recovery\n");
        return false;
    }

    const TimingStats ts = measure_reps(reps, [&]() {
        tape.step_forward(STEPS, force);
        tape.step_backward(STEPS, force);
    });

    const PhaseState post = tape.current_state();
    if (post.Q.P != init_state.Q.P || post.Q.N != init_state.Q.N ||
        post.P.P != init_state.P.P || post.P.N != init_state.P.N) {
        std::fprintf(stderr, "ERROR: ReversibleSymplecticTape post-timing state mismatch\n");
        return false;
    }

    const double msteps_s = (2.0 * static_cast<double>(STEPS)) / (ts.best_ms * 1e-3) / 1e6;

    std::printf("\n[4] ReversibleSymplecticTape (%llu fwd + %llu bwd steps)\n",
                static_cast<unsigned long long>(STEPS),
                static_cast<unsigned long long>(STEPS));
    std::printf("  Verified        : PASS (bit-exact Q,P recovery)\n");
    std::printf("  History RAM     : %zu B   vs Naive History Tape: %.2f MB (%.0fx smaller)\n",
                hist_bytes,
                static_cast<double>(virt_bytes) / (1024.0 * 1024.0),
                static_cast<double>(virt_bytes) / static_cast<double>(hist_bytes));
    std::printf("  Timing (ms)     : best=%.3f  median=%.3f  worst=%.3f  (%.2f MSteps/s)\n",
                ts.best_ms, ts.median_ms, ts.worst_ms, msteps_s);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    bool check_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--check") == 0) {
            check_mode = true;
        }
    }

    const int reps = check_mode ? 3 : 7;

    print_machine_header();
    std::printf("mode      : %s (reps=%d)\n", check_mode ? "check" : "full", reps);

    bool ok = true;
    ok = bench_cw_120_36(reps, check_mode) && ok;
    ok = bench_kronecker_cw120_cubed(reps) && ok;
    ok = bench_virtual_stencil_3d(reps) && ok;
    ok = bench_reversible_symplectic_tape(reps) && ok;

    const auto final_mem = bt::vmem::PhysicalMemoryMonitor::sample();
    if (final_mem.swaps > 0) {
        std::fprintf(stderr, "\nFAILED: OS swap occurred (ru_nswap=%llu)\n",
                     static_cast<unsigned long long>(final_mem.swaps));
        return 1;
    }

    if (!ok) {
        std::fprintf(stderr, "\nFAILED: One or more benchmark verification checks failed.\n");
        return 1;
    }

    std::printf("\nALL ALGEBRAIC VIRTUAL MEMORY BENCHMARKS VERIFIED & PASSED.\n");
    return 0;
}
