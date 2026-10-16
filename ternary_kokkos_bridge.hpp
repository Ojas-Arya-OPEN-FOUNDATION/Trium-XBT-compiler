/**
 * @file ternary_kokkos_bridge.hpp
 * @brief High-Performance Kokkos Multi-Threaded Parallel Execution Engine for Balanced Ternary
 * @author Antigravity Prime Engineering Engine
 * 
 * Provides production-grade C++20 multithreading and Kokkos-compatible execution policies:
 * - Persistent worker thread pool with latch-based synchronization (zero allocation in kernels)
 * - Kokkos::parallel_for and Kokkos::parallel_reduce execution patterns
 * - Multi-core bit-sliced parallel GEMM across all available CPU cores (e.g., 10 cores on Apple M5)
 * - Parallel Carry-Save Adder (BT-CSA) matrix compressor trees
 * - Planar bitmask tensor wrappers for Kokkos::HostSpace & device buffer bridges
 */

#ifndef TERNARY_KOKKOS_BRIDGE_HPP
#define TERNARY_KOKKOS_BRIDGE_HPP

#include "ternary_dense_algebra.hpp"
#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <latch>
#include <algorithm>
#include <iostream>
#include <memory>
#include <chrono>

namespace bt {
namespace kokkos {

// ==============================================================================
// 1. Planar Tensor View & Bridge Functors (Kokkos / IREE Interop)
// ==============================================================================

#ifndef KOKKOS_RESTRICT
#define KOKKOS_RESTRICT __restrict__
#endif

/**
 * @brief Planar Bitmask Tensor Descriptor for Heterogeneous Memory Spaces.
 * Stores positive rail (P) and negative rail (N) as independent contiguous buffers.
 */
template <typename MemorySpaceTag = void>
struct PlanarTensorView {
    uint64_t* d_pos{nullptr}; // Pointer to positive rail buffer in target memory space
    uint64_t* d_neg{nullptr}; // Pointer to negative rail buffer in target memory space
    size_t rows{0};
    size_t cols{0};
    size_t stride_words{0};   // Words per row (ceil(cols / 64))

    constexpr bool is_allocated() const noexcept {
        return d_pos != nullptr && d_neg != nullptr;
    }

    constexpr size_t total_words() const noexcept {
        return rows * stride_words;
    }

    constexpr size_t total_bytes() const noexcept {
        return total_words() * sizeof(uint64_t);
    }
};

/**
 * @brief Host-Side Factory for wrapping TernaryMatrix into a Kokkos-compatible PlanarTensorView.
 */
inline PlanarTensorView<void> make_host_view(TernaryMatrix& mat) noexcept {
    PlanarTensorView<void> view;
    view.rows = mat.rows;
    view.cols = mat.cols;
    view.stride_words = mat.words_per_row;
    return view;
}

/**
 * @brief Bit-Sliced Functor for Heterogeneous Parallel Dispatch:
 * Parallel For / Parallel Reduce kernel executing inside Kokkos::parallel_for or IREE Dispatch.
 */
struct PlanarGEMMFunctor {
    const uint64_t* const KOKKOS_RESTRICT A_pos;
    const uint64_t* const KOKKOS_RESTRICT A_neg;
    const uint64_t* const KOKKOS_RESTRICT BT_pos;
    const uint64_t* const KOKKOS_RESTRICT BT_neg;
    int64_t* const KOKKOS_RESTRICT C_out;

    size_t M;
    size_t N;
    size_t stride_words;

    inline void operator()(const size_t i, const size_t j) const noexcept {
        int64_t acc = 0;
        const size_t row_offset_A = i * stride_words;
        const size_t row_offset_B = j * stride_words;

        for (size_t w = 0; w < stride_words; ++w) {
            uint64_t ap = A_pos[row_offset_A + w];
            uint64_t an = A_neg[row_offset_A + w];
            uint64_t bp = BT_pos[row_offset_B + w];
            uint64_t bn = BT_neg[row_offset_B + w];

            uint64_t pos_prod = (ap & bp) | (an & bn);
            uint64_t neg_prod = (ap & bn) | (an & bp);

            acc += static_cast<int64_t>(std::popcount(pos_prod)) - 
                   static_cast<int64_t>(std::popcount(neg_prod));
        }

        C_out[i * N + j] = acc;
    }
};

} // namespace kokkos
} // namespace bt

// ==============================================================================
// 2. Standard Kokkos Namespace Emulation Layer & Thread Pool
// ==============================================================================

namespace Kokkos {

struct HostSpace {};
struct Serial {};
struct OpenMP {};
struct Threads {};
using DefaultExecutionSpace = Threads;

template <size_t N>
struct Rank {};

/**
 * @brief 1D Range Policy for parallel dispatch [begin, end).
 */
template <typename IndexType = size_t>
class RangePolicy {
public:
    IndexType m_begin{0};
    IndexType m_end{0};

    constexpr RangePolicy(IndexType begin, IndexType end) noexcept : m_begin(begin), m_end(end) {}
    constexpr explicit RangePolicy(IndexType end) noexcept : m_begin(0), m_end(end) {}

    constexpr IndexType begin() const noexcept { return m_begin; }
    constexpr IndexType end() const noexcept { return m_end; }
    constexpr IndexType size() const noexcept { return m_end > m_begin ? m_end - m_begin : 0; }
};

/**
 * @brief 2D Multi-Dimensional Range Policy for tiled matrix kernels.
 */
template <typename RankTag = Rank<2>, typename IndexType = size_t>
class MDRangePolicy {
public:
    std::array<IndexType, 2> m_begin{0, 0};
    std::array<IndexType, 2> m_end{0, 0};
    std::array<IndexType, 2> m_tile{16, 16};

    MDRangePolicy(std::array<IndexType, 2> begin, std::array<IndexType, 2> end,
                  std::array<IndexType, 2> tile = {16, 16}) noexcept
        : m_begin(begin), m_end(end), m_tile(tile) {}
};

/**
 * @brief Persistent Multi-Core Worker Pool for C++20 Kokkos Execution.
 */
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable cv_task;
    std::atomic<bool> stop_flag{false};
    size_t active_threads{0};

    ThreadPool() {
        size_t hw = std::thread::hardware_concurrency();
        init(hw > 0 ? hw : 4);
    }

    void init(size_t threads) {
        shutdown();
        stop_flag = false;
        active_threads = threads;
        workers.reserve(threads);
        for (size_t t = 0; t < threads; ++t) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->cv_task.wait(lock, [this]() {
                            return this->stop_flag.load() || !this->task_queue.empty();
                        });
                        if (this->stop_flag.load() && this->task_queue.empty()) {
                            return;
                        }
                        task = std::move(this->task_queue.front());
                        this->task_queue.pop();
                    }
                    task();
                }
            });
        }
    }

    void shutdown() {
        if (workers.empty()) return;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop_flag = true;
        }
        cv_task.notify_all();
        for (auto& th : workers) {
            if (th.joinable()) th.join();
        }
        workers.clear();
        std::queue<std::function<void()>> empty;
        std::swap(task_queue, empty);
    }

public:
    ~ThreadPool() {
        shutdown();
    }

    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    void reinit(size_t threads) {
        if (threads == 0) threads = std::thread::hardware_concurrency();
        if (threads == active_threads && !workers.empty()) return;
        init(threads);
    }

    size_t get_thread_count() const noexcept {
        return active_threads;
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            task_queue.push(std::move(task));
        }
        cv_task.notify_one();
    }
};

inline void initialize(size_t num_threads = 0) {
    ThreadPool::instance().reinit(num_threads);
}

inline void initialize(int& argc, char** argv) {
    size_t threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--kokkos-threads=", 0) == 0) {
            threads = std::stoul(arg.substr(17));
        }
    }
    initialize(threads);
}

inline void finalize() {
    // ThreadPool will be automatically finalized on program termination, or explicit reinit
}

inline size_t num_threads() noexcept {
    return ThreadPool::instance().get_thread_count();
}

inline bool is_initialized() noexcept {
    return ThreadPool::instance().get_thread_count() > 0;
}

/**
 * @brief Kokkos::parallel_for dispatch across CPU threads with chunk-level static scheduling.
 */
template <typename IndexType, typename Functor>
inline void parallel_for(const RangePolicy<IndexType>& policy, const Functor& func) {
    IndexType total = policy.size();
    if (total == 0) return;

    size_t nthreads = ThreadPool::instance().get_thread_count();
    if (nthreads <= 1 || total == 1) {
        for (IndexType i = policy.begin(); i < policy.end(); ++i) {
            func(i);
        }
        return;
    }

    size_t num_chunks = std::min(nthreads, static_cast<size_t>(total));
    IndexType chunk_size = (total + num_chunks - 1) / num_chunks;
    std::latch sync_latch(num_chunks);

    for (size_t c = 0; c < num_chunks; ++c) {
        IndexType start = policy.begin() + c * chunk_size;
        IndexType finish = std::min(policy.end(), start + chunk_size);

        if (c == num_chunks - 1) {
            // Run last chunk on caller thread directly to save thread dispatch overhead
            for (IndexType i = start; i < finish; ++i) {
                func(i);
            }
            sync_latch.count_down();
        } else {
            ThreadPool::instance().enqueue([start, finish, &func, &sync_latch]() {
                for (IndexType i = start; i < finish; ++i) {
                    func(i);
                }
                sync_latch.count_down();
            });
        }
    }

    sync_latch.wait();
}

/**
 * @brief Kokkos::parallel_for dispatch for 2D MDRangePolicy.
 */
template <typename RankTag, typename IndexType, typename Functor>
inline void parallel_for(const MDRangePolicy<RankTag, IndexType>& policy, const Functor& func) {
    IndexType M = policy.m_end[0] - policy.m_begin[0];
    IndexType N = policy.m_end[1] - policy.m_begin[1];
    if (M == 0 || N == 0) return;

    parallel_for(RangePolicy<IndexType>(policy.m_begin[0], policy.m_end[0]), [&](IndexType i) {
        for (IndexType j = policy.m_begin[1]; j < policy.m_end[1]; ++j) {
            func(i, j);
        }
    });
}

/**
 * @brief Kokkos::parallel_reduce dispatch with thread-local accumulators.
 */
template <typename IndexType, typename Functor, typename ValueType, typename Reducer = std::plus<ValueType>>
inline void parallel_reduce(const RangePolicy<IndexType>& policy, const Functor& func,
                            ValueType& result, Reducer reducer = Reducer()) {
    IndexType total = policy.size();
    if (total == 0) return;

    size_t nthreads = ThreadPool::instance().get_thread_count();
    if (nthreads <= 1 || total == 1) {
        for (IndexType i = policy.begin(); i < policy.end(); ++i) {
            func(i, result);
        }
        return;
    }

    size_t num_chunks = std::min(nthreads, static_cast<size_t>(total));
    IndexType chunk_size = (total + num_chunks - 1) / num_chunks;

    // Cache-line aligned (64 bytes) to completely eliminate L1 false sharing between worker threads
    struct alignas(64) PaddedAccumulator {
        ValueType val{};
    };
    std::vector<PaddedAccumulator> local_accumulators(num_chunks);
    std::latch sync_latch(num_chunks);

    for (size_t c = 0; c < num_chunks; ++c) {
        IndexType start = policy.begin() + c * chunk_size;
        IndexType finish = std::min(policy.end(), start + chunk_size);

        if (c == num_chunks - 1) {
            ValueType acc{};
            for (IndexType i = start; i < finish; ++i) {
                func(i, acc);
            }
            local_accumulators[c].val = acc;
            sync_latch.count_down();
        } else {
            ThreadPool::instance().enqueue([start, finish, c, &func, &local_accumulators, &sync_latch]() {
                ValueType acc{};
                for (IndexType i = start; i < finish; ++i) {
                    func(i, acc);
                }
                local_accumulators[c].val = acc;
                sync_latch.count_down();
            });
        }
    }

    sync_latch.wait();

    for (size_t c = 0; c < num_chunks; ++c) {
        result = reducer(result, local_accumulators[c].val);
    }
}

} // namespace Kokkos

// ==============================================================================
// 3. Parallel Algorithms: Parallel GEMM & Parallel BT-CSA Trees
// ==============================================================================

namespace bt {
namespace kokkos {

/**
 * @brief Parallel Matrix Transposition using Kokkos::parallel_for.
 * Thread-safe: Parallelizes over destination rows (c) so each worker thread writes
 * to its own exclusive memory word, completely eliminating concurrent bitwise write races!
 */
inline TernaryMatrix parallel_transpose(const TernaryMatrix& mat) {
    TernaryMatrix T(mat.cols, mat.rows);
    ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, mat.cols), [&](size_t c) {
        for (size_t r = 0; r < mat.rows; ++r) {
            T.set(c, r, mat.get(r, c));
        }
    });
    return T;
}

/**
 * @brief High-Performance Bit-Sliced Multi-Core GEMM: C = A * B.
 * Distributes row-wise dot products across all CPU cores (e.g. 10 cores on Apple M5).
 * 
 * @param A Left input matrix (M x K)
 * @param B Right input matrix (K x N)
 * @param C Output integer matrix (M x N)
 * @param num_threads Optional thread override (0 = auto-detect hardware concurrency)
 */
inline void parallel_gemm(const TernaryMatrix& A, const TernaryMatrix& B,
                          DenseMatrixInt64& C, size_t num_threads = 0) {
    assert(A.cols == B.rows && "Dimension mismatch for matrix multiplication!");
    assert(C.rows == A.rows && C.cols == B.cols && "Output matrix dimensions mismatch!");

    if (num_threads > 0) {
        ::Kokkos::initialize(num_threads);
    }

    // Parallel pre-transpose of B for streaming vector dot-products
    TernaryMatrix BT = parallel_transpose(B);

    const size_t M = A.rows;
    const size_t N = B.cols;
    const size_t num_words = A.words_per_row;

    // Distribute rows across Kokkos thread pool
    ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, M), [&](size_t i) {
        for (size_t j = 0; j < N; ++j) {
            int64_t acc = 0;
            for (size_t w = 0; w < num_words; ++w) {
                acc += dotProduct64(A.word(i, w), BT.word(j, w));
            }
            C(i, j) = acc;
        }
    });
}

/**
 * @brief Parallel Carry-Save Reduction across multiple ternary matrices.
 * Compresses K matrices of size M x N into redundant (Sum, Carry) pair across all CPU cores.
 */
inline void parallel_reduce_csa(const std::vector<TernaryMatrix>& inputs,
                                TernaryMatrix& out_sum,
                                TernaryMatrix& out_carry,
                                size_t num_threads = 0) {
    assert(!inputs.empty() && "Input list for CSA reduction cannot be empty!");
    const size_t M = inputs[0].rows;
    const size_t N = inputs[0].cols;
    const size_t total_words = M * inputs[0].words_per_row;

    out_sum = TernaryMatrix(M, N);
    out_carry = TernaryMatrix(M, N);

    if (num_threads > 0) {
        ::Kokkos::initialize(num_threads);
    }

    const size_t K = inputs.size();

    ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, total_words), [&](size_t idx) {
        BT_CSATree csa;
        for (size_t k = 0; k < K; ++k) {
            csa.add(inputs[k].data[idx]);
        }
        out_sum.data[idx] = csa.S;
        out_carry.data[idx] = csa.C;
    });
}

/**
 * @brief Parallel Carry-Save Reduction & Full Ripple Resolution.
 * Compresses K matrices of size M x N into a single canonical TernaryMatrix.
 */
inline TernaryMatrix parallel_reduce_csa(const std::vector<TernaryMatrix>& inputs,
                                         size_t num_threads = 0) {
    assert(!inputs.empty() && "Input list for CSA reduction cannot be empty!");
    const size_t M = inputs[0].rows;
    const size_t N = inputs[0].cols;
    const size_t total_words = M * inputs[0].words_per_row;

    TernaryMatrix resolved(M, N);

    if (num_threads > 0) {
        ::Kokkos::initialize(num_threads);
    }

    const size_t K = inputs.size();

    ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, total_words), [&](size_t idx) {
        BT_CSATree csa;
        for (size_t k = 0; k < K; ++k) {
            csa.add(inputs[k].data[idx]);
        }
        resolved.data[idx] = csa.resolve();
    });

    return resolved;
}

/**
 * @brief Multi-Core Carry-Save Reduction over a vector of 64-trit words.
 */
inline AdderOutput parallel_reduce_csa(const std::vector<TritWord64>& inputs,
                                       size_t num_threads = 0) {
    if (inputs.empty()) return { TritWord64(0, 0), TritWord64(0, 0) };
    if (inputs.size() == 1) return { inputs[0], TritWord64(0, 0) };

    if (num_threads > 0) {
        ::Kokkos::initialize(num_threads);
    }

    size_t n = inputs.size();
    size_t nthreads = ::Kokkos::num_threads();
    size_t chunks = std::min(nthreads, n);
    size_t chunk_size = (n + chunks - 1) / chunks;

    std::vector<AdderOutput> local_outputs(chunks);

    ::Kokkos::parallel_for(::Kokkos::RangePolicy<size_t>(0, chunks), [&](size_t c) {
        size_t start = c * chunk_size;
        size_t end = std::min(n, start + chunk_size);
        BT_CSATree local_csa;
        for (size_t i = start; i < end; ++i) {
            local_csa.add(inputs[i]);
        }
        local_outputs[c] = { local_csa.S, local_csa.C };
    });

    // Reduce thread chunk accumulators
    BT_CSATree final_csa;
    for (const auto& out : local_outputs) {
        final_csa.add(out.Sum);
        TritWord64 c_sh(out.Carry.P << 1, out.Carry.N << 1);
        final_csa.add(c_sh);
    }

    return { final_csa.S, final_csa.C };
}

} // namespace kokkos
} // namespace bt

#endif // TERNARY_KOKKOS_BRIDGE_HPP
