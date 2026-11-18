/**
 * @file AlgebraicVirtualMemory.hpp
 * @brief Algebraic Virtual Memory & Zero-Swap Out-of-Core Computing for Trium-XBT
 *
 * Provides four orthogonal primitives for computing on problem sizes far
 * exceeding physical RAM without OS swap or explicit sparse matrix storage:
 *  1. PhysicalMemoryMonitor   - Cross-platform (macOS/Linux) RSS, peak RSS, faults, swaps.
 *  2. ZeroSwapStreamStore<T>  - Out-of-core unlinked-file slab store bounded by slab_bytes.
 *  3. ReversibleSymplecticTape- O(1)-RAM reversible trajectory over SymplecticLeapfrogEngine.
 *  4. VirtualStencil3D        - Matrix-free 3D 7-point Laplacian via 3-plane rolling window.
 */

#ifndef BT_ALGEBRAIC_VIRTUAL_MEMORY_HPP
#define BT_ALGEBRAIC_VIRTUAL_MEMORY_HPP

#include "ternary_core.hpp"
#include "symplectic_ternary_engine.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/task_info.h>
#elif defined(__linux__)
#include <cstdio>
#endif

namespace bt {
namespace vmem {

// ============================================================================
// 1. PhysicalMemoryMonitor
// ============================================================================

class PhysicalMemoryMonitor {
public:
    struct MemorySnapshot {
        uint64_t resident_bytes{0};
        uint64_t peak_rss_bytes{0};
        uint64_t page_faults_major{0};
        uint64_t swaps{0};
    };

    static MemorySnapshot sample() noexcept {
        MemorySnapshot snap{};

        struct rusage ru{};
        if (::getrusage(RUSAGE_SELF, &ru) == 0) {
            snap.page_faults_major = static_cast<uint64_t>(ru.ru_majflt);
            snap.swaps = static_cast<uint64_t>(ru.ru_nswap);
#if defined(__APPLE__)
            // On macOS, ru_maxrss is reported in bytes.
            snap.peak_rss_bytes = static_cast<uint64_t>(ru.ru_maxrss);
#elif defined(__linux__)
            // On Linux, ru_maxrss is reported in kilobytes.
            snap.peak_rss_bytes = static_cast<uint64_t>(ru.ru_maxrss) * 1024ULL;
#endif
        }

#if defined(__APPLE__)
        mach_task_basic_info_data_t info{};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (::task_info(::mach_task_self(),
                        MACH_TASK_BASIC_INFO,
                        reinterpret_cast<task_info_t>(&info),
                        &count) == KERN_SUCCESS) {
            snap.resident_bytes = static_cast<uint64_t>(info.resident_size);
        }
#elif defined(__linux__)
        if (FILE* f = std::fopen("/proc/self/statm", "r")) {
            unsigned long long size_pages = 0;
            unsigned long long resident_pages = 0;
            if (std::fscanf(f, "%llu %llu", &size_pages, &resident_pages) == 2) {
                const long page_sz = ::sysconf(_SC_PAGESIZE);
                const uint64_t bytes_per_page =
                    static_cast<uint64_t>(page_sz > 0 ? page_sz : 4096);
                snap.resident_bytes =
                    static_cast<uint64_t>(resident_pages) * bytes_per_page;
            }
            std::fclose(f);
        }
#endif

        return snap;
    }
};

// ============================================================================
// 2. ZeroSwapStreamStore<T>
// ============================================================================

template <typename T>
    requires std::is_trivially_copyable_v<T>
class ZeroSwapStreamStore {
public:
    explicit ZeroSwapStreamStore(size_t num_elements,
                                 size_t slab_bytes = 2 * 1024 * 1024)
        : num_elements_(num_elements) {
        if (slab_bytes < sizeof(T)) {
            throw std::invalid_argument(
                "ZeroSwapStreamStore: slab_bytes must be >= sizeof(T)");
        }
        slab_capacity_elements_ = slab_bytes / sizeof(T);
        slab_bytes_ = slab_capacity_elements_ * sizeof(T);

        const char* tmpdir = std::getenv("TMPDIR");
        std::string pattern = (tmpdir && tmpdir[0] != '\0') ? std::string(tmpdir)
                                                            : std::string("/tmp");
        if (pattern.back() != '/') {
            pattern.push_back('/');
        }
        pattern += "bt_vmem_slab_XXXXXX";

        std::vector<char> path_buf(pattern.begin(), pattern.end());
        path_buf.push_back('\0');

        fd_ = ::mkstemp(path_buf.data());
        if (fd_ < 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "ZeroSwapStreamStore: mkstemp failed");
        }
        // Unlink immediately so the OS reclaims the backing file on close/exit.
        ::unlink(path_buf.data());

        if (num_elements_ > 0) {
            const off_t total_bytes =
                static_cast<off_t>(num_elements_ * sizeof(T));
            while (::ftruncate(fd_, total_bytes) < 0) {
                if (errno == EINTR) {
                    continue;
                }
                const int err = errno;
                ::close(fd_);
                fd_ = -1;
                throw std::system_error(err, std::generic_category(),
                                        "ZeroSwapStreamStore: ftruncate failed");
            }
        }
    }

    ~ZeroSwapStreamStore() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    ZeroSwapStreamStore(const ZeroSwapStreamStore&) = delete;
    ZeroSwapStreamStore& operator=(const ZeroSwapStreamStore&) = delete;

    ZeroSwapStreamStore(ZeroSwapStreamStore&& other) noexcept
        : fd_(other.fd_),
          num_elements_(other.num_elements_),
          slab_capacity_elements_(other.slab_capacity_elements_),
          slab_bytes_(other.slab_bytes_) {
        other.fd_ = -1;
        other.num_elements_ = 0;
        other.slab_capacity_elements_ = 0;
        other.slab_bytes_ = 0;
    }

    ZeroSwapStreamStore& operator=(ZeroSwapStreamStore&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) {
                ::close(fd_);
            }
            fd_ = other.fd_;
            num_elements_ = other.num_elements_;
            slab_capacity_elements_ = other.slab_capacity_elements_;
            slab_bytes_ = other.slab_bytes_;
            other.fd_ = -1;
            other.num_elements_ = 0;
            other.slab_capacity_elements_ = 0;
            other.slab_bytes_ = 0;
        }
        return *this;
    }

    size_t size() const noexcept { return num_elements_; }
    size_t slab_capacity_elements() const noexcept { return slab_capacity_elements_; }
    size_t virtual_bytes() const noexcept { return num_elements_ * sizeof(T); }
    size_t resident_budget_bytes() const noexcept { return slab_bytes_; }

    template <typename GeneratorFn>
    void generate(GeneratorFn&& fn) {
        if (num_elements_ == 0) {
            return;
        }
        const size_t buf_elems = std::min(num_elements_, slab_capacity_elements_);
        std::vector<T> slab(buf_elems);

        size_t offset = 0;
        while (offset < num_elements_) {
            const size_t chunk = std::min(buf_elems, num_elements_ - offset);
            for (size_t i = 0; i < chunk; ++i) {
                if constexpr (std::is_invocable_r_v<T, GeneratorFn&, size_t>) {
                    slab[i] = fn(offset + i);
                } else {
                    fn(offset + i, slab[i]);
                }
            }
            pwrite_exact(fd_, slab.data(), chunk * sizeof(T),
                         static_cast<off_t>(offset * sizeof(T)));
            offset += chunk;
        }
    }

    template <typename TransformFn>
    void transform_to(ZeroSwapStreamStore<T>& dst, TransformFn&& fn) const {
        if (dst.size() != num_elements_) {
            throw std::invalid_argument(
                "ZeroSwapStreamStore::transform_to: destination size mismatch");
        }
        if (num_elements_ == 0) {
            return;
        }
        const size_t buf_elems = std::min(
            num_elements_,
            std::min(slab_capacity_elements_, dst.slab_capacity_elements()));
        std::vector<T> in_slab(buf_elems);
        std::vector<T> out_slab(buf_elems);

        size_t offset = 0;
        while (offset < num_elements_) {
            const size_t chunk = std::min(buf_elems, num_elements_ - offset);
            pread_exact(fd_, in_slab.data(), chunk * sizeof(T),
                        static_cast<off_t>(offset * sizeof(T)));
            for (size_t i = 0; i < chunk; ++i) {
                if constexpr (std::is_invocable_v<TransformFn&, size_t, const T&>) {
                    out_slab[i] = fn(offset + i, in_slab[i]);
                } else {
                    out_slab[i] = fn(in_slab[i]);
                }
            }
            pwrite_exact(dst.fd_, out_slab.data(), chunk * sizeof(T),
                         static_cast<off_t>(offset * sizeof(T)));
            offset += chunk;
        }
    }

    template <typename Init, typename AccFn>
    Init stream_reduce(Init init, AccFn&& fn) const {
        if (num_elements_ == 0) {
            return init;
        }
        const size_t buf_elems = std::min(num_elements_, slab_capacity_elements_);
        std::vector<T> slab(buf_elems);

        size_t offset = 0;
        while (offset < num_elements_) {
            const size_t chunk = std::min(buf_elems, num_elements_ - offset);
            pread_exact(fd_, slab.data(), chunk * sizeof(T),
                        static_cast<off_t>(offset * sizeof(T)));
            for (size_t i = 0; i < chunk; ++i) {
                if constexpr (std::is_invocable_v<AccFn&, Init, size_t, const T&>) {
                    init = fn(std::move(init), offset + i, slab[i]);
                } else {
                    init = fn(std::move(init), slab[i]);
                }
            }
            offset += chunk;
        }
        return init;
    }

    template <typename AccFn>
    auto stream_reduce(AccFn&& fn) const {
        if constexpr (std::is_invocable_v<AccFn&, T, const T&>) {
            return stream_reduce(T{}, std::forward<AccFn>(fn));
        } else {
            if (num_elements_ == 0) {
                return;
            }
            const size_t buf_elems = std::min(num_elements_, slab_capacity_elements_);
            std::vector<T> slab(buf_elems);

            size_t offset = 0;
            while (offset < num_elements_) {
                const size_t chunk = std::min(buf_elems, num_elements_ - offset);
                pread_exact(fd_, slab.data(), chunk * sizeof(T),
                            static_cast<off_t>(offset * sizeof(T)));
                for (size_t i = 0; i < chunk; ++i) {
                    if constexpr (std::is_invocable_v<AccFn&, size_t, const T&>) {
                        fn(offset + i, slab[i]);
                    } else {
                        fn(slab[i]);
                    }
                }
                offset += chunk;
            }
        }
    }

    T read_element(size_t idx) const {
        if (idx >= num_elements_) {
            throw std::out_of_range("ZeroSwapStreamStore::read_element: index out of range");
        }
        T val{};
        pread_exact(fd_, &val, sizeof(T), static_cast<off_t>(idx * sizeof(T)));
        return val;
    }

    void write_element(size_t idx, const T& val) {
        if (idx >= num_elements_) {
            throw std::out_of_range("ZeroSwapStreamStore::write_element: index out of range");
        }
        pwrite_exact(fd_, &val, sizeof(T), static_cast<off_t>(idx * sizeof(T)));
    }

    void read_range(size_t start_idx, T* dst, size_t count) const {
        if (start_idx > num_elements_ || count > num_elements_ - start_idx) {
            throw std::out_of_range("ZeroSwapStreamStore::read_range: range out of bounds");
        }
        if (count == 0) {
            return;
        }
        pread_exact(fd_, dst, count * sizeof(T),
                    static_cast<off_t>(start_idx * sizeof(T)));
    }

    void write_range(size_t start_idx, const T* src, size_t count) {
        if (start_idx > num_elements_ || count > num_elements_ - start_idx) {
            throw std::out_of_range("ZeroSwapStreamStore::write_range: range out of bounds");
        }
        if (count == 0) {
            return;
        }
        pwrite_exact(fd_, src, count * sizeof(T),
                     static_cast<off_t>(start_idx * sizeof(T)));
    }

private:
    static void pread_exact(int fd, void* dst, size_t bytes, off_t offset) {
        auto* ptr = static_cast<uint8_t*>(dst);
        size_t remaining = bytes;
        off_t cur_off = offset;
        while (remaining > 0) {
            const ssize_t n = ::pread(fd, ptr, remaining, cur_off);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::system_error(errno, std::generic_category(),
                                        "ZeroSwapStreamStore: pread failed");
            }
            if (n == 0) {
                throw std::runtime_error("ZeroSwapStreamStore: unexpected EOF in pread");
            }
            ptr += n;
            remaining -= static_cast<size_t>(n);
            cur_off += n;
        }
    }

    static void pwrite_exact(int fd, const void* src, size_t bytes, off_t offset) {
        const auto* ptr = static_cast<const uint8_t*>(src);
        size_t remaining = bytes;
        off_t cur_off = offset;
        while (remaining > 0) {
            const ssize_t n = ::pwrite(fd, ptr, remaining, cur_off);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::system_error(errno, std::generic_category(),
                                        "ZeroSwapStreamStore: pwrite failed");
            }
            ptr += n;
            remaining -= static_cast<size_t>(n);
            cur_off += n;
        }
    }

    int fd_{-1};
    size_t num_elements_{0};
    size_t slab_capacity_elements_{0};
    size_t slab_bytes_{0};
};

// ============================================================================
// 3. ReversibleSymplecticTape
// ============================================================================

class ReversibleSymplecticTape {
public:
    using PhaseState = bt::symplectic::PhaseState;

    struct Checkpoint {
        uint64_t step{0};
        PhaseState state{};
    };

    explicit ReversibleSymplecticTape(PhaseState initial_state = {},
                                      uint64_t checkpoint_interval = 0) noexcept
        : state_(initial_state),
          checkpoint_interval_(checkpoint_interval) {}

    void step_forward(uint64_t steps, bt::TritWord64 force_mask) {
        for (uint64_t i = 0; i < steps; ++i) {
            state_ = bt::symplectic::SymplecticLeapfrogEngine::forward_step(
                state_, force_mask);
            ++step_count_;
            if (checkpoint_interval_ > 0 &&
                (step_count_ % checkpoint_interval_) == 0) {
                checkpoints_.push_back(Checkpoint{step_count_, state_});
            }
        }
    }

    void step_backward(uint64_t steps, bt::TritWord64 force_mask) {
        if (steps > step_count_) {
            throw std::underflow_error(
                "ReversibleSymplecticTape::step_backward: cannot reverse past step 0");
        }
        for (uint64_t i = 0; i < steps; ++i) {
            state_ = bt::symplectic::SymplecticLeapfrogEngine::reverse_step(
                state_, force_mask);
            --step_count_;
            while (!checkpoints_.empty() && checkpoints_.back().step > step_count_) {
                checkpoints_.pop_back();
            }
        }
    }

    uint64_t step_count() const noexcept { return step_count_; }

    size_t history_ram_bytes() const noexcept {
        return sizeof(PhaseState) + checkpoints_.size() * sizeof(Checkpoint);
    }

    uint64_t virtual_history_bytes() const noexcept {
        return step_count_ * static_cast<uint64_t>(sizeof(PhaseState));
    }

    PhaseState current_state() const noexcept { return state_; }

private:
    PhaseState state_{};
    uint64_t step_count_{0};
    uint64_t checkpoint_interval_{0};
    std::vector<Checkpoint> checkpoints_;
};

// ============================================================================
// 4. VirtualStencil3D
// ============================================================================

template <typename Scalar = double>
class VirtualStencil3D {
public:
    VirtualStencil3D(size_t nx, size_t ny, size_t nz)
        : nx_(nx), ny_(ny), nz_(nz) {
        if (nx_ == 0 || ny_ == 0 || nz_ == 0) {
            throw std::invalid_argument(
                "VirtualStencil3D: grid dimensions must be non-zero");
        }
    }

    size_t nx() const noexcept { return nx_; }
    size_t ny() const noexcept { return ny_; }
    size_t nz() const noexcept { return nz_; }
    size_t total_points() const noexcept { return nx_ * ny_ * nz_; }

    constexpr size_t matrix_ram_bytes() const noexcept { return 0; }

    size_t rolling_window_bytes() const noexcept {
        return 3 * nx_ * ny_ * sizeof(Scalar);
    }

    size_t explicit_csr_bytes_estimate() const noexcept {
        // 7-point stencil: 7 non-zeros/row (value + col_idx) + 1 row_ptr/row
        return total_points() *
               (7 * (sizeof(Scalar) + sizeof(uint64_t)) + sizeof(uint64_t));
    }

    /**
     * @brief Applies the exact 7-point 3D discrete Laplacian y = L * x
     * using zero matrix storage and a 3-plane rolling window in RAM.
     * Uses homogeneous Dirichlet (zero) boundary conditions outside the domain.
     */
    void apply_laplacian_3d(const ZeroSwapStreamStore<Scalar>& x,
                            ZeroSwapStreamStore<Scalar>& y) const {
        const size_t n_total = total_points();
        if (x.size() != n_total || y.size() != n_total) {
            throw std::invalid_argument(
                "VirtualStencil3D::apply_laplacian_3d: store size mismatch");
        }

        const size_t plane_elems = nx_ * ny_;
        std::vector<Scalar> prev_plane(plane_elems, Scalar{0});
        std::vector<Scalar> curr_plane(plane_elems, Scalar{0});
        std::vector<Scalar> next_plane(plane_elems, Scalar{0});
        std::vector<Scalar> out_plane(plane_elems, Scalar{0});

        // Preload z = 0 into curr_plane
        x.read_range(0, curr_plane.data(), plane_elems);

        for (size_t z = 0; z < nz_; ++z) {
            if (z + 1 < nz_) {
                x.read_range((z + 1) * plane_elems, next_plane.data(), plane_elems);
            } else {
                std::fill(next_plane.begin(), next_plane.end(), Scalar{0});
            }

            for (size_t iy = 0; iy < ny_; ++iy) {
                const size_t row_off = iy * nx_;
                for (size_t ix = 0; ix < nx_; ++ix) {
                    const size_t idx = row_off + ix;
                    const Scalar c = curr_plane[idx];
                    const Scalar xm = (ix > 0) ? curr_plane[idx - 1] : Scalar{0};
                    const Scalar xp = (ix + 1 < nx_) ? curr_plane[idx + 1] : Scalar{0};
                    const Scalar ym = (iy > 0) ? curr_plane[idx - nx_] : Scalar{0};
                    const Scalar yp = (iy + 1 < ny_) ? curr_plane[idx + nx_] : Scalar{0};
                    const Scalar zm = (z > 0) ? prev_plane[idx] : Scalar{0};
                    const Scalar zp = (z + 1 < nz_) ? next_plane[idx] : Scalar{0};

                    out_plane[idx] =
                        static_cast<Scalar>(-6) * c + xm + xp + ym + yp + zm + zp;
                }
            }

            y.write_range(z * plane_elems, out_plane.data(), plane_elems);

            // Rotate the 3-plane rolling window without reallocating
            std::swap(prev_plane, curr_plane);
            std::swap(curr_plane, next_plane);
        }
    }

private:
    size_t nx_{0};
    size_t ny_{0};
    size_t nz_{0};
};

VirtualStencil3D(size_t, size_t, size_t) -> VirtualStencil3D<double>;

} // namespace vmem
} // namespace bt

#endif // BT_ALGEBRAIC_VIRTUAL_MEMORY_HPP
