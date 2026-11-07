// ==============================================================================
//  include/Ternary/Memory/Memory.hpp -- the shared memory model.
//
//  docs/SCALING_PLAN.md 3.2. Allocation is currently ad-hoc std::vector in at
//  least six translation units, with no ownership model, no device awareness
//  and no lifetime information for a lowering to schedule against. Track A
//  cannot do async GPU transfers without this; Track B cannot target an
//  accelerator with a scratchpad.
//
//  Three layers, introduced bottom-up:
//    (a) Arena / Pool   -- host allocator with explicit lifecycles
//    (b) BufferDesc     -- the abstraction both tracks share
//    (c) View           -- non-owning re-interpretation, with aliasing rules
//
//  Verification (see the bottom of this file and test/Unit/test_unit_memory2.cpp):
//    * leak / double-free -- tracked by BufferTable, asserted at scope exit
//    * aliasing           -- Z3 model in verification/verify_memory.py
//    * reuse correctness  -- differential: bit-identical with reuse on/off
//
//  Per the plan, Alloc/Copy are OPTIONAL metadata at first: a lowering may
//  ignore them. They become mandatory once every backend honours them.
// ==============================================================================

#ifndef TERNARY_MEMORY_MEMORY_HPP
#define TERNARY_MEMORY_MEMORY_HPP

#include "Ternary/IR/Tensor.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt::mem {

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

enum class DeviceKind { Host, Gpu, Accelerator };

struct Device {
    DeviceKind kind{DeviceKind::Host};
    unsigned id{0};

    bool is_host() const { return kind == DeviceKind::Host; }
    bool operator==(const Device& o) const { return kind == o.kind && id == o.id; }
    bool operator!=(const Device& o) const { return !(*this == o); }

    std::string str() const {
        switch (kind) {
        case DeviceKind::Host: return "host";
        case DeviceKind::Gpu: return "gpu:" + std::to_string(id);
        case DeviceKind::Accelerator: return "acc:" + std::to_string(id);
        }
        return "?";
    }
};

inline Device host_device() { return Device{DeviceKind::Host, 0}; }

// ---------------------------------------------------------------------------
// Lifetimes -- first-use .. last-use, so lowering can reuse scratch
// ---------------------------------------------------------------------------

struct Lifetime {
    size_t first_use{0};
    size_t last_use{0};

    bool overlaps(const Lifetime& o) const {
        return first_use <= o.last_use && o.first_use <= last_use;
    }
    size_t length() const { return last_use - first_use + 1; }
};

// ---------------------------------------------------------------------------
// (a) Arena -- bump allocator; reset() reclaims everything in O(1)
// ---------------------------------------------------------------------------

/**
 * @brief Bump allocator. There is no individual free: reset() reclaims the
 *        whole arena at once, which is O(1) and is what makes scratch reuse
 *        cheap in a lowering.
 *
 * Every allocation is aligned; the returned pointer is always at least
 * `alignof(std::max_align_t)` aligned, and larger alignments are honoured.
 * `bytes_outstanding()` returns to zero after reset(), which is the invariant
 * the leak test asserts.
 */
class Arena {
public:
    explicit Arena(size_t capacity, size_t default_align = alignof(std::max_align_t))
        : default_align_(default_align ? default_align : alignof(std::max_align_t)) {
        if (capacity == 0) throw std::invalid_argument("arena: zero capacity");
        base_ = static_cast<uint8_t*>(::operator new(capacity));
        capacity_ = capacity;
    }
    ~Arena() { ::operator delete(base_); }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void* allocate(size_t bytes, size_t align = 0) {
        const size_t a = align ? align : default_align_;
        if (a == 0 || (a & (a - 1)) != 0)
            throw std::invalid_argument("arena: alignment is not a power of two");

        uintptr_t cur = reinterpret_cast<uintptr_t>(base_) + offset_;
        uintptr_t aligned = (cur + a - 1) & ~static_cast<uintptr_t>(a - 1);
        size_t pad = static_cast<size_t>(aligned - cur);

        if (offset_ + pad + bytes > capacity_)
            throw std::bad_alloc();

        offset_ += pad + bytes;
        if (offset_ > high_water_) high_water_ = offset_;
        return reinterpret_cast<void*>(aligned);
    }

    template <typename T>
    T* allocate_array(size_t n) {
        return static_cast<T*>(allocate(n * sizeof(T), alignof(T)));
    }

    /** O(1) bulk reclaim. Outstanding bytes return to zero. */
    void reset() noexcept { offset_ = 0; }

    size_t bytes_outstanding() const noexcept { return offset_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t high_water_mark() const noexcept { return high_water_; }
    double utilisation() const noexcept {
        return capacity_ ? static_cast<double>(high_water_) /
                               static_cast<double>(capacity_) : 0.0;
    }

private:
    uint8_t* base_{nullptr};
    size_t capacity_{0};
    size_t offset_{0};
    size_t high_water_{0};
    size_t default_align_;
};

// ---------------------------------------------------------------------------
// (a) Pool -- fixed-size block pool for hot temporaries
// ---------------------------------------------------------------------------

/**
 * @brief Fixed-size block pool. Allocate and free are O(1) and never touch
 *        the system allocator after construction.
 *
 * Unlike the Arena, this supports individual free -- which is what hot,
 * short-lived temporaries of a uniform size need.
 */
class Pool {
public:
    Pool(size_t block_size, size_t block_count)
        : block_size_(block_size), block_count_(block_count) {
        if (!block_size || !block_count)
            throw std::invalid_argument("pool: zero size or count");
        storage_ = static_cast<uint8_t*>(
            ::operator new(block_size * block_count));
        free_.reserve(block_count);
        for (size_t i = block_count; i-- > 0;) free_.push_back(i);
    }
    ~Pool() { ::operator delete(storage_); }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    void* acquire() {
        if (free_.empty()) throw std::bad_alloc();
        const size_t i = free_.back();
        free_.pop_back();
        ++live_;
        if (live_ > high_water_) high_water_ = live_;
        return storage_ + i * block_size_;
    }

    void release(void* p) noexcept {
        if (!p) return;
        const auto* q = static_cast<const uint8_t*>(p);
        if (q < storage_ || q >= storage_ + block_size_ * block_count_) return;
        const size_t i = static_cast<size_t>(q - storage_) / block_size_;
        free_.push_back(i);
        if (live_) --live_;
    }

    size_t block_size() const noexcept { return block_size_; }
    size_t capacity() const noexcept { return block_count_; }
    size_t live() const noexcept { return live_; }
    size_t high_water_mark() const noexcept { return high_water_; }

private:
    uint8_t* storage_{nullptr};
    size_t block_size_{0};
    size_t block_count_{0};
    size_t live_{0};
    size_t high_water_{0};
    std::vector<size_t> free_;
};

// ---------------------------------------------------------------------------
// (b) BufferDesc -- the abstraction both tracks share
// ---------------------------------------------------------------------------

struct BufferDesc {
    uint64_t handle{0};
    ir::TensorType type;          // element, dims, layout
    Device device;                // Host / Gpu / Accelerator(id)
    size_t bytes{0};
    Lifetime lifetime;            // first-use .. last-use, for reuse

    bool valid() const { return handle != 0; }
    bool on_host() const { return device.is_host(); }
};

// ---------------------------------------------------------------------------
// (c) View -- non-owning re-interpretation, with an aliasing rule
// ---------------------------------------------------------------------------

/**
 * @brief A non-owning window onto a buffer: offset, shape, strides.
 *
 * Two views ALIAS when their byte intervals intersect. That rule is proved
 * in verification/verify_memory.py; the point of proving it rather than
 * eyeballing it is that a strided view has a byte extent that is NOT
 * `product(shape)` -- the last stride can be larger than the element size,
 * and a naive extent computation misses overlaps.
 */
struct View {
    uint64_t buffer{0};
    size_t offset_bytes{0};
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;   // in ELEMENTS, matching shape

    /**
     * @brief Byte span of the view: highest touched address minus lowest,
     *        plus one element.
     *
     * This is the SUM over i of (shape[i]-1)*stride[i] -- NOT the max, and
     * NOT the element count.
     *
     * Why the sum: indices are independent, so a view can sit at the last
     * index of every dimension at once. For shape {3,2} strides {1,2} the
     * offsets are i0*1 + i1*2 over i0<3, i1<2, giving {0,1,2,3,4}: the max
     * offset is (3-1)*1 + (2-1)*2 = 4, whereas the max-over-dimensions
     * formula gives 2 and the element-count formula gives 6. Both are wrong,
     * and the max-over-dimensions error UNDER-reports the span, which makes
     * an aliasing check miss real overlaps. Found by
     * verification/verify_memory.py M1, not by inspection.
     *
     * For strided views the byte interval is a superset of the addresses
     * actually touched, so aliases() below is deliberately conservative:
     * it may report an overlap where the touched sets are disjoint. It must
     * never do the reverse -- see verify_memory.py M2.
     */
    size_t extent_bytes(size_t elem_size) const {
        if (shape.empty()) return elem_size;
        int64_t max_off = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            const int64_t s = (i < strides.size()) ? strides[i] : 1;
            if (shape[i] > 1) max_off += (shape[i] - 1) * s;
        }
        return static_cast<size_t>(max_off) * elem_size + elem_size;
    }

    size_t begin_bytes() const { return offset_bytes; }
    size_t end_bytes(size_t elem_size) const {
        return offset_bytes + extent_bytes(elem_size);
    }
};

/** True when two views of the same buffer overlap in bytes. */
inline bool aliases(const View& a, const View& b, size_t elem_size) {
    if (a.buffer != b.buffer) return false;
    return a.begin_bytes() < b.end_bytes(elem_size) &&
           b.begin_bytes() < a.end_bytes(elem_size);
}

// ---------------------------------------------------------------------------
// BufferTable -- ownership tracking, leak and double-free detection
// ---------------------------------------------------------------------------

/**
 * @brief Tracks live buffers so leaks and double-frees are reported rather
 *        than lurking. Assert `live() == 0` at scope exit.
 */
class BufferTable {
public:
    uint64_t alloc(const ir::TensorType& type, Device dev, size_t bytes,
                   Lifetime lt = {}) {
        BufferDesc d;
        d.handle = ++next_handle_;
        d.type = type;
        d.device = dev;
        d.bytes = bytes;
        d.lifetime = lt;
        live_[d.handle] = d;
        live_bytes_ += bytes;
        if (live_.size() > peak_) peak_ = live_.size();
        return d.handle;
    }

    void free(uint64_t handle) {
        auto it = live_.find(handle);
        if (it == live_.end())
            throw std::runtime_error("buffer table: double free or unknown "
                                     "handle " + std::to_string(handle));
        live_bytes_ -= it->second.bytes;
        live_.erase(it);
    }

    bool contains(uint64_t handle) const { return live_.count(handle) != 0; }
    const BufferDesc* get(uint64_t handle) const {
        auto it = live_.find(handle);
        return it == live_.end() ? nullptr : &it->second;
    }

    size_t count() const { return live_.size(); }
    size_t bytes() const { return live_bytes_; }
    size_t peak_count() const { return peak_; }

    /** Total bytes live on a given device. */
    size_t bytes_on(const Device& d) const {
        size_t n = 0;
        for (const auto& kv : live_)
            if (kv.second.device == d) n += kv.second.bytes;
        return n;
    }

    std::vector<BufferDesc> snapshot() const {
        std::vector<BufferDesc> out;
        for (const auto& kv : live_) out.push_back(kv.second);
        return out;
    }

private:
    uint64_t next_handle_{0};
    size_t peak_{0};
    std::map<uint64_t, BufferDesc> live_;
    size_t live_bytes_{0};
};

}  // namespace bt::mem

#endif  // TERNARY_MEMORY_MEMORY_HPP
