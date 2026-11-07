// ==============================================================================
//  include/Ternary/IR/Tensor.hpp -- shape-carrying types and shape algebra.
//
//  docs/SCALING_PLAN.md 3.1. `Type` stays a flat scalar enum -- turning it into
//  a class would touch every switch in the tree. Instead `Type` becomes the
//  ELEMENT kind and TensorType carries the shape alongside it.
//
//  Scope: STATIC SHAPES ONLY. Dynamic (data-dependent) dims are a large scope
//  increase and are explicitly out of scope; a dim <= 0 is treated as invalid.
//
//  Every function here is total: it returns std::nullopt rather than throwing
//  or asserting, so a shape error is a reported diagnostic instead of UB. The
//  shape algebra is verified in verification/verify_tensor_shapes.py.
// ==============================================================================

#ifndef TERNARY_IR_TENSOR_HPP
#define TERNARY_IR_TENSOR_HPP

#include "Core.hpp"
#include "Ternary/Target/GeneratedTargets.hpp"

#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace bt::ir {

// ---------------------------------------------------------------------------
// TensorType
// ---------------------------------------------------------------------------

/**
 * @brief A type with a shape.
 *
 * `dims` empty means scalar. `{n}` is a vector, `{m,n,...}` a tensor.
 * `layout` is the storage format, reusing bt::target::Layout so that the
 * layout vocabulary has exactly one definition (see docs/SCALING_PLAN.md 3.4).
 */
struct TensorType {
    Type element{Type::Int39};
    std::vector<int64_t> dims;
    bt::target::Layout layout{bt::target::Layout::DualRail};

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    bool is_scalar() const { return dims.empty(); }
    bool is_vector() const { return dims.size() == 1; }

    /**
     * @brief Number of elements, or -1 if any dim is not a positive static
     *        size. Never returns 0 for a valid type.
     */
    int64_t num_elements() const {
        if (dims.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : dims) {
            if (d <= 0) return -1;
            if (d > (INT64_MAX / n)) return -1;      // overflow guard
            n *= d;
        }
        return n;
    }

    /** True when every dim is a positive static size. */
    bool is_static() const { return num_elements() > 0; }

    /** Same element kind and same dims. Layout is NOT part of conformance:
     *  two views of the same logical tensor in different layouts conform. */
    bool conforms_to(const TensorType& o) const {
        return element == o.element && dims == o.dims;
    }

    bool operator==(const TensorType& o) const {
        return element == o.element && dims == o.dims && layout == o.layout;
    }
    bool operator!=(const TensorType& o) const { return !(*this == o); }

    std::string str() const {
        std::string s = "tensor<";
        if (dims.empty()) {
            s += "scalar";
        } else {
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i) s += "x";
                s += std::to_string(dims[i]);
            }
        }
        s += ":";
        switch (element) {
        case Type::Int39: s += "int39"; break;
        case Type::Trit: s += "trit"; break;
        case Type::Positional39: s += "pos39"; break;
        case Type::Packed64: s += "packed64"; break;
        case Type::Float64: s += "f64"; break;
        default: s += "void"; break;
        }
        s += ":";
        s += bt::target::layout_name(layout);
        s += ">";
        return s;
    }
};

// ---------------------------------------------------------------------------
// Shape algebra -- total functions returning nullopt on a shape error
// ---------------------------------------------------------------------------

/// Right-aligned NumPy-style broadcast. {} broadcasts to anything.
inline std::optional<std::vector<int64_t>> broadcast_shapes(
    const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    const size_t n = a.size() > b.size() ? a.size() : b.size();
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        // Right-align: index from the trailing dimension.
        const int64_t da = (i < a.size()) ? a[a.size() - 1 - i] : -1;
        const int64_t db = (i < b.size()) ? b[b.size() - 1 - i] : -1;
        if (da == -1 && db == -1) return std::nullopt;      // both absent
        if (da == -1) { out[n - 1 - i] = db; continue; }
        if (db == -1) { out[n - 1 - i] = da; continue; }
        if (da == db) { out[n - 1 - i] = da; continue; }
        if (da == 1) { out[n - 1 - i] = db; continue; }
        if (db == 1) { out[n - 1 - i] = da; continue; }
        return std::nullopt;                                 // incompatible
    }
    return out;
}

/// Matrix product shape: [...,m,k] x [...,k,n] -> [...,m,n].
/// Requires rank >= 2 and matching inner dims. Batch dims broadcast.
inline std::optional<std::vector<int64_t>> matmul_shape(
    const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    if (a.size() < 2 || b.size() < 2) return std::nullopt;
    if (a.back() != b[b.size() - 2]) return std::nullopt;    // k mismatch

    const size_t ranks = a.size() > b.size() ? a.size() : b.size();
    std::vector<int64_t> batch_a, batch_b;
    if (a.size() > 2) batch_a.assign(a.begin(), a.end() - 2);
    if (b.size() > 2) batch_b.assign(b.begin(), b.end() - 2);
    (void)ranks;

    std::optional<std::vector<int64_t>> batch;
    if (batch_a.empty() && batch_b.empty()) {
        batch = std::vector<int64_t>{};
    } else if (batch_a.empty()) {
        batch = batch_b;
    } else if (batch_b.empty()) {
        batch = batch_a;
    } else {
        batch = broadcast_shapes(batch_a, batch_b);
        if (!batch) return std::nullopt;
    }

    std::vector<int64_t> out = *batch;
    out.push_back(a[a.size() - 2]);      // m
    out.push_back(b.back());             // n
    return out;
}

/// Permute dims. `perm` must be a permutation of [0, rank).
inline std::optional<std::vector<int64_t>> transpose_shape(
    const std::vector<int64_t>& dims, const std::vector<int64_t>& perm) {
    if (perm.size() != dims.size()) return std::nullopt;
    std::vector<bool> seen(dims.size(), false);
    for (int64_t p : perm) {
        if (p < 0 || static_cast<size_t>(p) >= dims.size()) return std::nullopt;
        if (seen[static_cast<size_t>(p)]) return std::nullopt;   // not a perm
        seen[static_cast<size_t>(p)] = true;
    }
    std::vector<int64_t> out(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
        out[i] = dims[static_cast<size_t>(perm[i])];
    return out;
}

/// Drop the reduced axes (`keep` semantics: axes listed are removed).
inline std::optional<std::vector<int64_t>> reduce_shape(
    const std::vector<int64_t>& dims, const std::vector<int64_t>& axes,
    bool keep_dims = false) {
    std::vector<bool> drop(dims.size(), false);
    for (int64_t ax : axes) {
        int64_t a = ax < 0 ? ax + static_cast<int64_t>(dims.size()) : ax;
        if (a < 0 || a >= static_cast<int64_t>(dims.size()))
            return std::nullopt;
        if (drop[static_cast<size_t>(a)]) return std::nullopt;  // duplicate
        drop[static_cast<size_t>(a)] = true;
    }
    std::vector<int64_t> out;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (drop[i]) {
            if (keep_dims) out.push_back(1);
        } else {
            out.push_back(dims[i]);
        }
    }
    return out;
}

/// Reshape: element count must match. At most one -1, inferred.
inline std::optional<std::vector<int64_t>> reshape_shape(
    const std::vector<int64_t>& dims, const std::vector<int64_t>& newdims) {
    int64_t total = 1;
    bool any_dynamic = false;
    for (int64_t d : dims) {
        if (d <= 0) return std::nullopt;
        total *= d;
    }
    int64_t known = 1;
    size_t infer_at = SIZE_MAX;
    for (size_t i = 0; i < newdims.size(); ++i) {
        if (newdims[i] == -1) {
            if (any_dynamic) return std::nullopt;    // at most one -1
            any_dynamic = true;
            infer_at = i;
        } else if (newdims[i] <= 0) {
            return std::nullopt;
        } else {
            known *= newdims[i];
        }
    }
    std::vector<int64_t> out = newdims;
    if (any_dynamic) {
        if (known == 0 || total % known != 0) return std::nullopt;
        out[infer_at] = total / known;
    } else if (known != total) {
        return std::nullopt;
    }
    return out;
}

/// Inverse of a permutation.
inline std::vector<int64_t> invert_permutation(const std::vector<int64_t>& perm) {
    std::vector<int64_t> inv(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
        inv[static_cast<size_t>(perm[i])] = static_cast<int64_t>(i);
    return inv;
}

// ---------------------------------------------------------------------------
// Index arithmetic (row-major / C order)
// ---------------------------------------------------------------------------

/// Strides for a row-major layout of `dims`.
inline std::vector<int64_t> row_major_strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> st(dims.size(), 1);
    for (size_t i = dims.size(); i-- > 1;)
        st[i - 1] = st[i] * dims[i];
    return st;
}

inline int64_t linear_index(const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& idx) {
    if (dims.size() != idx.size()) return -1;
    const std::vector<int64_t> st = row_major_strides(dims);
    int64_t lin = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= dims[i]) return -1;      // out of bounds
        lin += idx[i] * st[i];
    }
    return lin;
}

inline std::vector<int64_t> unlinear_index(const std::vector<int64_t>& dims,
                                           int64_t lin) {
    std::vector<int64_t> idx(dims.size(), 0);
    if (dims.empty()) return idx;
    const std::vector<int64_t> st = row_major_strides(dims);
    int64_t total = 1;
    for (int64_t d : dims) total *= d;
    if (lin < 0 || lin >= total) return {};                  // empty == invalid
    for (size_t i = 0; i < dims.size(); ++i) {
        idx[i] = (st[i] == 0) ? 0 : (lin / st[i]) % dims[i];
    }
    return idx;
}

}  // namespace bt::ir

#endif  // TERNARY_IR_TENSOR_HPP
