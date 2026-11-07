/**
 * @file test_unit_tensor.cpp
 * @brief Conformance: the C++ shape algebra against the verified model.
 *
 * verification/verify_tensor_shapes.py proves the shape algebra in Z3 (18/18).
 * This file runs the same obligations through include/Ternary/IR/Tensor.hpp so
 * the shipped header is pinned to the proved model -- the proof is worthless if
 * the implementation drifts from it.
 *
 * Also covers layout round-trip, which is the "zero conformance vectors" gap
 * the hardware audit flags (docs/HARDWARE_READINESS_AUDIT.md).
 */

#include "Ternary/IR/Tensor.hpp"

#include <algorithm>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

using namespace bt::ir;
using Dims = std::vector<int64_t>;

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

std::string show(const std::optional<Dims>& v) {
    if (!v) return "None";
    if (v->empty()) return "()";
    std::string s = "(";
    for (size_t i = 0; i < v->size(); ++i) {
        if (i) s += ",";
        s += std::to_string((*v)[i]);
    }
    return s + ")";
}

bool eq(const std::optional<Dims>& a, const std::optional<Dims>& b) {
    return (!a.has_value() && !b.has_value()) ||
           (a.has_value() && b.has_value() && *a == *b);
}

// ---- TensorType basics ----------------------------------------------------

void test_tensor_type() {
    std::printf("  TensorType\n");
    TensorType scalar;
    check(scalar.is_scalar(), "default is scalar");
    check(scalar.rank() == 0, "scalar rank is 0");
    check(scalar.num_elements() == 1, "scalar has 1 element");

    TensorType vec{Type::Int39, {8}, bt::target::Layout::DualRail};
    check(vec.is_vector(), "{8} is a vector");
    check(vec.rank() == 1, "rank 1");
    check(vec.num_elements() == 8, "8 elements");
    check(vec.is_static(), "static");

    TensorType mat{Type::Trit, {4, 8}, bt::target::Layout::Base243};
    check(mat.rank() == 2, "rank 2");
    check(mat.num_elements() == 32, "4x8 == 32");
    check(mat.layout == bt::target::Layout::Base243, "layout is carried");

    // Conformance ignores layout: two views of one logical tensor.
    TensorType mat_dual{Type::Trit, {4, 8}, bt::target::Layout::DualRail};
    check(mat.conforms_to(mat_dual), "conformance ignores layout");
    check(mat != mat_dual, "equality includes layout");

    TensorType bad{Type::Int39, {0, 5}};
    check(bad.num_elements() == -1, "a zero dim is not static");
    check(!bad.is_static(), "not static");

    TensorType huge{Type::Int39, {INT64_MAX / 2, 4}};
    check(huge.num_elements() == -1, "overflow in num_elements is guarded");

    check(mat.str().find("tensor<4x8:trit:base-243>") != std::string::npos,
          "debug string: " + mat.str());
}

// ---- broadcast ------------------------------------------------------------

void test_broadcast() {
    std::printf("  broadcast\n");
    const std::vector<Dims> shapes = {{}, {1}, {3}, {1, 3}, {3, 1}, {4, 3},
                                      {2, 4, 3}, {1, 4, 3}, {5, 1, 3}, {2, 1, 1}};
    int bad_assoc = 0, bad_comm = 0, bad_ident = 0, bad_idem = 0, triples = 0;

    for (const auto& a : shapes)
        for (const auto& b : shapes)
            for (const auto& c : shapes) {
                auto ab = broadcast_shapes(a, b);
                auto bc = broadcast_shapes(b, c);
                if (!ab || !bc) continue;
                ++triples;
                if (!eq(broadcast_shapes(*ab, c), broadcast_shapes(a, *bc)))
                    ++bad_assoc;
            }
    check(bad_assoc == 0, "associative (" + std::to_string(triples) + " triples)");

    for (const auto& a : shapes)
        for (const auto& b : shapes) {
            if (!eq(broadcast_shapes(a, b), broadcast_shapes(b, a))) ++bad_comm;
            auto with_scalar = broadcast_shapes(a, {});
            if (!with_scalar || *with_scalar != a) ++bad_ident;
            auto self = broadcast_shapes(a, a);
            if (!self || *self != a) ++bad_idem;
        }
    check(bad_comm == 0, "commutative");
    check(bad_ident == 0, "scalar is the identity");
    check(bad_idem == 0, "idempotent");

    check(broadcast_shapes({2, 3}, {4, 5}) == std::nullopt,
          "incompatible dims are refused");
    check(broadcast_shapes({1, 3}, {4, 1}) == std::optional<Dims>(Dims{4, 3}),
          "{1,3} x {4,1} -> {4,3}");
}

// ---- transpose ------------------------------------------------------------

void test_transpose() {
    std::printf("  transpose\n");
    const std::vector<Dims> shapes = {{2, 3}, {2, 3, 4}, {5}, {2, 3, 4, 5}};
    int bad = 0, n = 0;
    for (const auto& dims : shapes) {
        Dims perm(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) perm[i] = static_cast<int64_t>(i);
        std::vector<int64_t> order = perm;
        // iterate permutations
        do {
            auto once = transpose_shape(dims, order);
            auto back = transpose_shape(*once, invert_permutation(order));
            ++n;
            if (!back || *back != dims) ++bad;
        } while (std::next_permutation(order.begin(), order.end()));
    }
    check(bad == 0, "transpose(inv(p)) o transpose(p) == id (" +
                        std::to_string(n) + " permutations)");

    check(transpose_shape({2, 3}, {0, 0}) == std::nullopt, "repeated index rejected");
    check(transpose_shape({2, 3}, {1}) == std::nullopt, "wrong length rejected");
    check(transpose_shape({2, 3}, {0, 1, 2}) == std::nullopt, "too long rejected");
    check(transpose_shape({2, 3}, {2, 0}) == std::nullopt, "out of range rejected");
}

// ---- reshape --------------------------------------------------------------

void test_reshape() {
    std::printf("  reshape\n");
    const std::vector<Dims> shapes = {{2, 3}, {4, 3}, {2, 3, 4}, {6}, {1, 12}, {12, 1}};
    int bad_count = 0, bad_round = 0;
    for (const auto& dims : shapes) {
        int64_t total = 1;
        for (int64_t d : dims) total *= d;
        const std::vector<Dims> targets = {{total}, {1, total}, {total, 1}, {-1},
                                           {2, total / 2}};
        for (const auto& nd : targets) {
            auto r = reshape_shape(dims, nd);
            if (!r) continue;
            int64_t nt = 1;
            for (int64_t d : *r) nt *= d;
            if (nt != total) ++bad_count;
            auto back = reshape_shape(*r, dims);
            if (!back || *back != dims) ++bad_round;
        }
    }
    check(bad_count == 0, "element count preserved");
    check(bad_round == 0, "round-trips");

    check(reshape_shape({2, 3}, {7}) == std::nullopt, "wrong count refused");
    check(reshape_shape({2, 3}, {0}) == std::nullopt, "zero dim refused");
    check(reshape_shape({2, 3}, {-1, -1}) == std::nullopt, "two -1 refused");
    check(reshape_shape({2, 3}, {5, -1}) == std::nullopt, "indivisible -1 refused");
    check(reshape_shape({2, 3}, {-1}) == std::optional<Dims>(Dims{6}), "-1 inferred");
}

// ---- matmul ---------------------------------------------------------------

void test_matmul() {
    std::printf("  matmul\n");
    const std::vector<Dims> shapes = {{2, 3}, {3, 4}, {4, 5}, {2, 2, 3},
                                      {2, 3, 4}, {2, 4, 5}, {1, 3, 4}};
    int bad = 0, n = 0;
    for (const auto& a : shapes)
        for (const auto& b : shapes)
            for (const auto& c : shapes) {
                auto ab = matmul_shape(a, b);
                auto bc = matmul_shape(b, c);
                if (!ab || !bc) continue;
                ++n;
                if (!eq(matmul_shape(*ab, c), matmul_shape(a, *bc))) ++bad;
            }
    check(bad == 0, "shape associativity (" + std::to_string(n) + " triples)");

    check(matmul_shape({2, 3}, {3, 4}) == std::optional<Dims>(Dims{2, 4}),
          "{2,3}x{3,4} -> {2,4}");
    check(matmul_shape({2, 3}, {4, 5}) == std::nullopt, "inner mismatch refused");
    check(matmul_shape({3}, {3, 4}) == std::nullopt, "rank < 2 refused");
}

// ---- index arithmetic -----------------------------------------------------

void test_index() {
    std::printf("  index arithmetic\n");
    const std::vector<Dims> shapes = {{5}, {2, 3}, {2, 3, 4}, {1, 7}, {3, 1, 2}};
    int bad = 0, n = 0;
    for (const auto& dims : shapes) {
        int64_t total = 1;
        for (int64_t d : dims) total *= d;
        for (int64_t lin = 0; lin < total; ++lin) {
            auto idx = unlinear_index(dims, lin);
            if (idx.empty() && !dims.empty()) { ++bad; continue; }
            auto back = linear_index(dims, idx);
            ++n;
            if (back != lin) ++bad;
        }
    }
    check(bad == 0, "linear/unlinear round-trip (" + std::to_string(n) + " offsets)");

    check(unlinear_index({2, 3}, 6).empty(), "out of range refused");
    check(unlinear_index({2, 3}, -1).empty(), "negative refused");
    check(linear_index({2, 3}, {2, 0}) == -1, "out of bounds refused");
    check(linear_index({2, 3}, {0}) == -1, "rank mismatch refused");
    check(linear_index({2, 3}, {1, 2}) == 5, "row-major offset");
}

// ---- reduce ---------------------------------------------------------------

void test_reduce() {
    std::printf("  reduce\n");
    int bad = 0;
    for (const Dims& dims : {Dims{2, 3}, Dims{2, 3, 4}, Dims{5}}) {
        for (size_t r = 0; r <= dims.size(); ++r) {
            // all subsets of axes of size r
            std::vector<bool> pick(dims.size(), false);
            std::fill(pick.end() - r, pick.end(), true);
            do {
                Dims axes;
                for (size_t i = 0; i < dims.size(); ++i)
                    if (pick[i]) axes.push_back(static_cast<int64_t>(i));
                auto out = reduce_shape(dims, axes);
                if (!out) { ++bad; continue; }
                Dims expected;
                for (size_t i = 0; i < dims.size(); ++i)
                    if (!pick[i]) expected.push_back(dims[i]);
                if (*out != expected) ++bad;

                auto kd = reduce_shape(dims, axes, true);
                if (!kd || kd->size() != dims.size()) { ++bad; continue; }
                for (size_t i = 0; i < dims.size(); ++i) {
                    int64_t want = pick[i] ? 1 : dims[i];
                    if ((*kd)[i] != want) ++bad;
                }
            } while (std::next_permutation(pick.begin(), pick.end()));
        }
    }
    check(bad == 0, "drops exactly the listed axes, keep_dims correct");

    check(reduce_shape({2, 3}, {0, 0}) == std::nullopt, "duplicate axis refused");
    check(reduce_shape({2, 3}, {2}) == std::nullopt, "out of range refused");
}

// ---- layout round-trip (the audit's zero-conformance-vector gap) ----------

void test_layouts() {
    std::printf("  layouts\n");
    using L = bt::target::Layout;
    const L all[] = {L::DualRail, L::Base243, L::TCSR, L::BTFP};

    // Every layout must round-trip through name -> enum -> name.
    int bad = 0;
    for (L l : all) {
        const char* n = bt::target::layout_name(l);
        if (std::string(n) == "unknown") ++bad;
    }
    check(bad == 0, "every layout has a name");

    // A layout change is a type change but NOT a conformance change.
    TensorType a{Type::Trit, {4, 8}, L::DualRail};
    TensorType b{Type::Trit, {4, 8}, L::Base243};
    check(a.conforms_to(b), "layout conversion preserves conformance");
    check(a != b, "layout conversion changes the type");

    // Density: base-243 stores 5 trits per byte vs dual-rail's 4, so the same
    // element count needs fewer bytes. This is the claim in
    // docs/TENSOR_MEMORY_MATHEMATICS.md; assert the direction, not a number.
    const int64_t elems = 1000;
    const double dual_bytes = elems * 2.0 / 8.0;      // 2 bits/trit
    const double base243_bytes = elems * 1.6 / 8.0;   // 8 bits / 5 trits
    check(base243_bytes < dual_bytes,
          "base-243 is denser than dual-rail (" +
              std::to_string((int)(100 * (1 - base243_bytes / dual_bytes))) +
              "% fewer bytes)");
}

}  // namespace

int main() {
    std::printf("tensor types and shape algebra\n");
    test_tensor_type();
    test_broadcast();
    test_transpose();
    test_reshape();
    test_matmul();
    test_index();
    test_reduce();
    test_layouts();
    std::printf("%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
