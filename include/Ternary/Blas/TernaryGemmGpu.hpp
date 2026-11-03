/**
 * @file TernaryGemmGpu.hpp
 * @brief GPU backend for the ternary micro-kernel: GLSL 450 compute shaders,
 *        a host-side operand/result contract, and a CPU oracle to validate
 *        the device against.
 *
 * Status of this file - read before trusting any GPU number:
 *
 *   DONE       The shaders below, the host interface, and the CPU oracle. The
 *              oracle is exact (integer MAC, no rounding) and is exercised by
 *              test/Unit/test_unit_microkernel.cpp.
 *   NOT DONE   Real Vulkan submission. ternary_gemm_gpu() is a CPU fallback
 *              stub: this sandbox has no Vulkan headers and no device, so the
 *              shaders have never been through glslangValidator or spirv-val.
 *              Treat the GLSL as design-level until compiled on a machine with
 *              the SDK shader toolchain.
 *
 * Why the shaders are shaped this way. The pre-existing gemm_shader.comp gives
 * one output element per invocation and re-reads both operands from global
 * memory for every K word - each loaded word is used once, which is the worst
 * possible ratio on a bandwidth-bound kernel. These shaders instead:
 *   - give each invocation a 4x4 register block (A word reused 4x, B word 4x),
 *   - stage a 32x8 slab of A and B^T into shared memory per K pass,
 *   - chunk K by TKW=8 words so the shared tile stays small,
 * which turns a memory-bound kernel into a compute-bound one.
 *
 * Exactly where ternary wins on a device: the weights are 1.6 bits/trit packed
 * versus 32 bits for FP32, so at large n the HBM traffic is ~20x lower. The
 * kernel is not faster per-FLOP than FP32 tensor cores; it is faster because
 * it moves far less data. Target large n.
 */
#pragma once

#include "Ternary/Blas/TernaryMicroKernel.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt {
namespace blas {

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/**
 * @brief Native path: 64-bit rails. Requires
 *        VkPhysicalDeviceFeatures::shaderInt64 (GK->Session sets `native64`
 *        from exactly this feature bit in lib/SDK/VulkanRuntime.cpp).
 *
 * Bindings 0..4 = A.P, A.N, B^T.P, B^T.N, C. Push constants = M, N, wordsPerRow.
 * One workgroup covers a 32x32 tile; local size is 8x8 with a 4x4 block each.
 */
inline const char* ternary_gemm_glsl_native() noexcept {
    return R"GLSL(#version 450
#extension GL_ARB_gpu_shader_int64 : require
//
// Tiled ternary GEMM, 64-bit rails. C[M,N] = A[M,K] * B[K,N].
// B is supplied transposed (BT is N rows x K).
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) readonly  buffer BufAP  { uint64_t a_p[];  };
layout(binding = 1) readonly  buffer BufAN  { uint64_t a_n[];  };
layout(binding = 2) readonly  buffer BufBTP { uint64_t bt_p[]; };
layout(binding = 3) readonly  buffer BufBTN { uint64_t bt_n[]; };
layout(binding = 4) writeonly buffer BufC   { int64_t  c[];    };

layout(push_constant) uniform PC { uint M; uint N; uint wordsPerRow; } pc;

const uint TILE = 32u;   // 8x8 invocations x 4x4 register block
const uint TKW  = 8u;    // K-words staged per pass
const uint HALVES = TKW * 2u;

shared uint64_t sAP[TILE][TKW];
shared uint64_t sAN[TILE][TKW];
shared uint64_t sBP[TILE][TKW];
shared uint64_t sBN[TILE][TKW];

void main() {
    const uint gx  = gl_WorkGroupID.x * TILE;
    const uint gy  = gl_WorkGroupID.y * TILE;
    const uint tx  = gl_LocalInvocationID.x * 4u;
    const uint ty  = gl_LocalInvocationID.y * 4u;
    const uint lid = gl_LocalInvocationID.y * 8u + gl_LocalInvocationID.x;

    int64_t acc[4][4];
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j)
            acc[i][j] = int64_t(0);

    for (uint kw0 = 0u; kw0 < pc.wordsPerRow; kw0 += TKW) {
        // Stage a TILE x TKW slab. 64 invocations x 4 entries = 256 = TILE*TKW.
        for (uint t = 0u; t < 4u; ++t) {
            const uint idx = lid * 4u + t;
            const uint row = idx / TKW;
            const uint w   = idx % TKW;
            const uint kk  = kw0 + w;
            const bool okw = kk < pc.wordsPerRow;
            const uint srcA = (gx + row) * pc.wordsPerRow + kk;
            const uint srcB = (gy + row) * pc.wordsPerRow + kk;
            sAP[row][w] = (okw && (gx + row) < pc.M) ? a_p[srcA]  : uint64_t(0);
            sAN[row][w] = (okw && (gx + row) < pc.M) ? a_n[srcA]  : uint64_t(0);
            sBP[row][w] = (okw && (gy + row) < pc.N) ? bt_p[srcB] : uint64_t(0);
            sBN[row][w] = (okw && (gy + row) < pc.N) ? bt_n[srcB] : uint64_t(0);
        }
        barrier();

        const uint wcount = min(TKW, pc.wordsPerRow - kw0);
        for (uint w = 0u; w < wcount; ++w) {
            uint64_t ap[4], an[4];
            for (uint i = 0u; i < 4u; ++i) {
                ap[i] = sAP[tx + i][w];
                an[i] = sAN[tx + i][w];
            }
            for (uint j = 0u; j < 4u; ++j) {
                const uint64_t bp = sBP[ty + j][w];
                const uint64_t bn = sBN[ty + j][w];
                for (uint i = 0u; i < 4u; ++i) {
                    // (a+ - a-)*(b+ - b-) expanded into four popcounts.
                    acc[i][j] += int64_t(bitCount(ap[i] & bp))
                               + int64_t(bitCount(an[i] & bn))
                               - int64_t(bitCount(ap[i] & bn))
                               - int64_t(bitCount(an[i] & bp));
                }
            }
        }
        barrier();
    }

    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j) {
            const uint r  = gx + tx + i;
            const uint co = gy + ty + j;
            if (r < pc.M && co < pc.N)
                c[r * pc.N + co] = acc[i][j];
        }
}
)GLSL";
}

/**
 * @brief Portable path: no 64-bit integers. Each 64-bit K-word is carried as
 *        two consecutive uint32 entries - index 2*w is the low half, 2*w+1 the
 *        high half - so the array has 2*wordsPerRow entries per row.
 *
 * Use this when shaderInt64 is unavailable. The popcount arithmetic is
 * identical, just applied to both halves.
 */
inline const char* ternary_gemm_glsl_portable() noexcept {
    return R"GLSL(#version 450
//
// Tiled ternary GEMM, 32-bit rails (two uint32 per 64-trit K-word).
// Runs on any Vulkan 1.1 device - no shaderInt64 required.
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) readonly  buffer BufAP  { uint a_p[];  };
layout(binding = 1) readonly  buffer BufAN  { uint a_n[];  };
layout(binding = 2) readonly  buffer BufBTP { uint bt_p[]; };
layout(binding = 3) readonly  buffer BufBTN { uint bt_n[]; };
layout(binding = 4) writeonly buffer BufC   { int64_t c[]; };

layout(push_constant) uniform PC { uint M; uint N; uint wordsPerRow; } pc;

const uint TILE = 32u;
const uint TKW  = 8u;
const uint HALVES = TKW * 2u;

shared uint sAP[TILE][HALVES];
shared uint sAN[TILE][HALVES];
shared uint sBP[TILE][HALVES];
shared uint sBN[TILE][HALVES];

void main() {
    const uint gx  = gl_WorkGroupID.x * TILE;
    const uint gy  = gl_WorkGroupID.y * TILE;
    const uint tx  = gl_LocalInvocationID.x * 4u;
    const uint ty  = gl_LocalInvocationID.y * 4u;
    const uint lid = gl_LocalInvocationID.y * 8u + gl_LocalInvocationID.x;

    int64_t acc[4][4];
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j)
            acc[i][j] = int64_t(0);

    for (uint kw0 = 0u; kw0 < pc.wordsPerRow; kw0 += TKW) {
        // 64 invocations x 8 entries = 512 = TILE * HALVES.
        for (uint t = 0u; t < 8u; ++t) {
            const uint idx  = lid * 8u + t;
            const uint row  = idx / HALVES;
            const uint hw   = idx % HALVES;
            const uint w    = hw / 2u;
            const uint half = hw % 2u;
            const uint kk   = kw0 + w;
            const bool okw  = kk < pc.wordsPerRow;
            const uint srcA = (gx + row) * pc.wordsPerRow + kk;
            const uint srcB = (gy + row) * pc.wordsPerRow + kk;
            sAP[row][hw] = (okw && (gx + row) < pc.M) ? a_p[2u  * srcA + half] : 0u;
            sAN[row][hw] = (okw && (gx + row) < pc.M) ? a_n[2u  * srcA + half] : 0u;
            sBP[row][hw] = (okw && (gy + row) < pc.N) ? bt_p[2u * srcB + half] : 0u;
            sBN[row][hw] = (okw && (gy + row) < pc.N) ? bt_n[2u * srcB + half] : 0u;
        }
        barrier();

        const uint wcount = min(TKW, pc.wordsPerRow - kw0);
        for (uint w = 0u; w < wcount; ++w) {
            for (uint i = 0u; i < 4u; ++i) {
                const uint apl = sAP[tx + i][2u * w],     aph = sAP[tx + i][2u * w + 1u];
                const uint anl = sAN[tx + i][2u * w],     anh = sAN[tx + i][2u * w + 1u];
                for (uint j = 0u; j < 4u; ++j) {
                    const uint bpl = sBP[ty + j][2u * w],     bph = sBP[ty + j][2u * w + 1u];
                    const uint bnl = sBN[ty + j][2u * w],     bnh = sBN[ty + j][2u * w + 1u];
                    int s = 0;
                    s += bitCount(apl & bpl) + bitCount(aph & bph);
                    s += bitCount(anl & bnl) + bitCount(anh & bnh);
                    s -= bitCount(apl & bnl) + bitCount(aph & bnh);
                    s -= bitCount(anl & bpl) + bitCount(anh & bph);
                    acc[i][j] += int64_t(s);
                }
            }
        }
        barrier();
    }

    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j) {
            const uint r  = gx + tx + i;
            const uint co = gy + ty + j;
            if (r < pc.M && co < pc.N)
                c[r * pc.N + co] = acc[i][j];
        }
}
)GLSL";
}

// ---------------------------------------------------------------------------
// Host interface
// ---------------------------------------------------------------------------

/** Operands for a device GEMM, in the same dual-rail layout as ternary_gemm.
 *  A is M x K row-major words; B is supplied transposed (N rows x K). */
struct GpuGemmOperands {
    size_t m{0}, n{0}, k{0}, kw{0};
    std::vector<uint64_t> Ap, An;   // A:  Ap[i*kw + w]
    std::vector<uint64_t> Bp, Bn;   // B^T: Bp[j*kw + w]
};

struct GpuGemmResult {
    bool        ok{false};
    std::string note;               // why a path was or was not taken
    double      ms{0.0};
    double      gmacs{0.0};
    std::vector<int64_t> C;
};

/** Thrown by ternary_gemm_gpu() when no device is available and the caller
 *  asked for one with require_gpu = true. */
class GpuUnavailable : public std::runtime_error {
public:
    explicit GpuUnavailable(const std::string& why) : std::runtime_error(why) {}
};

// ---------------------------------------------------------------------------
// CPU oracle - the thing every device must be measured against
// ---------------------------------------------------------------------------

/**
 * @brief Exact CPU result for the same operands, via the packed micro-kernel.
 *
 * Bit-identical to a device result is not a tolerance question: the ternary
 * MAC is exact integer arithmetic, so any mismatch is a real bug.
 */
inline std::vector<int64_t> ternary_gemm_cpu_reference(const GpuGemmOperands& op) {
    std::vector<int64_t> C(op.m * op.n, 0);
    DualRailPanel A  = pack_a(op.Ap.data(), op.An.data(), op.m, op.kw, 4);
    DualRailPanel BT{std::vector<uint64_t>(op.n * op.kw),
                     std::vector<uint64_t>(op.n * op.kw), op.n, op.kw};
    for (size_t j = 0; j < op.n; ++j)
        for (size_t w = 0; w < op.kw; ++w) {
            BT.P[j * op.kw + w] = op.Bp[j * op.kw + w];
            BT.N[j * op.kw + w] = op.Bn[j * op.kw + w];
        }
    for (size_t i0 = 0; i0 < ((op.m + 3) / 4) * 4; i0 += 4)
        for (size_t j0 = 0; j0 < ((op.n + 3) / 4) * 4; j0 += 4)
            MicroKernel<ScalarLane, 4, 4>::run(A, BT, i0, j0, op.kw,
                                               C.data(), op.n, op.m, op.n);
    return C;
}

// Forward declaration: validate_gpu_vs_cpu() below needs the dispatch, which
// is defined at the bottom of this header.
inline GpuGemmResult ternary_gemm_gpu(const GpuGemmOperands& op,
                                      bool require_gpu = false);

struct GpuValidation {
    bool        ran_on_gpu{false};
    bool        identical{false};
    std::string note;
    size_t      first_mismatch{0};
    int64_t     expected{0}, got{0};
};

/**
 * @brief Run the same GEMM on the device and the CPU and compare bit-for-bit.
 *
 * Call this once per target device before publishing any GPU number. It never
 * throws on a missing device: it returns ran_on_gpu=false with `note` saying
 * why, so a CI run without a GPU reports honestly instead of silently
 * "passing" a benchmark it never executed.
 */
inline GpuValidation validate_gpu_vs_cpu(const GpuGemmOperands& op) {
    GpuValidation v;
    const std::vector<int64_t> ref = ternary_gemm_cpu_reference(op);
    GpuGemmResult r;
    try {
        r = ternary_gemm_gpu(op, false);
    } catch (const std::exception& e) {
        v.note = std::string("GPU validation aborted: ") + e.what();
        return v;
    }
    v.ran_on_gpu = r.ok;
    v.note = r.note;
    if (!r.ok) return v;
    if (r.C.size() != ref.size()) {
        v.note = "GPU result size differs from the CPU reference";
        return v;
    }
    v.identical = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != r.C[i]) {
            v.identical = false; v.first_mismatch = i;
            v.expected = ref[i]; v.got = r.C[i];
            v.note = "GPU/CPU mismatch at element " + std::to_string(i);
            return v;
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// Device dispatch
// ---------------------------------------------------------------------------

/**
 * @brief Run the ternary GEMM on a GPU.
 *
 * NOTE: this is currently a CPU fallback. Building the real path requires the
 * Vulkan headers (absent here) and a dispatch point with 5 descriptor bindings
 * and a 12-byte push-constant range - lib/SDK/VulkanRuntime.cpp's
 * VulkanProgram::State allocates only 3 bindings and 4 bytes, so it cannot
 * host this shader as written. When those are in place, the body should:
 *   1. compile ternary_gemm_glsl_native() (or _portable() if !shaderInt64)
 *   2. allocate the 5 buffers and upload the operands
 *   3. dispatch ((M+31)/32, (N+31)/32) workgroups
 *   4. read back C
 * and the CPU reference above stays as the oracle.
 */
inline GpuGemmResult ternary_gemm_gpu(const GpuGemmOperands& op,
                                      bool require_gpu) {
    if (require_gpu)
        throw GpuUnavailable(
            "GPU unavailable: no GPU dispatch compiled into this build "
            "(Vulkan headers not present at compile time)");

    GpuGemmResult r;
    r.note = "GPU unavailable: no GPU dispatch compiled into this build "
             "(Vulkan headers not present at compile time)";
    const auto t0 = std::chrono::steady_clock::now();
    r.C = ternary_gemm_cpu_reference(op);
    const auto t1 = std::chrono::steady_clock::now();
    r.ok = false;
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.gmacs = r.ms > 0.0
            ? (double(op.m) * double(op.n) * double(op.k)) / (r.ms * 1e6) : 0.0;
    return r;
}

} // namespace blas
} // namespace bt
