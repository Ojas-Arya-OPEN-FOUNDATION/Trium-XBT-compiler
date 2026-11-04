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
#include <algorithm>
#include <bit>
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

// ---------------------------------------------------------------------------
// Everything about a dispatch that can be WRONG is computed here, as pure
// arithmetic on sizes - no Vulkan types, no device. That is deliberate: the
// API calls below are boilerplate, but off-by-one in a dispatch count or a
// buffer size is the bug that actually ships. Those are testable, so they are
// tested (see check_shader_contract / test_gpu_plan in the unit tests).
// ---------------------------------------------------------------------------
struct GpuGemmPlan {
    uint32_t M{0}, N{0}, words_per_row{0};
    uint32_t groups_x{0}, groups_y{0};    // workgroups to dispatch
    uint32_t local_x{8}, local_y{8};      // workgroup size (must match the shader)
    uint32_t tile{32};                    // rows/cols of C per workgroup
    uint32_t block{4};                    // register block per invocation
    uint32_t tkw{8};                      // K-words staged per pass
    bool     use_native64{false};         // native (uint64 rails) vs portable
    size_t   bytes_a_p{0}, bytes_a_n{0};
    size_t   bytes_bt_p{0}, bytes_bt_n{0};
    size_t   bytes_c{0};
    size_t   shared_bytes{0};             // per workgroup
    uint32_t binding_count{5};            // A.P A.N BT.P BT.N C
    uint32_t push_constant_bytes{12};     // M, N, wordsPerRow
};

/** Build the dispatch plan. Pure function of the shape - same inputs, same
 *  plan, on every machine. */
inline GpuGemmPlan gpu_gemm_plan(uint32_t M, uint32_t N, uint32_t K,
                                 bool shader_int64) noexcept {
    GpuGemmPlan p;
    p.M = M; p.N = N; p.words_per_row = (K + 63u) / 64u;
    p.use_native64 = shader_int64;
    p.groups_x = (M + p.tile - 1u) / p.tile;
    p.groups_y = (N + p.tile - 1u) / p.tile;
    const size_t words_a  = size_t(M) * p.words_per_row;
    const size_t words_bt = size_t(N) * p.words_per_row;
    // Native carries one uint64 per word; portable carries two uint32.
    // Native: one uint64 per K-word. Portable: two uint32 per K-word.
    const size_t elem = p.use_native64 ? sizeof(uint64_t) : 2u * sizeof(uint32_t);
    p.bytes_a_p  = words_a  * elem;
    p.bytes_a_n  = words_a  * elem;
    p.bytes_bt_p = words_bt * elem;
    p.bytes_bt_n = words_bt * elem;
    p.bytes_c    = size_t(M) * N * sizeof(int64_t);
    // Four planes (A.P A.N BT.P BT.N) of tile x tkw words, two uint32 each
    // on the portable path.
    p.shared_bytes = 4u * size_t(p.tile) * p.tkw * elem * (p.use_native64 ? 1u : 2u);
    return p;
}

/** Pack the push-constant block. Layout must match the shader's
 *  `layout(push_constant) uniform PC { uint M; uint N; uint wordsPerRow; }`. */
inline std::vector<uint32_t> pack_push_constants(const GpuGemmPlan& p) {
    return {p.M, p.N, p.words_per_row};
}

// ---------------------------------------------------------------------------
// Shader contract check. The GLSL cannot be compiled here (no glslangValidator
// in this environment), but the *contract between host and shader* is pure
// arithmetic and string structure - and a mismatch there is exactly the failure
// that produces silently wrong results rather than a compile error. This
// returns a list of violations; empty means the contract holds.
// ---------------------------------------------------------------------------
inline std::vector<std::string> check_shader_contract(const GpuGemmPlan& p) {
    std::vector<std::string> bad;
    const char* src = p.use_native64 ? ternary_gemm_glsl_native()
                                     : ternary_gemm_glsl_portable();
    const std::string g(src);

    if (g.rfind("#version 450", 0) != 0) bad.push_back("shader must start with #version 450");
    if (p.use_native64 && g.find("GL_ARB_gpu_shader_int64") == std::string::npos)
        bad.push_back("native shader must enable GL_ARB_gpu_shader_int64");
    if (!p.use_native64 && g.find("uint64_t") != std::string::npos)
        bad.push_back("portable shader must not use uint64_t");

    // Workgroup size in the source must equal the plan.
    if (g.find("layout(local_size_x = 8, local_size_y = 8) in;") == std::string::npos)
        bad.push_back("local size in shader does not match the plan (8x8)");
    if (p.local_x * p.block != p.tile)
        bad.push_back("local_x * block must equal the tile size");

    // Every binding the host will create must appear in the source.
    for (uint32_t b = 0; b < p.binding_count; ++b) {
        const std::string needle = "binding = " + std::to_string(b) + ")";
        if (g.find(needle) == std::string::npos)
            bad.push_back("shader is missing binding " + std::to_string(b));
    }
    if (g.find("uint M; uint N; uint wordsPerRow;") == std::string::npos)
        bad.push_back("push-constant block is not {M, N, wordsPerRow}");
    if (pack_push_constants(p).size() * sizeof(uint32_t) != p.push_constant_bytes)
        bad.push_back("push-constant packing does not match its declared size");

    // Staging must cover every shared element exactly once, or shared memory
    // is left uninitialised - a bug that reads as "works on my device".
    const uint32_t per_thread = p.use_native64 ? 4u : 8u;
    const uint32_t slots      = p.tile * p.tkw * (p.use_native64 ? 1u : 2u);
    if (p.local_x * p.local_y * per_thread != slots)
        bad.push_back("staging work does not cover the shared slab exactly once");

    // Vulkan guarantees at least 16 KiB of shared memory per workgroup.
    if (p.shared_bytes > 16384u)
        bad.push_back("shared memory exceeds the 16 KiB guaranteed minimum");

    int depth = 0;
    for (char c : g) { if (c == '{') ++depth; else if (c == '}') --depth;
                       if (depth < 0) { bad.push_back("unbalanced braces in shader"); break; } }
    if (depth != 0) bad.push_back("unbalanced braces in shader");
    return bad;
}

// ---------------------------------------------------------------------------
// Shader emulator. This executes the SAME algorithm the GLSL describes -
// workgroup tiling, shared-memory staging, TKW chunking, the 4x4 register
// block and the four-popcount MAC - in ordinary C++. It cannot validate GLSL
// syntax, but it does validate the algorithm and every index in it, which is
// where the real bugs live. If the emulator and the CPU oracle disagree, the
// shader is wrong on hardware too.
// ---------------------------------------------------------------------------
inline std::vector<int64_t> emulate_gpu_gemm(const GpuGemmOperands& op,
                                             const GpuGemmPlan& plan) {
    std::vector<int64_t> C(size_t(op.m) * op.n, 0);
    const uint32_t TILE = plan.tile, TKW = plan.tkw, BLK = plan.block;
    const uint32_t HALVES = TKW * 2u;
    const uint32_t kwr = plan.words_per_row;

    // Shared arrays, sized exactly as the shader declares them.
    const size_t S = size_t(TILE) * (plan.use_native64 ? TKW : HALVES);
    std::vector<uint64_t> sAP(S), sAN(S), sBP(S), sBN(S);

    for (uint32_t gwy = 0; gwy < plan.groups_y; ++gwy)
    for (uint32_t gwx = 0; gwx < plan.groups_x; ++gwx) {
        const uint32_t gx = gwx * TILE, gy = gwy * TILE;
        int64_t acc[8][8][4][4];   // [local_y][local_x][i][j]
        for (auto& a : acc) for (auto& b : a) for (auto& c : b) for (auto& d : c) d = 0;

        for (uint32_t kw0 = 0; kw0 < kwr; kw0 += TKW) {
            // --- staging: 64 invocations, per_thread entries each ---
            std::fill(sAP.begin(), sAP.end(), 0);
            std::fill(sAN.begin(), sAN.end(), 0);
            std::fill(sBP.begin(), sBP.end(), 0);
            std::fill(sBN.begin(), sBN.end(), 0);
            const uint32_t per_thread = plan.use_native64 ? 4u : 8u;
            for (uint32_t lid = 0; lid < plan.local_x * plan.local_y; ++lid)
                for (uint32_t t = 0; t < per_thread; ++t) {
                    const uint32_t idx = lid * per_thread + t;
                    const uint32_t row = plan.use_native64 ? (idx / TKW) : (idx / HALVES);
                    const uint32_t sub = plan.use_native64 ? (idx % TKW) : (idx % HALVES);
                    const uint32_t w  = plan.use_native64 ? sub : sub / 2u;
                    const uint32_t hf = plan.use_native64 ? 0u   : sub % 2u;
                    const uint32_t kk = kw0 + w;
                    const bool okw = kk < kwr;
                    const size_t sA = size_t(gx + row) * kwr + kk;
                    const size_t sB = size_t(gy + row) * kwr + kk;
                    const bool okA = okw && (gx + row) < op.m;
                    const bool okB = okw && (gy + row) < op.n;
                    if (plan.use_native64) {
                        const size_t o = size_t(row) * TKW + w;
                        sAP[o] = okA ? op.Ap[sA] : 0ULL;
                        sAN[o] = okA ? op.An[sA] : 0ULL;
                        sBP[o] = okB ? op.Bp[sB] : 0ULL;
                        sBN[o] = okB ? op.Bn[sB] : 0ULL;
                    } else {
                        const size_t o = size_t(row) * HALVES + sub;
                        sAP[o] = okA ? ((op.Ap[sA] >> (32ull * hf)) & 0xFFFFFFFFull) : 0ULL;
                        sAN[o] = okA ? ((op.An[sA] >> (32ull * hf)) & 0xFFFFFFFFull) : 0ULL;
                        sBP[o] = okB ? ((op.Bp[sB] >> (32ull * hf)) & 0xFFFFFFFFull) : 0ULL;
                        sBN[o] = okB ? ((op.Bn[sB] >> (32ull * hf)) & 0xFFFFFFFFull) : 0ULL;
                    }
                }

            // --- per-invocation 4x4 register block ---
            const uint32_t wcount = std::min<uint32_t>(TKW, kwr - kw0);
            for (uint32_t ly = 0; ly < plan.local_y; ++ly)
            for (uint32_t lx = 0; lx < plan.local_x; ++lx) {
                const uint32_t tx = lx * BLK, ty = ly * BLK;
                for (uint32_t w = 0; w < wcount; ++w) {
                    uint64_t ap[4], an[4], bp[4], bn[4];
                    for (uint32_t i = 0; i < BLK; ++i) {
                        const size_t o = size_t(tx + i) * (plan.use_native64 ? TKW : HALVES)
                                       + (plan.use_native64 ? w : 2u * w);
                        if (plan.use_native64) {
                            ap[i] = sAP[o]; an[i] = sAN[o];
                        } else {
                            ap[i] = sAP[o] | (sAP[o + 1] << 32);
                            an[i] = sAN[o] | (sAN[o + 1] << 32);
                        }
                    }
                    for (uint32_t j = 0; j < BLK; ++j) {
                        const size_t o = size_t(ty + j) * (plan.use_native64 ? TKW : HALVES)
                                       + (plan.use_native64 ? w : 2u * w);
                        if (plan.use_native64) {
                            bp[j] = sBP[o]; bn[j] = sBN[o];
                        } else {
                            bp[j] = sBP[o] | (sBP[o + 1] << 32);
                            bn[j] = sBN[o] | (sBN[o + 1] << 32);
                        }
                    }
                    for (uint32_t i = 0; i < BLK; ++i)
                        for (uint32_t j = 0; j < BLK; ++j)
                            acc[ly][lx][i][j] +=
                                  int64_t(std::popcount(ap[i] & bp[j]))
                                + int64_t(std::popcount(an[i] & bn[j]))
                                - int64_t(std::popcount(ap[i] & bn[j]))
                                - int64_t(std::popcount(an[i] & bp[j]));
                }
            }
        }
        // --- store, with the shader's bounds guard ---
        for (uint32_t ly = 0; ly < plan.local_y; ++ly)
        for (uint32_t lx = 0; lx < plan.local_x; ++lx) {
            const uint32_t tx = lx * BLK, ty = ly * BLK;
            for (uint32_t i = 0; i < BLK; ++i)
                for (uint32_t j = 0; j < BLK; ++j) {
                    const uint32_t r = gx + tx + i, c = gy + ty + j;
                    if (r < op.m && c < op.n)
                        C[size_t(r) * op.n + c] = acc[ly][lx][i][j];
                }
        }
    }
    return C;
}

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
    // pack_a pads A to a multiple of MR rows. BT must be padded to a multiple
    // of NR rows the same way, because the loops below run i0/j0 out to the
    // rounded-up bound and the kernel reads a whole 4x4 block each time.
    // Sizing BT at exactly op.n rows read past the end whenever n % 4 != 0 -
    // found by AddressSanitizer at n=23, and invisible without it because the
    // bytes past the end usually happen to be zero.
    constexpr int MR = 4, NR = 4;
    const size_t mp = ((op.m + MR - 1) / MR) * MR;
    const size_t np = ((op.n + NR - 1) / NR) * NR;
    DualRailPanel A  = pack_a(op.Ap.data(), op.An.data(), op.m, op.kw, MR);
    DualRailPanel BT{std::vector<uint64_t>(np * op.kw, 0),
                     std::vector<uint64_t>(np * op.kw, 0), np, op.kw};
    for (size_t j = 0; j < op.n; ++j)
        for (size_t w = 0; w < op.kw; ++w) {
            BT.P[j * op.kw + w] = op.Bp[j * op.kw + w];
            BT.N[j * op.kw + w] = op.Bn[j * op.kw + w];
        }
    (void)mp;  // A is already padded by pack_a; kept for symmetry with np
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

// ===========================================================================
// Packed weights -> device
//
// The weights are what you store. At 1.6 bits/trit they are ~20x smaller than
// the FP32 weights a conventional pipeline carries, and 20% smaller than even
// dual-rail (2.0 bits/trit). Activations are produced fresh each inference, so
// they stay dual-rail: there is nothing to compress, and unpacking them would
// cost more than it saves.
//
// The unpack happens ON the device, in shared memory, once per K pass. The
// base-243 decode below is a direct transcription of Base243Codec::unpack5_fast
// from ternary_core.hpp (Z3-verified magic multipliers 171/57/19/203), so host
// and device agree by construction - and the emulator proves they do.
// ===========================================================================

/** Weights as packed bytes, activations still dual-rail. */
struct GpuPackedOperands {
    uint32_t m{0}, n{0}, k{0}, kw{0}, bytes_per_row{0};
    std::vector<uint8_t>  A_packed;               // m * bytes_per_row
    std::vector<uint64_t> Bp, Bn;                 // n * kw  (B transposed)
};

struct GpuPackedGemmPlan {
    uint32_t M{0}, N{0}, words_per_row{0}, bytes_per_row{0};
    uint32_t groups_x{0}, groups_y{0};
    uint32_t local_x{8}, local_y{8}, tile{32}, block{4}, tkw{8};
    uint32_t pkt{104};                            // packed bytes staged per row
    uint32_t packed_entries_per_thread{52};       // TILE*PKT / (local_x*local_y)
    size_t   bytes_a_packed{0};
    size_t   bytes_bt_p{0}, bytes_bt_n{0}, bytes_c{0};
    size_t   shared_bytes{0};
    uint32_t binding_count{4};                    // A(packed) BT.P BT.N C
    uint32_t push_constant_bytes{20};             // M N wordsPerRow bytesPerRow K
};

inline GpuPackedGemmPlan gpu_gemm_plan_packed(uint32_t M, uint32_t N, uint32_t K) noexcept {
    GpuPackedGemmPlan p;
    p.M = M; p.N = N;
    p.words_per_row = (K + 63u) / 64u;
    p.bytes_per_row = (K + 4u) / 5u;              // 5 trits per byte
    p.groups_x = (M + p.tile - 1u) / p.tile;
    p.groups_y = (N + p.tile - 1u) / p.tile;
    // ceil(64*tkw/5) + 1: one extra byte so a group straddling the slab end
    // is always fully readable.
    p.pkt = (64u * p.tkw + 4u) / 5u + 1u;
    // Staging work must tile the slab EXACTLY: 64 threads x this many entries
    // = TILE * PKT. Get it wrong and only the first few rows are ever staged,
    // which reads as "wrong answers" rather than a crash. check below.
    const uint32_t threads = p.local_x * p.local_y;
    p.packed_entries_per_thread =
        (uint32_t)(size_t(p.tile) * p.pkt % threads == 0
                   ? size_t(p.tile) * p.pkt / threads : 0);
    p.bytes_a_packed = size_t(M) * p.bytes_per_row;
    const size_t words_bt = size_t(N) * p.words_per_row;
    p.bytes_bt_p = words_bt * sizeof(uint64_t);
    p.bytes_bt_n = words_bt * sizeof(uint64_t);
    p.bytes_c    = size_t(M) * N * sizeof(int64_t);
    // 4 dual-rail planes (A.P A.N BT.P BT.N) + the packed staging slab.
    p.shared_bytes = 4u * size_t(p.tile) * p.tkw * sizeof(uint64_t)
                   + size_t(p.tile) * p.pkt;
    return p;
}

// Forward declaration: the packed shader is defined further down, next to the
// rest of the packed path.
inline const char* ternary_gemm_glsl_packed() noexcept;

/** Contract check for the packed path. Same idea as check_shader_contract:
 *  the things that can be wrong are arithmetic and string structure. */
inline std::vector<std::string> check_packed_shader_contract(const GpuPackedGemmPlan& p) {
    std::vector<std::string> bad;
    const std::string g(ternary_gemm_glsl_packed());
    if (g.rfind("#version 450", 0) != 0) bad.push_back("packed shader must be #version 450");
    if (g.find("layout(local_size_x = 8, local_size_y = 8) in;") == std::string::npos)
        bad.push_back("packed shader local size does not match the plan");
    for (uint32_t b = 0; b < p.binding_count; ++b)
        if (g.find("binding = " + std::to_string(b) + ")") == std::string::npos)
            bad.push_back("packed shader is missing binding " + std::to_string(b));
    if (g.find("uint M; uint N; uint wordsPerRow; uint bytesPerRow; uint K;") == std::string::npos)
        bad.push_back("packed push-constant block is not {M,N,wordsPerRow,bytesPerRow,K}");
    if (p.push_constant_bytes != 5u * sizeof(uint32_t))
        bad.push_back("packed push-constant size does not match its block");
    // The staging must tile the slab exactly, or only the first rows are staged.
    if (p.packed_entries_per_thread == 0)
        bad.push_back("TILE*PKT is not a whole number of entries per thread");
    else if (size_t(p.local_x) * p.local_y * p.packed_entries_per_thread
             != size_t(p.tile) * p.pkt)
        bad.push_back("packed staging does not cover the slab exactly once");
    if (p.bytes_per_row != (p.M ? p.bytes_per_row : 0)) bad.push_back("unreachable");
    if (p.shared_bytes > 16384u)
        bad.push_back("packed shared memory exceeds the 16 KiB guarantee");
    // PKT in the source must match the plan.
    if (g.find("const uint PKT  = " + std::to_string(p.pkt) + "u;") == std::string::npos)
        bad.push_back("PKT in the shader does not match the plan (" +
                      std::to_string(p.pkt) + ")");
    if (g.find("const uint PACKED_ENTRIES = " +
               std::to_string(p.packed_entries_per_thread) + "u;") == std::string::npos)
        bad.push_back("PACKED_ENTRIES in the shader does not match the plan");
    int depth = 0;
    for (char c : g) { if (c == '{') ++depth; else if (c == '}') --depth;
                       if (depth < 0) { bad.push_back("unbalanced braces"); break; } }
    if (depth != 0) bad.push_back("unbalanced braces in packed shader");
    return bad;
}

/** Pack dual-rail weight planes into the base-243 byte stream the device reads.
 *  Uses the same Z3-verified codec as PackedTensor, so host and device agree. */
inline std::vector<uint8_t> pack_weights_for_gpu(const std::vector<uint64_t>& Ap,
                                                 const std::vector<uint64_t>& An,
                                                 uint32_t m, uint32_t k, uint32_t kw,
                                                 uint32_t bytes_per_row) {
    PackedTensor T = packed_from_dual_rail(Ap.data(), An.data(), m, k, kw);
    if (bytes_per_row == 0 || bytes_per_row != T.bytes_per_row())
        throw std::invalid_argument("pack_weights_for_gpu: row stride does not "
                                    "match the codec (expected " +
                                    std::to_string(T.bytes_per_row()) + ")");
    std::vector<uint8_t> out(size_t(m) * bytes_per_row, 0);
    for (uint32_t r = 0; r < m; ++r)
        for (uint32_t b = 0; b < bytes_per_row; ++b)
            out[size_t(r) * bytes_per_row + b] = T.row_byte(r, b);
    return out;
}

/**
 * @brief Packed-weight GEMM shader: unpack base-243 in shared memory, then
 *        run the same 4x4 tiled inner loop as the native path.
 *
 * Bindings: 0 = packed A bytes, 1 = BT.P, 2 = BT.N, 3 = C.
 * Push constants: M, N, wordsPerRow, bytesPerRow.
 */
inline const char* ternary_gemm_glsl_packed() noexcept {
    return R"GLSL(#version 450
//
// Tiled ternary GEMM with base-243 packed weights unpacked in shared memory.
// Weights: 1.6 bits/trit (20x smaller than FP32). Activations: dual-rail.
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) readonly  buffer BufAP  { uint a_packed[]; };
layout(binding = 1) readonly  buffer BufBTP { uint64_t bt_p[]; };
layout(binding = 2) readonly  buffer BufBTN { uint64_t bt_n[]; };
layout(binding = 3) writeonly buffer BufC   { int64_t  c[];    };

layout(push_constant) uniform PC {
    uint M; uint N; uint wordsPerRow; uint bytesPerRow; uint K;
} pc;

const uint TILE = 32u;
const uint TKW  = 8u;
const uint PKT  = 104u;   // ceil(64*TKW/5) + 1
const uint PACKED_ENTRIES = 52u;  // TILE*PKT / 64, must divide exactly

shared uint     sPacked[TILE][PKT];
shared uint64_t sAP[TILE][TKW];
shared uint64_t sAN[TILE][TKW];
shared uint64_t sBP[TILE][TKW];
shared uint64_t sBN[TILE][TKW];

// Base-243 decode: direct transcription of Base243Codec::unpack5_fast.
// Returns five digits in {0,1,2}; trit = digit - 1, so digit 2 is +1 (P rail)
// and digit 0 is -1 (N rail).
void unpack5_digits(uint b, out uint d0, out uint d1, out uint d2,
                            out uint d3, out uint d4) {
    uint q1 = (b * 171u) >> 9;
    uint q2 = (b * 57u)  >> 9;
    uint q3 = (b * 19u)  >> 9;
    uint q4 = (b * 203u) >> 14;
    d0 = b  - 3u * q1;
    d1 = q1 - 3u * q2;
    d2 = q2 - 3u * q3;
    d3 = q3 - 3u * q4;
    d4 = q4;
}

// Unpack one 64-trit word of row `row` from the packed slab into P/N rails.
//
// Byte b of the stream holds trits 5b..5b+4 - the grouping is GLOBAL, so a
// word starting at trit 64*w is not byte-aligned (64 is not a multiple of 5).
// Indexing the byte as (trit0 + g*5)/5 therefore reads the wrong slots; each
// trit must be located by its own absolute index. Decoding is still one pass:
// the byte is re-decoded only when the index changes.
//
// Bounded by K, not by the byte extent: a zero byte decodes as five -1 trits
// (digit 0 == trit -1) and PackedTensor leaves its tail at zero, so reading
// past K would invent negative weights out of pure padding.
void unpack_word(uint row, uint kw0, uint w, out uint64_t pp, out uint64_t nn) {
    pp = uint64_t(0); nn = uint64_t(0);
    uint d0, d1, d2, d3, d4;
    uint lastOff = 0xFFFFFFFFu;
    for (uint b = 0u; b < 64u; ++b) {
        const uint t = 64u * w + b;
        if (t >= pc.K) break;
        const uint off = (t - kw0 * 64u) / 5u;
        if (off != lastOff) {
            unpack5_digits(sPacked[row][off], d0, d1, d2, d3, d4);
            lastOff = off;
        }
        const uint slot = t % 5u;
        const uint d = (slot == 0u) ? d0 : (slot == 1u) ? d1 : (slot == 2u) ? d2
                     : (slot == 3u) ? d3 : d4;
        if (d == 2u)      pp |= uint64_t(1) << b;
        else if (d == 0u) nn |= uint64_t(1) << b;
    }
}

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
        // Stage the packed weight bytes for this K slab. 64 threads x
        // PACKED_ENTRIES entries must cover TILE x PKT exactly.
        for (uint t = 0u; t < PACKED_ENTRIES; ++t) {
            const uint idx = lid * PACKED_ENTRIES + t;
            const uint row = idx / PKT;
            const uint off = idx % PKT;
            const uint tri = kw0 * 64u + off * 5u;
            sPacked[row][off] = (tri < pc.K && (gx + row) < pc.M)
                              ? a_packed[(gx + row) * pc.bytesPerRow + tri / 5u]
                              : 0u;
        }
        barrier();

        // Unpack this slab into dual-rail. Thread (lx,ly) owns A rows tx..tx+3.
        const uint wcount = min(TKW, pc.wordsPerRow - kw0);
        for (uint i = 0u; i < 4u; ++i) {
            const uint row = tx + i;
            for (uint w = 0u; w < wcount; ++w) {
                uint64_t pp, nn;
                unpack_word(row, kw0, w, pp, nn);
                sAP[row][w] = pp;
                sAN[row][w] = nn;
            }
        }
        // Stage and keep BT dual-rail (activations are never compressed).
        for (uint t = 0u; t < 4u; ++t) {
            const uint idx = lid * 4u + t;
            const uint row = idx / TKW;
            const uint w   = idx % TKW;
            const uint kk  = kw0 + w;
            const bool okw = kk < pc.wordsPerRow;
            const uint src = (gy + row) * pc.wordsPerRow + kk;
            sBP[row][w] = (okw && (gy + row) < pc.N) ? bt_p[src] : uint64_t(0);
            sBN[row][w] = (okw && (gy + row) < pc.N) ? bt_n[src] : uint64_t(0);
        }
        barrier();

        const uint64_t zero = uint64_t(0);
        for (uint w = 0u; w < wcount; ++w) {
            uint64_t ap[4], an[4];
            for (uint i = 0u; i < 4u; ++i) {
                ap[i] = (tx + i < TILE) ? sAP[tx + i][w] : zero;
                an[i] = (tx + i < TILE) ? sAN[tx + i][w] : zero;
            }
            for (uint j = 0u; j < 4u; ++j) {
                const uint64_t bp = sBP[ty + j][w];
                const uint64_t bn = sBN[ty + j][w];
                for (uint i = 0u; i < 4u; ++i) {
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

// ---------------------------------------------------------------------------
// Emulator for the packed path - same algorithm as the GLSL above.
// ---------------------------------------------------------------------------

/** Mirror of the shader's unpack5_digits (digits in {0,1,2}, trit = digit-1). */
inline void gpu_unpack5_digits(uint32_t b, uint32_t d[5]) noexcept {
    const uint32_t q1 = (b * 171u) >> 9;
    const uint32_t q2 = (b *  57u) >> 9;
    const uint32_t q3 = (b *  19u) >> 9;
    const uint32_t q4 = (b * 203u) >> 14;
    d[0] = b  - 3u * q1;
    d[1] = q1 - 3u * q2;
    d[2] = q2 - 3u * q3;
    d[3] = q3 - 3u * q4;
    d[4] = q4;
}

inline std::vector<int64_t> emulate_gpu_gemm_packed(const GpuPackedOperands& op,
                                                    const GpuPackedGemmPlan& plan) {
    std::vector<int64_t> C(size_t(op.m) * op.n, 0);
    const uint32_t TILE = plan.tile, TKW = plan.tkw, BLK = plan.block;
    const uint32_t PKT = plan.pkt, kwr = plan.words_per_row;

    std::vector<uint32_t> sPacked(size_t(TILE) * PKT);
    std::vector<uint64_t> sAP(size_t(TILE) * TKW), sAN(size_t(TILE) * TKW);
    std::vector<uint64_t> sBP(size_t(TILE) * TKW), sBN(size_t(TILE) * TKW);

    for (uint32_t gwy = 0; gwy < plan.groups_y; ++gwy)
    for (uint32_t gwx = 0; gwx < plan.groups_x; ++gwx) {
        const uint32_t gx = gwx * TILE, gy = gwy * TILE;
        int64_t acc[8][8][4][4];
        for (auto& a : acc) for (auto& b : a) for (auto& c : b) for (auto& d : c) d = 0;

        for (uint32_t kw0 = 0; kw0 < kwr; kw0 += TKW) {
            std::fill(sPacked.begin(), sPacked.end(), 0u);
            const uint32_t pEnt = plan.packed_entries_per_thread;
            for (uint32_t lid = 0; lid < plan.local_x * plan.local_y; ++lid)
                for (uint32_t t = 0; t < pEnt; ++t) {
                    const uint32_t idx = lid * pEnt + t;
                    const uint32_t row = idx / PKT, off = idx % PKT;
                    const uint32_t tri = kw0 * 64u + off * 5u;
                    sPacked[size_t(row) * PKT + off] =
                        (tri < op.k && (gx + row) < op.m)
                        ? op.A_packed[size_t(gx + row) * op.bytes_per_row + tri / 5u] : 0u;
                }

            const uint32_t wcount = std::min<uint32_t>(TKW, kwr - kw0);
            std::fill(sAP.begin(), sAP.end(), 0);
            std::fill(sAN.begin(), sAN.end(), 0);
            for (uint32_t row = 0; row < TILE; ++row)
                for (uint32_t w = 0; w < wcount; ++w) {
                    uint64_t pp = 0, nn = 0;
                    uint32_t d[5];
                    uint32_t lastOff = 0xFFFFFFFFu;
                    // Byte b holds trits 5b..5b+4 globally, so locate each
                    // trit by its absolute index; re-decode only on change.
                    for (uint32_t b = 0; b < 64u; ++b) {
                        const uint32_t t = 64u * w + b;
                        if (t >= op.k) break;
                        const uint32_t off = (t - kw0 * 64u) / 5u;
                        if (off != lastOff) {
                            gpu_unpack5_digits(sPacked[size_t(row) * PKT + off], d);
                            lastOff = off;
                        }
                        const uint32_t slot = t % 5u;
                        if (d[slot] == 2u)      pp |= (1ULL << b);
                        else if (d[slot] == 0u) nn |= (1ULL << b);
                    }
                    sAP[size_t(row) * TKW + w] = pp;
                    sAN[size_t(row) * TKW + w] = nn;
                }

            std::fill(sBP.begin(), sBP.end(), 0);
            std::fill(sBN.begin(), sBN.end(), 0);
            for (uint32_t lid = 0; lid < plan.local_x * plan.local_y; ++lid)
                for (uint32_t t = 0; t < 4u; ++t) {
                    const uint32_t idx = lid * 4u + t;
                    const uint32_t row = idx / TKW, w = idx % TKW;
                    const uint32_t kk = kw0 + w;
                    const bool okw = kk < kwr;
                    const size_t src = size_t(gy + row) * kwr + kk;
                    sBP[size_t(row) * TKW + w] = (okw && (gy + row) < op.n) ? op.Bp[src] : 0ULL;
                    sBN[size_t(row) * TKW + w] = (okw && (gy + row) < op.n) ? op.Bn[src] : 0ULL;
                }

            for (uint32_t ly = 0; ly < plan.local_y; ++ly)
            for (uint32_t lx = 0; lx < plan.local_x; ++lx) {
                const uint32_t tx = lx * BLK, ty = ly * BLK;
                for (uint32_t w = 0; w < wcount; ++w) {
                    uint64_t ap[4], an[4], bp[4], bn[4];
                    for (uint32_t i = 0; i < BLK; ++i) {
                        ap[i] = sAP[size_t(tx + i) * TKW + w];
                        an[i] = sAN[size_t(tx + i) * TKW + w];
                    }
                    for (uint32_t j = 0; j < BLK; ++j) {
                        bp[j] = sBP[size_t(ty + j) * TKW + w];
                        bn[j] = sBN[size_t(ty + j) * TKW + w];
                    }
                    for (uint32_t i = 0; i < BLK; ++i)
                        for (uint32_t j = 0; j < BLK; ++j)
                            acc[ly][lx][i][j] +=
                                  int64_t(std::popcount(ap[i] & bp[j]))
                                + int64_t(std::popcount(an[i] & bn[j]))
                                - int64_t(std::popcount(ap[i] & bn[j]))
                                - int64_t(std::popcount(an[i] & bp[j]));
                }
            }
        }
        for (uint32_t ly = 0; ly < plan.local_y; ++ly)
        for (uint32_t lx = 0; lx < plan.local_x; ++lx) {
            const uint32_t tx = lx * BLK, ty = ly * BLK;
            for (uint32_t i = 0; i < BLK; ++i)
                for (uint32_t j = 0; j < BLK; ++j) {
                    const uint32_t r = gx + tx + i, c = gy + ty + j;
                    if (r < op.m && c < op.n)
                        C[size_t(r) * op.n + c] = acc[ly][lx][i][j];
                }
        }
    }
    return C;
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
