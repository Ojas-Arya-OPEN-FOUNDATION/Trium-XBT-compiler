// Auto-generated Apple Metal Shading Language Compute Shader
// Target: Apple Silicon M-Series GPU (Unified Memory Architecture)
#include <metal_stdlib>
using namespace metal;

struct TernaryBitplane {
    uint64_t P;
    uint64_t N;
};

kernel void bt_gemm_metal(
    device const uint64_t* A_pos      [[buffer(0)]],
    device const uint64_t* A_neg      [[buffer(1)]],
    device const uint64_t* BT_pos     [[buffer(2)]],
    device const uint64_t* BT_neg     [[buffer(3)]],
    device int64_t*        C_out      [[buffer(4)]],
    constant uint&         M          [[buffer(5)]],
    constant uint&         N          [[buffer(6)]],
    constant uint&         wordsPerRow[[buffer(7)]],
    uint2                  gid        [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i >= M || j >= N) return;

    int64_t acc = 0;
    uint row_offset_A = i * wordsPerRow;
    uint row_offset_B = j * wordsPerRow;

    for (uint w = 0; w < wordsPerRow; ++w) {
        uint64_t ap = A_pos[row_offset_A + w];
        uint64_t an = A_neg[row_offset_A + w];
        uint64_t bp = BT_pos[row_offset_B + w];
        uint64_t bn = BT_neg[row_offset_B + w];

        uint64_t p_prod = (ap & bp) | (an & bn);
        uint64_t n_prod = (ap & bn) | (an & bp);

        // Hardware SIMD Popcount in Metal
        acc += static_cast<int64_t>(popcount(p_prod)) - static_cast<int64_t>(popcount(n_prod));
    }

    C_out[i * N + j] = acc;
}
