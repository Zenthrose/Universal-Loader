#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_extended_types : require

layout(local_size_x = 64, local_size_y = 1) in;

layout(binding = 0) readonly buffer A { float a[]; };
layout(binding = 1) readonly buffer B { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint K_tiles;
    float alpha;
    float beta;
    uint padding[2];
} params;

const uint TILE_M = 64;
const uint TILE_N = 64;
const uint TILE_K = 16;

shared float16_t s_a[TILE_M][TILE_K + 1];
shared float16_t s_b[TILE_K][TILE_N + 1];

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint ty = gl_WorkGroupID.y;
    const uint tx = gl_LocalInvocationID.x;

    const uint row = ty * TILE_M + tx;
    const uint col = bx * TILE_N;

    float16_t acc[8];
    for (int i = 0; i < 8; ++i) {
        acc[i] = float16_t(0.0);
    }

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        const uint a_row = row;
        const uint a_col = k_base + (tx / 8);
        const uint a_offset = (tx / 8) + (tx % 8) * 2;

        if (a_row < params.M && a_col + a_offset < params.K) {
            const uint idx = a_row * params.K + a_col + a_offset;
            float16_t val = float16_t(a[idx]);
            s_a[tx][a_offset] = val;
            if (a_offset + 1 < TILE_K && a_col + a_offset + 1 < params.K) {
                s_a[tx][a_offset + 1] = float16_t(a[idx + 1]);
            } else {
                s_a[tx][a_offset + 1] = float16_t(0.0);
            }
        } else {
            s_a[tx][a_offset] = float16_t(0.0);
            s_a[tx][a_offset + 1] = float16_t(0.0);
        }

        const uint b_row = k_base + (tx / 8);
        const uint b_col = col + (tx % 8) * 8;

        if (b_row < params.K) {
            for (uint j = 0; j < 8 && b_col + j < params.N; ++j) {
                const uint idx = b_row * params.N + b_col + j;
                s_b[b_row - k_base][b_col - bx * TILE_N + j] = float16_t(b[idx]);
            }
            for (uint j = 8; j < 16 && b_col + j < params.N; ++j) {
                const uint idx = (b_row + 1) * params.N + b_col + j;
                s_b[b_row - k_base + 1][b_col - bx * TILE_N + j] = float16_t(b[idx]);
            }
        }

        memoryBarrierShared();
        barrier();

        float16_t a0 = s_a[tx][0];
        float16_t a1 = s_a[tx][1];
        float16_t a2 = s_a[tx][2];
        float16_t a3 = s_a[tx][3];
        float16_t a4 = s_a[tx][4];
        float16_t a5 = s_a[tx][5];
        float16_t a6 = s_a[tx][6];
        float16_t a7 = s_a[tx][7];

        for (uint k = 0; k < TILE_K; ++k) {
            float16_t b0 = subgroupShuffleXor(s_b[k][0], tx);
            float16_t b1 = subgroupShuffleXor(s_b[k][1], tx);
            float16_t b2 = subgroupShuffleXor(s_b[k][2], tx);
            float16_t b3 = subgroupShuffleXor(s_b[k][3], tx);
            float16_t b4 = subgroupShuffleXor(s_b[k][4], tx);
            float16_t b5 = subgroupShuffleXor(s_b[k][5], tx);
            float16_t b6 = subgroupShuffleXor(s_b[k][6], tx);
            float16_t b7 = subgroupShuffleXor(s_b[k][7], tx);

            acc[0] = fma(a0, b0, acc[0]);
            acc[1] = fma(a1, b1, acc[1]);
            acc[2] = fma(a2, b2, acc[2]);
            acc[3] = fma(a3, b3, acc[3]);
            acc[4] = fma(a4, b4, acc[4]);
            acc[5] = fma(a5, b5, acc[5]);
            acc[6] = fma(a6, b6, acc[6]);
            acc[7] = fma(a7, b7, acc[7]);
        }

        memoryBarrierShared();
        barrier();
    }

    if (row < params.M) {
        const uint out_row = row;
        for (uint i = 0; i < 8; ++i) {
            const uint out_col = col + i * 8 + tx;
            if (out_col < params.N) {
                const uint out_idx = out_row * params.N + out_col;
                const uint acc_idx = i + (tx / 8);
                const float result = float(acc[acc_idx % 8]) * params.alpha;
                c[out_idx] = result;
            }
        }
    }
}
