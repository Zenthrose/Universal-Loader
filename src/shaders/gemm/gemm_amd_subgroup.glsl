#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_extended_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(constant_id = 0) const uint TILE_M = 128;
layout(constant_id = 1) const uint TILE_N = 64;
layout(constant_id = 2) const uint TILE_K = 16;

layout(local_size_x = 64, local_size_y = 1) in;

layout(binding = 0) readonly buffer A { uint a[]; };
layout(binding = 1) readonly buffer B { uint b[]; };
layout(binding = 2) buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint K_tiles;
    float alpha;
    float beta;
    uint lda;
    uint ldb;
    uint ldc;
} params;

shared float16_t s_a[TILE_M][TILE_K];
shared float16_t s_b[TILE_K][TILE_N];

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;
    const uint tx = gl_LocalInvocationID.x;
    const uint sg_size = gl_SubgroupSize;

    const uint warp_id = tx / sg_size;
    const uint lane_id = tx % sg_size;

    const uint rows_per_warp = TILE_M / (64 / sg_size);
    const uint cols_per_warp = TILE_N / sg_size;

    const uint row = by * TILE_M + warp_id * rows_per_warp;
    const uint col = bx * TILE_N + lane_id * cols_per_warp;

    const uint num_acc = rows_per_warp * cols_per_warp;
    float16_t acc[16];

    for (uint i = 0; i < num_acc; ++i) {
        acc[i] = float16_t(0.0);
    }

    const uint num_loads_a = (TILE_M * TILE_K) / 64;
    const uint num_loads_b = (TILE_K * TILE_N) / 64;

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        for (uint i = 0; i < num_loads_a; ++i) {
            const uint load_idx = tx * num_loads_a + i;
            const uint local_row = load_idx / TILE_K;
            const uint local_k = load_idx % TILE_K;
            const uint global_row = by * TILE_M + local_row;
            const uint global_k = k_base + local_k;

            if (global_row < params.M && global_k < params.K) {
                uint32_t packed = a[global_row * params.lda + global_k];
                s_a[local_row][local_k] = float16_t(unpackHalf2x16(packed).x);
            } else {
                s_a[local_row][local_k] = float16_t(0.0);
            }
        }

        for (uint i = 0; i < num_loads_b; ++i) {
            const uint load_idx = tx * num_loads_b + i;
            const uint local_k = load_idx / TILE_N;
            const uint local_n = load_idx % TILE_N;
            const uint global_k = k_base + local_k;
            const uint global_n = bx * TILE_N + local_n;

            if (global_k < params.K && global_n < params.N) {
                uint32_t packed = b[global_k * params.ldb + global_n];
                s_b[local_k][local_n] = float16_t(unpackHalf2x16(packed).x);
            } else {
                s_b[local_k][local_n] = float16_t(0.0);
            }
        }

        memoryBarrierShared();
        barrier();

        for (uint k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (uint m = 0; m < rows_per_warp; ++m) {
                const float16_t a_val = s_a[warp_id * rows_per_warp + m][k];

                #pragma unroll
                for (uint n = 0; n < cols_per_warp; ++n) {
                    const float16_t b_val = subgroupShuffle(s_b[k][lane_id * cols_per_warp + n], lane_id);
                    acc[m * cols_per_warp + n] = fma(a_val, b_val, acc[m * cols_per_warp + n]);
                }
            }
        }

        memoryBarrierShared();
        barrier();
    }

    const float alpha = params.alpha;
    const float beta = params.beta;

    for (uint m = 0; m < rows_per_warp; ++m) {
        const uint out_row = row + m;
        if (out_row < params.M) {
            for (uint n = 0; n < cols_per_warp; ++n) {
                const uint out_col = col + n;
                if (out_col < params.N) {
                    const uint out_idx = out_row * params.ldc + out_col;
                    const float c_val = (beta != 0.0) ? beta * c[out_idx] : 0.0;
                    c[out_idx] = c_val + alpha * float(acc[m * cols_per_warp + n]);
                }
            }
        }
    }
}
