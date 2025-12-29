#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_extended_types : require

layout(constant_id = 0) const uint TILE_M = 128;
layout(constant_id = 1) const uint TILE_N = 64;
layout(constant_id = 2) const uint TILE_K = 16;

layout(local_size_x = 64, local_size_y = 2) in;

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

shared float s_a[TILE_M][TILE_K];
shared float s_b[TILE_K][TILE_N];

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;
    const uint tx = gl_LocalInvocationID.x;
    const uint ty = gl_LocalInvocationID.y;
    const uint tid = ty * 64 + tx;

    const uint row = by * TILE_M + ty * (TILE_M / 2);
    const uint col = bx * TILE_N;

    const uint acc_per_thread = (TILE_M / 2) * (TILE_N / 64);
    float acc[32];

    for (uint i = 0; i < acc_per_thread; ++i) {
        acc[i] = 0.0;
    }

    const uint num_loads_a = (TILE_M * TILE_K) / 128;
    const uint num_loads_b = (TILE_K * TILE_N) / 128;

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        for (uint i = 0; i < num_loads_a; ++i) {
            const uint load_idx = tid * num_loads_a + i;
            const uint local_row = load_idx / TILE_K;
            const uint local_k = load_idx % TILE_K;
            const uint global_row = by * TILE_M + local_row;
            const uint global_k = k_base + local_k;

            if (global_row < params.M && global_k < params.K) {
                s_a[local_row][local_k] = uintBitsToFloat(a[global_row * params.lda + global_k]);
            } else {
                s_a[local_row][local_k] = 0.0;
            }
        }

        for (uint i = 0; i < num_loads_b; ++i) {
            const uint load_idx = tid * num_loads_b + i;
            const uint local_k = load_idx / TILE_N;
            const uint local_n = load_idx % TILE_N;
            const uint global_k = k_base + local_k;
            const uint global_n = bx * TILE_N + local_n;

            if (global_k < params.K && global_n < params.N) {
                s_b[local_k][local_n] = uintBitsToFloat(b[global_k * params.ldb + global_n]);
            } else {
                s_b[local_k][local_n] = 0.0;
            }
        }

        memoryBarrierShared();
        barrier();

        #pragma unroll
        for (uint k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (uint m = 0; m < TILE_M / 2; ++m) {
                const float a_val = s_a[ty * (TILE_M / 2) + m][k];
                const vec4 b_vec0 = vec4(s_b[k][tx * 4], s_b[k][tx * 4 + 1], s_b[k][tx * 4 + 2], s_b[k][tx * 4 + 3]);
                const vec4 b_vec1 = vec4(s_b[k][tx * 4 + 32], s_b[k][tx * 4 + 33], s_b[k][tx * 4 + 34], s_b[k][tx * 4 + 35]);
                const vec4 b_vec2 = vec4(s_b[k][tx * 4 + 64], s_b[k][tx * 4 + 65], s_b[k][tx * 4 + 66], s_b[k][tx * 4 + 67]);
                const vec4 b_vec3 = vec4(s_b[k][tx * 4 + 96], s_b[k][tx * 4 + 97], s_b[k][tx * 4 + 98], s_b[k][tx * 4 + 99]);

                acc[m * 16 + 0] = fma(a_val, b_vec0.x, acc[m * 16 + 0]);
                acc[m * 16 + 1] = fma(a_val, b_vec0.y, acc[m * 16 + 1]);
                acc[m * 16 + 2] = fma(a_val, b_vec0.z, acc[m * 16 + 2]);
                acc[m * 16 + 3] = fma(a_val, b_vec0.w, acc[m * 16 + 3]);
                acc[m * 16 + 4] = fma(a_val, b_vec1.x, acc[m * 16 + 4]);
                acc[m * 16 + 5] = fma(a_val, b_vec1.y, acc[m * 16 + 5]);
                acc[m * 16 + 6] = fma(a_val, b_vec1.z, acc[m * 16 + 6]);
                acc[m * 16 + 7] = fma(a_val, b_vec1.w, acc[m * 16 + 7]);
                acc[m * 16 + 8] = fma(a_val, b_vec2.x, acc[m * 16 + 8]);
                acc[m * 16 + 9] = fma(a_val, b_vec2.y, acc[m * 16 + 9]);
                acc[m * 16 + 10] = fma(a_val, b_vec2.z, acc[m * 16 + 10]);
                acc[m * 16 + 11] = fma(a_val, b_vec2.w, acc[m * 16 + 11]);
                acc[m * 16 + 12] = fma(a_val, b_vec3.x, acc[m * 16 + 12]);
                acc[m * 16 + 13] = fma(a_val, b_vec3.y, acc[m * 16 + 13]);
                acc[m * 16 + 14] = fma(a_val, b_vec3.z, acc[m * 16 + 14]);
                acc[m * 16 + 15] = fma(a_val, b_vec3.w, acc[m * 16 + 15]);
            }
        }

        memoryBarrierShared();
        barrier();
    }

    const float alpha = params.alpha;
    const float beta = params.beta;

    for (uint m = 0; m < TILE_M / 2; ++m) {
        const uint out_row = row + m;
        if (out_row < params.M) {
            for (uint n = 0; n < 4; ++n) {
                const uint out_col_base = col + n * 16;
                for (uint j = 0; j < 4; ++j) {
                    const uint out_col = out_col_base + j;
                    if (out_col < params.N) {
                        const uint out_idx = out_row * params.ldc + out_col;
                        const float c_val = (beta != 0.0) ? beta * c[out_idx] : 0.0;
                        c[out_idx] = c_val + alpha * acc[m * 16 + n * 4 + j];
                    }
                }
            }
        }
    }
}
