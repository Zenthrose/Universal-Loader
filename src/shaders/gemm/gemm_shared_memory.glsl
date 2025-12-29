#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(constant_id = 0) const uint TILE_M = 128;
layout(constant_id = 1) const uint TILE_N = 64;
layout(constant_id = 2) const uint TILE_K = 16;

layout(local_size_x = 16, local_size_y = 8) in;

layout(binding = 0) readonly buffer A { float a[]; };
layout(binding = 1) readonly buffer B { float b[]; };
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
    const uint tid = ty * 16 + tx;

    const uint warp_row = ty * 4;
    const uint warp_col = tx * 4;

    const uint row = by * TILE_M + warp_row;
    const uint col = bx * TILE_N + warp_col;

    vec4 acc[16];
    for (uint i = 0; i < 16; ++i) {
        acc[i] = vec4(0.0);
    }

    const uint num_loads_per_thread_a = (TILE_M * TILE_K) / 128;
    const uint num_loads_per_thread_b = (TILE_K * TILE_N) / 128;

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        for (uint i = 0; i < num_loads_per_thread_a; ++i) {
            const uint load_idx = tid * num_loads_per_thread_a + i;
            const uint a_row_shared = load_idx / TILE_K;
            const uint a_col_shared = load_idx % TILE_K;
            const uint a_row_global = by * TILE_M + a_row_shared;
            const uint a_col_global = k_base + a_col_shared;

            if (a_row_global < params.M && a_col_global < params.K) {
                s_a[a_row_shared][a_col_shared] = a[a_row_global * params.lda + a_col_global];
            } else {
                s_a[a_row_shared][a_col_shared] = 0.0;
            }
        }

        for (uint i = 0; i < num_loads_per_thread_b; ++i) {
            const uint load_idx = tid * num_loads_per_thread_b + i;
            const uint b_row_shared = load_idx / TILE_N;
            const uint b_col_shared = load_idx % TILE_N;
            const uint b_row_global = k_base + b_row_shared;
            const uint b_col_global = bx * TILE_N + b_col_shared;

            if (b_row_global < params.K && b_col_global < params.N) {
                s_b[b_row_shared][b_col_shared] = b[b_row_global * params.ldb + b_col_global];
            } else {
                s_b[b_row_shared][b_col_shared] = 0.0;
            }
        }

        memoryBarrierShared();
        barrier();

        for (uint k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (uint m = 0; m < 4; ++m) {
                const float a_val = s_a[warp_row + m][k];
                const vec4 b_vec = vec4(s_b[k][warp_col], s_b[k][warp_col + 1], s_b[k][warp_col + 2], s_b[k][warp_col + 3]);
                acc[m * 4 + 0] += vec4(a_val) * b_vec;
                acc[m * 4 + 1] += vec4(a_val) * b_vec;
                acc[m * 4 + 2] += vec4(a_val) * b_vec;
                acc[m * 4 + 3] += vec4(a_val) * b_vec;
            }
        }

        memoryBarrierShared();
        barrier();
    }

    const float alpha = params.alpha;
    const float beta = params.beta;

    for (uint m = 0; m < 4; ++m) {
        const uint out_row = row + m;
        if (out_row < params.M) {
            const vec4 c_vals = (beta != 0.0) ?
                vec4(c[out_row * params.ldc + col], c[out_row * params.ldc + col + 1],
                     c[out_row * params.ldc + col + 2], c[out_row * params.ldc + col + 3]) * beta :
                vec4(0.0);

            for (uint n = 0; n < 4; ++n) {
                const uint out_col = col + n;
                if (out_col < params.N) {
                    c[out_row * params.ldc + out_col] = c_vals[n] + alpha * acc[m * 4 + n][n];
                }
            }
        }
    }
}
