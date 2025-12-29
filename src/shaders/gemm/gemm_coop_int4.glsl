#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_KHR_shader_subgroups : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(constant_id = 0) const uint TILE_M = 32;
layout(constant_id = 1) const uint TILE_N = 32;
layout(constant_id = 2) const uint TILE_K = 32;

layout(local_size_x = 32, local_size_y = 1) in;

layout(binding = 0) readonly buffer A_Q { uint a_q[]; };
layout(binding = 1) readonly buffer A_Scale { float a_scale[]; };
layout(binding = 2) readonly buffer B_Q { uint b_q[]; };
layout(binding = 3) readonly buffer B_Scale { float b_scale[]; };
layout(binding = 4) buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint K_tiles;
    float alpha;
    float beta;
    uint block_size_a;
    uint block_size_b;
    uint lda;
    uint ldb;
    uint ldc;
} params;

const uint ScopeSubgroup = 1;
const uint ScopeWorkgroup = 2;

const uint LayoutRowMajor = 0;
const uint LayoutColMajor = 1;

cooperative_matrixKHR<int8_t, ScopeSubgroup, UseA, TILE_M, TILE_K> matA_q;
cooperative_matrixKHR<int8_t, ScopeSubgroup, UseB, TILE_K, TILE_N> matB_q;
cooperative_matrixKHR<float, ScopeSubgroup, UseC, TILE_M, TILE_N> matC;

shared int8_t s_a_q[TILE_M][TILE_K];
shared int8_t s_b_q[TILE_K][TILE_N];
shared float s_a_scale[TILE_M];
shared float s_b_scale[TILE_N];

int8_t unpack_int4(uint32_t packed, uint idx) {
    uint byte_idx = idx / 2;
    uint8_t byte = uint8_t((packed >> (byte_idx * 8)) & 0xFF);
    int8_t val = int8_t((idx % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F));
    return (val < 8) ? val : (val - 16);
}

void load_int4_tile_a(uint base_row, uint base_k) {
    const uint tx = gl_LocalInvocationID.x;
    const uint num_elems = (TILE_M * TILE_K) / 32;

    for (uint i = 0; i < num_elems; ++i) {
        const uint idx = tx * num_elems + i;
        const uint local_row = idx / TILE_K;
        const uint local_k = idx % TILE_K;
        const uint global_row = base_row + local_row;
        const uint global_k = base_k + local_k;

        if (global_row < params.M && global_k < params.K) {
            const uint global_idx = global_row * params.lda + global_k;
            const uint block = global_idx / params.block_size_a;
            const uint idx_in_block = global_idx % params.block_size_a;
            const uint packed_idx = block * (params.block_size_a / 2) + idx_in_block / 2;
            const uint32_t packed = a_q[packed_idx];

            s_a_q[local_row][local_k] = unpack_int4(packed, idx_in_block);

            if (local_k == 0) {
                s_a_scale[local_row] = a_scale[block];
            }
        } else {
            s_a_q[local_row][local_k] = int8_t(0);
            if (local_k == 0) {
                s_a_scale[local_row] = 1.0;
            }
        }
    }
}

void load_int4_tile_b(uint base_k, uint base_col) {
    const uint tx = gl_LocalInvocationID.x;
    const uint num_elems = (TILE_K * TILE_N) / 32;

    for (uint i = 0; i < num_elems; ++i) {
        const uint idx = tx * num_elems + i;
        const uint local_k = idx / TILE_N;
        const uint local_n = idx % TILE_N;
        const uint global_k = base_k + local_k;
        const uint global_n = base_col + local_n;

        if (global_k < params.K && global_n < params.N) {
            const uint global_idx = global_k * params.ldb + global_n;
            const uint block = global_idx / params.block_size_b;
            const uint idx_in_block = global_idx % params.block_size_b;
            const uint packed_idx = block * (params.block_size_b / 2) + idx_in_block / 2;
            const uint32_t packed = b_q[packed_idx];

            s_b_q[local_k][local_n] = unpack_int4(packed, idx_in_block);

            if (local_n == 0) {
                s_b_scale[local_n] = b_scale[block];
            }
        } else {
            s_b_q[local_k][local_n] = int8_t(0);
            if (local_n == 0) {
                s_b_scale[local_n] = 1.0;
            }
        }
    }
}

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;

    const uint row = by * TILE_M;
    const uint col = bx * TILE_N;

    cooperativeMatrixLoadKHR(matC, c, row * params.ldc + col, params.ldc, LayoutColMajor);

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        load_int4_tile_a(row, k_base);
        load_int4_tile_b(k_base, col);

        memoryBarrierShared();
        barrier();

        cooperativeMatrixLoadKHR(matA_q, s_a_q, 0, TILE_K, LayoutRowMajor);
        cooperativeMatrixLoadKHR(matB_q, s_b_q, 0, TILE_N, LayoutRowMajor);

        cooperative_matrix<int, ScopeSubgroup, UseC, TILE_M, TILE_N> matC_int;
        cooperativeMatrixMulAddKHR(matA_q, matB_q, matC_int, matC_int);

        for (uint m = 0; m < TILE_M; ++m) {
            for (uint n = 0; n < TILE_N; ++n) {
                const float scale_prod = s_a_scale[m] * s_b_scale[n];
                matC[m * TILE_N + n] += float(matC_int[m * TILE_N + n]) * scale_prod;
            }
        }

        memoryBarrierShared();
        barrier();
    }

    const float alpha = params.alpha;
    const float beta = params.beta;

    if (beta != 0.0) {
        cooperative_matrix<float, ScopeSubgroup, UseC, TILE_M, TILE_N> matC_beta;
        cooperativeMatrixLoadKHR(matC_beta, c, row * params.ldc + col, params.ldc, LayoutColMajor);

        for (uint i = 0; i < TILE_M * TILE_N; ++i) {
            matC[i] = alpha * matC[i] + beta * matC_beta[i];
        }
    } else {
        for (uint i = 0; i < TILE_M * TILE_N; ++i) {
            matC[i] = alpha * matC[i];
        }
    }

    cooperativeMatrixStoreKHR(matC, c, row * params.ldc + col, params.ldc, LayoutColMajor);
}
