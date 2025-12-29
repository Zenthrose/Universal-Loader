#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_KHR_shader_subgroups : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(constant_id = 0) const uint TILE_M = 32;
layout(constant_id = 1) const uint TILE_N = 32;
layout(constant_id = 2) const uint TILE_K = 16;
layout(constant_id = 3) const uint USE_FLOAT16 = 0;
layout(constant_id = 4) const uint USE_INT8 = 0;
layout(constant_id = 5) const uint USE_INT4 = 0;

layout(local_size_x = 32, local_size_y = 1) in;

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

const uint ScopeSubgroup = 1;
const uint ScopeWorkgroup = 2;

const uint LayoutRowMajor = 0;
const uint LayoutColMajor = 1;

#if USE_FLOAT16
    cooperative_matrixKHR<float16_t, ScopeSubgroup, UseA, TILE_M, TILE_K> matA;
    cooperative_matrixKHR<float16_t, ScopeSubgroup, UseB, TILE_K, TILE_N> matB;
    cooperative_matrixKHR<float, ScopeSubgroup, UseC, TILE_M, TILE_N> matC;

    shared float16_t s_a[TILE_M][TILE_K];
    shared float16_t s_b[TILE_K][TILE_N];

    void load_tile_to_shared_a(uint row, uint k_base) {
        const uint tx = gl_LocalInvocationID.x;
        const uint num_elems = (TILE_M * TILE_K) / 32;

        for (uint i = 0; i < num_elems; ++i) {
            const uint idx = tx * num_elems + i;
            const uint local_row = idx / TILE_K;
            const uint local_k = idx % TILE_K;
            const uint global_row = row + local_row;
            const uint global_k = k_base + local_k;

            if (global_row < params.M && global_k < params.K) {
                uint16_t val = uint16_t(a[(global_row * params.lda + global_k) / 2]);
                s_a[local_row][local_k] = float16_t(unpackHalf2x16(val).x);
            } else {
                s_a[local_row][local_k] = float16_t(0.0);
            }
        }
    }

    void load_tile_to_shared_b(uint k_base, uint col) {
        const uint tx = gl_LocalInvocationID.x;
        const uint num_elems = (TILE_K * TILE_N) / 32;

        for (uint i = 0; i < num_elems; ++i) {
            const uint idx = tx * num_elems + i;
            const uint local_k = idx / TILE_N;
            const uint local_n = idx % TILE_N;
            const uint global_k = k_base + local_k;
            const uint global_n = col + local_n;

            if (global_k < params.K && global_n < params.N) {
                uint16_t val = uint16_t(b[(global_k * params.ldb + global_n) / 2]);
                s_b[local_k][local_n] = float16_t(unpackHalf2x16(val).x);
            } else {
                s_b[local_k][local_n] = float16_t(0.0);
            }
        }
    }
#else
    cooperative_matrixKHR<float, ScopeSubgroup, UseA, TILE_M, TILE_K> matA;
    cooperative_matrixKHR<float, ScopeSubgroup, UseB, TILE_K, TILE_N> matB;
    cooperative_matrixKHR<float, ScopeSubgroup, UseC, TILE_M, TILE_N> matC;

    void load_tile_to_shared_a(uint row, uint k_base) {
        const uint tx = gl_LocalInvocationID.x;
        const uint num_elems = (TILE_M * TILE_K) / 32;

        for (uint i = 0; i < num_elems; ++i) {
            const uint idx = tx * num_elems + i;
            const uint local_row = idx / TILE_K;
            const uint local_k = idx % TILE_K;
            const uint global_row = row + local_row;
            const uint global_k = k_base + local_k;

            float val = (global_row < params.M && global_k < params.K) ?
                uintBitsToFloat(a[global_row * params.lda + global_k]) : 0.0;
            cooperativeMatrixLoadKHR(matA, &val, 0, TILE_K, LayoutRowMajor);
        }
    }

    void load_tile_to_shared_b(uint k_base, uint col) {
        const uint tx = gl_LocalInvocationID.x;
        const uint num_elems = (TILE_K * TILE_N) / 32;

        for (uint i = 0; i < num_elems; ++i) {
            const uint idx = tx * num_elems + i;
            const uint local_k = idx / TILE_N;
            const uint local_n = idx % TILE_N;
            const uint global_k = k_base + local_k;
            const uint global_n = col + local_n;

            float val = (global_k < params.K && global_n < params.N) ?
                uintBitsToFloat(b[global_k * params.ldb + global_n]) : 0.0;
            cooperativeMatrixLoadKHR(matB, &val, 0, TILE_N, LayoutRowMajor);
        }
    }
#endif

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;

    const uint row = by * TILE_M;
    const uint col = bx * TILE_N;

    cooperativeMatrixLoadKHR(matC, c, row * params.ldc + col, params.ldc, LayoutColMajor);

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        load_tile_to_shared_a(row, k_base);
        load_tile_to_shared_b(k_base, col);

        memoryBarrierShared();
        barrier();

        cooperativeMatrixLoadKHR(matA, s_a, 0, TILE_K, LayoutRowMajor);
        cooperativeMatrixLoadKHR(matB, s_b, 0, TILE_N, LayoutRowMajor);

        cooperativeMatrixMulAddKHR(matA, matB, matC, matC);

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
