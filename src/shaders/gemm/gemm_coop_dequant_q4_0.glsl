#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroups : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(constant_id = 0) const uint TILE_M = 32;
layout(constant_id = 1) const uint TILE_N = 32;
layout(constant_id = 2) const uint TILE_K = 16;
layout(constant_id = 3) const uint QUANT_TYPE = 0;

layout(local_size_x = 32, local_size_y = 1) in;

layout(binding = 0) readonly buffer A_Q { uint a_q[]; };
layout(binding = 1) readonly buffer A_Scale { float a_scale[]; };
layout(binding = 2) readonly buffer A_Min { float a_min[]; };
layout(binding = 3) readonly buffer B { uint b[]; };
layout(binding = 4) buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint K_tiles;
    float alpha;
    float beta;
    uint block_size;
    uint lda;
    uint ldb;
    uint ldc;
} params;

const uint ScopeSubgroup = 1;
const uint ScopeWorkgroup = 2;
const uint LayoutRowMajor = 0;
const uint LayoutColMajor = 1;

const uint QUANT_Q4_0 = 0;
const uint QUANT_Q4_K = 1;
const uint QUANT_Q5_K = 2;

cooperative_matrixKHR<float, ScopeSubgroup, UseA, TILE_M, TILE_K> matA;
cooperative_matrixKHR<float, ScopeSubgroup, UseB, TILE_K, TILE_N> matB;
cooperative_matrixKHR<float, ScopeSubgroup, UseC, TILE_M, TILE_N> matC;

shared float s_a_dequant[TILE_M][TILE_K];
shared float s_b[TILE_K][TILE_N];

float dequant_q4_0(uint16_t packed, uint idx, float scale) {
    uint8_t q = uint8_t((idx % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F));
    return (float(q) - 8.0) * scale;
}

float dequant_q4_k(uint32_t packed, uint32_t packed2, uint idx, float scale, float min_val) {
    uint idx_in_block = idx % 32;

    if (idx_in_block < 8) {
        uint8_t q = uint8_t((idx_in_block % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F));
        uint8_t qh_bit = uint8_t((idx_in_block < 4) ? ((packed >> (idx_in_block * 2)) & 0x3) : ((packed >> (8 + (idx_in_block - 4) * 2)) & 0x3));
        uint16_t combined = uint16_t(q) | (uint16_t(qh_bit) << 4);
        return (float(combined) - 8.0) * scale + min_val;
    } else if (idx_in_block < 16) {
        uint idx2 = idx_in_block - 8;
        uint8_t q = uint8_t((idx2 % 2 == 0) ? ((packed >> 8) & 0x0F) : ((packed >> 12) & 0x0F));
        uint8_t qh_bit = uint8_t((idx2 < 4) ? ((packed >> (16 + idx2 * 2)) & 0x3) : ((packed >> (24 + (idx2 - 4) * 2)) & 0x3));
        uint16_t combined = uint16_t(q) | (uint16_t(qh_bit) << 4);
        return (float(combined) - 8.0) * scale + min_val;
    } else {
        uint idx2 = idx_in_block - 16;
        uint8_t q = uint8_t((idx2 % 2 == 0) ? (packed2 & 0x0F) : ((packed2 >> 4) & 0x0F));
        return (float(q) - 8.0) * scale + min_val;
    }
}

float dequant_q5_k(uint32_t packed, uint32_t packed2, uint idx, float scale) {
    uint idx_in_block = idx % 32;

    if (idx_in_block < 8) {
        uint8_t q = uint8_t(packed & 0x1F);
        uint8_t qh = uint8_t((idx_in_block % 2 == 0) ? ((packed2 & 0xF0) >> 4) : (packed2 & 0x0F));
        return (float(q) | (float(qh) << 5) - 16.0) * scale;
    } else if (idx_in_block < 16) {
        uint8_t q = uint8_t((packed >> 5) & 0x1F);
        uint8_t qh = uint8_t(((packed2 >> 8) & 0xF0) >> 4);
        return (float(q) | (float(qh) << 5) - 16.0) * scale;
    } else {
        uint idx2 = idx_in_block - 16;
        uint8_t q = uint8_t((idx2 < 8) ? ((packed >> 10) & 0x1F) : ((packed >> 15) & 0x1F));
        uint8_t qh = uint8_t((idx2 < 8) ? ((packed2 >> 16) & 0xF) : ((packed2 >> 20) & 0xF));
        return (float(q) | (float(qh) << 5) - 16.0) * scale;
    }
}

void load_quant_tile(uint base_row, uint base_k) {
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
            const uint block = global_idx / params.block_size;
            const uint idx_in_block = global_idx % params.block_size;
            const float scale = a_scale[block];

            float val;
            if (QUANT_TYPE == QUANT_Q4_0) {
                const uint packed_idx = block * (params.block_size / 2) + idx_in_block / 2;
                const uint16_t packed = uint16_t(a_q[packed_idx]);
                val = dequant_q4_0(packed, idx_in_block, scale);
            } else if (QUANT_TYPE == QUANT_Q4_K) {
                const uint packed_idx = (block * 128) + (idx_in_block / 2);
                const uint32_t packed = a_q[packed_idx];
                const uint32_t packed2 = a_q[packed_idx + 16];
                const float min_val = a_min[block / 2];
                val = dequant_q4_k(packed, packed2, idx_in_block, scale, min_val);
            } else if (QUANT_TYPE == QUANT_Q5_K) {
                const uint packed_idx = (block * 64) + (idx_in_block / 2);
                const uint32_t packed = a_q[packed_idx];
                const uint32_t packed2 = a_q[packed_idx + 32];
                val = dequant_q5_k(packed, packed2, idx_in_block, scale);
            }

            s_a_dequant[local_row][local_k] = val;
        } else {
            s_a_dequant[local_row][local_k] = 0.0;
        }
    }
}

void load_b_tile(uint base_k, uint base_col) {
    const uint tx = gl_LocalInvocationID.x;
    const uint num_elems = (TILE_K * TILE_N) / 32;

    for (uint i = 0; i < num_elems; ++i) {
        const uint idx = tx * num_elems + i;
        const uint local_k = idx / TILE_N;
        const uint local_n = idx % TILE_N;
        const uint global_k = base_k + local_k;
        const uint global_n = base_col + local_n;

        if (global_k < params.K && global_n < params.N) {
            s_b[local_k][local_n] = uintBitsToFloat(b[global_k * params.ldb + global_n]);
        } else {
            s_b[local_k][local_n] = 0.0;
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

        load_quant_tile(row, k_base);
        load_b_tile(k_base, col);

        memoryBarrierShared();
        barrier();

        cooperativeMatrixLoadKHR(matA, s_a_dequant, 0, TILE_K, LayoutRowMajor);
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
