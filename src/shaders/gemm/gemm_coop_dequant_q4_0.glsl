#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroups : require

layout(local_size_x = 32, local_size_y = 1) in;

layout(binding = 0) readonly buffer A_Q4 { uint16_t a_q4[]; };
layout(binding = 1) readonly buffer A_Scale { float a_scale[]; };
layout(binding = 2) readonly buffer B { float b[]; };
layout(binding = 3) writeonly buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint K_tiles;
    float alpha;
    float beta;
    uint block_size;
    uint mat_layout;
} params;

layout(constant_id = 0) const uint TILE_M = 16;
layout(constant_id = 1) const uint TILE_N = 16;
layout(constant_id = 2) const uint TILE_K = 16;

const uint ScopeSubgroup = 1;
const uint ScopeWorkgroup = 2;
const uint LayoutRowMajor = 0;
const uint LayoutColMajor = 1;

cooperative_matrixKHR<float, ScopeSubgroup, UseA, TILE_M, TILE_K> matA;
cooperative_matrixKHR<float, ScopeSubgroup, UseB, TILE_K, TILE_N> matB;
cooperative_matrixKHR<float, ScopeSubgroup, UseC, TILE_M, TILE_N> matC;

shared float s_a_dequant[TILE_M][TILE_K];
shared float s_b[TILE_K][TILE_N];

float dequant_q4_value(uint16_t packed, uint idx, float scale) {
    if (idx % 2 == 0) {
        return (float(packed & 0x0F) - 8.0) * scale;
    } else {
        return (float((packed >> 4) & 0x0F) - 8.0) * scale;
    }
}

void load_q4_tile(uint base_row, uint base_k) {
    const uint tx = gl_LocalInvocationID.x;

    for (uint i = 0; i < TILE_M; i += 32) {
        uint row = base_row + i + tx;
        if (row < params.M) {
            for (uint j = 0; j < TILE_K; j += 2) {
                uint k = base_k + j;
                if (k < params.K) {
                    uint idx = row * params.K + k;
                    uint block = idx / params.block_size;
                    uint idx_in_block = idx % params.block_size;
                    uint packed_idx = block * (params.block_size / 2) + idx_in_block / 2;
                    uint16_t packed = a_q4[packed_idx];
                    float scale = a_scale[block];
                    float val = dequant_q4_value(packed, idx_in_block, scale);

                    if (i + tx < TILE_M && j < TILE_K) {
                        s_a_dequant[i + tx][j] = val;
                    }
                } else if (j < TILE_K) {
                    s_a_dequant[i + tx][j] = 0.0;
                }

                if (k + 1 < params.K) {
                    uint idx1 = row * params.K + k + 1;
                    uint block1 = idx1 / params.block_size;
                    uint idx_in_block1 = idx1 % params.block_size;
                    uint packed_idx1 = block1 * (params.block_size / 2) + idx_in_block1 / 2;
                    uint16_t packed1 = a_q4[packed_idx1];
                    float scale1 = a_scale[block1];
                    float val1 = dequant_q4_value(packed1, idx_in_block1, scale1);

                    if (i + tx < TILE_M && j + 1 < TILE_K) {
                        s_a_dequant[i + tx][j + 1] = val1;
                    }
                } else if (j + 1 < TILE_K) {
                    s_a_dequant[i + tx][j + 1] = 0.0;
                }
            }
        } else {
            for (uint j = 0; j < TILE_K; ++j) {
                if (i + tx < TILE_M) {
                    s_a_dequant[i + tx][j] = 0.0;
                }
            }
        }
    }
}

void load_b_tile(uint base_k, uint base_col) {
    const uint tx = gl_LocalInvocationID.x;

    for (uint i = 0; i < TILE_K; ++i) {
        uint k = base_k + i;
        if (k < params.K) {
            for (uint j = 0; j < TILE_N; j += 32) {
                uint col = base_col + j + tx;
                if (col < params.N) {
                    uint offset = k * params.N + col;
                    if (j + tx < TILE_N) {
                        s_b[i][j + tx] = b[offset];
                    }
                } else if (j + tx < TILE_N) {
                    s_b[i][j + tx] = 0.0;
                }
            }
        } else {
            for (uint j = 0; j < TILE_N; ++j) {
                s_b[i][j] = 0.0;
            }
        }
    }
}

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;

    const uint row = by * TILE_M;
    const uint col = bx * TILE_N;

    cooperativeMatrixLoadKHR(matC, c, row * params.N + col, params.N, LayoutColMajor);

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        load_q4_tile(row, k_base);
        load_b_tile(k_base, col);

        memoryBarrierShared();
        barrier();

        cooperativeMatrixLoadKHR(matA, s_a_dequant, 0, TILE_K, LayoutRowMajor);
        cooperativeMatrixLoadKHR(matB, s_b, 0, TILE_N, LayoutRowMajor);

        cooperativeMatrixMulAddKHR(matA, matB, matC, matC);

        memoryBarrierShared();
        barrier();
    }

    cooperativeMatrixStoreKHR(matC, c, row * params.N + col, params.N, LayoutColMajor);
}
