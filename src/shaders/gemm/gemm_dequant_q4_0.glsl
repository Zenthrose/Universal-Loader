#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 8, local_size_y = 8) in;

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
    uint padding[3];
} params;

const uint TILE_M = 64;
const uint TILE_N = 64;
const uint TILE_K = 16;

shared float s_a[TILE_M][TILE_K + 1];
shared float s_b[TILE_K][TILE_N + 1];

float dequant_q4_0(uint16_t packed, float scale) {
    float lo = float(packed & 0x0F);
    float hi = float((packed >> 4) & 0x0F);
    return (lo - 8.0) * scale;
}

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;
    const uint tx = gl_LocalInvocationID.x;
    const uint ty = gl_LocalInvocationID.y;

    const uint row = by * TILE_M + ty * 8;
    const uint col = bx * TILE_N + tx * 8;

    vec4 c0 = vec4(0.0);
    vec4 c1 = vec4(0.0);
    vec4 c2 = vec4(0.0);
    vec4 c3 = vec4(0.0);
    vec4 c4 = vec4(0.0);
    vec4 c5 = vec4(0.0);
    vec4 c6 = vec4(0.0);
    vec4 c7 = vec4(0.0);

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        const uint a_row = row;
        const uint a_col = k_base + tx * 2;
        const uint b_row = k_base + ty * 2;
        const uint b_col = col;

        if (a_row < params.M && a_col < params.K) {
            uint idx = a_row * params.K + a_col;
            uint block_a = idx / params.block_size;
            uint idx_a = idx % params.block_size;
            uint packed_idx = block_a * (params.block_size / 2) + idx_a / 2;
            uint16_t packed = a_q4[packed_idx];
            float scale = a_scale[block_a];

            s_a[ty * 8][tx * 2] = (idx_a % 2 == 0) ? 
                ((float(packed & 0x0F) - 8.0) * scale) : 
                ((float((packed >> 4) & 0x0F) - 8.0) * scale);

            if (a_col + 1 < params.K) {
                uint idx1 = a_row * params.K + a_col + 1;
                uint block_a1 = idx1 / params.block_size;
                uint idx_a1 = idx1 % params.block_size;
                uint packed_idx1 = block_a1 * (params.block_size / 2) + idx_a1 / 2;
                uint16_t packed1 = a_q4[packed_idx1];
                float scale1 = a_scale[block_a1];

                s_a[ty * 8][tx * 2 + 1] = (idx_a1 % 2 == 0) ? 
                    ((float(packed1 & 0x0F) - 8.0) * scale1) : 
                    ((float((packed1 >> 4) & 0x0F) - 8.0) * scale1);
            } else {
                s_a[ty * 8][tx * 2 + 1] = 0.0;
            }
        } else {
            s_a[ty * 8][tx * 2] = 0.0;
            s_a[ty * 8][tx * 2 + 1] = 0.0;
        }

        if (ty * 8 < TILE_K) {
            if (b_row < params.K && b_col < params.N) {
                uint offset = b_row * params.N + b_col;
                s_b[ty * 8][tx * 8] = b[offset];
                if (b_col + 1 < params.N) {
                    s_b[ty * 8][tx * 8 + 1] = b[offset + 1];
                }
                if (b_col + 2 < params.N) {
                    s_b[ty * 8][tx * 8 + 2] = b[offset + 2];
                }
                if (b_col + 3 < params.N) {
                    s_b[ty * 8][tx * 8 + 3] = b[offset + 3];
                }
            } else {
                s_b[ty * 8][tx * 8] = 0.0;
                s_b[ty * 8][tx * 8 + 1] = 0.0;
                s_b[ty * 8][tx * 8 + 2] = 0.0;
                s_b[ty * 8][tx * 8 + 3] = 0.0;
            }
        }

        if (ty * 8 + 1 < TILE_K) {
            if (b_row + 1 < params.K && b_col < params.N) {
                uint offset = (b_row + 1) * params.N + b_col;
                s_b[ty * 8 + 1][tx * 8] = b[offset];
                if (b_col + 1 < params.N) {
                    s_b[ty * 8 + 1][tx * 8 + 1] = b[offset + 1];
                }
                if (b_col + 2 < params.N) {
                    s_b[ty * 8 + 1][tx * 8 + 2] = b[offset + 2];
                }
                if (b_col + 3 < params.N) {
                    s_b[ty * 8 + 1][tx * 8 + 3] = b[offset + 3];
                }
            } else {
                s_b[ty * 8 + 1][tx * 8] = 0.0;
                s_b[ty * 8 + 1][tx * 8 + 1] = 0.0;
                s_b[ty * 8 + 1][tx * 8 + 2] = 0.0;
                s_b[ty * 8 + 1][tx * 8 + 3] = 0.0;
            }
        }

        memoryBarrierShared();
        barrier();

        for (uint k = 0; k < TILE_K; ++k) {
            const float a0 = s_a[ty * 8][k];
            const float a1 = s_a[ty * 8 + 1][k];
            const float a2 = s_a[ty * 8 + 2][k];
            const float a3 = s_a[ty * 8 + 3][k];
            const float a4 = s_a[ty * 8 + 4][k];
            const float a5 = s_a[ty * 8 + 5][k];
            const float a6 = s_a[ty * 8 + 6][k];
            const float a7 = s_a[ty * 8 + 7][k];

            const vec4 b0 = vec4(s_b[k][tx * 8], s_b[k][tx * 8 + 1], s_b[k][tx * 8 + 2], s_b[k][tx * 8 + 3]);
            const vec4 b1 = vec4(s_b[k][tx * 8 + 4], s_b[k][tx * 8 + 5], s_b[k][tx * 8 + 6], s_b[k][tx * 8 + 7]);

            c0 += vec4(a0) * b0;
            c1 += vec4(a1) * b0;
            c2 += vec4(a2) * b0;
            c3 += vec4(a3) * b0;
            c4 += vec4(a4) * b1;
            c5 += vec4(a5) * b1;
            c6 += vec4(a6) * b1;
            c7 += vec4(a7) * b1;
        }

        memoryBarrierShared();
        barrier();
    }

    if (row < params.M && col < params.N) {
        c[row * params.N + col] = c0.x * params.alpha;
    }
    if (row < params.M && col + 1 < params.N) {
        c[row * params.N + col + 1] = c0.y * params.alpha;
    }
    if (row < params.M && col + 2 < params.N) {
        c[row * params.N + col + 2] = c0.z * params.alpha;
    }
    if (row < params.M && col + 3 < params.N) {
        c[row * params.N + col + 3] = c0.w * params.alpha;
    }

    if (row + 1 < params.M && col < params.N) {
        c[(row + 1) * params.N + col] = c1.x * params.alpha;
    }
    if (row + 1 < params.M && col + 1 < params.N) {
        c[(row + 1) * params.N + col + 1] = c1.y * params.alpha;
    }
    if (row + 1 < params.M && col + 2 < params.N) {
        c[(row + 1) * params.N + col + 2] = c1.z * params.alpha;
    }
    if (row + 1 < params.M && col + 3 < params.N) {
        c[(row + 1) * params.N + col + 3] = c1.w * params.alpha;
    }

    if (row + 2 < params.M && col < params.N) {
        c[(row + 2) * params.N + col] = c2.x * params.alpha;
    }
    if (row + 2 < params.M && col + 1 < params.N) {
        c[(row + 2) * params.N + col + 1] = c2.y * params.alpha;
    }
    if (row + 2 < params.M && col + 2 < params.N) {
        c[(row + 2) * params.N + col + 2] = c2.z * params.alpha;
    }
    if (row + 2 < params.M && col + 3 < params.N) {
        c[(row + 2) * params.N + col + 3] = c2.w * params.alpha;
    }

    if (row + 3 < params.M && col < params.N) {
        c[(row + 3) * params.N + col] = c3.x * params.alpha;
    }
    if (row + 3 < params.M && col + 1 < params.N) {
        c[(row + 3) * params.N + col + 1] = c3.y * params.alpha;
    }
    if (row + 3 < params.M && col + 2 < params.N) {
        c[(row + 3) * params.N + col + 2] = c3.z * params.alpha;
    }
    if (row + 3 < params.M && col + 3 < params.N) {
        c[(row + 3) * params.N + col + 3] = c3.w * params.alpha;
    }

    if (row + 4 < params.M && col < params.N) {
        c[(row + 4) * params.N + col] = c4.x * params.alpha;
    }
    if (row + 4 < params.M && col + 1 < params.N) {
        c[(row + 4) * params.N + col + 1] = c4.y * params.alpha;
    }
    if (row + 4 < params.M && col + 2 < params.N) {
        c[(row + 4) * params.N + col + 2] = c4.z * params.alpha;
    }
    if (row + 4 < params.M && col + 3 < params.N) {
        c[(row + 4) * params.N + col + 3] = c4.w * params.alpha;
    }

    if (row + 5 < params.M && col < params.N) {
        c[(row + 5) * params.N + col] = c5.x * params.alpha;
    }
    if (row + 5 < params.M && col + 1 < params.N) {
        c[(row + 5) * params.N + col + 1] = c5.y * params.alpha;
    }
    if (row + 5 < params.M && col + 2 < params.N) {
        c[(row + 5) * params.N + col + 2] = c5.z * params.alpha;
    }
    if (row + 5 < params.M && col + 3 < params.N) {
        c[(row + 5) * params.N + col + 3] = c5.w * params.alpha;
    }

    if (row + 6 < params.M && col < params.N) {
        c[(row + 6) * params.N + col] = c6.x * params.alpha;
    }
    if (row + 6 < params.M && col + 1 < params.N) {
        c[(row + 6) * params.N + col + 1] = c6.y * params.alpha;
    }
    if (row + 6 < params.M && col + 2 < params.N) {
        c[(row + 6) * params.N + col + 2] = c6.z * params.alpha;
    }
    if (row + 6 < params.M && col + 3 < params.N) {
        c[(row + 6) * params.N + col + 3] = c6.w * params.alpha;
    }

    if (row + 7 < params.M && col < params.N) {
        c[(row + 7) * params.N + col] = c7.x * params.alpha;
    }
    if (row + 7 < params.M && col + 1 < params.N) {
        c[(row + 7) * params.N + col + 1] = c7.y * params.alpha;
    }
    if (row + 7 < params.M && col + 2 < params.N) {
        c[(row + 7) * params.N + col + 2] = c7.z * params.alpha;
    }
    if (row + 7 < params.M && col + 3 < params.N) {
        c[(row + 7) * params.N + col + 3] = c7.w * params.alpha;
    }
}
