#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(constant_id = 0) const uint BLOCK_SIZE = 64;
layout(constant_id = 1) const uint VEC_WIDTH = 4;

layout(local_size_x = BLOCK_SIZE, local_size_y = 1) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer GateW { float gate[]; };
layout(binding = 2) readonly buffer UpW { float up[]; };
layout(binding = 3) readonly buffer DownW { float down[]; };
layout(binding = 4) buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint hidden_dim;
    uint intermediate_dim;
    uint num_tokens;
    uint padding[1];
};

shared float s_gate[BLOCK_SIZE];
shared float s_up[BLOCK_SIZE];

void main() {
    const uint tx = gl_LocalInvocationID.x;
    const uint token_idx = gl_WorkGroupID.x;

    if (token_idx >= params.num_tokens) return;

    const uint hidden_base = token_idx * params.hidden_dim;
    const uint num_hidden_tiles = (params.hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint num_inter_tiles = (params.intermediate_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float gate_accum = 0.0;
    float up_accum = 0.0;

    for (uint h_tile = 0; h_tile < num_hidden_tiles; ++h_tile) {
        const uint h_base = h_tile * BLOCK_SIZE;
        const uint h_idx = hidden_base + h_base + tx;

        const float x_val = (h_base + tx < params.hidden_dim) ? x[h_idx] : 0.0;

        for (uint i_tile = 0; i_tile < num_inter_tiles; ++i_tile) {
            const uint i_base = i_tile * BLOCK_SIZE;

            const uint gate_idx = h_base * params.intermediate_dim + i_base + tx;
            const uint up_idx = h_base * params.intermediate_dim + i_base + tx;

            if (h_base + tx < params.hidden_dim && i_base + tx < params.intermediate_dim) {
                gate_accum += x_val * gate[gate_idx];
                up_accum += x_val * up[up_idx];
            }
        }
    }

    s_gate[tx] = gate_accum;
    s_up[tx] = up_accum;

    memoryBarrierShared();
    barrier();

    const float silu = s_gate[tx] * subgroupInversesqrt(1.0 + exp(-s_gate[tx]));
    const float gated = silu * s_up[tx];

    float output = 0.0;

    for (uint i_tile = 0; i_tile < num_inter_tiles; ++i_tile) {
        const uint i_base = i_tile * BLOCK_SIZE;

        const float gated_val = (i_base + tx < params.intermediate_dim) ? gated : 0.0;
        const uint down_idx_base = i_base * params.hidden_dim + tx;

        if (i_base + tx < params.intermediate_dim && tx < params.hidden_dim) {
            const vec4 down_vec = vec4(down[down_idx_base], down[down_idx_base + 1],
                                       down[down_idx_base + 2], down[down_idx_base + 3]);
            const vec4 gated_vec = vec4(gated_val);
            output += dot(down_vec, gated_vec);
        }
    }

    const uint out_idx = hidden_base + tx;
    if (tx < params.hidden_dim) {
        out[out_idx] = output;
    }
}
