#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(constant_id = 0) const uint TILE_SIZE = 64;
layout(constant_id = 1) const uint VEC_SIZE = 4;

layout(local_size_x = TILE_SIZE, local_size_y = 1) in;

layout(binding = 0) readonly buffer Input { float input_data[]; };
layout(binding = 1) readonly buffer Gate { float gate[]; };
layout(binding = 2) readonly buffer Up { float up[]; };
layout(binding = 3) readonly buffer Down { float down[]; };
layout(binding = 4) buffer Output { float output_data[]; };
layout(binding = 5) readonly buffer NormScale { float norm_scale[]; };

layout(push_constant) uniform Params {
    uint hidden_dim;
    uint ffn_dim;
    float epsilon;
    uint num_tokens;
    uint padding[3];
};

shared float s_input[TILE_SIZE];
shared float s_gate_intermediate[TILE_SIZE];
shared float s_up_intermediate[TILE_SIZE];
shared float s_rms[TILE_SIZE];

void main() {
    const uint tx = gl_LocalInvocationID.x;
    const uint token_idx = gl_WorkGroupID.x;
    const uint batch_base = token_idx * hidden_dim;

    if (token_idx >= params.num_tokens) return;

    const uint num_hidden_tiles = (params.hidden_dim + TILE_SIZE - 1) / TILE_SIZE;

    float sum_sq = 0.0;
    for (uint tile = 0; tile < num_hidden_tiles; ++tile) {
        const uint hidden_base = tile * TILE_SIZE;
        const uint idx = batch_base + hidden_base + tx;

        float x_val = (hidden_base + tx < params.hidden_dim) ? input_data[idx] : 0.0;
        sum_sq += x_val * x_val;

        s_input[tx] = x_val;
        memoryBarrierShared();
        barrier();
    }

    const float variance = subgroupAdd(sum_sq);
    const float rms = rsqrt(variance / float(params.hidden_dim) + params.epsilon);
    s_rms[tx] = rms;

    memoryBarrierShared();
    barrier();

    float gate_accum = 0.0;
    float up_accum = 0.0;

    for (uint tile = 0; tile < num_hidden_tiles; ++tile) {
        const uint hidden_base = tile * TILE_SIZE;
        const uint idx = batch_base + hidden_base + tx;

        const float x_norm = (hidden_base + tx < params.hidden_dim) ?
            input_data[idx] * s_rms[tx] * norm_scale[hidden_base + tx] : 0.0;

        const uint num_ffn_tiles = (params.ffn_dim + TILE_SIZE - 1) / TILE_SIZE;

        for (uint ffn_tile = 0; ffn_tile < num_ffn_tiles; ++ffn_tile) {
            const uint ffn_base = ffn_tile * TILE_SIZE;

            const uint gate_idx = hidden_base * params.ffn_dim + ffn_base + tx;
            const uint up_idx = hidden_base * params.ffn_dim + ffn_base + tx;

            if (hidden_base + tx < params.hidden_dim && ffn_base + tx < params.ffn_dim) {
                gate_accum += x_norm * gate[gate_idx];
                up_accum += x_norm * up[up_idx];
            }
        }
    }

    const float silu = gate_accum * subgroupInversesqrt(1.0 + exp(-gate_accum));
    const float gated = silu * up_accum;

    float down_accum = 0.0;

    for (uint ffn_tile = 0; ffn_tile < num_ffn_tiles; ++ffn_tile) {
        const uint ffn_base = ffn_tile * TILE_SIZE;

        float x_ffn = 0.0;
        const uint down_idx_base = ffn_base * params.hidden_dim + tx;

        if (ffn_base + tx < params.ffn_dim && tx < params.hidden_dim) {
            x_ffn = gated * down[down_idx_base];
        }

        down_accum += subgroupAdd(x_ffn);
    }

    const uint out_idx = batch_base + tx;
    if (tx < params.hidden_dim) {
        const float residual = input_data[out_idx];
        output_data[out_idx] = down_accum + residual;
    }
}
