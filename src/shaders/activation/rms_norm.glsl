#version 460
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(constant_id = 0) const uint TILE_SIZE = 256;
layout(constant_id = 1) const uint HEAD_DIM = 128;

layout(local_size_x = TILE_SIZE, local_size_y = 1) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer Weight { float w[]; };
layout(binding = 2) buffer Output { float y[]; };

layout(push_constant) uniform Params {
    uint hidden_dim;
    uint num_tokens;
    float epsilon;
    uint padding[2];
};

shared float s_x[TILE_SIZE];

void main() {
    const uint tx = gl_LocalInvocationID.x;
    const uint token_idx = gl_WorkGroupID.x;

    if (token_idx >= params.num_tokens) return;

    const uint hidden_base = token_idx * params.hidden_dim;
    const uint num_tiles = (params.hidden_dim + TILE_SIZE - 1) / TILE_SIZE;

    float sum_sq = 0.0;

    for (uint tile = 0; tile < num_tiles; ++tile) {
        const uint hidden_idx = tile * TILE_SIZE + tx;
        const uint idx = hidden_base + hidden_idx;

        if (hidden_idx < params.hidden_dim) {
            const float x_val = x[idx];
            s_x[tx] = x_val * x_val;
        } else {
            s_x[tx] = 0.0;
        }

        memoryBarrierShared();
        barrier();

        if (hidden_idx < params.hidden_dim) {
            sum_sq += s_x[tx];
        }

        memoryBarrierShared();
        barrier();
    }

    const float total_sum_sq = subgroupAdd(sum_sq);
    const float rms = rsqrt(total_sum_sq / float(params.hidden_dim) + params.epsilon);

    for (uint tile = 0; tile < num_tiles; ++tile) {
        const uint hidden_idx = tile * TILE_SIZE + tx;
        const uint idx = hidden_base + hidden_idx;

        if (hidden_idx < params.hidden_dim) {
            y[idx] = x[idx] * w[hidden_idx] * rms;
        }
    }
}
