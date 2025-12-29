#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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
};

shared float gate_shared[64];
shared float up_shared[64];

void rms_norm(uint idx, float eps) {
    float sum_sq = 0.0f;
    for (uint i = 0; i < hidden_dim; i++) {
        sum_sq += input_data[idx * hidden_dim + i] * input_data[idx * hidden_dim + i];
    }
    float rms = rsqrt(sum_sq / float(hidden_dim) + eps);

    for (uint i = 0; i < hidden_dim; i++) {
        input_data[idx * hidden_dim + i] *= norm_scale[i] * rms;
    }
}

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint row_idx = gl_LocalInvocationID.x;

    rms_norm(batch_idx, epsilon);

    float gate_val = 0.0f;
    float up_val = 0.0f;

    for (uint i = 0; i < hidden_dim; i++) {
        gate_val += input_data[batch_idx * hidden_dim + i] * gate[i * hidden_dim + row_idx];
        up_val += input_data[batch_idx * hidden_dim + i] * up[i * hidden_dim + row_idx];
    }

    gate_shared[row_idx] = gate_val;
    up_shared[row_idx] = up_val;

    barrier();

    float silu = gate_shared[row_idx] / (1.0f + exp(-gate_shared[row_idx]));
    float gated = silu * up_shared[row_idx];

    float output = 0.0f;
    for (uint i = 0; i < ffn_dim; i++) {
        output += gated * down[i * hidden_dim + row_idx];
    }

    if (row_idx < hidden_dim) {
        output_data[batch_idx * hidden_dim + row_idx] = output + input_data[batch_idx * hidden_dim + row_idx];
    }
}
