#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer Q { float q_data[]; };
layout(binding = 1) readonly buffer K { float k_data[]; };
layout(binding = 2) readonly buffer V { float v_data[]; };
layout(binding = 3) buffer Output { float output_data[]; };
layout(binding = 4) uniform Params {
    uint seq_len;
    uint head_dim;
    uint num_heads;
    float scale;
};

shared float q_shared[64];
shared float k_shared[64];
shared float v_shared[64][64];

const float SOFTMAX_MAX = 100.0;

void main() {
    uint head_idx = gl_WorkGroupID.x;
    uint row_idx = gl_GlobalInvocationID.x;
    uint local_idx = gl_LocalInvocationID.x;

    uint q_offset = head_idx * seq_len * head_dim + row_idx * head_dim;
    uint output_offset = head_idx * seq_len * head_dim + row_idx * head_dim;

    float sum_exp = 0.0;
    float max_val = -SOFTMAX_MAX;
    float output[64] = float[](0.0);

    for (uint j = 0; j < seq_len; j++) {
        if (local_idx < head_dim) {
            q_shared[local_idx] = q_data[q_offset + local_idx];
            k_shared[local_idx] = k_data[j * head_dim + local_idx];
        }

        barrier();

        if (local_idx == 0) {
            float s = 0.0;
            for (uint k = 0; k < head_dim; k++) {
                s += q_shared[k] * k_shared[k];
            }
            s *= scale;

            max_val = max(max_val, s);
        }

        barrier();

        float s;
        if (local_idx == 0) {
            s = scale;
            for (uint k = 0; k < head_dim; k++) {
                s *= exp(q_shared[k] * k_shared[k] - max_val);
            }
        }

        barrier();

        float exp_s;
        if (local_idx == 0) {
            exp_s = s;
        } else {
            exp_s = 0.0;
        }

        exp_s = subgroupAdd(exp_s);
        sum_exp += exp_s;

        barrier();

        if (local_idx < head_dim) {
            v_shared[local_idx][j] = v_data[j * head_dim + local_idx];
        }

        barrier();

        for (uint k = 0; k < head_dim; k++) {
            output[k] += (s / sum_exp) * v_shared[k][row_idx % seq_len];
        }
    }

    if (row_idx < seq_len && local_idx < head_dim) {
        output_data[output_offset + local_idx] = output[local_idx];
    }
}
