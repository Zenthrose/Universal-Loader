#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer Input { float input_data[]; };
layout(binding = 1) readonly buffer QProj { float q_proj[]; };
layout(binding = 2) readonly buffer KProj { float k_proj[]; };
layout(binding = 3) readonly buffer VProj { float v_proj[]; };
layout(binding = 4) readonly buffer NormScale { float norm_scale[]; };
layout(binding = 5) buffer Output { float output_data[]; };

layout(binding = 6) readonly buffer KVCacheK { float kv_cache_k[]; };
layout(binding = 7) readonly buffer KVCacheV { float kv_cache_v[]; };

layout(push_constant) uniform Params {
    uint hidden_dim;
    uint head_dim;
    uint num_heads;
    uint seq_len;
    uint position;
    float epsilon;
};

shared float q_shared[64];
shared float k_shared[64];
shared float v_shared[64];

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

float attention_score(uint head, uint pos_q, uint pos_k) {
    uint q_offset = pos_q * hidden_dim + head * head_dim;
    uint k_offset = pos_k * hidden_dim + head * head_dim;

    float score = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        score += q_shared[gl_LocalInvocationID.x % head_dim] * kv_cache_k[k_offset + i];
    }
    return score * rsqrt(float(head_dim));
}

void main() {
    uint batch_idx = gl_WorkGroupID.y;
    uint head_idx = gl_WorkGroupID.x / num_heads;
    uint row_idx = gl_LocalInvocationID.x;

    rms_norm(batch_idx, epsilon);

    if (row_idx < hidden_dim) {
        q_shared[row_idx] = input_data[batch_idx * hidden_dim + row_idx];
    }
    barrier();

    float attn_output = 0.0f;
    float sum_exp = 0.0f;
    float max_score = -10000.0f;

    for (uint pos = 0; pos < seq_len; pos++) {
        float score = attention_score(head_idx, batch_idx, pos);
        max_score = max(max_score, score);
    }

    for (uint pos = 0; pos < seq_len; pos++) {
        float score = attention_score(head_idx, batch_idx, pos);
        float exp_score = exp(score - max_score);
        sum_exp += exp_score;

        uint v_offset = pos * hidden_dim + head_idx * head_dim;
        float v_val = kv_cache_v[v_offset + row_idx % head_dim];
        attn_output += (exp_score / sum_exp) * v_val;
    }

    output_data[batch_idx * hidden_dim + row_idx] = attn_output;

    barrier();

    float output = 0.0f;
    if (row_idx < hidden_dim) {
        for (uint i = 0; i < head_dim; i++) {
            output += q_shared[i] * v_proj[head_idx * hidden_dim * hidden_dim + i * hidden_dim + row_idx];
        }
    }

    if (row_idx < hidden_dim) {
        output_data[batch_idx * hidden_dim + row_idx] = output + input_data[batch_idx * hidden_dim + row_idx];
    }
}
