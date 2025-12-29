#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 128, local_size_y = 1) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer Q { float q[]; };
layout(binding = 2) readonly buffer K { float k[]; };
layout(binding = 3) readonly buffer V { float v[]; };
layout(binding = 4) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint seq_len;
    uint head_dim;
    uint num_heads;
    uint num_kv_heads;
} params;

shared float s_qk[128];

void main() {
    uint h = gl_GlobalInvocationID.y;
    uint t = gl_GlobalInvocationID.x;
    
    if (t >= params.seq_len || h >= params.num_heads) return;
    
    uint head_idx = h % params.num_kv_heads;
    uint head_offset = head_idx * params.head_dim;
    
    uint q_offset = t * params.head_dim + head_offset;
    uint k_offset = t * params.seq_len * params.head_dim + head_offset;
    uint v_offset = t * params.seq_len * params.head_dim + head_offset;
    
    float sum = 0.0;
    
    for (uint j = 0; j < params.head_dim; ++j) {
        float q_val = q[q_offset + j];
        float k_val = k[k_offset + j];
        sum += q_val * k_val;
    }
    
    float scale = 1.0 / sqrt(float(params.head_dim));
    float attn_weight = sum * scale;
    
    out[t * params.num_heads + h] = attn_weight;
}
