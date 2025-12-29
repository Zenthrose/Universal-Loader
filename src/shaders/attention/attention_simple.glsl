#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Q { float q[]; };
layout(binding = 1) readonly buffer K { float k[]; };
layout(binding = 2) readonly buffer V { float v[]; };
layout(binding = 3) readonly buffer KVCache_K { float kv_k[]; };
layout(binding = 4) readonly buffer KVCache_V { float kv_v[]; };
layout(binding = 5) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint seq_len;
    uint head_dim;
    uint num_heads;
    uint pos;
} params;

shared float s_k[32];

void main() {
    uint h = gl_GlobalInvocationID.y;
    uint t = gl_GlobalInvocationID.x;
    
    if (h >= params.num_heads) return;
    
    uint head_offset = h * params.head_dim;
    uint q_offset = t * params.seq_len * params.head_dim + head_offset;
    uint k_offset = t * params.seq_len * params.head_dim + head_offset;
    uint v_offset = t * params.seq_len * params.head_dim + head_offset;
    
    if (gl_LocalInvocationID.x == 0) {
        for (uint j = 0; j < params.head_dim; ++j) {
            uint idx = j + head_offset;
            s_k[j] = k[k_offset + idx];
        }
        memoryBarrierShared();
        barrier();
    }
    
    float sum = 0.0;
    
    for (uint j = 0; j < params.head_dim; ++j) {
        sum += q[q_offset + j] * s_k[j];
    }
    
    float scale = 1.0 / sqrt(float(params.head_dim));
    
    out[q_offset + h] = sum * scale;
}
