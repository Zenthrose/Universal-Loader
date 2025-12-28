#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer AttnWeights { float attn[]; };
layout(binding = 1) readonly buffer V { float v[]; };
layout(binding = 2) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint seq_len;
    uint head_dim;
    uint num_heads;
    uint padding;
} params;

void main() {
    uint t = gl_GlobalInvocationID.x; // Token index
    
    if (t >= params.seq_len) return;
    
    float sum = 0.0;
    
    for (uint h = 0; h < params.num_heads; ++h) {
        float weight = attn[t * params.num_heads + h];
        
        for (uint d = 0; d < params.head_dim; ++d) {
            uint v_offset = t * params.num_heads * params.head_dim + h * params.head_dim + d;
            sum += weight * v[v_offset];
        }
    }
    
    out[t] = sum;
}
