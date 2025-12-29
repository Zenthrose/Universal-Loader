#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer KVCache { float kv[]; };
layout(binding = 2) writeonly buffer Output { float y[]; };

layout(push_constant) uniform Params {
    uint N;
    uint seq_len;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float x_i = x[i];
    
    float sum = 0.0;
    float sum_sq = 0.0;
    
    for (uint j = 0; j < params.seq_len; ++j) {
        float kv = kv[j * params.N + i];
        sum += kv;
        sum_sq += kv * kv;
    }
    
    float mean = sum / float(params.seq_len);
    float var = (sum_sq / float(params.seq_len)) - (mean * mean);
    float epsilon = 1e-6;
    
    y[i] = x_i * (1.0 / sqrt(var + epsilon));
}
