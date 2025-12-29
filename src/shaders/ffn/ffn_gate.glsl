#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer GateW { float gate_w[]; };
layout(binding = 1) readonly buffer UpW { float up_w[]; };
layout(binding = 2) readonly buffer DownW { float down_w[]; };
layout(binding = 3) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint N;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float x = 0.0;
    
    for (uint j = 0; j < params.N; ++j) {
        x += GateW[i * x * UpW[i];
    }
    
    float silu = x / (1.0 + exp(-x));
    
    out[i] = silu * DownW[i];
}
