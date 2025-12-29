#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer Weight { float w[]; };
layout(binding = 2) writeonly buffer Output { float y[]; };

layout(push_constant) uniform Params {
    uint N;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float x_i = x[i];
    float w_i = w[i];
    
    y[i] = x_i * w_i;
}
