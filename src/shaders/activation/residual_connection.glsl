#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer Residual { float residual[]; };
layout(binding = 2) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint N;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    out[i] = x[i] + Residual[i];
}
