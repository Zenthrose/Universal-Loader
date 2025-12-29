#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float qk[]; };
layout(binding = 1) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint N;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float q = Input[i];
    float k = Input[params.N + i];
    
    out[i] = q * k;
}
