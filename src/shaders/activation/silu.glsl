#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float y[]; };

layout(binding = 2) readonly buffer Params {
    uint N;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.N) return;
    
    float x_i = x[i];
    float silu = x_i / (1.0 + exp(-x_i));
    
    y[i] = silu;
}
