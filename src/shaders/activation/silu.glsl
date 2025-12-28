#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float y[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    float x_i = x[i];
    float silu = x_i / (1.0 + exp(-x_i));
    
    y[i] = silu;
}
