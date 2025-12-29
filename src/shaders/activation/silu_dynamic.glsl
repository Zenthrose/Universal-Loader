#version 460

layout(local_size_x_id = 0) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float y[]; };

layout(push_constant) uniform Params {
    uint n_elements;
    uint padding0;
    uint padding1;
    uint padding2;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;

    if (i >= params.n_elements) return;

    float x_i = x[i];
    float silu = x_i / (1.0 + exp(-x_i));

    y[i] = silu;
}
