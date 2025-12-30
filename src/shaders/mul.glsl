#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer A { float a[]; };
layout(binding = 1) readonly buffer B { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };

layout(binding = 3) readonly buffer Params {
    uint N;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.N) return;
    
    c[i] = a[i] * b[i];
}
