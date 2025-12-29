#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) readonly buffer A { float a[]; };
layout(binding = 1) readonly buffer B { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint padding;
} params;

shared float s_a[16][16];
shared float s_b[16][16];

void main() {
    uint bx = gl_WorkGroupID.x;
    uint by = gl_WorkGroupID.y;
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;

    uint row = by * 16 + ty;
    uint col = bx * 16 + tx;

    float sum = 0.0;

    for (uint k = 0; k < params.K; k += 16) {
        if (row < params.M && k + tx < params.K) {
            s_a[ty][tx] = a[row * params.K + k + tx];
        } else {
            s_a[ty][tx] = 0.0;
        }

        if (k + ty < params.K && col < params.N) {
            s_b[ty][tx] = b[(k + ty) * params.N + col];
        } else {
            s_b[ty][tx] = 0.0;
        }

        memoryBarrierShared();
        barrier();

        for (uint i = 0; i < 16; ++i) {
            sum += s_a[ty][i] * s_b[i][tx];
        }

        memoryBarrierShared();
        barrier();
    }

    if (row < params.M && col < params.N) {
        c[row * params.N + col] = sum;
    }
}
