#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) readonly buffer W8 { int8_t w8_data[]; };
layout(binding = 1) readonly buffer A8 { int8_t a8_data[]; };
layout(binding = 2) readonly buffer ScaleW { float scale_w[]; };
layout(binding = 3) readonly buffer ScaleA { float scale_a[]; };
layout(binding = 4) buffer Output { float output_data[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
};

shared int8_t w_shared[64][8];
shared int8_t a_shared[64][8];

void main() {
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint row = gl_WorkGroupID.y * 64 + ty;
    uint col = gl_WorkGroupID.x * 64 + tx;

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;

    for (uint k = 0; k < K; k += 8) {
        if (ty < 8 && tx < 8) {
            w_shared[tx][ty] = w8_data[(k + ty) * N + col];
            a_shared[tx][ty] = a8_data[row * K + k + tx];
        }

        barrier();

        for (uint i = 0; i < 8; ++i) {
            int32_t w_val = int32_t(w_shared[ty][i]);
            int32_t a_val = int32_t(a_shared[i][tx]);
            acc += float(w_val * a_val);
        }

        barrier();
    }

    float scale = scale_w[col] * scale_a[row];
    output_data[row * N + col] = acc * scale;
}
