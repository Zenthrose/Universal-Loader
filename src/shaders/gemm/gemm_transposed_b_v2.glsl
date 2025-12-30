#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) readonly buffer A { float a[]; };
layout(binding = 1) readonly buffer B { float b[]; }; // Dequantized weights [Out, In] (Transposed logic)
layout(binding = 2) writeonly buffer C { float c[]; };

layout(binding = 3) readonly buffer Params {
    uint M;
    uint N;
    uint K;
} params;

void main() {
    uint m = gl_GlobalInvocationID.x;
    uint n = gl_GlobalInvocationID.y;
    
    if (m >= params.M || n >= params.N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < params.K; k++) {
        // A is [M, K] -> a[m * K + k]
        // B is [N, K] (Transposed relative to standard GEMM) -> b[n * K + k]
        sum += a[m * params.K + k] * b[n * params.K + k];
    }
    
    c[m * params.N + n] = sum;
}
