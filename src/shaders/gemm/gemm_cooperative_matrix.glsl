#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroups : require

layout(local_size_x = 32, local_size_y = 1) in;

layout(binding = 0) readonly buffer A { float a[]; };
layout(binding = 1) readonly buffer B { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint K_tiles;
    float alpha;
    float beta;
    uint mat_scope;
    uint mat_layout;
} params;

layout(constant_id = 0) const uint TILE_M = 16;
layout(constant_id = 1) const uint TILE_N = 16;
layout(constant_id = 2) const uint TILE_K = 16;

const uint ScopeSubgroup = 1;
const uint ScopeWorkgroup = 2;
const uint ScopeDevice = 3;

const uint LayoutRowMajor = 0;
const uint LayoutColMajor = 1;

cooperative_matrixKHR<float, ScopeSubgroup, UseA, TILE_M, TILE_K> matA;
cooperative_matrixKHR<float, ScopeSubgroup, UseB, TILE_K, TILE_N> matB;
cooperative_matrixKHR<float, ScopeSubgroup, UseC, TILE_M, TILE_N> matC;

void main() {
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;
    const uint tx = gl_LocalInvocationID.x;

    const uint row = by * TILE_M;
    const uint col = bx * TILE_N;

    cooperativeMatrixLoadKHR(matC, c, row * params.N + col, params.N, LayoutColMajor);

    for (uint k_tile = 0; k_tile < params.K_tiles; ++k_tile) {
        const uint k_base = k_tile * TILE_K;

        cooperativeMatrixLoadKHR(matA, a, row * params.K + k_base, params.K, LayoutRowMajor);
        cooperativeMatrixLoadKHR(matB, b, k_base * params.N + col, params.N, LayoutRowMajor);

        cooperativeMatrixMulAddKHR(matA, matB, matC, matC);
    }

    cooperativeMatrixStoreKHR(matC, c, row * params.N + col, params.N, LayoutColMajor);
}
