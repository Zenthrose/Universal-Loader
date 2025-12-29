#version 460
#extension GL_KHR_cooperative_matrix : require

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

cooperative_matrixKHR matA;
cooperative_matrixKHR matB;
cooperative_matrixKHR matC;

void main() {
    uint bx = gl_WorkGroupID.x;
    uint by = gl_WorkGroupID.y;
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;

    uint row = by * 16 + ty;
    uint col = bx * 16 + tx;

    uint M_tile = (params.M + 15) / 16;
    uint N_tile = (params.N + 15) / 16;
    uint K_tile = (params.K + 15) / 16;

    uint subM = (M_tile + 1) / 2;
    uint subK = (K_tile + 1) / 2;
    uint subN = (N_tile + 1) / 2;

    coopMatLoadA(matA, 0, row, subM, subK);
    coopMatLoadB(matB, 0, col, subK, subN);
    coopMatLoadC(matC, 0, row, subM, subN);

    coopMatMulAdd(matA, matB, matC);

    coopMatStoreC(matC, 0, row, col, subM, subN);
}
