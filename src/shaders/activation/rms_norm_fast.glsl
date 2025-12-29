#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float y[]; };

layout(push_constant) uniform Params {
    uint N;
    uint padding1;
    uint padding2;
    uint padding3;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float x_i = x[i];
    
    float sum_sq = x_i * x_i;
    sum_sq = subgroupAdd(sum_sq);
    
    float mean_sq = subgroupBroadcastFirst(sum_sq) / float(subgroupBarrier());
    
    float rms = inversesqrt(mean_sq + 0.000001f);
    
    y[i] = x_i * rms;
}
