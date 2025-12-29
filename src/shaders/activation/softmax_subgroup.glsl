#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

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
    
    float x_val = x[i];
    
    float max_val = subgroupMax(x_val);
    
    if (subgroupElect()) {
        max_val = max(x_val, subgroupShuffleUp(x_val, 1));
        max_val = max(max_val, subgroupShuffleUp(max_val, 2));
        max_val = max(max_val, subgroupShuffleUp(max_val, 4));
        max_val = max(max_val, subgroupShuffleUp(max_val, 8));
        max_val = max(max_val, subgroupShuffleUp(max_val, 16));
        max_val = max(max_val, subgroupShuffleUp(max_val, 32));
        max_val = max(max_val, subgroupShuffleUp(max_val, 64));
        max_val = max(max_val, subgroupShuffleUp(max_val, 128));
    }
    
    max_val = subgroupBroadcastFirst(max_val);
    
    float exp_x = exp(x_val - max_val);
    float sum_exp = subgroupAdd(exp_x);
    
    float prob = exp_x / sum_exp;
    
    y[i] = prob;
}
