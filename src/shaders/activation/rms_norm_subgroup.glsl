#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer Weight { float w[]; };
layout(binding = 2) writeonly buffer Output { float y[]; };

layout(binding = 3) readonly buffer Params {
    uint N;
    uint padding1;
    uint padding2;
    uint padding3;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float x_i = x[i];
    float w_i = w[i];
    float prod = x_i * w_i;
    
    float sum = subgroupAdd(prod);
    sum = subgroupExclusiveScanAdd(prod) + prod;
    
    float rms = inversesqrt(sum / float(params.N));
    
    y[i] = x_i * rms * w_i;
}
