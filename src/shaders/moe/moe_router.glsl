#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint n_elements;
    uint num_experts;
    uint top_k;
    uint padding;
} params;

shared float s_max[256];

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.n_elements) return;
    
    float x_i = x[i];
    s_max[gl_LocalInvocationID.x] = x_i;
    
    memoryBarrierShared();
    barrier();
    
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (gl_LocalInvocationID.x < stride) {
            s_max[gl_LocalInvocationID.x] = max(s_max[gl_LocalInvocationID.x], s_max[gl_LocalInvocationID.x + stride]);
        }
        memoryBarrierShared();
        barrier();
    }
    
    float max_val = s_max[0];
    
    float score = x_i - max_val;
    out[i] = score;
}
