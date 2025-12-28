#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float y[]; };

shared float s_max[256];

void main() {
    uint local_id = gl_LocalInvocationID.x;
    
    float x_val = x[gl_GlobalInvocationID.x];
    s_max[local_id] = x_val;
    
    memoryBarrierShared();
    barrier();
    
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (local_id < stride) {
            s_max[local_id] = max(s_max[local_id], s_max[local_id + stride]);
        }
        memoryBarrierShared();
        barrier();
    }
    
    float max_val = s_max[0];
    
    s_max[local_id] = exp(x_val - max_val);
    
    memoryBarrierShared();
    barrier();
    
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (local_id < stride) {
            s_max[local_id] += s_max[local_id + stride];
        }
        memoryBarrierShared();
        barrier();
    }
    
    float sum = s_max[0];
    
    y[gl_GlobalInvocationID.x] = exp(x_val - max_val) / sum;
}
