#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer ExpertWeights { float weights[]; };
layout(binding = 2) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint n_elements;
    uint expert_id;
    uint in_features;
    uint out_features;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.n_elements) return;
    
    float x_i = x[i];
    float sum = 0.0;
    
    uint weight_offset = params.expert_id * params.in_features * params.out_features;
    
    for (uint j = 0; j < params.out_features; ++j) {
        float dot = 0.0;
        for (uint k = 0; k < params.in_features; ++k) {
            dot += x_i * weights[weight_offset + k * params.out_features + j];
        }
        sum += dot;
    }
    
    out[i] = sum;
}
