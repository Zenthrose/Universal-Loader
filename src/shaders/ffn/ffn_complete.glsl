#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer GateW { float gate[]; };
layout(binding = 2) readonly buffer UpW { float up[]; };
layout(binding = 3) readonly buffer DownW { float down[]; };
layout(binding = 4) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint N;
    uint intermediate_dim;
} params;

shared float s_gate_up[256];

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float gate = 0.0;
    float up = 0.0;
    
    for (uint j = 0; j < params.intermediate_dim; ++j) {
        gate += x[i] * GateW[i * params.intermediate_dim + j];
        up += x[i] * UpW[i * params.intermediate_dim + j];
    }
    
    float silu = gate / (1.0 + exp(-gate));
    
    float sum = 0.0;
    for (uint j = 0; j < params.intermediate_dim; ++j) {
        sum += silu * up * DownW[j * params.N + i];
    }
    
    out[i] = sum;
}
