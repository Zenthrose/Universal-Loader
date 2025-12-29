#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) readonly buffer Residual { float residual[]; };
layout(binding = 2) readonly buffer Norm { float norm_w[]; };
layout(binding = 3) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint N;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    float x = x[i] + Residual[i];
    
    float sum = 0.0;
    for (uint j = 0; j < params.N; ++j) {
        sum += x * Norm[j];
    }
    
    float mean = sum / float(params.N);
    float var = 0.0;
    for (uint j = 0; j < params.N; ++j) {
        float diff = x - mean;
        var += diff * diff;
    }
    
    float epsilon = 1e-6;
    float norm = x / sqrt(var + epsilon);
    
    out[i] = norm;
}
