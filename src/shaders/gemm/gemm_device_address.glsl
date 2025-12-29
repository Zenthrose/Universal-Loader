#version 460
#extension GL_KHR_buffer_device_address : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256, local_size_y = 1) in;

layout(buffer_reference, std430) readonly buffer Tensor {
    float data[];
};

layout(push_constant) uniform Params {
    uint M;
    uint N;
    uint K;
    uint padding;
} params;

layout(buffer_reference, std430) readonly buffer MatmulParams {
    float a_ptr;
    float b_ptr;
    float c_ptr;
};

shared float s_tile[256];

void main() {
    uint global_id = gl_GlobalInvocationID.x;
    
    if (global_id >= params.M * params.N) return;
    
    uint row = global_id / params.N;
    uint col = global_id % params.N;
    
    float a_ptr = MatmulParams.a_ptr + row * params.K * 4;
    float b_ptr = MatmulParams.b_ptr + col * params.K * 4;
    
    float sum = 0.0;
    
    for (uint k = 0; k < params.K; ++k) {
        float a_val = *reinterpret_cast<float*>(a_ptr + k * 4);
        float b_val = *reinterpret_cast<float*>(b_ptr + k * 4);
        sum += a_val * b_val;
    }
    
    float c_ptr = MatmulParams.c_ptr + global_id * 4;
    *reinterpret_cast<float*>(c_ptr) = sum;
}
