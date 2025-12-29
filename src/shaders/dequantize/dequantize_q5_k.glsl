#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { uint8_t qk[]; };
layout(binding = 1) readonly buffer Scale { float scales[]; };
layout(binding = 2) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint N;
    uint block_size;
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    uint32_t block = i / params.block_size;
    uint32_t idx_in_block = i % params.block_size;
    
    uint8_t q = Input[block * (params.block_size / 2) + idx_in_block];
    uint8_t k = Input[block * (params.block_size / 2) + (params.block_size / 2) + idx_in_block];
    
    float scale = Scale[block / 2];
    
    uint8_t qh = q >> 4;
    uint8_t ql = q & 0xF;
    uint8_t kh = k >> 4;
    uint8_t kl = k & 0xF;
    
    float d0 = float(ql);
    float d1 = float(qh);
    float d2 = float(kl);
    float d3 = float(kh);
    
    out[i] = scale * ((d0 * d2) + (d1 * d3) * 16.0);
}
