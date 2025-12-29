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
    
    uint16_t q = Input[(block * (params.block_size / 2)) + idx_in_block];
    uint16_t k = Input[(block * (params.block_size / 2)) + (params.block_size / 2) + idx_in_block];
    
    float scale = Scale[block / 2];
    
    float d0 = float((q >> 0) & 0xF);
    float d1 = float((q >> 4) & 0xF);
    float d2 = float((k >> 0) & 0xF);
    float d3 = float((k >> 4) & 0xF);
    float d4 = float((q >> 8) & 0xF);
    float d5 = float((k >> 12) & 0xF);
    float d6 = float((k >> 8) & 0xF);
    float d7 = float((k >> 12) & 0xF);
    
    float min_q = min(d0, d1) - 8.0;
    float max_q = max(d0, d1) - 8.0;
    
    float min_k = min(d2, d3) - 8.0;
    float max_k = max(d2, d3) - 8.0;
    
    float a = min_q + min_k;
    float b = max_q + max_k;
    float c = min(d4, d5) - 8.0;
    float d = max(d4, d5) - 8.0;
    float e = min(d6, d7) - 8.0;
    float f = max(d6, d7) - 8.0;
    
    float min_cd = min(c, e);
    float max_cd = max(c, d);
    
    out[i] = scale * ((a & 0xF) - ((min_cd + max_cd) << 4) - 0.5 * (b & 0xF) - 8.0);
}
