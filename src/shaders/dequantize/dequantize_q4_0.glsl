#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { uint16_t q4[]; };
layout(binding = 1) readonly buffer Scales { float d[]; };
layout(binding = 2) writeonly buffer Output { float out[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint block_id = i / 32;
    uint block_offset = i % 32;
    
    uint16_t packed = q4[block_id * 16 + block_offset / 2];
    uint8_t q4_val = (block_offset % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
    
    float d_val = d[block_id];
    out[i] = (float(q4_val) - 8.0) * d_val;
}
