#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer QK_Packed { uint32_t qk_packed[]; };
layout(binding = 1) readonly buffer Scales { float scales[]; };
layout(binding = 2) readonly buffer Mins { float mins[]; };
layout(binding = 3) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint N;
    uint block_size;
    uint padding[2];
} params;

const uint K_Q4 = 32;
const uint K_Q4_K = 256;

uint16_t get_bitfield(uint32_t word, uint bit_offset, uint nbits) {
    return uint16_t((word >> bit_offset) & ((1u << nbits) - 1));
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    
    if (i >= params.N) return;
    
    uint block = i / params.block_size;
    uint idx_in_block = i % params.block_size;
    
    uint16_t qh;
    uint16_t ql;
    uint16_t qh2;
    
    if (params.block_size == K_Q4_K) {
        uint packed_idx = (block * 128) + (idx_in_block / 2);
        if (packed_idx < qk_packed.length()) {
            uint32_t packed = qk_packed[packed_idx];
            
            if (idx_in_block % 2 == 0) {
                ql = uint16_t(packed & 0xFFFF);
                qh = uint16_t((packed >> 16) & 0xFFFF);
            } else {
                ql = uint16_t((packed >> 16) & 0xFFFF);
                qh = uint16_t((packed >> 32) & 0xFFFF);
            }
            
            packed_idx += 16;
            if (packed_idx < qk_packed.length()) {
                uint32_t packed2 = qk_packed[packed_idx];
                uint idx2 = idx_in_block / 2;
                qh2 = uint16_t((packed2 >> (idx2 % 2 == 0 ? 0 : 16)) & 0xFFFF);
            }
        }
    } else {
        uint packed_idx = (block * (params.block_size / 2)) + (idx_in_block / 2);
        if (packed_idx < qk_packed.length()) {
            uint32_t packed = qk_packed[packed_idx];
            ql = uint16_t((idx_in_block % 2 == 0) ? (packed & 0xFFFF) : ((packed >> 16) & 0xFFFF));
            qh = uint16_t(0);
            qh2 = uint16_t(0);
        }
    }
    
    float scale;
    float min_val;
    
    uint scale_idx = (block / 2);
    uint group_idx = block % 2;
    
    if (scale_idx < scales.length()) {
        scale = scales[scale_idx];
    } else {
        scale = 1.0;
    }
    
    if (scale_idx < mins.length()) {
        min_val = mins[scale_idx];
    } else {
        min_val = 0.0;
    }
    
    if (idx_in_block < 8) {
        uint8_t q = uint8_t((idx_in_block % 2 == 0) ? (ql & 0x0F) : ((ql >> 4) & 0x0F));
        uint8_t qh_bit = uint8_t((idx_in_block < 4) ? ((qh >> (idx_in_block * 2)) & 0x3) : ((qh >> (8 + (idx_in_block - 4) * 2)) & 0x3));
        uint16_t combined = uint16_t(q) | (uint16_t(qh_bit) << 4);
        out[i] = (float(combined) - 8.0) * scale + min_val;
    } else if (idx_in_block < 16) {
        uint idx2 = idx_in_block - 8;
        uint8_t q = uint8_t((idx2 % 2 == 0) ? (ql >> 8) & 0x0F : ((ql >> 12) & 0x0F));
        uint8_t qh_bit = uint8_t((idx2 < 4) ? ((qh >> (16 + idx2 * 2)) & 0x3) : ((qh >> (24 + (idx2 - 4) * 2)) & 0x3));
        uint16_t combined = uint16_t(q) | (uint16_t(qh_bit) << 4);
        out[i] = (float(combined) - 8.0) * scale + min_val;
    } else {
        uint idx2 = idx_in_block - 16;
        uint8_t q = uint8_t((idx2 % 2 == 0) ? (qh2 & 0x0F) : ((qh2 >> 4) & 0x0F));
        out[i] = (float(q) - 8.0) * scale + min_val;
    }
}
