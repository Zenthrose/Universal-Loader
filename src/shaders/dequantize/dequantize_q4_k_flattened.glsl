#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(local_size_x = 256) in;

// Bindings
// 0: Packed Quants (from GGUF Q4_K blocks) - treating 128 bytes as 32 uints
layout(binding = 0) readonly buffer Quants { uint quants[]; };

// 1: Expanded Scales (8 per superblock) - computed on CPU
layout(binding = 1) readonly buffer Scales { float scales[]; };

// 2: Expanded Mins (8 per superblock) - computed on CPU
layout(binding = 2) readonly buffer Mins { float mins[]; };

// 3: Output (FP32)
layout(binding = 3) writeonly buffer Output { float out_opt[]; };

layout(binding = 4) readonly buffer Params {
    uint N;
} params;

// Constants for Q4_K
const uint SUPERBLOCK_SIZE = 256;
const uint SUBBLOCK_SIZE = 32;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.N) return;

    // Superblock index
    uint sb_idx = i / SUPERBLOCK_SIZE;
    
    // Index within superblock (0-255)
    uint within_sb = i % SUPERBLOCK_SIZE;
    
    // Subblock index (0-7)
    uint sub_idx = within_sb / SUBBLOCK_SIZE;
    
    // Index within subblock (0-31)
    uint within_sub = within_sb % SUBBLOCK_SIZE;

    // 1. Get Scale and Min (pre-computed on CPU)
    // There are 8 scales/mins per superblock (one per subblock)
    uint param_idx = sb_idx * 8 + sub_idx;
    float scale = scales[param_idx];
    float min_val = mins[param_idx];

    // 2. Extract Quant
    // Quants are stored densely in the last 128 bytes of the block.
    // In our `quants` array (uint32), each superblock takes 32 uints (128 bytes).
    // Accessing `quants` buffer: 
    // Format on GPU: [SB0_Quants (32 uints)] [SB1_Quants (32 uints)] ...
    // Note: This assumes we STRIPPED the headers on CPU and only uploaded the 128-byte bodies packed together!
    // Or we upload full blocks and calculate offset?
    // Stripping is cleaner. CPU creates a dedicated dense quants buffer.
    
    uint q_word_idx = (sb_idx * 32) + (within_sb / 8); 
    // Each uint32 contains 8 quants (4 bits each).
    // within_sb / 8 gives the uint index 0..31 inside the block.
    
    uint32_t word = quants[q_word_idx];
    
    // Shift: (item % 8) * 4
    uint shift = (within_sb % 8) * 4;
    uint8_t q = uint8_t((word >> shift) & 0xF);

    // 3. Dequantize
    out_opt[i] = float(q) * scale + min_val;
}
