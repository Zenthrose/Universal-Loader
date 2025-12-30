#version 460

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float y[]; };

layout(binding = 2) readonly buffer Params {
    uint N;           // Total elements (heads * head_dim * seq_len)
    uint head_dim;    // Dimension of each head
    uint pos_offset;  // Starting position index
    uint seq_len;     // Number of tokens in this batch
} params;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= params.N) return;
    
    // Calculate logical indices
    // i = token_idx * (num_heads * head_dim) + head_idx * head_dim + dim_idx
    // Simplified: We just need 'dim_idx' and 'token_idx'.
    // Assuming packed layout: [Batch, Seq, Head, Dim] or [Batch, Head, Seq, Dim]?
    // LLaMA usually [Batch, Seq, Head, Dim] or similar.
    // Let's assume contiguous heads per token.
    // Dimensions: i % head_dim = d.
    // Token index: i / (total_dim_per_token).
    // Start simple: i corresponds to a specific float.
    
    // RoPE rotates PAIRS of floats.
    // We process pairs.
    uint head_dim = params.head_dim;
    uint d = i % head_dim; // Dimension index within head
    
    // RoPE is usually applied to first `rotary_dim` elements. Assuming full head_dim for now.
    // We need pairs (2k, 2k+1).
    // If we launch threads for each element, we need synchronization or careful access.
    // Easier: Launch threads for PAIRS? Or just handle both in one thread?
    // Let's process pairs. 
    // Thread i handles element i?
    // Need to peer with i+1 or i-1.
    // If d is even: peer is i+1.
    // If d is odd: peer is i-1.
    
    // Calculate rotation angle theta.
    // theta = pos * freq_base ^ (-2 * (d/2) / head_dim)
    
    // Need 'pos'.
    // Assuming x is flat [Seq, Heads, HeadDim].
    // Elements per token = (N / seq_len).
    uint elements_per_token = params.N / params.seq_len;
    uint token_idx = i / elements_per_token;
    uint pos = params.pos_offset + token_idx;
    
    float theta_base = 10000.0;
    float theta_exp = -float(2 * (d / 2)) / float(head_dim);
    float theta = float(pos) * pow(theta_base, theta_exp);
    
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    
    float val = x[i];
    
    // Check if d is even or odd
    if ((d % 2) == 0) {
        // Even: This is x0. Needs y0 = x0*cos - x1*sin
        // Read x1 (at i+1)
        float val_pair = x[i + 1];
        y[i] = val * cos_theta - val_pair * sin_theta;
    } else {
        // Odd: This is x1. Needs y1 = x0*sin + x1*cos
        // Read x0 (at i-1)
        float val_pair = x[i - 1];
        y[i] = val_pair * sin_theta + val * cos_theta;
    }
}
