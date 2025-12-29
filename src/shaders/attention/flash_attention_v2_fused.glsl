#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

layout(local_size_x = 128, local_size_y = 4) in;

layout(binding = 0) readonly buffer Q { float q[]; };
layout(binding = 1) readonly buffer K_Cache { float k_cache[]; };
layout(binding = 2) readonly buffer V_Cache { float v_cache[]; };
layout(binding = 3) writeonly buffer Output { float out[]; };

layout(push_constant) uniform Params {
    uint batch_size;
    uint seq_len;
    uint head_dim;
    uint num_q_heads;
    uint num_kv_heads;
    uint num_kv_groups;
    uint max_seq_len;
    float scale;
    uint is_causal;
    uint padding[3];
} params;

const uint BLOCK_M = 128;
const uint BLOCK_N = 64;
const uint BLOCK_D = 64;

shared float s_q[BLOCK_M][BLOCK_D];
shared float s_k[BLOCK_N][BLOCK_D];
shared float s_v[BLOCK_N][BLOCK_D];

void main() {
    const uint tx = gl_LocalInvocationID.x;
    const uint ty = gl_LocalInvocationID.y;

    const uint batch_idx = gl_WorkGroupID.z;
    const uint head_idx = gl_WorkGroupID.y;
    const uint q_block_idx = gl_WorkGroupID.x;

    const uint q_head_offset = head_idx * params.head_dim;
    const uint kv_head_idx = head_idx / params.num_kv_groups;
    const uint kv_head_offset = kv_head_idx * params.head_dim;

    const uint seq_start = q_block_idx * BLOCK_M;
    const uint seq_end = min(seq_start + BLOCK_M, params.seq_len);

    const uint row = seq_start + tx;

    vec4 acc[16];
    for (int i = 0; i < 16; ++i) {
        acc[i] = vec4(0.0);
    }

    float row_max[BLOCK_M / 128];
    float row_sum[BLOCK_M / 128];
    for (int i = 0; i < BLOCK_M / 128; ++i) {
        row_max[i] = -1e30;
        row_sum[i] = 0.0;
    }

    const uint n_blocks = (params.max_seq_len + BLOCK_N - 1) / BLOCK_N;

    for (uint kv_block = 0; kv_block < n_blocks; ++kv_block) {
        const uint kv_start = kv_block * BLOCK_N;
        const uint kv_end = min(kv_start + BLOCK_N, params.max_seq_len);

        for (uint m = 0; m < BLOCK_M; m += 128) {
            const uint seq_pos = seq_start + m + tx;
            if (seq_pos < seq_start + BLOCK_M && seq_pos < params.seq_len) {
                const uint q_idx = batch_idx * params.num_q_heads * params.head_dim + 
                                  q_head_offset + seq_pos * params.num_q_heads * params.head_dim;

                for (uint d = 0; d < params.head_dim; d += 4) {
                    if (tx < 32) {
                        vec4 q_vec;
                        if (q_idx + d + 3 < q.length()) {
                            q_vec = vec4(q[q_idx + d], q[q_idx + d + 1], q[q_idx + d + 2], q[q_idx + d + 3]);
                        } else {
                            q_vec = vec4(0.0);
                        }
                        if (m + tx * 4 < BLOCK_M) {
                            s_q[m + tx * 4][d] = q_vec.x;
                            s_q[m + tx * 4 + 1][d] = q_vec.y;
                            s_q[m + tx * 4 + 2][d] = q_vec.z;
                            s_q[m + tx * 4 + 3][d] = q_vec.w;
                        }
                    }
                }
            }
        }

        for (uint n = 0; n < BLOCK_N; n += 128) {
            const uint kv_seq = kv_start + n + tx;
            if (kv_seq < kv_end) {
                const uint k_idx = batch_idx * params.num_kv_heads * params.head_dim + 
                                  kv_head_offset + kv_seq * params.num_kv_heads * params.head_dim;

                for (uint d = 0; d < params.head_dim; d += 4) {
                    if (tx < 32) {
                        vec4 k_vec;
                        if (k_idx + d + 3 < k_cache.length()) {
                            k_vec = vec4(k_cache[k_idx + d], k_cache[k_idx + d + 1], k_cache[k_idx + d + 2], k_cache[k_idx + d + 3]);
                        } else {
                            k_vec = vec4(0.0);
                        }
                        if (n + tx * 4 < BLOCK_N) {
                            s_k[n + tx * 4][d] = k_vec.x;
                            s_k[n + tx * 4 + 1][d] = k_vec.y;
                            s_k[n + tx * 4 + 2][d] = k_vec.z;
                            s_k[n + tx * 4 + 3][d] = k_vec.w;
                        }
                    }
                }

                const uint v_idx = batch_idx * params.num_kv_heads * params.head_dim + 
                                  kv_head_offset + kv_seq * params.num_kv_heads * params.head_dim;

                for (uint d = 0; d < params.head_dim; d += 4) {
                    if (tx < 32) {
                        vec4 v_vec;
                        if (v_idx + d + 3 < v_cache.length()) {
                            v_vec = vec4(v_cache[v_idx + d], v_cache[v_idx + d + 1], v_cache[v_idx + d + 2], v_cache[v_idx + d + 3]);
                        } else {
                            v_vec = vec4(0.0);
                        }
                        if (n + tx * 4 < BLOCK_N) {
                            s_v[n + tx * 4][d] = v_vec.x;
                            s_v[n + tx * 4 + 1][d] = v_vec.y;
                            s_v[n + tx * 4 + 2][d] = v_vec.z;
                            s_v[n + tx * 4 + 3][d] = v_vec.w;
                        }
                    }
                }
            }
        }

        memoryBarrierShared();
        barrier();

        const uint local_m = (tx % (BLOCK_M / 128)) * 128 + (tx / (BLOCK_M / 128));
        const uint local_q_row = local_m;

        if (local_q_row < BLOCK_M && seq_start + local_q_row < params.seq_len) {
            float new_max = row_max[local_q_row / 128];
            float new_sum = row_sum[local_q_row / 128];

            for (uint n_idx = 0; n_idx < BLOCK_N; ++n_idx) {
                const uint kv_pos = kv_start + n_idx;
                if (kv_pos >= params.max_seq_len) break;

                const bool is_causal = (params.is_causal != 0) && (kv_pos > (seq_start + local_q_row));
                if (is_causal) continue;

                float qk = 0.0;
                for (uint d = 0; d < params.head_dim; ++d) {
                    qk += s_q[local_q_row][d] * s_k[n_idx][d];
                }
                qk *= params.scale;

                const float old_max = new_max;
                new_max = max(new_max, qk);
                const float scale_val = exp(old_max - new_max);
                new_sum = new_sum * scale_val + exp(qk - new_max);
            }

            const float scale_final = exp(row_max[local_q_row / 128] - new_max);
            for (int i = 0; i < 16; ++i) {
                acc[local_q_row / 8] *= scale_final;
            }

            row_max[local_q_row / 128] = new_max;
            row_sum[local_q_row / 128] = new_sum;

            for (uint n_idx = 0; n_idx < BLOCK_N; ++n_idx) {
                const uint kv_pos = kv_start + n_idx;
                if (kv_pos >= params.max_seq_len) break;

                const bool is_causal = (params.is_causal != 0) && (kv_pos > (seq_start + local_q_row));
                if (is_causal) continue;

                float qk = 0.0;
                for (uint d = 0; d < params.head_dim; ++d) {
                    qk += s_q[local_q_row][d] * s_k[n_idx][d];
                }
                qk *= params.scale;

                const float attn_weight = exp(qk - new_max);

                const uint acc_idx = local_q_row / 8;
                const uint d_start = (local_q_row % 8) * 8;
                for (uint d = 0; d < 8 && d_start + d < params.head_dim; ++d) {
                    acc[acc_idx][d] += attn_weight * s_v[n_idx][d_start + d];
                }
            }
        }

        memoryBarrierShared();
        barrier();
    }

    const uint out_idx_base = batch_idx * params.num_q_heads * params.head_dim + 
                             head_idx * params.head_dim;

    for (uint m = 0; m < BLOCK_M; ++m) {
        const uint seq_pos = seq_start + m;
        if (seq_pos >= params.seq_len) continue;

        const uint out_idx = out_idx_base + seq_pos * params.num_q_heads * params.head_dim;

        const uint acc_base = m / 8;
        const uint d_base = (m % 8) * 8;

        if (row_sum[m / 128] > 0.0) {
            const float inv_sum = 1.0 / row_sum[m / 128];
            for (uint d = 0; d < 8 && d_base + d < params.head_dim; ++d) {
                const uint write_idx = (m % 8) * 32 + d;
                const uint global_write_idx = out_idx + d_base + d;
                if (global_write_idx < out.length()) {
                    out[global_write_idx] = acc[acc_base][d] * inv_sum;
                }
            }
        }
    }
}
