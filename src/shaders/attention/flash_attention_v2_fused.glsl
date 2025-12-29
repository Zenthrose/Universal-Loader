#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(constant_id = 0) const uint BLOCK_M = 128;
layout(constant_id = 1) const uint BLOCK_N = 128;
layout(constant_id = 2) const uint BLOCK_D = 64;

layout(local_size_x = BLOCK_N, local_size_y = 1) in;

layout(binding = 0) readonly buffer Q { uint q[]; };
layout(binding = 1) readonly buffer K_Cache { uint k_cache[]; };
layout(binding = 2) readonly buffer V_Cache { uint v_cache[]; };
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
    uint rope_theta;
    uint padding[2];
} params;

shared float s_q[BLOCK_M][BLOCK_D];
shared float s_k[BLOCK_N][BLOCK_D];
shared float s_v[BLOCK_N][BLOCK_D];

vec2 rope_cis(uint pos, uint dim) {
    const float freq = params.rope_theta * exp(-float(dim / 2) * log(10000.0 / float(params.head_dim)));
    return vec2(cos(float(pos) * freq), sin(float(pos) * freq));
}

void apply_rope(inout vec2 x, vec2 cis, uint dim_idx) {
    float new_x = x.x * cis.x - x.y * cis.y;
    x.y = x.x * cis.y + x.y * cis.x;
    x.x = new_x;
}

void main() {
    const uint tx = gl_LocalInvocationID.x;
    const uint bx = gl_WorkGroupID.x;
    const uint by = gl_WorkGroupID.y;
    const uint bz = gl_WorkGroupID.z;

    const uint batch_idx = bz;
    const uint head_idx = by;
    const uint kv_head_idx = head_idx / params.num_kv_groups;

    const uint q_head_offset = head_idx * params.head_dim;
    const uint kv_head_offset = kv_head_idx * params.head_dim;

    const uint seq_start = bx * BLOCK_M;
    const uint seq_end = min(seq_start + BLOCK_M, params.seq_len);

    const uint n_blocks = (params.max_seq_len + BLOCK_N - 1) / BLOCK_N;

    const uint row_per_thread = (BLOCK_M + BLOCK_N - 1) / BLOCK_N;
    const uint my_row_start = tx * row_per_thread;
    const uint my_row_end = min(my_row_start + row_per_thread, BLOCK_M);

    float m_i[16];
    float l_i[16];
    vec4 o[16][16];

    for (uint r = 0; r < row_per_thread; ++r) {
        const uint local_row = my_row_start + r;
        if (local_row < 16) {
            m_i[local_row] = -1e30;
            l_i[local_row] = 0.0;
            for (uint d = 0; d < 16; ++d) {
                o[local_row][d] = vec4(0.0);
            }
        }
    }

    for (uint kv_block = 0; kv_block < n_blocks; ++kv_block) {
        const uint kv_start = kv_block * BLOCK_N;
        const uint kv_end = min(kv_start + BLOCK_N, params.max_seq_len);

        for (uint m = 0; m < BLOCK_M; m += BLOCK_N) {
            const uint load_idx = m + tx;
            const uint seq_pos = seq_start + load_idx;

            if (load_idx < BLOCK_M && seq_pos < params.seq_len) {
                const uint q_global_idx = batch_idx * params.num_q_heads * params.head_dim + 
                                        q_head_offset + seq_pos * params.num_q_heads * params.head_dim;

                for (uint d = 0; d < params.head_dim; d += 4) {
                    const uint idx = q_global_idx + d;
                    if (idx + 3 < q.length()) {
                        vec4 q_vals = vec4(uintBitsToFloat(q[idx]), uintBitsToFloat(q[idx + 1]),
                                           uintBitsToFloat(q[idx + 2]), uintBitsToFloat(q[idx + 3]));

                        for (uint i = 0; i < 4 && d + i < params.head_dim; ++i) {
                            vec2 cis = rope_cis(seq_pos, d + i);
                            vec2 q_pair = vec2(q_vals[i], (i + 1 < 4 && d + i + 1 < params.head_dim) ? q_vals[i + 1] : 0.0);
                            apply_rope(q_pair, cis, d + i);
                            s_q[load_idx][d + i] = q_pair.x;
                            if (d + i + 1 < params.head_dim) {
                                s_q[load_idx][d + i + 1] = q_pair.y;
                            }
                        }
                    } else {
                        for (uint i = 0; i < 4 && d + i < params.head_dim; ++i) {
                            s_q[load_idx][d + i] = 0.0;
                        }
                    }
                }
            }
        }

        for (uint n = 0; n < BLOCK_N; n += BLOCK_N) {
            const uint kv_seq = kv_start + n + tx;
            if (kv_seq < kv_end) {
                const uint k_global_idx = batch_idx * params.num_kv_heads * params.head_dim + 
                                        kv_head_offset + kv_seq * params.num_kv_heads * params.head_dim;

                for (uint d = 0; d < params.head_dim; d += 4) {
                    const uint idx = k_global_idx + d;
                    if (idx + 3 < k_cache.length()) {
                        vec4 k_vals = vec4(uintBitsToFloat(k_cache[idx]), uintBitsToFloat(k_cache[idx + 1]),
                                           uintBitsToFloat(k_cache[idx + 2]), uintBitsToFloat(k_cache[idx + 3]));

                        for (uint i = 0; i < 4 && d + i < params.head_dim; ++i) {
                            vec2 cis = rope_cis(kv_seq, d + i);
                            vec2 k_pair = vec2(k_vals[i], (i + 1 < 4 && d + i + 1 < params.head_dim) ? k_vals[i + 1] : 0.0);
                            apply_rope(k_pair, cis, d + i);
                            s_k[n + tx][d + i] = k_pair.x;
                            if (d + i + 1 < params.head_dim) {
                                s_k[n + tx][d + i + 1] = k_pair.y;
                            }
                        }
                    } else {
                        for (uint i = 0; i < 4 && d + i < params.head_dim; ++i) {
                            s_k[n + tx][d + i] = 0.0;
                        }
                    }
                }

                const uint v_global_idx = batch_idx * params.num_kv_heads * params.head_dim + 
                                        kv_head_offset + kv_seq * params.num_kv_heads * params.head_dim;

                for (uint d = 0; d < params.head_dim; d += 4) {
                    const uint idx = v_global_idx + d;
                    if (idx + 3 < v_cache.length()) {
                        s_v[n + tx][d] = uintBitsToFloat(v_cache[idx]);
                        s_v[n + tx][d + 1] = uintBitsToFloat(v_cache[idx + 1]);
                        s_v[n + tx][d + 2] = uintBitsToFloat(v_cache[idx + 2]);
                        s_v[n + tx][d + 3] = uintBitsToFloat(v_cache[idx + 3]);
                    } else {
                        for (uint i = 0; i < 4 && d + i < params.head_dim; ++i) {
                            s_v[n + tx][d + i] = 0.0;
                        }
                    }
                }
            } else if (tx < BLOCK_N) {
                for (uint d = 0; d < params.head_dim; ++d) {
                    s_k[tx][d] = 0.0;
                    s_v[tx][d] = 0.0;
                }
            }
        }

        memoryBarrierShared();
        barrier();

        for (uint r = 0; r < row_per_thread; ++r) {
            const uint local_m = my_row_start + r;
            if (local_m >= BLOCK_M) continue;

            const uint q_seq_pos = seq_start + local_m;
            if (q_seq_pos >= params.seq_len) continue;

            float new_m_i = m_i[local_m];
            float new_l_i = l_i[local_m];

            const vec4 q_vec[BLOCK_D / 4];
            for (uint d = 0; d < params.head_dim; d += 4) {
                q_vec[d / 4] = vec4(s_q[local_m][d], s_q[local_m][d + 1],
                                     s_q[local_m][d + 2], s_q[local_m][d + 3]);
            }

            for (uint n = 0; n < BLOCK_N; ++n) {
                const uint kv_seq_pos = kv_start + n;
                if (kv_seq_pos >= params.max_seq_len) break;

                const bool is_causal = (params.is_causal != 0) && (kv_seq_pos > q_seq_pos);
                if (is_causal) continue;

                float qk = 0.0;
                for (uint d = 0; d < params.head_dim; d += 4) {
                    const vec4 k_vec = vec4(s_k[n][d], s_k[n][d + 1],
                                           s_k[n][d + 2], s_k[n][d + 3]);
                    qk += dot(q_vec[d / 4], k_vec);
                }
                qk *= params.scale;

                const float old_m_i = new_m_i;
                new_m_i = max(new_m_i, qk);
                const float scale_m = exp(old_m_i - new_m_i);
                new_l_i = new_l_i * scale_m + exp(qk - new_m_i);
            }

            const float rescale_o = exp(m_i[local_m] - new_m_i);
            for (uint d = 0; d < (params.head_dim + 3) / 4; ++d) {
                o[local_m][d] *= rescale_o;
            }

            m_i[local_m] = new_m_i;
            l_i[local_m] = new_l_i;

            for (uint n = 0; n < BLOCK_N; ++n) {
                const uint kv_seq_pos = kv_start + n;
                if (kv_seq_pos >= params.max_seq_len) break;

                const bool is_causal = (params.is_causal != 0) && (kv_seq_pos > q_seq_pos);
                if (is_causal) continue;

                float qk = 0.0;
                for (uint d = 0; d < params.head_dim; d += 4) {
                    const vec4 k_vec = vec4(s_k[n][d], s_k[n][d + 1],
                                           s_k[n][d + 2], s_k[n][d + 3]);
                    qk += dot(q_vec[d / 4], k_vec);
                }
                qk *= params.scale;

                const float attn_weight = exp(qk - new_m_i);

                for (uint d = 0; d < params.head_dim; d += 4) {
                    const vec4 v_vec = vec4(s_v[n][d], s_v[n][d + 1],
                                           s_v[n][d + 2], s_v[n][d + 3]);
                    o[local_m][d / 4] += attn_weight * v_vec;
                }
            }
        }

        memoryBarrierShared();
        barrier();
    }

    const uint out_base_idx = batch_idx * params.num_q_heads * params.head_dim + 
                              head_idx * params.head_dim;

    for (uint r = 0; r < row_per_thread; ++r) {
        const uint local_m = my_row_start + r;
        if (local_m >= BLOCK_M) continue;

        const uint seq_pos = seq_start + local_m;
        if (seq_pos >= params.seq_len) continue;

        const float inv_l_i = 1.0 / l_i[local_m];
        const uint out_idx = out_base_idx + seq_pos * params.num_q_heads * params.head_dim;

        for (uint d = 0; d < params.head_dim; d += 4) {
            const uint idx = out_idx + d;
            if (idx + 3 < out.length()) {
                out[idx] = o[local_m][d / 4].x * inv_l_i;
                out[idx + 1] = o[local_m][d / 4].y * inv_l_i;
                out[idx + 2] = o[local_m][d / 4].z * inv_l_i;
                out[idx + 3] = o[local_m][d / 4].w * inv_l_i;
            }
        }
    }
}
