#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_ray_query : require
#extension GL_KHR_cooperative_matrix : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer K { float k[]; };
layout(binding = 1) readonly buffer V { float v[]; };
layout(binding = 2) readonly buffer Q { float q[]; };
layout(binding = 3) writeonly buffer O { float o[]; };

layout(push_constant) uniform Params {
    uint seq_len;
    uint hidden_dim;
    uint num_heads;
    uint head_dim;
    float scale;
    float sparsity_threshold;
} params;

layout(binding = 3) readonly buffer TLAS { accelerationStructureEXT tlas; };

void main() {
    uint head_id = gl_GlobalInvocationID.x / params.head_dim;
    uint dim_id = gl_GlobalInvocationID.x % params.head_dim;
    uint seq_id = gl_GlobalInvocationID.y;

    if (head_id >= params.num_heads || seq_id >= params.seq_len) return;

    float q_val = q[seq_id * params.hidden_dim + head_id * params.head_dim + dim_id];

    // Initialize ray query for sparse key lookup
    rayQueryEXT ray_query;
    vec3 origin = vec3(q_val, float(seq_id), float(head_id));
    vec3 direction = vec3(0.0, 1.0, 0.0); // Trace along sequence dimension
    rayQueryInitializeEXT(ray_query, tlas, gl_RayFlagsNoneEXT, 0xFF, origin, 0.0, direction, 1000.0);

    float attn_sum = 0.0;
    float weight_sum = 0.0;

    while (rayQueryProceedEXT(ray_query)) {
        if (rayQueryGetIntersectionTypeEXT(ray_query) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint key_idx = rayQueryGetIntersectionInstanceIdEXT(ray_query);
            if (key_idx >= params.seq_len) continue;

            float k_val = k[key_idx * params.hidden_dim + head_id * params.head_dim + dim_id];
            float attn_score = (q_val * k_val) * params.scale;

            // Apply dynamic sparsity threshold
            if (attn_score > params.sparsity_threshold) {
                float attn = exp(attn_score);
                float v_val = v[key_idx * params.hidden_dim + head_id * params.head_dim + dim_id];

                // Simple weighted accumulation (cooperative matrix fusion would be more complex)
                attn_sum += attn * v_val;
                weight_sum += attn;
            }

            rayQueryConfirmIntersectionEXT(ray_query);
        }
    }

    o[seq_id * params.hidden_dim + head_id * params.head_dim + dim_id] = attn_sum / max(weight_sum, 1e-6);
}