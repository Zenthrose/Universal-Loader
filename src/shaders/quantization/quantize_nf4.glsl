#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

const float NF4_TABLE[16] = float[](
    -1.0, -0.696192803978, -0.525072057247, -0.394917488694,
    -0.284441382885, -0.184773430284, -0.0910502361056, 0.0,
    0.079580299556, 0.160930201411, 0.246112301945, 0.337915241718,
    0.440709829331, 0.562617003918, 0.722956836224, 1.0
);

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { float input_data[]; };
layout(binding = 1) buffer Output { uint8_t output_data[]; };
layout(binding = 2) uniform Params { uint size; };

void main() {
    uint idx = gl_GlobalInvocationID.x * 2;

    if (idx >= size) {
        return;
    }

    float f0 = input_data[idx];
    float f1 = (idx + 1 < size) ? input_data[idx + 1] : 0.0;

    float min_dist0 = 10000.0;
    float min_dist1 = 10000.0;
    uint q0 = 0;
    uint q1 = 0;

    for (uint j = 0; j < 16; j++) {
        float nf4_val = NF4_TABLE[j];

        float dist0 = abs(f0 - nf4_val);
        if (dist0 < min_dist0) {
            min_dist0 = dist0;
            q0 = j;
        }

        float dist1 = abs(f1 - nf4_val);
        if (dist1 < min_dist1) {
            min_dist1 = dist1;
            q1 = j;
        }
    }

    output_data[idx / 2] = (q0 << 4) | q1;
}
