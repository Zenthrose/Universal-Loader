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

layout(binding = 0) buffer Input { uint8_t input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) uniform Params { uint size; };

void main() {
    uint idx = gl_GlobalInvocationID.x * 2;

    if (idx >= size) {
        return;
    }

    uint8_t packed = input_data[idx / 2];
    uint8_t q0 = (packed >> 4) & 0x0F;
    uint8_t q1 = packed & 0x0F;

    output_data[idx] = NF4_TABLE[q0];
    if (idx + 1 < size) {
        output_data[idx + 1] = NF4_TABLE[q1];
    }
}
