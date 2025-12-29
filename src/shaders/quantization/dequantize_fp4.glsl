#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { uint8_t input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) uniform Params { float scale; uint size; };

void main() {
    uint idx = gl_GlobalInvocationID.x * 2;

    if (idx >= size) {
        return;
    }

    uint8_t packed = input_data[idx / 2];
    uint8_t q0 = (packed >> 4) & 0x0F;
    uint8_t q1 = packed & 0x0F;

    output_data[idx] = (float(q0) - 8.0) * scale;
    if (idx + 1 < size) {
        output_data[idx + 1] = (float(q1) - 8.0) * scale;
    }
}
