#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { float input_data[]; };
layout(binding = 1) buffer Output { uint8_t output_data[]; };
layout(binding = 2) uniform Params { float scale; uint size; };

void main() {
    uint idx = gl_GlobalInvocationID.x * 2;

    if (idx >= size) {
        return;
    }

    float f0 = input_data[idx];
    float f1 = (idx + 1 < size) ? input_data[idx + 1] : 0.0;

    float scaled0 = f0 / scale;
    float scaled1 = f1 / scale;

    uint8_t q0 = uint8_t(clamp(round(scaled0 + 8.0), 0.0, 7.0));
    uint8_t q1 = uint8_t(clamp(round(scaled1 + 8.0), 0.0, 7.0));

    output_data[idx / 2] = (q0 << 4) | q1;
}
