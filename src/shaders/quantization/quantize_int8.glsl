#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { float input_data[]; };
layout(binding = 1) buffer Output { int8_t output_data[]; };
layout(binding = 2) uniform Params { float scale; uint size; };

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= size) {
        return;
    }

    float value = input_data[idx];
    float scaled = value / scale;

    int8_t quantized = int8_t(clamp(scaled, -128.0, 127.0));
    output_data[idx] = quantized;
}
