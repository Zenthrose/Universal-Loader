#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { int8_t input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) uniform Params { float scale; uint size; };

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= size) {
        return;
    }

    int8_t quantized = input_data[idx];
    float dequantized = float(quantized) * scale;
    output_data[idx] = dequantized;
}
