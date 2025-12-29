#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer Input { int8_t input_data[]; };
layout(binding = 1) buffer Output { float output_data[]; };
layout(binding = 2) readonly buffer Scale { float scale_data[]; };
layout(binding = 3) readonly buffer ZeroPoint { int32_t zero_point_data[]; };

layout(push_constant) uniform Params {
    uint size;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= size) {
        return;
    }

    int32_t quantized = int32_t(input_data[idx]);
    float scale = scale_data[0];
    int32_t zp = zero_point_data[0];

    float dequantized = float(quantized - zp) * scale;
    output_data[idx] = dequantized;
}
