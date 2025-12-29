#pragma once
#include <cstddef>

namespace cpu {

void dequantize_q4_0_f32(const void* src, float* dst, size_t n);
void dequantize_q4_k_f32(const void* src, float* dst, size_t n);

}
