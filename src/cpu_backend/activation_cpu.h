#pragma once
#include <cstddef>

namespace cpu {

void gelu_f32(float* dst, const float* src, size_t n);
void silu_f32(float* dst, const float* src, size_t n);
void softmax_f32(float* dst, const float* src, size_t n);

}
