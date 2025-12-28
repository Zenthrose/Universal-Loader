#include <immintrin.h>
#include <cmath>

namespace cpu {

void gelu_f32(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = src[i];
        float gelu = 0.5f * x * (1.0f + std::tanh(0.79788456f * x * (1.0f + 0.044715f * x * x)));
        dst[i] = gelu;
    }
}

void silu_f32(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = src[i];
        float silu = x * (1.0f / (1.0f + std::exp(-x)));
        dst[i] = silu;
    }
}

void softmax_f32(float* dst, const float* src, size_t n) {
    float max_val = src[0];
    for (size_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, src[i]);
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::exp(src[i] - max_val);
        sum += dst[i];
    }
    
    for (size_t i = 0; i < n; ++i) {
        dst[i] /= sum;
    }
}

}
