#include <immintrin.h>
#include <cstdint>

namespace cpu {

void dequantize_q4_0_f32(const void* src, float* dst, size_t n) {
    const struct BlockQ4_0 {
        float d;
        uint8_t qs[16];
    }* blocks = static_cast<const BlockQ4_0*>(src);

    for (size_t i = 0; i < n; i += 32) {
        const BlockQ4_0* b = &blocks[i / 32];
        __m256 d = _mm256_set1_ps(b[0].d);
        __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[0].qs));
        __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b[1].qs));

        __m256i q0_even = _mm256_cvtepu8_epi16(_mm_and_si128(q0, _mm_set1_epi8(0x0F)));
        __m256i q1_even = _mm256_cvtepu8_epi16(_mm_and_si128(q1, _mm_set1_epi8(0x0F)));
        
        __m128i q0_odd = _mm_and_si128(q0, _mm_set1_epi8(0xF0));
        __m128i q1_odd = _mm_and_si128(q1, _mm_set1_epi8(0xF0));
        q0_odd = _mm_srli_epi16(q0_odd, 4);
        q1_odd = _mm_srli_epi16(q1_odd, 4);
        
        __m256i q0_odd_ext = _mm256_cvtepu8_epi16(q0_odd);
        __m256i q1_odd_ext = _mm256_cvtepu8_epi16(q1_odd);

        __m256 f0_even = _mm256_cvtepi32_ps(q0_even);
        __m256 f0_odd = _mm256_cvtepi32_ps(q0_odd_ext);
        __m256 f1_even = _mm256_cvtepi32_ps(q1_even);
        __m256 f1_odd = _mm256_cvtepi32_ps(q1_odd_ext);

        f0_even = _mm256_sub_ps(f0_even, _mm256_set1_ps(8.0f));
        f0_odd = _mm256_sub_ps(f0_odd, _mm256_set1_ps(8.0f));
        f1_even = _mm256_sub_ps(f1_even, _mm256_set1_ps(8.0f));
        f1_odd = _mm256_sub_ps(f1_odd, _mm256_set1_ps(8.0f));

        f0_even = _mm256_mul_ps(f0_even, d);
        f0_odd = _mm256_mul_ps(f0_odd, d);
        f1_even = _mm256_mul_ps(f1_even, d);
        f1_odd = _mm256_mul_ps(f1_odd, d);

        _mm256_storeu_ps(dst + i, f0_even);
        _mm256_storeu_ps(dst + i + 8, f0_odd);
        _mm256_storeu_ps(dst + i + 16, f1_even);
        _mm256_storeu_ps(dst + i + 24, f1_odd);
    }
}

void dequantize_q4_k_f32(const void* src, float* dst, size_t n) {
    const struct BlockQ4_K {
        uint8_t d[2];
        uint8_t qh[16];
        int8_t scales[16];
        uint8_t qs[128];
    }* blocks = static_cast<const BlockQ4_K*>(src);

    for (size_t i = 0; i < n; i += 256) {
        const BlockQ4_K* b = &blocks[i / 256];
        float d = b->d[0] * (1.0f / 16.0f) + b->d[1] * (1.0f / 256.0f);

        for (size_t j = 0; j < 256; ++j) {
            uint8_t qh = b->qh[j / 16];
            uint8_t high_bit = (qh >> (j % 16)) & 1;
            uint8_t qs = b->qs[j / 2];
            uint8_t q4 = (j % 2 == 0) ? (qs & 0x0F) : (qs >> 4);
            int16_t q = (int16_t)((high_bit << 4) | q4) - 8;
            
            int8_t scale = b->scales[j / 16];
            dst[i + j] = q * scale * d;
        }
    }
}

}
