#include "gemm_avx2.h"
#include <cstring>

namespace cpu {

void gemm_avx2_f32(const float* A, const float* B, float* C,
                   uint32_t M, uint32_t N, uint32_t K) {
    memset(C, 0, M * N * sizeof(float));

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; k += 8) {
            __m256 a_vec = _mm256_loadu_ps(A + m * K + k);

            for (uint32_t n = 0; n < N; n += 8) {
                __m256 b_vec = _mm256_loadu_ps(B + k * N + n);
                __m256 c_vec = _mm256_loadu_ps(C + m * N + n);

                __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                c_vec = _mm256_add_ps(c_vec, prod);

                _mm256_storeu_ps(C + m * N + n, c_vec);
            }
        }
    }
}

void gemm_avx2_f16_f32(const uint16_t* A, const float* B, float* C,
                       uint32_t M, uint32_t N, uint32_t K) {
    memset(C, 0, M * N * sizeof(float));

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < K; k += 8) {
            __m128i a_i16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(A + m * K + k));
            __m256 a_vec = _mm256_cvtph_ps(a_i16);

            for (uint32_t n = 0; n < N; n += 8) {
                __m256 b_vec = _mm256_loadu_ps(B + k * N + n);
                __m256 c_vec = _mm256_loadu_ps(C + m * N + n);

                __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                c_vec = _mm256_add_ps(c_vec, prod);

                _mm256_storeu_ps(C + m * N + n, c_vec);
            }
        }
    }
}

}
