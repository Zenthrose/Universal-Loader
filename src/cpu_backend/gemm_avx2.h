#pragma once
#include <immintrin.h>
#include <cstdint>

namespace cpu {

void gemm_avx2_f32(const float* A, const float* B, float* C, 
                   uint32_t M, uint32_t N, uint32_t K);

void gemm_avx2_f16_f32(const uint16_t* A, const float* B, float* C,
                       uint32_t M, uint32_t N, uint32_t K);

}
