#pragma once
#include <cstdint>

namespace ggml {

enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ2_S = 18,
    IQ3_XXS = 19,
    IQ1_S = 24,
    IQ4_NL = 25,
    IQ4_XS = 27,
    I32 = 4,
    I16 = 5,
    I8 = 22,
    NF4 = 26,
    FP4 = 28,
};

inline size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGMLType::F32: return 4;
        case GGMLType::F16: return 2;
        case GGMLType::Q4_0: return sizeof(float) + sizeof(uint16_t);
        case GGMLType::Q4_1: return 2 * sizeof(float) + sizeof(uint16_t);
        case GGMLType::Q5_0: return sizeof(float) + sizeof(uint16_t);
        case GGMLType::Q5_1: return 2 * sizeof(float) + sizeof(uint16_t);
        case GGMLType::Q8_0: return sizeof(float) + sizeof(int16_t) * 8;
        case GGMLType::Q8_1: return 2 * sizeof(float) + sizeof(int8_t) * 16;
        case GGMLType::Q2_K: return sizeof(uint16_t) + sizeof(uint8_t) * 4 + 2;
        case GGMLType::Q3_K: return sizeof(uint16_t) + sizeof(uint8_t) * 2 + sizeof(uint8_t) * 4;
        case GGMLType::Q4_K: return sizeof(uint16_t) + sizeof(uint8_t) * 2 + sizeof(uint8_t) * 8;
        case GGMLType::Q5_K: return sizeof(uint16_t) + sizeof(uint8_t) * 2 + sizeof(uint8_t) * 8;
        case GGMLType::Q6_K: return sizeof(uint16_t) + sizeof(uint8_t) * 2 + sizeof(uint8_t) * 8;
        case GGMLType::Q8_K: return sizeof(uint16_t) + sizeof(uint8_t) * 2 + sizeof(uint8_t) * 16;
        case GGMLType::IQ2_XXS: return sizeof(uint32_t);
        case GGMLType::IQ2_XS: return sizeof(uint32_t);
        case GGMLType::IQ3_XXS: return sizeof(uint8_t) * 4;
        case GGMLType::IQ4_NL: return sizeof(uint8_t) * 4;
        case GGMLType::IQ4_XS: return sizeof(uint16_t) + sizeof(uint8_t) * 4;
        default: return 4;
    }
}

inline int ggml_blck_size(GGMLType type) {
    switch (type) {
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q4_1: return 32;
        case GGMLType::Q5_0: return 32;
        case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q8_1: return 32;
        case GGMLType::Q2_K: return 256;
        case GGMLType::Q3_K: return 256;
        case GGMLType::Q4_K: return 256;
        case GGMLType::Q5_K: return 256;
        case GGMLType::Q6_K: return 256;
        case GGMLType::Q8_K: return 256;
        case GGMLType::IQ2_XXS: return 32;
        case GGMLType::IQ2_XS: return 32;
        case GGMLType::IQ3_XXS: return 32;
        case GGMLType::IQ4_NL: return 32;
        case GGMLType::IQ4_XS: return 32;
        default: return 1;
    }
}

}
