#include "quantization_manager.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <map>

namespace inference {

QuantizationManager::QuantizationManager()
    : initialized_(false) {
}

QuantizationManager::~QuantizationManager() {
}

void QuantizationManager::initialize(const QuantizationConfig& config) {
    config_ = config;
    initialized_ = true;
}

void QuantizationManager::quantize_tensor(ggml::Tensor* tensor, QuantizationType target_type) {
    if (!tensor || target_type == QuantizationType::NONE || target_type == QuantizationType::FP32) {
        return;
    }

    const float* src_data = static_cast<const float*>(tensor->get_cpu_data());
    if (!src_data) {
        return;
    }

    const auto& shape = tensor->get_shape();
    uint32_t size = 1;
    for (uint32_t dim : shape) {
        size *= dim;
    }

    std::string tensor_name = tensor->get_name();

    if (target_type == QuantizationType::INT8) {
        float scale = compute_per_tensor_scale(src_data, size, target_type);
        tensor_scales_[tensor_name] = scale;

        std::vector<int8_t> quantized_data(size);
        quantize_fp32_to_int8(src_data, quantized_data.data(), size, scale);

        tensor->set_quantized_data(quantized_data.data(), size * sizeof(int8_t));
        tensor->set_quantized(true);
        tensor->set_quantized_type(ggml::GGMLType::Q8_0);
    }
    else if (target_type == QuantizationType::FP4) {
        float scale = compute_per_tensor_scale(src_data, size, target_type);
        tensor_scales_[tensor_name] = scale;

        uint32_t fp4_size = (size + 1) / 2;
        std::vector<uint8_t> quantized_data(fp4_size);
        quantize_fp32_to_fp4(src_data, quantized_data.data(), size, scale);

        tensor->set_quantized_data(quantized_data.data(), fp4_size);
        tensor->set_quantized(true);
        tensor->set_quantized_type(ggml::GGMLType::Q4_0);
    }
    else if (target_type == QuantizationType::NF4) {
        uint32_t nf4_size = (size + 1) / 2;
        std::vector<uint8_t> quantized_data(nf4_size);
        quantize_nf4(src_data, quantized_data.data(), size);

        tensor->set_quantized_data(quantized_data.data(), nf4_size);
        tensor->set_quantized(true);
        tensor->set_quantized_type(ggml::GGMLType::NF4);
    }
}

void QuantizationManager::dequantize_tensor(ggml::Tensor* tensor, QuantizationType source_type) {
    if (!tensor || !tensor->is_quantized()) {
        return;
    }

    const uint8_t* quantized_data = static_cast<const uint8_t*>(tensor->get_quantized_data());
    if (!quantized_data) {
        return;
    }

    const auto& shape = tensor->get_shape();
    uint32_t size = 1;
    for (uint32_t dim : shape) {
        size *= dim;
    }

    std::string tensor_name = tensor->get_name();
    std::vector<float> dequantized_data(size);

    if (source_type == QuantizationType::INT8) {
        float scale = tensor_scales_.at(tensor_name);
        dequantize_int8_to_fp32(reinterpret_cast<const int8_t*>(quantized_data),
                               dequantized_data.data(), size, scale);
    }
    else if (source_type == QuantizationType::FP4) {
        float scale = tensor_scales_.at(tensor_name);
        dequantize_fp4_to_fp32(quantized_data, dequantized_data.data(), size, scale);
    }
    else if (source_type == QuantizationType::NF4) {
        dequantize_nf4(quantized_data, dequantized_data.data(), size);
    }

    tensor->set_cpu_data(dequantized_data.data(), size * sizeof(float));
}

void QuantizationManager::quantize_weights(ggml::Tensor* weights, QuantizationType type) {
    quantize_tensor(weights, type);
}

void QuantizationManager::quantize_activations(std::vector<float>& activations, QuantizationType type) {
    if (activations.empty() || type == QuantizationType::NONE || type == QuantizationType::FP32) {
        return;
    }

    uint32_t size = static_cast<uint32_t>(activations.size());

    if (type == QuantizationType::INT8) {
        float scale = compute_per_tensor_scale(activations.data(), size, type);
        std::vector<int8_t> quantized(size);
        quantize_fp32_to_int8(activations.data(), quantized.data(), size, scale);

        for (uint32_t i = 0; i < size; ++i) {
            activations[i] = static_cast<float>(quantized[i]) * scale;
        }
    }
    else if (type == QuantizationType::FP16) {
        for (uint32_t i = 0; i < size; ++i) {
            float f = std::max(-65504.0f, std::min(65504.0f, activations[i]));
            uint16_t* fp16 = reinterpret_cast<uint16_t*>(&f);
            uint32_t f32 = *reinterpret_cast<uint32_t*>(&f);
            uint32_t f16 = ((f32 >> 16) & 0x8000) | ((((f32 & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((f32 >> 13) & 0x03ff);
            activations[i] = *reinterpret_cast<float*>(&f16);
        }
    }
}

std::vector<float> QuantizationManager::dequantize_to_fp32(const ggml::Tensor* tensor, QuantizationType source_type) {
    if (!tensor) {
        return std::vector<float>();
    }

    ggml::Tensor* tensor_copy = const_cast<ggml::Tensor*>(tensor);
    dequantize_tensor(tensor_copy, source_type);

    const float* data = static_cast<const float*>(tensor_copy->get_cpu_data());
    if (!data) {
        return std::vector<float>();
    }

    const auto& shape = tensor->get_shape();
    uint32_t size = 1;
    for (uint32_t dim : shape) {
        size *= dim;
    }

    return std::vector<float>(data, data + size);
}

void QuantizationManager::calibrate_quantization(const std::map<std::string, ggml::Tensor*>& weights) {
    if (!config_.calibration_enabled) {
        return;
    }

    std::cout << "[QuantizationManager] Starting quantization calibration with "
              << config_.calibration_steps << " steps" << std::endl;

    for (const auto& pair : weights) {
        const std::string& name = pair.first;
        ggml::Tensor* tensor = pair.second;

        if (!tensor || !tensor->get_cpu_data()) {
            continue;
        }

        const float* data = static_cast<const float*>(tensor->get_cpu_data());
        const auto& shape = tensor->get_shape();
        uint32_t size = 1;
        for (uint32_t dim : shape) {
            size *= dim;
        }

        if (config_.weight_quantization == QuantizationType::INT8) {
            float scale = compute_entropy_scale(data, size);
            tensor_scales_[name] = scale;
        }
        else if (config_.weight_quantization == QuantizationType::FP4) {
            float scale = compute_entropy_scale(data, size);
            tensor_scales_[name] = scale;
        }
        else if (config_.weight_quantization == QuantizationType::NF4) {
            float scale = compute_entropy_scale(data, size);
            tensor_scales_[name] = scale;
        }
    }

    std::cout << "[QuantizationManager] Calibration complete for "
              << tensor_scales_.size() << " tensors" << std::endl;
}

bool QuantizationManager::is_quantization_supported(QuantizationType type) const {
    switch (type) {
        case QuantizationType::NONE:
        case QuantizationType::FP32:
        case QuantizationType::FP16:
        case QuantizationType::INT8:
            return true;
        case QuantizationType::FP4:
        case QuantizationType::NF4:
            return initialized_;
        default:
            return false;
    }
}

float QuantizationManager::compute_per_tensor_scale(const float* data, uint32_t size, QuantizationType type) {
    float min_val, max_val;
    return compute_min_max_scale(data, size, min_val, max_val);
}

std::vector<float> QuantizationManager::compute_per_channel_scale(const float* data, uint32_t rows, uint32_t cols, QuantizationType type) {
    std::vector<float> scales(rows);

    for (uint32_t i = 0; i < rows; ++i) {
        const float* channel_data = data + i * cols;
        float min_val, max_val;
        scales[i] = compute_min_max_scale(channel_data, cols, min_val, max_val);
    }

    return scales;
}

void QuantizationManager::quantize_fp32_to_int8(const float* input, int8_t* output, uint32_t size, float scale) {
    for (uint32_t i = 0; i < size; ++i) {
        float clamped = std::max(-128.0f, std::min(127.0f, input[i] / scale));
        output[i] = static_cast<int8_t>(std::round(clamped));
    }
}

void QuantizationManager::dequantize_int8_to_fp32(const int8_t* input, float* output, uint32_t size, float scale) {
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

void QuantizationManager::quantize_fp32_to_fp4(const float* input, uint8_t* output, uint32_t size, float scale) {
    for (uint32_t i = 0; i < size; i += 2) {
        float f0 = input[i] / scale;
        float f1 = (i + 1 < size) ? input[i + 1] / scale : 0.0f;

        uint8_t q0 = static_cast<uint8_t>(std::max(0.0f, std::min(7.0f, std::round(f0 + 8.0f)));
        uint8_t q1 = static_cast<uint8_t>(std::max(0.0f, std::min(7.0f, std::round(f1 + 8.0f))));

        output[i / 2] = (q0 << 4) | (q1 & 0x0F);
    }
}

void QuantizationManager::dequantize_fp4_to_fp32(const uint8_t* input, float* output, uint32_t size, float scale) {
    for (uint32_t i = 0; i < size; i += 2) {
        uint8_t packed = input[i / 2];
        uint8_t q0 = (packed >> 4) & 0x0F;
        uint8_t q1 = packed & 0x0F;

        output[i] = (static_cast<float>(q0) - 8.0f) * scale;
        if (i + 1 < size) {
            output[i + 1] = (static_cast<float>(q1) - 8.0f) * scale;
        }
    }
}

void QuantizationManager::quantize_nf4(const float* input, uint8_t* output, uint32_t size) {
    for (uint32_t i = 0; i < size; i += 2) {
        float f0 = input[i];
        float f1 = (i + 1 < size) ? input[i + 1] : 0.0f;

        uint8_t q0 = 0;
        uint8_t q1 = 0;
        float min_dist0 = std::numeric_limits<float>::max();
        float min_dist1 = std::numeric_limits<float>::max();

        for (int j = 0; j < 16; ++j) {
            float nf4_val = NF4_TABLE[j];
            float dist0 = std::abs(f0 - nf4_val);
            if (dist0 < min_dist0) {
                min_dist0 = dist0;
                q0 = static_cast<uint8_t>(j);
            }

            float dist1 = std::abs(f1 - nf4_val);
            if (dist1 < min_dist1) {
                min_dist1 = dist1;
                q1 = static_cast<uint8_t>(j);
            }
        }

        output[i / 2] = (q0 << 4) | (q1 & 0x0F);
    }
}

void QuantizationManager::dequantize_nf4(const uint8_t* input, float* output, uint32_t size) {
    for (uint32_t i = 0; i < size; i += 2) {
        uint8_t packed = input[i / 2];
        uint8_t q0 = (packed >> 4) & 0x0F;
        uint8_t q1 = packed & 0x0F;

        output[i] = NF4_TABLE[q0];
        if (i + 1 < size) {
            output[i + 1] = NF4_TABLE[q1];
        }
    }
}

std::vector<float> QuantizationManager::generate_nf4_lookup_table() {
    std::vector<float> table(16);
    for (int i = 0; i < 16; ++i) {
        table[i] = NF4_TABLE[i];
    }
    return table;
}

float QuantizationManager::compute_min_max_scale(const float* data, uint32_t size, float& min_val, float& max_val) {
    min_val = data[0];
    max_val = data[0];

    for (uint32_t i = 1; i < size; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    float range = max_val - min_val;
    if (range < 1e-6f) {
        return 1.0f;
    }

    return range / 255.0f;
}

float QuantizationManager::compute_entropy_scale(const float* data, uint32_t size) {
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint32_t i = 0; i < size; ++i) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }

    float mean = sum / size;
    float std_dev = std::sqrt((sum_sq / size) - (mean * mean));

    return std_dev * 3.0f / 127.0f;
}

}
