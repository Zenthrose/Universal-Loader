#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include "../core/tensor.h"

namespace inference {

enum class QuantizationType {
    NONE,
    FP32,
    FP16,
    INT8,
    FP4,
    NF4
};

struct QuantizationConfig {
    QuantizationType weight_quantization;
    QuantizationType activation_quantization;
    QuantizationType kv_cache_quantization;

    bool calibration_enabled;
    uint32_t calibration_steps;

    float fp4_block_size;
    float nf4_block_size;

    QuantizationConfig()
        : weight_quantization(QuantizationType::FP16)
        , activation_quantization(QuantizationType::FP16)
        , kv_cache_quantization(QuantizationType::FP16)
        , calibration_enabled(false)
        , calibration_steps(100)
        , fp4_block_size(256.0f)
        , nf4_block_size(256.0f)
    {}
};

class QuantizationManager {
public:
    QuantizationManager();
    ~QuantizationManager();

    void initialize(const QuantizationConfig& config);

    void quantize_tensor(ggml::Tensor* tensor, QuantizationType target_type);
    void dequantize_tensor(ggml::Tensor* tensor, QuantizationType source_type);

    void quantize_weights(ggml::Tensor* weights, QuantizationType type);
    void quantize_activations(std::vector<float>& activations, QuantizationType type);

    std::vector<float> dequantize_to_fp32(const ggml::Tensor* tensor, QuantizationType source_type);

    void calibrate_quantization(const std::map<std::string, ggml::Tensor*>& weights);

    bool is_quantization_supported(QuantizationType type) const;

    QuantizationConfig get_config() const { return config_; }

    static float compute_per_tensor_scale(const float* data, uint32_t size, QuantizationType type);
    static std::vector<float> compute_per_channel_scale(const float* data, uint32_t rows, uint32_t cols, QuantizationType type);

private:
    void quantize_fp32_to_int8(const float* input, int8_t* output, uint32_t size, float scale);
    void dequantize_int8_to_fp32(const int8_t* input, float* output, uint32_t size, float scale);

    void quantize_fp32_to_fp4(const float* input, uint8_t* output, uint32_t size, float scale);
    void dequantize_fp4_to_fp32(const uint8_t* input, float* output, uint32_t size, float scale);

    void quantize_nf4(const float* input, uint8_t* output, uint32_t size);
    void dequantize_nf4(const uint8_t* input, float* output, uint32_t size);

    std::vector<float> generate_nf4_lookup_table();

    static float compute_min_max_scale(const float* data, uint32_t size, float& min_val, float& max_val);
    static float compute_entropy_scale(const float* data, uint32_t size);

    QuantizationConfig config_;
    std::map<std::string, float> tensor_scales_;
    std::map<std::string, std::vector<float>> per_channel_scales_;

    bool initialized_;

    static constexpr float NF4_TABLE[16] = {
        -1.0f, -0.696192803978, -0.525072057247, -0.394917488694,
        -0.284441382885, -0.184773430284, -0.0910502361056, 0.0f,
        0.079580299556, 0.160930201411, 0.246112301945, 0.337915241718,
        0.440709829331, 0.562617003918, 0.722956836224, 1.0f
    };
};

}
