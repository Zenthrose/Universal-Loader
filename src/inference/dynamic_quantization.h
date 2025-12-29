#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "quantization_manager.h"

namespace inference {

enum class TensorType {
    WEIGHT_Q_PROJ,
    WEIGHT_K_PROJ,
    WEIGHT_V_PROJ,
    WEIGHT_O_PROJ,
    WEIGHT_GATE_PROJ,
    WEIGHT_UP_PROJ,
    WEIGHT_DOWN_PROJ,
    WEIGHT_NORM1,
    WEIGHT_NORM2,
    ACTIVATION,
    KV_CACHE,
    UNKNOWN
};

struct DynamicQuantizationConfig {
    bool enable_adaptive;
    float sensitivity_threshold;
    uint32_t min_layer_quantization;
    uint32_t max_layer_quantization;

    std::map<TensorType, QuantizationType> per_tensor_precision;
    std::map<uint32_t, QuantizationType> per_layer_precision;

    DynamicQuantizationConfig()
        : enable_adaptive(false)
        , sensitivity_threshold(0.01f)
        , min_layer_quantization(0)
        , max_layer_quantization(UINT32_MAX)
    {
        per_tensor_precision[TensorType::WEIGHT_Q_PROJ] = QuantizationType::INT8;
        per_tensor_precision[TensorType::WEIGHT_K_PROJ] = QuantizationType::INT8;
        per_tensor_precision[TensorType::WEIGHT_V_PROJ] = QuantizationType::INT8;
        per_tensor_precision[TensorType::WEIGHT_O_PROJ] = QuantizationType::INT8;
        per_tensor_precision[TensorType::WEIGHT_GATE_PROJ] = QuantizationType::FP4;
        per_tensor_precision[TensorType::WEIGHT_UP_PROJ] = QuantizationType::FP4;
        per_tensor_precision[TensorType::WEIGHT_DOWN_PROJ] = QuantizationType::INT8;
        per_tensor_precision[TensorType::WEIGHT_NORM1] = QuantizationType::NONE;
        per_tensor_precision[TensorType::WEIGHT_NORM2] = QuantizationType::NONE;
        per_tensor_precision[TensorType::ACTIVATION] = QuantizationType::FP16;
        per_tensor_precision[TensorType::KV_CACHE] = QuantizationType::FP4;
    }
};

class DynamicQuantizationManager {
public:
    DynamicQuantizationManager();
    ~DynamicQuantizationManager();

    void initialize(const DynamicQuantizationConfig& config);

    QuantizationType get_layer_quantization(uint32_t layer_id, TensorType tensor_type) const;

    void analyze_layer_sensitivity(const std::vector<float>& weights, uint32_t layer_id);
    void calibrate_quantization_levels(uint32_t num_layers);

    void update_quantization_for_layer(uint32_t layer_id, TensorType tensor_type, QuantizationType type);
    void set_tensor_precision(TensorType tensor_type, QuantizationType type);

    float get_layer_importance(uint32_t layer_id) const;
    std::vector<uint32_t> get_critical_layers() const;

    DynamicQuantizationConfig get_config() const { return config_; }

    void export_quantization_map(const std::string& path);
    void import_quantization_map(const std::string& path);

private:
    float compute_sensitivity_score(const std::vector<float>& weights);
    QuantizationType select_quantization_for_score(float score, uint32_t layer_id);
    void update_layer_importance(uint32_t layer_id, float importance);

    DynamicQuantizationConfig config_;

    std::map<uint32_t, QuantizationType> layer_quantization_map_;
    std::map<uint32_t, float> layer_importance_scores_;
    std::map<std::pair<uint32_t, TensorType>, QuantizationType> tensor_quantization_map_;

    uint32_t total_layers_;
    bool initialized_;

    static constexpr float CRITICAL_LAYER_THRESHOLD = 0.7f;
    static constexpr float HIGH_IMPORTANCE_THRESHOLD = 0.5f;
};

}
