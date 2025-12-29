#include "dynamic_quantization.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace inference {

using json = nlohmann::json;

DynamicQuantizationManager::DynamicQuantizationManager()
    : total_layers_(0)
    , initialized_(false) {
}

DynamicQuantizationManager::~DynamicQuantizationManager() {
}

void DynamicQuantizationManager::initialize(const DynamicQuantizationConfig& config) {
    config_ = config;
    initialized_ = true;

    std::cout << "[DynamicQuantization] Initialized with adaptive quantization: "
              << (config_.enable_adaptive ? "enabled" : "disabled") << std::endl;
}

QuantizationType DynamicQuantizationManager::get_layer_quantization(uint32_t layer_id, TensorType tensor_type) const {
    if (!initialized_) {
        return QuantizationType::FP16;
    }

    auto key = std::make_pair(layer_id, tensor_type);
    auto it = tensor_quantization_map_.find(key);
    if (it != tensor_quantization_map_.end()) {
        return it->second;
    }

    auto layer_it = layer_quantization_map_.find(layer_id);
    if (layer_it != layer_quantization_map_.end()) {
        return layer_it->second;
    }

    auto tensor_it = config_.per_tensor_precision.find(tensor_type);
    if (tensor_it != config_.per_tensor_precision.end()) {
        return tensor_it->second;
    }

    return QuantizationType::FP16;
}

void DynamicQuantizationManager::analyze_layer_sensitivity(const std::vector<float>& weights, uint32_t layer_id) {
    if (!initialized_ || !config_.enable_adaptive) {
        return;
    }

    float sensitivity = compute_sensitivity_score(weights);
    update_layer_importance(layer_id, sensitivity);

    QuantizationType q_type = select_quantization_for_score(sensitivity, layer_id);
    layer_quantization_map_[layer_id] = q_type;

    std::cout << "[DynamicQuantization] Layer " << layer_id
              << " sensitivity: " << sensitivity
              << ", quantization: " << static_cast<int>(q_type) << std::endl;
}

void DynamicQuantizationManager::calibrate_quantization_levels(uint32_t num_layers) {
    total_layers_ = num_layers;

    if (config_.enable_adaptive) {
        std::vector<std::pair<float, uint32_t>> importance_list;
        for (const auto& pair : layer_importance_scores_) {
            importance_list.push_back({pair.second, pair.first});
        }

        std::sort(importance_list.begin(), importance_list.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        uint32_t critical_count = static_cast<uint32_t>(num_layers * 0.3f);
        for (uint32_t i = 0; i < std::min(critical_count, static_cast<uint32_t>(importance_list.size())); ++i) {
            uint32_t layer_id = importance_list[i].second;
            if (layer_id < config_.min_layer_quantization || layer_id > config_.max_layer_quantization) {
                continue;
            }
            layer_quantization_map_[layer_id] = QuantizationType::FP16;
        }

        uint32_t high_importance_count = static_cast<uint32_t>(num_layers * 0.3f);
        for (uint32_t i = critical_count; i < critical_count + high_importance_count && i < importance_list.size(); ++i) {
            uint32_t layer_id = importance_list[i].second;
            if (layer_id < config_.min_layer_quantization || layer_id > config_.max_layer_quantization) {
                continue;
            }
            layer_quantization_map_[layer_id] = QuantizationType::INT8;
        }

        for (uint32_t i = critical_count + high_importance_count; i < importance_list.size(); ++i) {
            uint32_t layer_id = importance_list[i].second;
            if (layer_id < config_.min_layer_quantization || layer_id > config_.max_layer_quantization) {
                continue;
            }
            layer_quantization_map_[layer_id] = QuantizationType::FP4;
        }

        std::cout << "[DynamicQuantization] Calibrated " << num_layers
                  << " layers with adaptive quantization" << std::endl;
    }
}

void DynamicQuantizationManager::update_quantization_for_layer(uint32_t layer_id, TensorType tensor_type, QuantizationType type) {
    if (!initialized_) {
        return;
    }

    auto key = std::make_pair(layer_id, tensor_type);
    tensor_quantization_map_[key] = type;
}

void DynamicQuantizationManager::set_tensor_precision(TensorType tensor_type, QuantizationType type) {
    if (!initialized_) {
        return;
    }

    config_.per_tensor_precision[tensor_type] = type;
}

float DynamicQuantizationManager::get_layer_importance(uint32_t layer_id) const {
    auto it = layer_importance_scores_.find(layer_id);
    if (it != layer_importance_scores_.end()) {
        return it->second;
    }
    return 0.0f;
}

std::vector<uint32_t> DynamicQuantizationManager::get_critical_layers() const {
    std::vector<uint32_t> critical_layers;

    for (const auto& pair : layer_importance_scores_) {
        if (pair.second >= CRITICAL_LAYER_THRESHOLD) {
            critical_layers.push_back(pair.first);
        }
    }

    std::sort(critical_layers.begin(), critical_layers.end());
    return critical_layers;
}

void DynamicQuantizationManager::export_quantization_map(const std::string& path) {
    std::ofstream file(path);
    file << "enable_adaptive:" << (config_.enable_adaptive ? "1" : "0") << "\n";
    file << "sensitivity_threshold:" << config_.sensitivity_threshold << "\n";
    file << "min_layer_quantization:" << config_.min_layer_quantization << "\n";
    file << "max_layer_quantization:" << config_.max_layer_quantization << "\n";

    file << "per_tensor_precision:\n";
    for (const auto& pair : config_.per_tensor_precision) {
        file << "  " << static_cast<int>(pair.first) << ":" << static_cast<int>(pair.second) << "\n";
    }

    file << "layer_quantization_map:\n";
    for (const auto& pair : layer_quantization_map_) {
        file << "  " << pair.first << ":" << static_cast<int>(pair.second) << "\n";
    }

    file << "layer_importance_scores:\n";
    for (const auto& pair : layer_importance_scores_) {
        file << "  " << pair.first << ":" << pair.second << "\n";
    }

    file.close();
    std::cout << "[DynamicQuantization] Exported quantization map to " << path << std::endl;
}

void DynamicQuantizationManager::import_quantization_map(const std::string& path) {
    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);

        if (key == "enable_adaptive") {
            config_.enable_adaptive = (value == "1");
        } else if (key == "sensitivity_threshold") {
            config_.sensitivity_threshold = std::stof(value);
        } else if (key == "min_layer_quantization") {
            config_.min_layer_quantization = std::stoul(value);
        } else if (key == "max_layer_quantization") {
            config_.max_layer_quantization = std::stoul(value);
        }
    }

    file.close();
    std::cout << "[DynamicQuantization] Imported quantization map from " << path << std::endl;
}

float DynamicQuantizationManager::compute_sensitivity_score(const std::vector<float>& weights) {
    if (weights.empty()) {
        return 0.0f;
    }

    float mean = 0.0f;
    for (float w : weights) {
        mean += w;
    }
    mean /= weights.size();

    float variance = 0.0f;
    for (float w : weights) {
        variance += (w - mean) * (w - mean);
    }
    variance /= weights.size();

    float std_dev = std::sqrt(variance);

    float skewness = 0.0f;
    for (float w : weights) {
        float diff = (w - mean) / (std_dev + 1e-6f);
        skewness += diff * diff * diff;
    }
    skewness /= weights.size();

    float kurtosis = 0.0f;
    for (float w : weights) {
        float diff = (w - mean) / (std_dev + 1e-6f);
        kurtosis += diff * diff * diff * diff;
    }
    kurtosis /= weights.size();

    float sensitivity = 0.4f * std_dev + 0.3f * std::abs(skewness) + 0.2f * kurtosis + 0.1f * (variance / (mean * mean + 1e-6f));

    sensitivity = std::max(0.0f, std::min(1.0f, sensitivity));

    return sensitivity;
}

QuantizationType DynamicQuantizationManager::select_quantization_for_score(float score, uint32_t layer_id) {
    if (layer_id < config_.min_layer_quantization || layer_id > config_.max_layer_quantization) {
        return QuantizationType::FP16;
    }

    if (score >= CRITICAL_LAYER_THRESHOLD) {
        return QuantizationType::FP16;
    }

    if (score >= HIGH_IMPORTANCE_THRESHOLD) {
        return QuantizationType::INT8;
    }

    if (score >= config_.sensitivity_threshold) {
        return QuantizationType::FP4;
    }

    return QuantizationType::NF4;
}

void DynamicQuantizationManager::update_layer_importance(uint32_t layer_id, float importance) {
    layer_importance_scores_[layer_id] = importance;
}

}
