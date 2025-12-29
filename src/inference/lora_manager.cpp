#include "lora_manager.h"
#include <algorithm>
#include <cstring>

namespace inference {

LoRAManager::LoRAManager(uint32_t hidden_dim, uint32_t vocab_size)
    : hidden_dim_(hidden_dim), vocab_size_(vocab_size) {
}

LoRAManager::~LoRAManager() {
}

bool LoRAManager::load_adapter(const std::string& filepath, const std::string& name) {
    std::lock_guard<std::mutex> lock(adapter_mutex_);

    LoRAAdapter adapter;
    adapter.name = name;
    adapter.alpha = 1.0f;
    adapter.rank = 0.0f;
    adapter.target_layer = "";

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint32_t dims[2];
    file.read(reinterpret_cast<char*>(dims), sizeof(dims));
    adapter.rank = static_cast<float>(dims[0]);

    uint32_t num_elements = dims[1];
    adapter.weights_A.resize(num_elements);
    adapter.weights_B.resize(num_elements);

    file.read(reinterpret_cast<char*>(adapter.weights_A.data()),
            num_elements * sizeof(float));
    file.read(reinterpret_cast<char*>(adapter.weights_B.data()),
            num_elements * sizeof(float));

    file.close();

    loaded_adapters_[name] = adapter;
    return true;
}

void LoRAManager::unload_adapter(const std::string& name) {
    std::lock_guard<std::mutex> lock(adapter_mutex_);

    auto it = loaded_adapters_.find(name);
    if (it != loaded_adapters_.end()) {
        loaded_adapters_.erase(it);
    }

    enabled_adapters_.erase(name);
    adapter_alphas_.erase(name);
}

void LoRAManager::enable_adapter(const std::string& name, float alpha) {
    std::lock_guard<std::mutex> lock(adapter_mutex_);

    if (loaded_adapters_.find(name) == loaded_adapters_.end()) {
        return;
    }

    enabled_adapters_[name] = true;
    adapter_alphas_[name] = alpha;
}

void LoRAManager::disable_adapter(const std::string& name) {
    std::lock_guard<std::mutex> lock(adapter_mutex_);
    enabled_adapters_.erase(name);
}

std::vector<float> LoRAManager::apply_adapters(const float* base_weights,
                                             const std::string& layer_name,
                                             uint32_t tensor_size) {
    std::lock_guard<std::mutex> lock(adapter_mutex_);

    std::vector<float> merged_weights(tensor_size);
    memcpy(merged_weights.data(), base_weights, tensor_size * sizeof(float));

    for (const auto& [name, adapter] : loaded_adapters_) {
        if (enabled_adapters_[name]) {
            float alpha = adapter_alphas_[name];

            uint32_t A_size = static_cast<uint32_t>(adapter.weights_A.size());
            uint32_t B_size = static_cast<uint32_t>(adapter.weights_B.size());

            uint32_t rank = static_cast<uint32_t>(adapter.rank);

            std::vector<float> delta(tensor_size);

            for (uint32_t i = 0; i < tensor_size; ++i) {
                float sum = 0.0f;

                for (uint32_t r = 0; r < rank; ++r) {
                    sum += adapter.weights_A[i * rank + r] * adapter.weights_B[r * tensor_size + i];
                }

                delta[i] = sum * alpha;
            }

            for (uint32_t i = 0; i < tensor_size; ++i) {
                merged_weights[i] += delta[i];
            }
        }
    }

    return merged_weights;
}

std::vector<std::string> LoRAManager::get_loaded_adapters() const {
    std::lock_guard<std::mutex> lock(adapter_mutex_);

    std::vector<std::string> names;
    for (const auto& [name, _] : loaded_adapters_) {
        names.push_back(name);
    }

    return names;
}

std::vector<std::string> LoRAManager::get_enabled_adapters() const {
    std::lock_guard<std::mutex> lock(adapter_mutex_);

    std::vector<std::string> names;
    for (const auto& [name, enabled] : enabled_adapters_) {
        if (enabled) {
            names.push_back(name);
        }
    }

    return names;
}

float LoRAManager::get_effective_rank(const std::string& name) const {
    std::lock_guard<std::mutex> lock(adapter_mutex_);

    auto it = loaded_adapters_.find(name);
    if (it != loaded_adapters_.end()) {
        return it->second.rank;
    }

    return 0.0f;
}

void LoRAManager::set_adapter_alpha(const std::string& name, float alpha) {
    std::lock_guard<std::mutex> lock(adapter_mutex_);
    adapter_alphas_[name] = alpha;
}

std::vector<float> WeightMergingStrategy::merge_weights(
    const std::vector<std::pair<std::string, float>>& weighted_adapters,
    const float* base_weights,
    uint32_t tensor_size,
    Method method) {

    std::vector<float*> adapter_weight_ptrs;

    for (const auto& [name, weight] : weighted_adapters) {
        adapter_weight_ptrs.push_back(const_cast<float*>(weight));
    }

    std::vector<float> weights;
    for (const auto& [_, weight] : weighted_adapters) {
        weights.push_back(weight);
    }

    switch (method) {
        case Method::LINEAR:
            return linear_merge(adapter_weight_ptrs, weights, base_weights, tensor_size);

        case Method::ADDITIVE:
            return additive_merge(adapter_weight_ptrs, weights, base_weights, tensor_size);

        case Method::WEIGHTED_AVERAGE:
            return weighted_average(adapter_weight_ptrs, weights, base_weights, tensor_size);

        case Method::TIES:
            return ties_merge(adapter_weight_ptrs, weights, base_weights, tensor_size);
    }

    return std::vector<float>(tensor_size);
}

std::vector<float> WeightMergingStrategy::linear_merge(
    const std::vector<float*>& adapter_weights,
    const std::vector<float>& weights,
    const float* base,
    uint32_t size) {

    std::vector<float> result(size);

    for (uint32_t i = 0; i < size; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < adapter_weights.size(); ++j) {
            sum += adapter_weights[j][i];
        }
        result[i] = base[i] + sum / static_cast<float>(adapter_weights.size());
    }

    return result;
}

std::vector<float> WeightMergingStrategy::additive_merge(
    const std::vector<float*>& adapter_weights,
    const std::vector<float>& weights,
    const float* base,
    uint32_t size) {

    std::vector<float> result(size);
    memcpy(result.data(), base, size * sizeof(float));

    for (uint32_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < adapter_weights.size(); ++j) {
            result[i] += adapter_weights[j][i];
        }
    }

    return result;
}

std::vector<float> WeightMergingStrategy::weighted_average(
    const std::vector<float*>& adapter_weights,
    const std::vector<float>& weights,
    const float* base,
    uint32_t size) {

    std::vector<float> result(size);

    for (uint32_t i = 0; i < size; ++i) {
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;

        for (size_t j = 0; j < adapter_weights.size(); ++j) {
            weighted_sum += weights[j] * adapter_weights[j][i];
            weight_sum += weights[j];
        }

        result[i] = base[i] + weighted_sum / weight_sum;
    }

    return result;
}

std::vector<float> WeightMergingStrategy::ties_merge(
    const std::vector<float*>& adapter_weights,
    const std::vector<float>& weights,
    const float* base,
    uint32_t size) {

    std::vector<float> result(size);

    for (uint32_t i = 0; i < size; ++i) {
        float sum = 0.0f;
        float abs_sum = 0.0f;

        for (size_t j = 0; j < adapter_weights.size(); ++j) {
            float delta = adapter_weights[j][i] - base[i];
            sum += weights[j] * delta;
            abs_sum += weights[j] * std::abs(delta);
        }

        float sign = (sum >= 0) ? 1.0f : -1.0f;
        result[i] = base[i] + sign * std::min(std::abs(sum), abs_sum) / abs_sum;
    }

    return result;
}

}
