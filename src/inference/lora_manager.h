#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <mutex>

namespace inference {

struct LoRAAdapter {
    std::string name;
    std::vector<float> weights_A;
    std::vector<float> weights_B;
    float alpha;
    float rank;
    std::string target_layer;
};

class LoRAManager {
public:
    LoRAManager(uint32_t hidden_dim, uint32_t vocab_size);
    ~LoRAManager();

    bool load_adapter(const std::string& filepath, const std::string& name);
    void unload_adapter(const std::string& name);
    void enable_adapter(const std::string& name, float alpha = 1.0f);
    void disable_adapter(const std::string& name);

    std::vector<float> apply_adapters(const float* base_weights,
                                     const std::string& layer_name,
                                     uint32_t tensor_size);

    std::vector<std::string> get_loaded_adapters() const;
    std::vector<std::string> get_enabled_adapters() const;

    float get_effective_rank(const std::string& name) const;
    void set_adapter_alpha(const std::string& name, float alpha);

private:
    float apply_lora_weights(const float* base, const float* A, const float* B,
                           uint32_t size, float alpha);

    uint32_t hidden_dim_;
    uint32_t vocab_size_;

    std::unordered_map<std::string, LoRAAdapter> loaded_adapters_;
    std::unordered_map<std::string, bool> enabled_adapters_;
    std::unordered_map<std::string, float> adapter_alphas_;

    mutable std::mutex adapter_mutex_;
};

class WeightMergingStrategy {
public:
    enum class Method {
        LINEAR,
        ADDITIVE,
        WEIGHTED_AVERAGE,
        TIES
    };

    static std::vector<float> merge_weights(
        const std::vector<std::pair<std::string, float>>& weighted_adapters,
        const float* base_weights,
        uint32_t tensor_size,
        Method method = Method::WEIGHTED_AVERAGE
    );

private:
    static std::vector<float> linear_merge(
        const std::vector<float*>& adapter_weights,
        const std::vector<float>& weights,
        const float* base,
        uint32_t size
    );

    static std::vector<float> weighted_average(
        const std::vector<float*>& adapter_weights,
        const std::vector<float>& weights,
        const float* base,
        uint32_t size
    );

    static std::vector<float> ties_merge(
        const std::vector<float*>& adapter_weights,
        const std::vector<float>& weights,
        const float* base,
        uint32_t size
    );
};

}
