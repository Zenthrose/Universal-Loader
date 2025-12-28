#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/tensor.h"
#include "lru_cache.h"

namespace inference {

class OffloadManager {
public:
    OffloadManager();
    ~OffloadManager();

    void set_model(const std::unordered_map<std::string, ggml::Tensor*>& weights);
    void set_gpu_cache_size(size_t size_mb);

    bool should_offload(ggml::Tensor* tensor) const;
    void offload_tensor(ggml::Tensor* tensor, bool to_gpu);
    void update_layer_access(uint32_t layer_id);

private:
    size_t estimate_size(ggml::Tensor* tensor) const;
    void calculate_layer_priorities();

    std::unordered_map<std::string, ggml::Tensor*> weights_;
    size_t gpu_cache_size_;
    std::vector<std::pair<uint32_t, float>> layer_priorities_;
};

}
