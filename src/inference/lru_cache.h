#pragma once
#include <vulkan/vulkan.h>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/tensor.h"

namespace inference {

struct CachedLayer {
    ggml::Tensor* q_proj;
    ggml::Tensor* k_proj;
    ggml::Tensor* v_proj;
    ggml::Tensor* o_proj;
    ggml::Tensor* gate_proj;
    ggml::Tensor* up_proj;
    ggml::Tensor* down_proj;
    uint32_t layer_id;
    uint32_t access_count;
    size_t size_bytes;
};

class LRU_Cache {
public:
    explicit LRU_Cache(size_t max_layers);
    ~LRU_Cache();

    bool prefetch_layer(uint32_t layer_id, ggml::Tensor* layer_weights[7]);
    bool get_layer(uint32_t layer_id, CachedLayer& layer);
    void evict_layer(uint32_t layer_id);

    size_t get_cached_count() const { return cache_.size(); }
    size_t get_total_size_bytes() const { return total_size_bytes_; }

    void defragment();
    void compact_cache();
    void evict_least_used(uint32_t target_bytes);

private:
    void update_access(uint32_t layer_id);

    size_t max_layers_;
    std::list<uint32_t> access_order_;
    std::unordered_map<uint32_t, CachedLayer> cache_;
    size_t total_size_bytes_;
};

}
