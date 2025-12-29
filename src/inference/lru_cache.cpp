#include "lru_cache.h"
#include <algorithm>

namespace inference {

LRU_Cache::LRU_Cache(size_t max_layers) : max_layers_(max_layers), total_size_bytes_(0) {
}

LRU_Cache::~LRU_Cache() {
}

bool LRU_Cache::prefetch_layer(uint32_t layer_id, ggml::Tensor* layer_weights[7]) {
    if (cache_.find(layer_id) != cache_.end()) {
        update_access(layer_id);
        return true;
    }
    
    if (cache_.size() >= max_layers_) {
        uint32_t evict_id = access_order_.front();
        access_order_.pop_front();
        cache_.erase(evict_id);
    }
    
    CachedLayer cached;
    cached.layer_id = layer_id;
    cached.q_proj = layer_weights[0];
    cached.k_proj = layer_weights[1];
    cached.v_proj = layer_weights[2];
    cached.o_proj = layer_weights[3];
    cached.gate_proj = layer_weights[4];
    cached.up_proj = layer_weights[5];
    cached.down_proj = layer_weights[6];
    
    cache_[layer_id] = cached;
    access_order_.push_back(layer_id);
    
    return true;
}

bool LRU_Cache::get_layer(uint32_t layer_id, CachedLayer& layer) {
    auto it = cache_.find(layer_id);
    if (it != cache_.end()) {
        layer = it->second;
        update_access(layer_id);
        return true;
    }
    return false;
}

void LRU_Cache::evict_layer(uint32_t layer_id) {
    cache_.erase(layer_id);
    access_order_.remove(layer_id);
}

void LRU_Cache::update_access(uint32_t layer_id) {
    access_order_.remove(layer_id);
    access_order_.push_back(layer_id);
}

void LRU_Cache::defragment() {
    std::vector<uint32_t> hot_layers;

    for (auto& [id, layer] : cache_) {
        if (layer.access_count > 10) {
            hot_layers.push_back(id);
        }
    }

    access_order_.clear();
    for (uint32_t id : hot_layers) {
        access_order_.push_back(id);
    }
}

void LRU_Cache::compact_cache() {
    std::vector<std::pair<uint32_t, uint32_t>> layer_access_counts;

    for (auto& [id, layer] : cache_) {
        layer_access_counts.push_back({id, layer.access_count});
    }

    std::sort(layer_access_counts.begin(), layer_access_counts.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    size_t target_keep = max_layers_ * 0.8;
    size_t keep_count = std::min(layer_access_counts.size(), target_keep);

    std::unordered_map<uint32_t, CachedLayer> new_cache;
    std::list<uint32_t> new_order;

    for (size_t i = 0; i < keep_count; ++i) {
        uint32_t id = layer_access_counts[i].first;
        new_cache[id] = cache_[id];
        new_order.push_back(id);
    }

    cache_ = std::move(new_cache);
    access_order_ = std::move(new_order);
}

void LRU_Cache::evict_least_used(uint32_t target_bytes) {
    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes;

    for (auto& [id, layer] : cache_) {
        layer_sizes.push_back({id, layer.access_count});
    }

    std::sort(layer_sizes.begin(), layer_sizes.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });

    for (auto& [id, count] : layer_sizes) {
        if (total_size_bytes_ <= target_bytes) {
            break;
        }

        auto it = cache_.find(id);
        if (it != cache_.end()) {
            total_size_bytes_ -= it->second.size_bytes;
            cache_.erase(it);
            access_order_.remove(id);
        }
    }
}

}
