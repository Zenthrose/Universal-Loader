#include "lru_cache.h"

namespace inference {

LRU_Cache::LRU_Cache(size_t max_layers) : max_layers_(max_layers) {
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

}
