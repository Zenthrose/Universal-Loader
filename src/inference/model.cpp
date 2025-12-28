#include "model.h"
#include "../core/gguf_parser.h"
#include <algorithm>
#include <sstream>

namespace inference {

KVCache::KVCache(uint32_t layers, uint32_t hidden_dim, uint32_t context_len)
    : layers_(layers), hidden_dim_(hidden_dim), context_len_(context_len) {
    k_cache_.resize(layers_);
    v_cache_.resize(layers_);
}

KVCache::~KVCache() {
    for (auto& k : k_cache_) {
        if (k) {
            delete[] k;
        }
    }
    for (auto& v : v_cache_) {
        if (v) {
            delete[] v;
        }
    }
}

void KVCache::allocate_gpu() {
    for (uint32_t layer = 0; layer < layers_; ++layer) {
        k_cache_[layer] = new float[context_len_ * hidden_dim_]();
        v_cache_[layer] = new float[context_len_ * hidden_dim_]();
    }
}

void KVCache::store(uint32_t layer, uint32_t pos, const float* k, const float* v) {
    if (layer >= layers_ || pos >= context_len_) return;
    
    float* k_layer = k_cache_[layer];
    float* v_layer = v_cache_[layer];
    
    memcpy(k_layer + pos * hidden_dim_, k, hidden_dim_ * sizeof(float));
    memcpy(v_layer + pos * hidden_dim_, v, hidden_dim_ * sizeof(float));
}

void KVCache::retrieve(uint32_t layer, uint32_t pos, float* k, float* v) const {
    if (layer >= layers_ || pos >= context_len_) return;
    
    const float* k_layer = k_cache_[layer];
    const float* v_layer = v_cache_[layer];
    
    memcpy(k, k_layer + pos * hidden_dim_, hidden_dim_ * sizeof(float));
    memcpy(v, v_layer + pos * hidden_dim_, hidden_dim_ * sizeof(float));
}

Model::Model() 
    : kv_cache_(nullptr), num_layers_(0), hidden_dim_(0), 
      num_heads_(0), context_len_(0) {
}

Model::~Model() {
    free_tensors();
    if (kv_cache_) {
        delete kv_cache_;
    }
}

bool Model::load_from_gguf(const std::string& filepath) {
    ggml::GGUFParser parser;
    if (!parser.parse(filepath)) {
        return false;
    }
    
    const auto& tensors = parser.get_tensors();
    const auto& metadata = parser.get_metadata();
    
    auto get_meta = [&](const std::string& key) -> std::string {
        auto it = metadata.find(key);
        return it != metadata.end() ? it->second : "";
    };
    
    num_layers_ = std::stoul(get_meta("llama.block_count"));
    hidden_dim_ = std::stoul(get_meta("llama.embedding_length"));
    num_heads_ = std::stoul(get_meta("llama.attention.head_count"));
    context_len_ = std::stoul(get_meta("llama.context_length"));
    
    layers_.resize(num_layers_);
    
    for (const auto& tensor_info : tensors) {
        std::string name = tensor_info.name;
        ggml::Tensor* tensor = new ggml::Tensor(name, tensor_info.type, tensor_info.shape);
        tensor->allocate_cpu();
        
        std::vector<uint8_t> data = parser.read_tensor_data(tensor_info);
        memcpy(tensor->get_cpu_data(), data.data(), data.size());
        
        weights_[name] = tensor;
        
        uint32_t layer_id = UINT32_MAX;
        std::string weight_name;
        
        size_t layer_pos = name.find(".layers.");
        if (layer_pos != std::string::npos) {
            size_t next_dot = name.find(".", layer_pos + 8);
            if (next_dot != std::string::npos) {
                layer_id = std::stoul(name.substr(layer_pos + 8, next_dot - (layer_pos + 8)));
                weight_name = name.substr(next_dot + 1);
            }
        }
        
        if (layer_id < num_layers_) {
            LayerWeights& layer = layers_[layer_id];
            if (weight_name == "attention.wq") layer.q_proj = tensor;
            else if (weight_name == "attention.wk") layer.k_proj = tensor;
            else if (weight_name == "attention.wv") layer.v_proj = tensor;
            else if (weight_name == "attention.wo") layer.o_proj = tensor;
            else if (weight_name == "feed_forward.w1") layer.gate_proj = tensor;
            else if (weight_name == "feed_forward.w3") layer.up_proj = tensor;
            else if (weight_name == "feed_forward.w2") layer.down_proj = tensor;
            else if (weight_name == "attention_norm") layer.norm1 = tensor;
            else if (weight_name == "ffn_norm") layer.norm2 = tensor;
        }
    }
    
    kv_cache_ = new KVCache(num_layers_, hidden_dim_, context_len_);
    kv_cache_->allocate_gpu();
    
    allocate_tensors();
    return true;
}

void Model::allocate_tensors() {
    for (auto& pair : weights_) {
        pair.second->allocate_cpu();
    }
}

void Model::free_tensors() {
    for (auto& pair : weights_) {
        pair.second->free_cpu();
        delete pair.second;
    }
    weights_.clear();
}

const LayerWeights& Model::get_layer(uint32_t idx) const {
    return layers_[idx];
}

}
