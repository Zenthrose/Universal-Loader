#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "../core/gguf_types.h"
#include "../core/tensor.h"

namespace inference {

struct LayerWeights {
    ggml::Tensor* q_proj;
    ggml::Tensor* k_proj;
    ggml::Tensor* v_proj;
    ggml::Tensor* o_proj;
    ggml::Tensor* gate_proj;
    ggml::Tensor* up_proj;
    ggml::Tensor* down_proj;
    ggml::Tensor* norm1;
    ggml::Tensor* norm2;
};

class KVCache {
public:
    KVCache(uint32_t layers, uint32_t hidden_dim, uint32_t context_len);
    ~KVCache();

    void allocate_gpu();
    void store(uint32_t layer, uint32_t pos, const float* k, const float* v);
    void retrieve(uint32_t layer, uint32_t pos, float* k, float* v) const;

private:
    uint32_t layers_;
    uint32_t hidden_dim_;
    uint32_t context_len_;
    std::vector<float*> k_cache_;
    std::vector<float*> v_cache_;
};

class Model {
public:
    Model();
    ~Model();

    bool load_from_gguf(const std::string& filepath);
    void allocate_tensors();
    void free_tensors();

    uint32_t get_num_layers() const { return num_layers_; }
    uint32_t get_hidden_dim() const { return hidden_dim_; }
    uint32_t get_num_heads() const { return num_heads_; }
    uint32_t get_context_len() const { return context_len_; }

    const std::unordered_map<std::string, ggml::Tensor*>& get_weights() const { return weights_; }
    const LayerWeights& get_layer(uint32_t idx) const;

    KVCache* get_kv_cache() const { return kv_cache_; }

private:
    std::unordered_map<std::string, ggml::Tensor*> weights_;
    std::vector<LayerWeights> layers_;
    KVCache* kv_cache_;

    uint32_t num_layers_;
    uint32_t hidden_dim_;
    uint32_t num_heads_;
    uint32_t context_len_;
};

}
