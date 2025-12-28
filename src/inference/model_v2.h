#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "../core/gguf_types.h"
#include "../core/tensor.h"

namespace inference {

enum class ModelArchitecture {
    UNKNOWN,
    LLAMA,
    LLAMA2,
    LLAMA3,
    MISTRAL,
    MIXTRAL,
    GEMMA,
    GEMMA2,
    QWEN,
    QWEN2,
    PHI,
    PHI2,
    PHI3,
    GPT_NEOX,
    GPTJ,
    STABLELM,
    FALCON
};

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
    
    uint32_t num_kv_heads;
    uint32_t num_experts;
};

struct MoELayerWeights {
    ggml::Tensor* gate;
    ggml::Tensor* up_proj;
    ggml::Tensor* down_proj;
    std::vector<LayerWeights> experts;
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

    ModelArchitecture get_architecture() const { return architecture_; }
    bool is_moe() const { return is_moe_; }
    bool has_gqa() const { return has_gqa_; }

    uint32_t get_num_layers() const { return num_layers_; }
    uint32_t get_hidden_dim() const { return hidden_dim_; }
    uint32_t get_num_heads() const { return num_heads_; }
    uint32_t get_num_kv_heads() const { return num_kv_heads_; }
    uint32_t get_context_len() const { return context_len_; }
    uint32_t get_vocab_size() const { return vocab_size_; }
    uint32_t get_num_experts() const { return num_experts_; }

    const std::unordered_map<std::string, ggml::Tensor*>& get_weights() const { return weights_; }
    const LayerWeights& get_layer(uint32_t idx) const;
    const MoELayerWeights& get_moe_layer(uint32_t idx) const;

    KVCache* get_kv_cache() const { return kv_cache_; }

private:
    bool detect_architecture(const std::map<std::string, std::string>& metadata);
    bool parse_llama_weights(const std::vector<ggml::TensorInfo>& tensors);
    bool parse_mistral_weights(const std::vector<ggml::TensorInfo>& tensors);
    bool parse_mixtral_weights(const std::vector<ggml::TensorInfo>& tensors);
    bool parse_gemma_weights(const std::vector<ggml::TensorInfo>& tensors);
    bool parse_qwen_weights(const std::vector<ggml::TensorInfo>& tensors);
    bool parse_phi_weights(const std::vector<ggml::TensorInfo>& tensors);

    std::unordered_map<std::string, ggml::Tensor*> weights_;
    std::vector<LayerWeights> layers_;
    std::vector<MoELayerWeights> moe_layers_;
    KVCache* kv_cache_;

    ModelArchitecture architecture_;
    bool is_moe_;
    bool has_gqa_;

    uint32_t num_layers_;
    uint32_t hidden_dim_;
    uint32_t num_heads_;
    uint32_t num_kv_heads_;
    uint32_t context_len_;
    uint32_t vocab_size_;
    uint32_t num_experts_;
};

}
