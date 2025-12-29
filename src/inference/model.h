#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <cstdint>
#include "../core/gguf_types.h"
#include "../core/tensor.h"
#include "../vulkan_backend/memory.h"

namespace vulkan {
class VulkanMemory;
}

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
};

class KVCache {
public:
    KVCache(uint32_t layers, uint32_t hidden_dim, uint32_t context_len);
    ~KVCache();

    void set_vulkan_memory(vulkan::VulkanMemory* vulkan_memory);
    void allocate_gpu();
    void cleanup_gpu();
    void allocate_gpu_up_to(uint32_t max_pos);
    void store(uint32_t layer, uint32_t pos, const float* k, const float* v);
    void retrieve(uint32_t layer, uint32_t pos, float* k, float* v) const;

    VkBuffer get_gpu_k_buffer(uint32_t layer) const;
    VkBuffer get_gpu_v_buffer(uint32_t layer) const;

private:
    uint32_t layers_;
    uint32_t hidden_dim_;
    uint32_t context_len_;
    uint32_t allocated_len_;

    vulkan::VulkanMemory* vulkan_memory_;

    std::vector<vulkan::VulkanBuffer> k_gpu_buffers_;
    std::vector<vulkan::VulkanBuffer> v_gpu_buffers_;

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
    uint32_t get_vocab_size() const { return vocab_size_; }

    ModelArchitecture get_architecture() const { return architecture_; }

    const std::unordered_map<std::string, ggml::Tensor*>& get_weights() const { return weights_; }
    const LayerWeights& get_layer(uint32_t idx) const;

    KVCache* get_kv_cache() const { return kv_cache_; }

private:
    bool detect_architecture(const std::map<std::string, std::string>& metadata);

    std::unordered_map<std::string, ggml::Tensor*> weights_;
    std::vector<LayerWeights> layers_;
    KVCache* kv_cache_;

    uint32_t num_layers_;
    uint32_t hidden_dim_;
    uint32_t num_heads_;
    uint32_t vocab_size_;
    uint32_t context_len_;
    ModelArchitecture architecture_;
};

}
