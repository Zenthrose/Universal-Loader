#pragma once
#include <memory>
#include <mutex>
#include <atomic>
#include "../core/gguf_parser.h"
#include "model.h"
#include "offload_manager.h"
#include "prefetch_engine.h"

namespace inference {

enum class BackendType {
    CPU,
    GPU,
    HYBRID
};

struct InferenceConfig {
    uint32_t num_threads;
    size_t gpu_memory_pool_mb;
    size_t gpu_cache_mb;
    uint32_t context_len;
    uint32_t prefetch_layers;
    BackendType backend;
    bool enable_validation;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    bool initialize(const InferenceConfig& config);
    bool load_model(const std::string& filepath);
    
    std::string generate(const std::string& prompt, uint32_t max_tokens);
    
    void set_num_threads(uint32_t num_threads);
    void set_gpu_enabled(bool enabled);
    void set_gpu_memory_pool(size_t size_mb);
    void set_gpu_cache_size(size_t size_mb);
    void set_kv_cache_size(uint32_t size);
    void set_prefetch_layers(uint32_t count);
    
    bool is_gpu_enabled() const;
    uint32_t get_num_threads() const;
    size_t get_model_size_bytes() const;
    
private:
    void tokenized(const std::string& prompt, std::vector<uint32_t>& tokens);
    float sample_token(const float* logits, uint32_t vocab_size);
    void forward_layer(uint32_t layer_id, const float* input, float* output);
    
    std::unique_ptr<Model> model_;
    std::unique_ptr<OffloadManager> offload_manager_;
    std::unique_ptr<PrefetchEngine> prefetch_engine_;
    
    std::mutex model_mutex_;
    std::mutex cache_mutex_;
    std::mutex transfer_mutex_;
    
    InferenceConfig config_;
    std::atomic<bool> gpu_enabled_;
    std::atomic<bool> initialized_;
    
    uint32_t vocab_size_;
    uint32_t context_len_;
};

}
