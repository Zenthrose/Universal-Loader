#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include "../core/gguf_parser.h"
#include "../vulkan_backend/context.h"
#include "../vulkan_backend/compute.h"
#include "../vulkan_backend/memory.h"
#include "../vulkan_backend/descriptors.h"
#include "../vulkan_backend/pipeline_cache.h"
#include "../vulkan_backend/transfer.h"
#include "../vulkan_backend/shader_compiler.h"
#include "../vulkan_backend/timeline_semaphores.h"
#include "../vulkan_backend/async_pipeline.h"
#include "model.h"
#include "offload_manager.h"
#include "prefetch_engine.h"
#include "tokenizer.h"
#include "quantization_manager.h"

namespace inference {

enum class BackendType {
    CPU,
    GPU,
    HYBRID
};

struct InferenceConfig {
    uint32_t max_tokens;
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
    using StreamCallback = std::function<void(const std::string&)>;

    struct GenerationProgress {
        uint32_t current_step;
        uint32_t total_steps;
        float progress_pct;
        float time_elapsed_ms;
        float estimated_remaining_ms;
        float tokens_per_second;
        std::string last_token;
    };

    using ProgressCallback = std::function<void(const GenerationProgress&)>;

    InferenceEngine();
    ~InferenceEngine();

    bool initialize(const InferenceConfig& config);
    bool load_model(const std::string& filepath);

    std::string generate(const std::string& prompt, uint32_t max_tokens);
    std::string generate_streaming(const std::string& prompt, uint32_t max_tokens,
                                  StreamCallback callback);
    std::string generate_with_progress(const std::string& prompt, uint32_t max_tokens,
                                      ProgressCallback callback);

    void set_num_threads(uint32_t num_threads);
    void set_gpu_enabled(bool enabled);
    void set_gpu_memory_pool(size_t size_mb);
    void set_gpu_cache_size(size_t size_mb);
    void set_kv_cache_size(uint32_t size);
    void set_prefetch_layers(uint32_t count);

    bool is_gpu_enabled() const;
    uint32_t get_num_threads() const;
    size_t get_model_size_bytes() const;

    uint32_t get_model_size() const;
    uint32_t get_model_layers() const;
    uint32_t get_hidden_dim() const;
    uint32_t get_context_len() const;
    uint32_t get_num_heads() const;

    float get_acceptance_rate() const;
    float get_throughput() const;

    bool save_pipeline_cache(const std::string& path);
    bool load_pipeline_cache(const std::string& path);

    void enable_profiling(bool enable);
    std::string get_performance_report() const;

    std::vector<float> get_token_logits(const std::vector<uint32_t>& tokens);

    void set_adapter_alpha(const std::string& adapter_name, float alpha);
    void enable_adapter(const std::string& adapter_name);
    void disable_adapter(const std::string& adapter_name);
    std::vector<std::string> get_loaded_adapters() const;

    void set_quantization_enabled(bool enabled);
    void set_weight_quantization(QuantizationType type);
    void set_activation_quantization(QuantizationType type);
    QuantizationConfig get_quantization_config() const;

    bool reload_shaders();

private:
    void tokenized(const std::string& prompt, std::vector<uint32_t>& tokens);
    float sample_token(const float* logits, uint32_t vocab_size);
    void forward_layer(uint32_t layer_id, uint32_t position, const float* input, float* output);
    void forward_layer_gpu(uint32_t layer_id, const float* input, float* output,
                          const LayerWeights& layer_weights, uint32_t hidden_dim);
    void forward_layer_cpu(uint32_t layer_id, const float* input, float* output,
                          const LayerWeights& layer_weights, uint32_t hidden_dim);

    vulkan::ComputeWork calculate_workgroups(uint32_t n, uint32_t m, uint32_t batch = 1);
    void upload_to_gpu(const float* data, VkDeviceSize size,
                      vulkan::VulkanBuffer& gpu_buffer);
    void download_from_gpu(VkDeviceSize size, vulkan::VulkanBuffer& gpu_buffer,
                          float* output);

    bool supports_flash_attention() const;
    bool should_use_flash_attention(uint32_t layer_id) const;

    std::unique_ptr<Model> model_;
    std::unique_ptr<OffloadManager> offload_manager_;
    std::unique_ptr<PrefetchEngine> prefetch_engine_;
    std::unique_ptr<Tokenizer> tokenizer_;

    std::unique_ptr<vulkan::VulkanContext> vulkan_context_;
    std::unique_ptr<vulkan::ComputeDispatcher> compute_dispatcher_;
    std::unique_ptr<vulkan::VulkanMemory> vulkan_memory_;
    std::unique_ptr<vulkan::DescriptorPool> descriptor_pool_;
    std::unique_ptr<vulkan::PipelineCache> pipeline_cache_;
    std::unique_ptr<vulkan::TransferEngine> transfer_engine_;
    std::unique_ptr<vulkan::ShaderCompiler> shader_compiler_;
    std::unique_ptr<vulkan::TimelineSemaphores> timeline_semaphores_;
    std::unique_ptr<vulkan::AsyncPipelineManager> async_pipeline_;
    std::unique_ptr<QuantizationManager> quantization_manager_;

    std::mutex model_mutex_;
    std::mutex cache_mutex_;
    std::mutex transfer_mutex_;

    InferenceConfig config_;
    std::atomic<bool> gpu_enabled_;
    std::atomic<bool> initialized_;

    uint32_t vocab_size_;
    uint32_t context_len_;

    struct GPUBuffers {
        vulkan::VulkanBuffer input_buffer;
        vulkan::VulkanBuffer output_buffer;
        VkDeviceSize tensor_size;
    };

    std::unordered_map<std::string, GPUBuffers> gpu_buffers_;
};

}
