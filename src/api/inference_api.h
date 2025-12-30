#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../inference/inference_engine.h"

namespace py {

struct GenerationConfig {
    uint32_t max_tokens;
    float temperature;
    float top_p;
    uint32_t top_k;
    float frequency_penalty;
    float presence_penalty;
    bool do_sample;
};

struct GenerationResult {
    std::vector<uint32_t> tokens;
    std::string text;
    uint32_t num_tokens;
    float time_ms;
    float tokens_per_second;
};

struct GenerationProgress {
    uint32_t current_step;
    uint32_t total_steps;
    float progress_pct;
    float time_elapsed_ms;
    float estimated_remaining_ms;
    float tokens_per_second;
    std::string last_token;
};

struct ModelMetrics {
    uint32_t vocab_size;
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t context_len;
    uint32_t num_heads;
    size_t model_size_bytes;
    std::string architecture;
};

class InferenceAPI {
public:
    InferenceAPI();
    ~InferenceAPI();

    bool load_model(const std::string& model_path);
    bool unload_model();

    std::string generate(const std::string& prompt, const GenerationConfig& config);
    GenerationResult generate_default(const std::string& prompt);

    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts,
                                           const GenerationConfig& config);

    std::string generate_streaming(const std::string& prompt, const GenerationConfig& config,
                                   std::function<void(const std::string&)> token_callback);

    std::string generate_with_progress(const std::string& prompt, const GenerationConfig& config,
                                      std::function<void(const GenerationProgress&)> progress_callback);

    bool enable_gpu(bool enable);
    bool is_gpu_enabled() const;

    ModelMetrics get_model_info() const;
    std::string get_model_architecture() const;

    void set_max_batch_size(uint32_t max_batch);
    void enable_speculative_decoding(bool enable);
    void enable_multi_gpu(bool enable);

    float get_acceptance_rate() const;
    float get_throughput() const;

    void set_temperature(float temp);
    void set_top_p(float top_p);
    void set_top_k(uint32_t top_k);

    void set_cache_size_mb(size_t size_mb);
    void set_num_threads(uint32_t num_threads);

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
    void set_weight_quantization(const std::string& type);
    void set_activation_quantization(const std::string& type);
    std::string get_quantization_config() const;

    void register_custom_tokenizer(std::function<std::vector<uint32_t>(const std::string&)> tokenizer_func);
    void register_custom_kernel(std::function<void(const std::string&, const std::string&)> kernel_compiler);

private:
    std::unique_ptr<inference::InferenceEngine> engine_;
    mutable std::mutex api_mutex_;

    GenerationConfig config_;
    std::string model_path_;

    void convert_config(const GenerationConfig& py_config, inference::InferenceConfig& cpp_config);
public:
    static std::string get_quantization_type_name(inference::QuantizationType type);
};

}
