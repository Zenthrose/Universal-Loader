#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include "../inference/inference_engine.h"

namespace api {

struct LoadResult {
    bool success;
    std::string error_message;
    size_t model_size_bytes;
    std::string architecture;
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t num_parameters;
};

struct GenerationConfig {
    float temperature;
    float top_p;
    uint32_t top_k;
    float repetition_penalty;
    bool stop_on_eos;
    std::vector<uint32_t> stop_tokens;
};

class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();

    LoadResult load_model(const std::string& filepath);

    std::string get_model_info() const;
    uint32_t get_vocab_size() const;
    uint32_t get_context_length() const;

private:
    std::unique_ptr<inference::InferenceEngine> engine_;
    LoadResult last_result_;
};

class TextGenerator {
public:
    TextGenerator(std::shared_ptr<inference::InferenceEngine> engine);
    ~TextGenerator();

    std::string generate(const std::string& prompt,
                       uint32_t max_tokens,
                       const GenerationConfig& config = {});

    void set_stop_token(uint32_t token_id);
    void clear_stop_tokens();

private:
    std::shared_ptr<inference::InferenceEngine> engine_;
    std::vector<uint32_t> stop_tokens_;
    float apply_temperature(float* logits, uint32_t vocab_size, float temperature);
    float apply_top_p(float* logits, uint32_t vocab_size, float top_p);
    uint32_t apply_top_k(const float* logits, uint32_t vocab_size, uint32_t top_k);
};

class InferenceSession {
public:
    InferenceSession();
    ~InferenceSession();

    bool initialize(const inference::InferenceConfig& config);
    LoadResult load_model(const std::string& filepath);

    std::string generate(const std::string& prompt,
                       uint32_t max_tokens,
                       const GenerationConfig& gen_config = {});

    void set_backend(inference::BackendType backend);
    void set_num_threads(uint32_t num_threads);
    void set_gpu_memory_pool(size_t size_mb);
    void set_context_length(uint32_t context_len);

    std::string get_model_info() const;

private:
    std::unique_ptr<inference::InferenceEngine> engine_;
    std::unique_ptr<GGUFLoader> loader_;
    std::unique_ptr<TextGenerator> generator_;
    bool initialized_;
};

}