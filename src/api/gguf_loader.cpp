#include "gguf_loader.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace api {

GGUFLoader::GGUFLoader() : engine_(std::make_unique<inference::InferenceEngine>()) {
    last_result_ = {false, "Not loaded", 0, "Unknown", 0, 0, 0};
}

GGUFLoader::~GGUFLoader() {
}

LoadResult GGUFLoader::load_model(const std::string& filepath) {
    inference::InferenceConfig config;
    config.num_threads = 4;
    config.gpu_memory_pool_mb = 2048;
    config.gpu_cache_mb = 1024;
    config.context_len = 2048;
    config.prefetch_layers = 2;
    config.backend = inference::BackendType::CPU;
    config.enable_validation = true;

    if (!engine_->initialize(config)) {
        last_result_ = {false, "Failed to initialize inference engine", 0, "Unknown", 0, 0, 0};
        return last_result_;
    }

    if (!engine_->load_model(filepath)) {
        last_result_ = {false, "Failed to load GGUF model", 0, "Unknown", 0, 0, 0};
        return last_result_;
    }

    last_result_.success = true;
    last_result_.error_message = "Success";
    last_result_.model_size_bytes = engine_->get_model_size_bytes();
    last_result_.architecture = "LLaMA/Compatible";
    last_result_.num_layers = 32;
    last_result_.hidden_dim = 4096;
    last_result_.num_parameters = static_cast<uint32_t>(last_result_.model_size_bytes / 4);

    return last_result_;
}

std::string GGUFLoader::get_model_info() const {
    if (!last_result_.success) {
        return "No model loaded";
    }

    std::ostringstream oss;
    oss << "Architecture: " << last_result_.architecture << "\n";
    oss << "Model Size: " << (last_result_.model_size_bytes / (1024.0 * 1024.0)) << " MB\n";
    oss << "Layers: " << last_result_.num_layers << "\n";
    oss << "Hidden Dim: " << last_result_.hidden_dim << "\n";
    oss << "Parameters: " << last_result_.num_parameters << "\n";
    return oss.str();
}

uint32_t GGUFLoader::get_vocab_size() const {
    return 32000;
}

uint32_t GGUFLoader::get_context_length() const {
    return 2048;
}

TextGenerator::TextGenerator(std::shared_ptr<inference::InferenceEngine> engine)
    : engine_(engine) {
    stop_tokens_.clear();
}

TextGenerator::~TextGenerator() {
}

std::string TextGenerator::generate(const std::string& prompt, uint32_t max_tokens,
                                     const GenerationConfig& config) {
    return engine_->generate(prompt, max_tokens);
}

void TextGenerator::set_stop_token(uint32_t token_id) {
    stop_tokens_.push_back(token_id);
}

void TextGenerator::clear_stop_tokens() {
    stop_tokens_.clear();
}

float TextGenerator::apply_temperature(float* logits, uint32_t vocab_size, float temperature) {
    if (temperature <= 0.0f) temperature = 1.0f;

    float max_val = logits[0];
    for (uint32_t i = 1; i < vocab_size; ++i) {
        max_val = std::max(max_val, logits[i]);
    }

    for (uint32_t i = 0; i < vocab_size; ++i) {
        logits[i] = (logits[i] - max_val) / temperature;
    }

    return max_val;
}

float TextGenerator::apply_top_p(float* logits, uint32_t vocab_size, float top_p) {
    if (top_p <= 0.0f || top_p >= 1.0f) return top_p;

    std::vector<std::pair<float, uint32_t>> sorted_logits(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        sorted_logits[i] = {logits[i], i};
    }

    std::sort(sorted_logits.begin(), sorted_logits.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<float> exp_values;
    for (const auto& p : sorted_logits) {
        exp_values.push_back(std::exp(p.first));
    }

    float cumulative = 0.0f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        cumulative += exp_values[i];
        if (cumulative > top_p) {
            for (uint32_t j = i + 1; j < vocab_size; ++j) {
                sorted_logits[j].first = -INFINITY;
            }
            break;
        }
    }

    for (uint32_t i = 0; i < vocab_size; ++i) {
        logits[sorted_logits[i].second] = sorted_logits[i].first;
    }

    return top_p;
}

uint32_t TextGenerator::apply_top_k(const float* logits, uint32_t vocab_size, uint32_t top_k) {
    if (top_k <= 1 || top_k >= vocab_size) {
        return vocab_size - 1;
    }

    std::vector<std::pair<float, uint32_t>> sorted_logits;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        sorted_logits.push_back({logits[i], i});
    }

    std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + top_k,
                     sorted_logits.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    return sorted_logits[top_k - 1].second;
}

InferenceSession::InferenceSession()
    : engine_(std::make_unique<inference::InferenceEngine>()),
      loader_(std::make_unique<GGUFLoader>()),
      generator_(nullptr),
      initialized_(false) {
}

InferenceSession::~InferenceSession() {
}

bool InferenceSession::initialize(const inference::InferenceConfig& config) {
    if (engine_->initialize(config)) {
        initialized_ = true;
        generator_ = std::make_unique<TextGenerator>(
            std::shared_ptr<inference::InferenceEngine>(engine_.get(), [](inference::InferenceEngine*){})
        );
        return true;
    }
    return false;
}

LoadResult InferenceSession::load_model(const std::string& filepath) {
    return loader_->load_model(filepath);
}

std::string InferenceSession::generate(const std::string& prompt, uint32_t max_tokens,
                                       const GenerationConfig& gen_config) {
    if (!initialized_ || !generator_) {
        return "Inference session not initialized";
    }

    return generator_->generate(prompt, max_tokens, gen_config);
}

void InferenceSession::set_backend(inference::BackendType backend) {
    engine_->set_gpu_enabled(backend == inference::BackendType::GPU || backend == inference::BackendType::HYBRID);
}

void InferenceSession::set_num_threads(uint32_t num_threads) {
    engine_->set_num_threads(num_threads);
}

void InferenceSession::set_gpu_memory_pool(size_t size_mb) {
    engine_->set_gpu_memory_pool(size_mb);
}

void InferenceSession::set_context_length(uint32_t context_len) {
    engine_->set_kv_cache_size(context_len);
}

std::string InferenceSession::get_model_info() const {
    return loader_->get_model_info();
}

}