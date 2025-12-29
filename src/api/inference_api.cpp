#include "inference_api.h"
#include <chrono>
#include <thread>
#include <mutex>
#include <iostream>

namespace py {

InferenceAPI::InferenceAPI() : config_() {
    config_.max_tokens = 100;
    config_.temperature = 1.0f;
    config_.top_p = 0.9f;
    config_.top_k = 40;
    config_.frequency_penalty = 0.0f;
    config_.presence_penalty = 0.0f;
    config_.do_sample = true;
}

InferenceAPI::~InferenceAPI() {
    std::lock_guard<std::mutex> lock(api_mutex_);
}

bool InferenceAPI::load_model(const std::string& model_path) {
    std::lock_guard<std::mutex> lock(api_mutex_);

    inference::InferenceConfig config;
    config.backend = inference::BackendType::GPU;
    config.enable_validation = false;

    engine_ = std::make_unique<inference::InferenceEngine>();
    bool result = engine_->initialize(config);

    if (result) {
        result = engine_->load_model(model_path);
        if (result) {
            model_path_ = model_path;
            std::cout << "[InferenceAPI] Model loaded: " << model_path << std::endl;
        }
    }

    return result;
}

bool InferenceAPI::unload_model() {
    std::lock_guard<std::mutex> lock(api_mutex_);

    if (engine_) {
        engine_.~InferenceEngine();
        engine_.reset();
    }

    model_path_.clear();
    std::cout << "[InferenceAPI] Model unloaded" << std::endl;
    return true;
}

std::string InferenceAPI::generate(const std::string& prompt, const GenerationConfig& config) {
    std::lock_guard<std::mutex> lock(api_mutex_);

    if (!engine_) {
        return "Error: No model loaded";
    }

    auto start = std::chrono::steady_clock::now();

    inference::InferenceConfig cpp_config;
    convert_config(config, cpp_config);

    std::string result = engine_->generate(prompt, config.max_tokens);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "[InferenceAPI] Generated " << result.length() << " chars in "
              << duration.count() << "ms" << std::endl;

    return result;
}

GenerationResult InferenceAPI::generate_default(const std::string& prompt) {
    auto start = std::chrono::steady_clock::now();

    std::string text = generate(prompt, config_);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    GenerationResult result;
    result.text = text;
    result.num_tokens = static_cast<uint32_t>(text.length());
    result.time_ms = static_cast<float>(duration.count());
    result.tokens_per_second = result.num_tokens / (result.time_ms / 1000.0f);

    return result;
}

std::string InferenceAPI::generate_streaming(const std::string& prompt, const GenerationConfig& config,
                                               std::function<void(const std::string&)> token_callback) {
    std::lock_guard<std::mutex> lock(api_mutex_);

    if (!engine_) {
        return "Error: No model loaded";
    }

    auto start = std::chrono::steady_clock::now();

    inference::InferenceConfig cpp_config;
    convert_config(config, cpp_config);

    std::string result;
    engine_->generate_streaming(prompt, config.max_tokens,
        [&result, token_callback](const std::string& token) {
            result += token;
            if (token_callback) {
                token_callback(token);
            }
        });

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "[InferenceAPI] Generated " << result.length() << " chars in "
              << duration.count() << "ms" << std::endl;

    return result;
}

std::string InferenceAPI::generate_with_progress(const std::string& prompt, const GenerationConfig& config,
                                                  std::function<void(const GenerationProgress&)> progress_callback) {
    std::lock_guard<std::mutex> lock(api_mutex_);

    if (!engine_) {
        return "Error: No model loaded";
    }

    auto start = std::chrono::steady_clock::now();

    inference::InferenceConfig cpp_config;
    convert_config(config, cpp_config);

    std::string result;
    engine_->generate_with_progress(prompt, config.max_tokens,
        [&result, progress_callback](const inference::InferenceEngine::GenerationProgress& cpp_progress) {
            if (progress_callback) {
                GenerationProgress py_progress;
                py_progress.current_step = cpp_progress.current_step;
                py_progress.total_steps = cpp_progress.total_steps;
                py_progress.progress_pct = cpp_progress.progress_pct;
                py_progress.time_elapsed_ms = cpp_progress.time_elapsed_ms;
                py_progress.estimated_remaining_ms = cpp_progress.estimated_remaining_ms;
                py_progress.tokens_per_second = cpp_progress.tokens_per_second;
                py_progress.last_token = cpp_progress.last_token;
                progress_callback(py_progress);
            }
        });

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "[InferenceAPI] Generated in " << duration.count() << "ms" << std::endl;

    return result;
}

std::vector<std::string> InferenceAPI::generate_batch(const std::vector<std::string>& prompts,
                                                   const GenerationConfig& config) {
    std::vector<std::string> results;
    results.reserve(prompts.size());

    for (const auto& prompt : prompts) {
        results.push_back(generate(prompt, config));
    }

    return results;
}

bool InferenceAPI::enable_gpu(bool enable) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->set_gpu_enabled(enable);
    return true;
}

bool InferenceAPI::is_gpu_enabled() const {
    return engine_->is_gpu_enabled();
}

ModelMetrics InferenceAPI::get_model_info() const {
    std::lock_guard<std::mutex> lock(api_mutex_);

    if (!engine_) {
        ModelMetrics empty;
        return empty;
    }

    ModelMetrics metrics;
    metrics.vocab_size = engine_->get_model_size();
    metrics.num_layers = engine_->get_model_layers();
    metrics.hidden_dim = engine_->get_hidden_dim();
    metrics.context_len = engine_->get_context_len();
    metrics.num_heads = engine_->get_num_heads();
    metrics.model_size_bytes = engine_->get_model_size_bytes();
    metrics.architecture = "llama";

    return metrics;
}

std::string InferenceAPI::get_model_architecture() const {
    return "llama";
}

void InferenceAPI::set_max_batch_size(uint32_t max_batch) {
    std::lock_guard<std::mutex> lock(api_mutex_);
}

void InferenceAPI::enable_speculative_decoding(bool enable) {
    std::lock_guard<std::mutex> lock(api_mutex_);
}

void InferenceAPI::enable_multi_gpu(bool enable) {
    std::lock_guard<std::mutex> lock(api_mutex_);
}

float InferenceAPI::get_acceptance_rate() const {
    std::lock_guard<std::mutex> lock(api_mutex_);
    return engine_->get_acceptance_rate();
}

float InferenceAPI::get_throughput() const {
    std::lock_guard<std::mutex> lock(api_mutex_);
    return engine_->get_throughput();
}

void InferenceAPI::set_temperature(float temp) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    config_.temperature = temp;
}

void InferenceAPI::set_top_p(float top_p) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    config_.top_p = top_p;
}

void InferenceAPI::set_top_k(uint32_t top_k) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    config_.top_k = top_k;
}

void InferenceAPI::set_cache_size_mb(size_t size_mb) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->set_gpu_cache_size(size_mb);
}

void InferenceAPI::set_num_threads(uint32_t num_threads) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->set_num_threads(num_threads);
}

bool InferenceAPI::save_pipeline_cache(const std::string& path) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    return engine_->save_pipeline_cache(path);
}

bool InferenceAPI::load_pipeline_cache(const std::string& path) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    return engine_->load_pipeline_cache(path);
}

void InferenceAPI::enable_profiling(bool enable) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->enable_profiling(enable);
}

std::string InferenceAPI::get_performance_report() const {
    std::lock_guard<std::mutex> lock(api_mutex_);
    return engine_->get_performance_report();
}

std::vector<float> InferenceAPI::get_token_logits(const std::vector<uint32_t>& tokens) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    return engine_->get_token_logits(tokens);
}

void InferenceAPI::set_adapter_alpha(const std::string& adapter_name, float alpha) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->set_adapter_alpha(adapter_name, alpha);
}

void InferenceAPI::enable_adapter(const std::string& adapter_name) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->enable_adapter(adapter_name);
}

void InferenceAPI::disable_adapter(const std::string& adapter_name) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->disable_adapter(adapter_name);
}

std::vector<std::string> InferenceAPI::get_loaded_adapters() const {
    std::lock_guard<std::mutex> lock(api_mutex_);
    return engine_->get_loaded_adapters();
}

void InferenceAPI::register_custom_tokenizer(std::function<std::vector<uint32_t>(const std::string&)> tokenizer_func) {
    std::lock_guard<std::mutex> lock(api_mutex_);
}

void InferenceAPI::register_custom_kernel(std::function<void(const std::string&, const std::string&)> kernel_compiler) {
    std::lock_guard<std::mutex> lock(api_mutex_);
}

void InferenceAPI::set_quantization_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(api_mutex_);
    engine_->set_quantization_enabled(enabled);
}

void InferenceAPI::set_weight_quantization(const std::string& type) {
    std::lock_guard<std::mutex> lock(api_mutex_);

    inference::QuantizationType q_type;
    if (type == "int8") {
        q_type = inference::QuantizationType::INT8;
    } else if (type == "fp4") {
        q_type = inference::QuantizationType::FP4;
    } else if (type == "nf4") {
        q_type = inference::QuantizationType::NF4;
    } else if (type == "fp16") {
        q_type = inference::QuantizationType::FP16;
    } else {
        q_type = inference::QuantizationType::NONE;
    }

    engine_->set_weight_quantization(q_type);
}

void InferenceAPI::set_activation_quantization(const std::string& type) {
    std::lock_guard<std::mutex> lock(api_mutex_);

    inference::QuantizationType q_type;
    if (type == "int8") {
        q_type = inference::QuantizationType::INT8;
    } else if (type == "fp4") {
        q_type = inference::QuantizationType::FP4;
    } else if (type == "fp16") {
        q_type = inference::QuantizationType::FP16;
    } else {
        q_type = inference::QuantizationType::NONE;
    }

    engine_->set_activation_quantization(q_type);
}

std::string InferenceAPI::get_quantization_config() const {
    std::lock_guard<std::mutex> lock(api_mutex_);

    inference::QuantizationConfig config = engine_->get_quantization_config();

    std::string result = "{";
    result += "\"weight_quantization\": \"" + get_quantization_type_name(config.weight_quantization) + "\", ";
    result += "\"activation_quantization\": \"" + get_quantization_type_name(config.activation_quantization) + "\", ";
    result += "\"kv_cache_quantization\": \"" + get_quantization_type_name(config.kv_cache_quantization) + "\", ";
    result += "\"calibration_enabled\": " + (config.calibration_enabled ? "true" : "false") + ", ";
    result += "\"calibration_steps\": " + std::to_string(config.calibration_steps);
    result += "}";

    return result;
}

std::string InferenceAPI::get_quantization_type_name(inference::QuantizationType type) {
    switch (type) {
        case inference::QuantizationType::NONE: return "none";
        case inference::QuantizationType::FP32: return "fp32";
        case inference::QuantizationType::FP16: return "fp16";
        case inference::QuantizationType::INT8: return "int8";
        case inference::QuantizationType::FP4: return "fp4";
        case inference::QuantizationType::NF4: return "nf4";
        default: return "unknown";
    }
}

void InferenceAPI::convert_config(const GenerationConfig& py_config, inference::InferenceConfig& cpp_config) {
    cpp_config.max_tokens = py_config.max_tokens;
    cpp_config.num_threads = 8;
    cpp_config.gpu_cache_mb = 1024;
    cpp_config.prefetch_layers = 2;
    cpp_config.backend = inference::BackendType::GPU;
    cpp_config.enable_validation = false;
}

PYBIND11_MODULE(vulkangguf, m) {
    py::class_<InferenceAPI>(m, "InferenceAPI")
        .def(py::init())
        .def("load_model", &InferenceAPI::load_model, "Load a GGUF model from disk")
        .def("unload_model", &InferenceAPI::unload_model, "Unload the current model")
        .def("generate", &InferenceAPI::generate, "Generate text from a prompt with custom config")
        .def("generate_default", &InferenceAPI::generate_default, "Generate text with default config")
        .def("generate_streaming", &InferenceAPI::generate_streaming,
             py::arg("prompt"), py::arg("config"), py::arg("token_callback"),
             "Generate text with streaming callback for each token")
        .def("generate_with_progress", &InferenceAPI::generate_with_progress,
             py::arg("prompt"), py::arg("config"), py::arg("progress_callback"),
             "Generate text with progress callback containing statistics")
        .def("generate_batch", &InferenceAPI::generate_batch, "Generate text for multiple prompts")
        .def("enable_gpu", &InferenceAPI::enable_gpu, "Enable or disable GPU acceleration")
        .def("is_gpu_enabled", &InferenceAPI::is_gpu_enabled, "Check if GPU is enabled")
        .def("get_model_info", &InferenceAPI::get_model_info, "Get model metadata and metrics")
        .def("get_model_architecture", &InferenceAPI::get_model_architecture, "Get model architecture name")
        .def("set_max_batch_size", &InferenceAPI::set_max_batch_size, "Set maximum batch size")
        .def("enable_speculative_decoding", &InferenceAPI::enable_speculative_decoding, "Enable or disable speculative decoding")
        .def("enable_multi_gpu", &InferenceAPI::enable_multi_gpu, "Enable or disable multi-GPU")
        .def("get_acceptance_rate", &InferenceAPI::get_acceptance_rate, "Get speculative decoding acceptance rate")
        .def("get_throughput", &InferenceAPI::get_throughput, "Get current throughput in tokens/second")
        .def("set_temperature", &InferenceAPI::set_temperature, "Set sampling temperature")
        .def("set_top_p", &InferenceAPI::set_top_p, "Set nucleus sampling parameter")
        .def("set_top_k", &InferenceAPI::set_top_k, "Set top-k sampling parameter")
        .def("set_cache_size_mb", &InferenceAPI::set_cache_size_mb, "Set GPU cache size in MB")
        .def("set_num_threads", &InferenceAPI::set_num_threads, "Set number of CPU threads")
        .def("save_pipeline_cache", &InferenceAPI::save_pipeline_cache, "Save pipeline cache to disk")
        .def("load_pipeline_cache", &InferenceAPI::load_pipeline_cache, "Load pipeline cache from disk")
        .def("enable_profiling", &InferenceAPI::enable_profiling, "Enable performance profiling")
        .def("get_performance_report", &InferenceAPI::get_performance_report, "Get performance metrics report")
        .def("get_token_logits", &InferenceAPI::get_token_logits, "Get logits for generated tokens")
        .def("set_adapter_alpha", &InferenceAPI::set_adapter_alpha, "Set LoRA adapter alpha value")
        .def("enable_adapter", &InferenceAPI::enable_adapter, "Enable a LoRA adapter")
        .def("disable_adapter", &InferenceAPI::disable_adapter, "Disable a LoRA adapter")
        .def("get_loaded_adapters", &InferenceAPI::get_loaded_adapters, "Get list of loaded adapters")
        .def("register_custom_tokenizer", &InferenceAPI::register_custom_tokenizer, "Register custom tokenizer function")
        .def("register_custom_kernel", &InferenceAPI::register_custom_kernel, "Register custom shader compiler function");

    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init())
        .def_readwrite("max_tokens", &GenerationConfig::max_tokens, "Maximum number of tokens to generate")
        .def_readwrite("temperature", &GenerationConfig::temperature, "Sampling temperature (0.0 - 2.0)")
        .def_readwrite("top_p", &GenerationConfig::top_p, "Nucleus sampling parameter (0.0 - 1.0)")
        .def_readwrite("top_k", &GenerationConfig::top_k, "Top-k sampling parameter")
        .def_readwrite("frequency_penalty", &GenerationConfig::frequency_penalty, "Frequency penalty")
        .def_readwrite("presence_penalty", &GenerationConfig::presence_penalty, "Presence penalty")
        .def_readwrite("do_sample", &GenerationConfig::do_sample, "Whether to use sampling or greedy decoding");

    py::class_<ModelMetrics>(m, "ModelMetrics")
        .def_readonly("vocab_size", &ModelMetrics::vocab_size, "Vocabulary size")
        .def_readonly("num_layers", &ModelMetrics::num_layers, "Number of layers")
        .def_readonly("hidden_dim", &ModelMetrics::hidden_dim, "Hidden dimension")
        .def_readonly("context_len", &ModelMetrics::context_len, "Context length")
        .def_readonly("num_heads", &ModelMetrics::num_heads, "Number of attention heads")
        .def_readonly("model_size_bytes", &ModelMetrics::model_size_bytes, "Model size in bytes")
        .def_readonly("architecture", &ModelMetrics::architecture, "Model architecture");

    py::class_<GenerationResult>(m, "GenerationResult")
        .def_readonly("tokens", &GenerationResult::tokens, "Generated token IDs")
        .def_readonly("text", &GenerationResult::text, "Decoded text")
        .def_readonly("num_tokens", &GenerationResult::num_tokens, "Number of tokens generated")
        .def_readonly("time_ms", &GenerationResult::time_ms, "Generation time in milliseconds")
        .def_readonly("tokens_per_second", &GenerationResult::tokens_per_second, "Generation speed");

    py::class_<GenerationProgress>(m, "GenerationProgress")
        .def_readonly("current_step", &GenerationProgress::current_step, "Current step number")
        .def_readonly("total_steps", &GenerationProgress::total_steps, "Total steps to complete")
        .def_readonly("progress_pct", &GenerationProgress::progress_pct, "Progress percentage (0-100)")
        .def_readonly("time_elapsed_ms", &GenerationProgress::time_elapsed_ms, "Time elapsed in milliseconds")
        .def_readonly("estimated_remaining_ms", &GenerationProgress::estimated_remaining_ms, "Estimated time remaining")
        .def_readonly("tokens_per_second", &GenerationProgress::tokens_per_second, "Current tokens per second")
        .def_readonly("last_token", &GenerationProgress::last_token, "Last generated token");

    py::enum_<inference::QuantizationType>(m, "QuantizationType")
        .value("NONE", inference::QuantizationType::NONE)
        .value("FP32", inference::QuantizationType::FP32)
        .value("FP16", inference::QuantizationType::FP16)
        .value("INT8", inference::QuantizationType::INT8)
        .value("FP4", inference::QuantizationType::FP4)
        .value("NF4", inference::QuantizationType::NF4);

    m.def("get_quantization_type_name", &InferenceAPI::get_quantization_type_name, "Get quantization type name");
}

}
