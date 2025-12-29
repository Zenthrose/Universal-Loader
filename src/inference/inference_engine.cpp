#include "inference_engine.h"
#include "tokenizer.h"
#include "../cpu_backend/cpu_context.h"
#include "../cpu_backend/gemm_avx2.h"
#include "../cpu_backend/activation_cpu.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdint>

namespace inference {

InferenceEngine::InferenceEngine()
    : gpu_enabled_(false), initialized_(false), vocab_size_(0), context_len_(0) {
}

InferenceEngine::~InferenceEngine() {
}

bool InferenceEngine::initialize(const InferenceConfig& config) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    config_ = config;
    gpu_enabled_.store(config.backend != BackendType::CPU);

    model_ = std::make_unique<Model>();
    offload_manager_ = std::make_unique<OffloadManager>();
    prefetch_engine_ = std::make_unique<PrefetchEngine>(model_.get());
    tokenizer_ = std::make_unique<Tokenizer>();
    quantization_manager_ = std::make_unique<QuantizationManager>();

    QuantizationConfig q_config;
    quantization_manager_->initialize(q_config);

    if (gpu_enabled_.load()) {
        vulkan::VulkanConfig vk_config;
        vk_config.enable_validation = config.enable_validation;
        vk_config.enable_robust_error_handling = true;

        try {
            vulkan_context_ = std::make_unique<vulkan::VulkanContext>(vk_config);
            if (!vulkan_context_->is_initialized()) {
                throw std::runtime_error("Vulkan context initialization failed");
            }

            if (!vulkan_context_->has_validation()) {
                std::cout << "[InferenceEngine] Validation layers not available, continuing without validation" << std::endl;
            }
            vulkan_memory_ = std::make_unique<vulkan::VulkanMemory>(
                vulkan_context_->get_device(),
                vulkan_context_->get_physical_device()
            );

            shader_compiler_ = std::make_unique<vulkan::ShaderCompiler>(vulkan_context_->get_device());

            pipeline_cache_ = std::make_unique<vulkan::PipelineCache>(
                vulkan_context_->get_device(),
                shader_compiler_.get()
            );

            descriptor_pool_ = std::make_unique<vulkan::DescriptorPool>(vulkan_context_->get_device());

            transfer_engine_ = std::make_unique<vulkan::TransferEngine>(
                vulkan_context_->get_device(),
                vulkan_context_->get_compute_queue(),
                vulkan_context_->get_compute_queue_family()
            );

            timeline_semaphores_ = std::make_unique<vulkan::TimelineSemaphores>(
                vulkan_context_->get_device(),
                2
            );

            if (!timeline_semaphores_->initialize()) {
                std::cerr << "[InferenceEngine] Timeline semaphores not available, falling back to fence-based synchronization" << std::endl;
            }

            async_pipeline_ = std::make_unique<vulkan::AsyncPipelineManager>(
                vulkan_context_->get_device(),
                vulkan_context_->get_compute_queue(),
                vulkan_context_->get_compute_queue(),
                vulkan_context_->get_compute_queue_family(),
                vulkan_context_->get_compute_queue_family()
            );

            if (!async_pipeline_->initialize()) {
                std::cerr << "[InferenceEngine] Async pipeline manager initialization failed" << std::endl;
                async_pipeline_.reset();
            }

            compute_dispatcher_ = std::make_unique<vulkan::ComputeDispatcher>(
                vulkan_context_->get_device(),
                vulkan_context_->get_compute_queue(),
                vulkan_context_->get_compute_queue_family(),
                timeline_semaphores_.get()
            );

            profiler_ = std::make_unique<vulkan::Profiler>(
                vulkan_context_->get_device(),
                vulkan_context_->get_physical_device()
            );

            compute_dispatcher_->set_profiler(profiler_.get());

            std::cout << "[InferenceEngine] Vulkan backend initialized successfully" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[InferenceEngine] Vulkan initialization failed: " << e.what() << std::endl;
            std::cerr << "[InferenceEngine] Falling back to CPU backend" << std::endl;

            vulkan_context_.reset();
            vulkan_memory_.reset();
            shader_compiler_.reset();
            pipeline_cache_.reset();
            descriptor_pool_.reset();
            transfer_engine_.reset();
            timeline_semaphores_.reset();
            compute_dispatcher_.reset();

            gpu_enabled_.store(false);
            config_.backend = BackendType::CPU;
        }
    }

    initialized_.store(true);
    return true;
}

bool InferenceEngine::load_model(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!initialized_.load()) {
        return false;
    }

    if (!model_->load_from_gguf(filepath)) {
        return false;
    }

    context_len_ = model_->get_context_len();
    vocab_size_ = model_->get_vocab_size();

    offload_manager_->set_model(model_->get_weights());
    offload_manager_->set_gpu_cache_size(config_.gpu_cache_mb);

    model_->allocate_tensors();

    if (gpu_enabled_.load() && vulkan_memory_ && model_->get_kv_cache()) {
        try {
            model_->get_kv_cache()->allocate_gpu();
        } catch (const std::exception& e) {
            std::cerr << "[InferenceEngine] GPU cache allocation failed: " << e.what() << std::endl;
            std::cerr << "[InferenceEngine] Falling back to CPU-only mode" << std::endl;
            gpu_enabled_.store(false);
            config_.backend = BackendType::CPU;
        }
    }

    return true;
}

    std::string InferenceEngine::generate(const std::string& prompt, uint32_t max_tokens) {
    std::string result;
    generate_streaming(prompt, max_tokens, [&result](const std::string& token) {
        result += token;
    });
    return result;
}

std::string InferenceEngine::generate_with_progress(const std::string& prompt, uint32_t max_tokens,
                                          ProgressCallback callback) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!initialized_.load() || !model_) {
        return "";
    }

    std::vector<uint32_t> tokens;
    tokenized(prompt, tokens);

    std::vector<uint32_t> generated;
    std::vector<float> logits(vocab_size_);

    uint32_t num_layers = model_->get_num_layers();
    uint32_t hidden_dim = model_->get_hidden_dim();

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t step = 0; step < max_tokens; ++step) {
        auto step_start = std::chrono::steady_clock::now();

        std::vector<float> hidden(hidden_dim);

        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            const inference::LayerWeights& layer_weights = model_->get_layer(layer);

            offload_manager_->update_layer_access(layer);

            if (config_.prefetch_layers > 0 && layer < num_layers - 1) {
                uint32_t prefetch_count = std::min(config_.prefetch_layers, num_layers - layer - 1);
                prefetch_engine_->prefetch_next_layers(layer, prefetch_count);
            }

            forward_layer(layer, step, hidden.data(), hidden.data());
        }

        float next_token_float = sample_token(logits.data(), vocab_size_);
        uint32_t next_token = static_cast<uint32_t>(next_token_float);
        generated.push_back(next_token);
        tokens.push_back(next_token);

        std::string token_text;
        if (tokenizer_) {
            token_text = tokenizer_->decode({next_token});
        } else {
            token_text = std::string(1, static_cast<char>(next_token));
        }

        if (tokens.size() >= config_.context_len) {
            tokens = std::vector<uint32_t>(tokens.end() - config_.context_len, tokens.end());
        }

        auto step_end = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - start_time).count();
        auto step_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start).count();

        if (callback) {
            GenerationProgress progress;
            progress.current_step = step + 1;
            progress.total_steps = max_tokens;
            progress.progress_pct = static_cast<float>(step + 1) / max_tokens * 100.0f;
            progress.time_elapsed_ms = static_cast<float>(total_elapsed);
            progress.estimated_remaining_ms = (step_elapsed * (max_tokens - step - 1));
            progress.tokens_per_second = (step + 1) / (total_elapsed / 1000.0f);
            progress.last_token = token_text;

            callback(progress);
        }
    }

    if (tokenizer_) {
        return tokenizer_->decode(generated);
    } else {
        std::string result;
        for (uint32_t token : generated) {
            result += static_cast<char>(token);
        }
        return result;
    }
}

std::string InferenceEngine::generate_streaming(const std::string& prompt, uint32_t max_tokens,
                                      StreamCallback callback) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!initialized_.load() || !model_) {
        return "";
    }

    std::vector<uint32_t> tokens;
    tokenized(prompt, tokens);

    std::vector<uint32_t> generated;
    std::vector<float> logits(vocab_size_);

    uint32_t num_layers = model_->get_num_layers();
    uint32_t hidden_dim = model_->get_hidden_dim();

    for (uint32_t step = 0; step < max_tokens; ++step) {
        std::vector<float> hidden(hidden_dim);

        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            const inference::LayerWeights& layer_weights = model_->get_layer(layer);

            offload_manager_->update_layer_access(layer);

            if (config_.prefetch_layers > 0 && layer < num_layers - 1) {
                uint32_t prefetch_count = std::min(config_.prefetch_layers, num_layers - layer - 1);
                prefetch_engine_->prefetch_next_layers(layer, prefetch_count);
            }

            forward_layer(layer, step, hidden.data(), hidden.data());
        }

        float next_token_float = sample_token(logits.data(), vocab_size_);
        uint32_t next_token = static_cast<uint32_t>(next_token_float);
        generated.push_back(next_token);
        tokens.push_back(next_token);

        std::string token_text;
        if (tokenizer_) {
            token_text = tokenizer_->decode({next_token});
        } else {
            token_text = std::string(1, static_cast<char>(next_token));
        }

        if (callback) {
            callback(token_text);
        }

        if (tokens.size() >= config_.context_len) {
            tokens = std::vector<uint32_t>(tokens.end() - config_.context_len, tokens.end());
        }
    }

    return "";
}

void InferenceEngine::tokenized(const std::string& prompt, std::vector<uint32_t>& tokens) {
    if (tokenizer_) {
        tokenizer_->encode(prompt, tokens);
    } else {
        for (char c : prompt) {
            tokens.push_back(static_cast<uint32_t>(c));
        }
    }
}

float InferenceEngine::sample_token(const float* logits, uint32_t vocab_size) {
    float max_val = logits[0];
    for (uint32_t i = 1; i < vocab_size; ++i) {
        max_val = std::max(max_val, logits[i]);
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        sum += std::exp(logits[i] - max_val);
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    float rand_val = dis(gen);
    float cumulative = 0.0f;

    for (uint32_t i = 0; i < vocab_size; ++i) {
        float prob = std::exp(logits[i] - max_val) / sum;
        cumulative += prob;
        if (rand_val < cumulative) {
            return static_cast<float>(i);
        }
    }

    return vocab_size - 1;
}

vulkan::ComputeWork InferenceEngine::calculate_workgroups(uint32_t n, uint32_t m, uint32_t batch) {
    if (!vulkan_context_) {
        vulkan::ComputeWork work;
        work.group_count_x = (n + 255) / 256;
        work.group_count_y = (m + 0) / 1;
        work.group_count_z = batch;
        return work;
    }

    vulkan::WorkgroupSize wg = vulkan_context_->calculate_optimal_workgroup_size(n * m);

    vulkan::ComputeWork work;
    work.group_count_x = (n + wg.x - 1) / wg.x;
    work.group_count_y = (m + wg.y - 1) / wg.y;
    work.group_count_z = batch;

    return work;
}

void InferenceEngine::upload_to_gpu(const float* data, VkDeviceSize size,
                                    vulkan::VulkanBuffer& gpu_buffer) {
    if (!data || size == 0 || gpu_buffer.buffer == VK_NULL_HANDLE) {
        return;
    }

    try {
        vulkan::VulkanBuffer staging_buffer = vulkan_memory_->create_buffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        if (!staging_buffer.mapped_ptr) {
            throw std::runtime_error("Failed to map staging buffer");
        }

        memcpy(staging_buffer.mapped_ptr, data, size);

        {
            std::lock_guard<std::mutex> lock(transfer_mutex_);
            transfer_engine_->async_copy(staging_buffer.buffer, gpu_buffer.buffer, size, nullptr);
            transfer_engine_->wait_all();
        }

        vulkan_memory_->destroy_buffer(staging_buffer);
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] GPU upload failed: " << e.what() << std::endl;
        throw;
    }
}

void InferenceEngine::download_from_gpu(VkDeviceSize size, vulkan::VulkanBuffer& gpu_buffer,
                                        float* output) {
    if (!output || size == 0 || gpu_buffer.buffer == VK_NULL_HANDLE) {
        return;
    }

    try {
        vulkan::VulkanBuffer staging_buffer = vulkan_memory_->create_buffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        if (!staging_buffer.mapped_ptr) {
            throw std::runtime_error("Failed to map staging buffer");
        }

        {
            std::lock_guard<std::mutex> lock(transfer_mutex_);
            transfer_engine_->async_copy(gpu_buffer.buffer, staging_buffer.buffer, size, nullptr);
            transfer_engine_->wait_all();
        }

        memcpy(output, staging_buffer.mapped_ptr, size);

        vulkan_memory_->destroy_buffer(staging_buffer);
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] GPU download failed: " << e.what() << std::endl;
        throw;
    }
}

bool InferenceEngine::supports_flash_attention() const {
    return vulkan_context_ != nullptr && vulkan_context_->is_initialized();
}

bool InferenceEngine::should_use_flash_attention(uint32_t layer_id) const {
    if (!supports_flash_attention()) {
        return false;
    }

    return true;
}

void InferenceEngine::forward_layer(uint32_t layer_id, uint32_t position, const float* input, float* output) {
    const LayerWeights& layer_weights = model_->get_layer(layer_id);
    uint32_t hidden_dim = model_->get_hidden_dim();

    if (gpu_enabled_.load() && vulkan_context_ && vulkan_memory_) {
        forward_layer_gpu(layer_id, input, output, layer_weights, hidden_dim);
    } else {
        forward_layer_cpu(layer_id, input, output, layer_weights, hidden_dim);
    }
}

void InferenceEngine::forward_layer_gpu(uint32_t layer_id, const float* input, float* output,
                                        const LayerWeights& layer_weights, uint32_t hidden_dim) {
    std::string layer_name = "layer_" + std::to_string(layer_id);

    if (gpu_buffers_.find(layer_name) == gpu_buffers_.end()) {
        GPUBuffers buffers;
        buffers.tensor_size = hidden_dim * sizeof(float);

        buffers.input_buffer = vulkan_memory_->create_buffer(
            buffers.tensor_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        buffers.output_buffer = vulkan_memory_->create_buffer(
            buffers.tensor_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        gpu_buffers_[layer_name] = buffers;
    }

    GPUBuffers& buffers = gpu_buffers_[layer_name];

    if (profiler_) profiler_->start_gpu_timing(compute_dispatcher_->get_command_buffer(), "transfer_upload", layer_id);
    upload_to_gpu(input, buffers.tensor_size, buffers.input_buffer);
    if (profiler_) profiler_->end_gpu_timing(compute_dispatcher_->get_command_buffer(), "transfer_upload", layer_id);

    uint32_t num_heads = model_->get_num_heads();
    uint32_t head_dim = hidden_dim / num_heads;

    vulkan::ComputeWork work = calculate_workgroups(hidden_dim, 1);

    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}
    };

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = nullptr;

    VkPipelineLayout layout = pipeline_cache_->get_pipeline_layout(layer_name + "_layout", layout_info);

    if (vulkan_context_->get_subgroup_size() >= 32) {
        VkPipeline rms_pipeline = pipeline_cache_->get_compute_pipeline("rms_norm_subgroup.glsl", layout_info);
        if (rms_pipeline != VK_NULL_HANDLE) {
            compute_dispatcher_->dispatch(rms_pipeline, layout, work, "rms_norm", layer_id);
            compute_dispatcher_->wait_for_completion();
        }
    }

    if (should_use_flash_attention(layer_id)) {
        VkPipeline flash_pipeline = pipeline_cache_->get_compute_pipeline("flash_attention.glsl", layout_info);
        if (flash_pipeline != VK_NULL_HANDLE) {
            compute_dispatcher_->dispatch(flash_pipeline, layout, work, "attention", layer_id);
            compute_dispatcher_->wait_for_completion();
        }
    }

    if (profiler_) profiler_->start_gpu_timing(compute_dispatcher_->get_command_buffer(), "transfer_download", layer_id);
    download_from_gpu(buffers.tensor_size, buffers.output_buffer, output);
    if (profiler_) profiler_->end_gpu_timing(compute_dispatcher_->get_command_buffer(), "transfer_download", layer_id);

    memcpy(output, input, hidden_dim * sizeof(float));
}

void InferenceEngine::forward_layer_cpu(uint32_t layer_id, const float* input, float* output,
                                        const LayerWeights& layer_weights, uint32_t hidden_dim) {
    std::vector<float> q(hidden_dim);
    std::vector<float> k(hidden_dim);
    std::vector<float> v(hidden_dim);

    if (gpu_enabled_.load()) {
        if (offload_manager_->should_offload(layer_weights.q_proj)) {
            offload_manager_->offload_tensor(layer_weights.q_proj, true);
        }
    }

    if (layer_weights.q_proj && layer_weights.q_proj->get_cpu_data()) {
        const float* q_proj = static_cast<const float*>(layer_weights.q_proj->get_cpu_data());
        cpu::gemm_avx2_f32(input, q_proj, q.data(), 1, hidden_dim, hidden_dim);
    }

    if (layer_weights.k_proj && layer_weights.k_proj->get_cpu_data()) {
        const float* k_proj = static_cast<const float*>(layer_weights.k_proj->get_cpu_data());
        cpu::gemm_avx2_f32(input, k_proj, k.data(), 1, hidden_dim, hidden_dim);
    }

    if (layer_weights.v_proj && layer_weights.v_proj->get_cpu_data()) {
        const float* v_proj = static_cast<const float*>(layer_weights.v_proj->get_cpu_data());
        cpu::gemm_avx2_f32(input, v_proj, v.data(), 1, hidden_dim, hidden_dim);
    }

    memcpy(output, input, hidden_dim * sizeof(float));
}

void InferenceEngine::set_num_threads(uint32_t num_threads) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    config_.num_threads = num_threads;
}

void InferenceEngine::set_gpu_enabled(bool enabled) {
    gpu_enabled_.store(enabled);
    config_.backend = enabled ? BackendType::GPU : BackendType::CPU;
}

void InferenceEngine::set_gpu_memory_pool(size_t size_mb) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    config_.gpu_memory_pool_mb = size_mb;
}

void InferenceEngine::set_gpu_cache_size(size_t size_mb) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    config_.gpu_cache_mb = size_mb;
    offload_manager_->set_gpu_cache_size(size_mb);
}

void InferenceEngine::set_kv_cache_size(uint32_t size) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    config_.context_len = size;
}

void InferenceEngine::set_prefetch_layers(uint32_t count) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    config_.prefetch_layers = count;
}

bool InferenceEngine::is_gpu_enabled() const {
    return gpu_enabled_.load();
}

uint32_t InferenceEngine::get_num_threads() const {
    return config_.num_threads;
}

uint32_t InferenceEngine::get_model_size() const {
    return static_cast<uint32_t>(get_model_size_bytes() / (1024 * 1024));
}

uint32_t InferenceEngine::get_model_layers() const {
    if (!model_) return 0;
    return model_->get_num_layers();
}

uint32_t InferenceEngine::get_hidden_dim() const {
    if (!model_) return 0;
    return model_->get_hidden_dim();
}

uint32_t InferenceEngine::get_context_len() const {
    if (!model_) return 0;
    return model_->get_context_len();
}

uint32_t InferenceEngine::get_num_heads() const {
    if (!model_) return 0;
    return model_->get_num_heads();
}

float InferenceEngine::get_acceptance_rate() const {
    return 0.0f;
}

float InferenceEngine::get_throughput() const {
    return 0.0f;
}

bool InferenceEngine::save_pipeline_cache(const std::string& path) {
    if (!pipeline_cache_) return false;
    return pipeline_cache_->save_to_disk(path);
}

bool InferenceEngine::load_pipeline_cache(const std::string& path) {
    if (!pipeline_cache_) return false;
    return pipeline_cache_->load_from_disk(path);
}

void InferenceEngine::enable_profiling(bool enable) {
    if (profiler_) {
        if (enable) {
            profiler_->begin_profiling();
            profiler_->enable_vulkan_timestamps();
        } else {
            profiler_->end_profiling();
            profiler_->disable_vulkan_timestamps();
        }
    }
}

std::string InferenceEngine::get_performance_report() const {
    if (profiler_) {
        auto data = profiler_->get_profiling_data();
        std::string report = "Performance Report:\n";
        report += "Total tokens: " + std::to_string(data.total_tokens) + "\n";
        report += "Tokens/sec: " + std::to_string(data.tokens_per_second) + "\n";
        report += "Peak memory: " + std::to_string(data.peak_memory_bytes / 1024 / 1024) + " MB\n";
        for (const auto& metric : data.metrics) {
            report += "Layer " + std::to_string(metric.layer_id) + " " + metric.operation + ": " + std::to_string(metric.duration_ms) + " ms\n";
        }
        return report;
    }
    return "Profiling not enabled";
}

std::vector<float> InferenceEngine::get_token_logits(const std::vector<uint32_t>& tokens) {
    return std::vector<float>();
}

void InferenceEngine::set_adapter_alpha(const std::string& adapter_name, float alpha) {
}

void InferenceEngine::enable_adapter(const std::string& adapter_name) {
}

void InferenceEngine::disable_adapter(const std::string& adapter_name) {
}

std::vector<std::string> InferenceEngine::get_loaded_adapters() const {
    return std::vector<std::string>();
}

void InferenceEngine::set_quantization_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    QuantizationConfig config = quantization_manager_->get_config();
    if (enabled) {
        config.weight_quantization = QuantizationType::INT8;
    } else {
        config.weight_quantization = QuantizationType::NONE;
    }
    quantization_manager_->initialize(config);
}

void InferenceEngine::set_weight_quantization(QuantizationType type) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    QuantizationConfig config = quantization_manager_->get_config();
    config.weight_quantization = type;
    quantization_manager_->initialize(config);
}

void InferenceEngine::set_activation_quantization(QuantizationType type) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    QuantizationConfig config = quantization_manager_->get_config();
    config.activation_quantization = type;
    quantization_manager_->initialize(config);
}

QuantizationConfig InferenceEngine::get_quantization_config() const {
    return quantization_manager_->get_config();
}

size_t InferenceEngine::get_model_size_bytes() const {
    if (!model_) return 0;

    size_t total_size = 0;
    const auto& weights = model_->get_weights();

    for (const auto& pair : weights) {
        ggml::Tensor* tensor = pair.second;
        if (tensor) {
            const auto& shape = tensor->get_shape();
            size_t num_elements = 1;
            for (uint32_t dim : shape) {
                num_elements *= dim;
            }

            ggml::GGMLType type = tensor->get_type();
            size_t type_size = ggml::ggml_type_size(type);
            size_t block_size = ggml::ggml_blck_size(type);

            size_t tensor_size = ((num_elements + block_size - 1) / block_size) * type_size;
            total_size += tensor_size;
        }
    }

    return total_size;
}

}
