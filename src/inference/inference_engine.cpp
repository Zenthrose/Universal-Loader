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
#include <stdexcept>

// Helper for FP16 to FP32 conversion
static float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 0x00000001;
    uint32_t e = (h >> 10) & 0x0000001f;
    uint32_t m = h & 0x000003ff;

    if (e == 0) {
        if (m == 0) {
            return s ? -0.0f : 0.0f;
        } else {
            // Denormalized
            return (s ? -1.0f : 1.0f) * std::ldexp((float)m, -24);
        }
    } else if (e == 31) {
        if (m == 0) {
            return s ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }

    float f = std::ldexp((float)(m + 1024), e - 25);
    return s ? -f : f;
}

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
    // Helper for creating params buffer and dispatching
    auto dispatch_simple = [&](const std::string& pipeline_name, 
                               const std::vector<vulkan::VulkanBuffer*>& buffers, 
                               void* params, size_t params_size, 
                               uint32_t gx, uint32_t gy, uint32_t gz,
                               const std::string& layout_name) {
        vulkan::DescriptorPool pool(vulkan_context_->get_device());
        std::vector<VkDescriptorPoolSize> pool_sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)buffers.size() + 1}};
        pool.create_descriptor_pool(pool_sizes, 1);
        
        std::vector<vulkan::DescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < buffers.size(); i++) {
            bindings.push_back({i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT});
        }
        // Params binding (last)
        uint32_t params_binding = (uint32_t)buffers.size();
        bindings.push_back({params_binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT});
        
        pool.create_descriptor_set_layouts(bindings);
        VkDescriptorSet set;
        VkDescriptorSetLayout layout = pool.get_layout(0);
        pool.allocate_descriptor_set(set, layout);
        
        for (uint32_t i = 0; i < buffers.size(); i++) {
            pool.update_buffer_descriptor(set, i, buffers[i]->buffer, buffers[i]->size);
        }
        
        vulkan::VulkanBuffer p_buf = vulkan_memory_->create_buffer(params_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        memcpy(p_buf.mapped_ptr, params, params_size);
        pool.update_buffer_descriptor(set, params_binding, p_buf.buffer, p_buf.size);
        
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &layout;
        
        VkPipeline pipeline = pipeline_cache_->get_compute_pipeline(pipeline_name, layout_info);
        VkPipelineLayout pipe_layout = pipeline_cache_->get_pipeline_layout(layout_name, layout_info);

        if (pipeline != VK_NULL_HANDLE) {
            vulkan::ComputeWork work = {gx, gy, gz};
            compute_dispatcher_->dispatch(pipeline, pipe_layout, set, work);
        }
        compute_dispatcher_->wait_for_completion();
        vulkan_memory_->destroy_buffer(p_buf);
    };

    auto run_norm = [&](vulkan::VulkanBuffer& in, vulkan::VulkanBuffer& w, vulkan::VulkanBuffer& out, uint32_t n_elements) {
        struct Params { uint32_t N; uint32_t p1; uint32_t p2; uint32_t p3; } p = {n_elements, 0, 0, 0};
        std::vector<vulkan::VulkanBuffer*> bufs = {&in, &w, &out};
        dispatch_simple("activation/rms_norm_subgroup.glsl", bufs, &p, sizeof(p), (n_elements + 255)/256, 1, 1, "rms_layout_v2");
    };
    
    // Q4_K Linear Helper (Dequant + GEMM)
    auto run_linear_q4k = [&](ggml::Tensor* weight, vulkan::VulkanBuffer& input_buf, vulkan::VulkanBuffer& output_buf) {
        if (!weight) return;
        std::string w_name = weight->get_name();
        
        if (gpu_buffers_.find(w_name) == gpu_buffers_.end()) {
             GPUBuffers w_bufs;
             std::vector<uint32_t> q; std::vector<float> s, m;
             repack_q4k_tensor(weight, q, s, m);
             w_bufs.q4k_quants = vulkan_memory_->create_buffer(q.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
             w_bufs.q4k_scales = vulkan_memory_->create_buffer(s.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
             w_bufs.q4k_mins = vulkan_memory_->create_buffer(m.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
             upload_to_gpu((float*)q.data(), q.size()*4, w_bufs.q4k_quants);
             upload_to_gpu(s.data(), s.size()*4, w_bufs.q4k_scales);
             upload_to_gpu(m.data(), m.size()*4, w_bufs.q4k_mins);
             w_bufs.is_uploaded = true;
             gpu_buffers_[w_name] = w_bufs;
        }
        GPUBuffers& w_bufs = gpu_buffers_[w_name];
        size_t n_in = weight->get_shape()[0];
        size_t n_out = weight->get_shape()[1];
        size_t num_elements = n_in * n_out;

        if (w_bufs.dequantized_buffer.buffer == VK_NULL_HANDLE) {
             w_bufs.dequantized_buffer = vulkan_memory_->create_buffer(num_elements*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        
        // Dequantize (0:Q, 1:S, 2:M, 3:Out, 4:Params)
        struct DQParams { uint32_t N; } dq_p = {(uint32_t)num_elements};
        std::vector<vulkan::VulkanBuffer*> dq_bufs = {&w_bufs.q4k_quants, &w_bufs.q4k_scales, &w_bufs.q4k_mins, &w_bufs.dequantized_buffer};
        dispatch_simple("dequantize/dequantize_q4_k_flattened.glsl", dq_bufs, &dq_p, sizeof(dq_p), (uint32_t)(num_elements+255)/256, 1, 1, "dq_layout_v2");
        
        // GEMM (0:In, 1:W, 2:Out, 3:Params)
        struct GemmParams { uint32_t M, N, K; } g_p = {1, (uint32_t)n_out, (uint32_t)n_in};
        std::vector<vulkan::VulkanBuffer*> g_bufs = {&input_buf, &w_bufs.dequantized_buffer, &output_buf};
        dispatch_simple("gemm/gemm_transposed_b_v2.glsl", g_bufs, &g_p, sizeof(g_p), (g_p.M+7)/8, (g_p.N+7)/8, 1, "gemm_layout_v2");
        
        vulkan_memory_->destroy_buffer(w_bufs.dequantized_buffer);
        w_bufs.dequantized_buffer = {VK_NULL_HANDLE, VK_NULL_HANDLE, 0, nullptr};
    };
    
    // Preparation
    std::string layer_name = "layer_" + std::to_string(layer_id);
    if (gpu_buffers_.find(layer_name) == gpu_buffers_.end()) {
        GPUBuffers buffers;
        buffers.tensor_size = hidden_dim * sizeof(float);
        buffers.input_buffer = vulkan_memory_->create_buffer(buffers.tensor_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffers.output_buffer = vulkan_memory_->create_buffer(buffers.tensor_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        gpu_buffers_[layer_name] = buffers;
    }
    GPUBuffers& l_buf = gpu_buffers_[layer_name];
    upload_to_gpu(input, l_buf.tensor_size, l_buf.input_buffer);
    
    auto create_temp = [&](size_t size) { return vulkan_memory_->create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); };
    
    uint32_t head_dim = 128;
    uint32_t num_heads = model_->get_num_heads();
    if (hidden_dim > 0 && num_heads > 0) head_dim = hidden_dim / num_heads;
    
    // Intermediate Buffers
    vulkan::VulkanBuffer norm_1 = create_temp(hidden_dim * 4);
    vulkan::VulkanBuffer q = create_temp(hidden_dim * 4);
    vulkan::VulkanBuffer k = create_temp(hidden_dim * 4); 
    vulkan::VulkanBuffer v = create_temp(hidden_dim * 4);
    vulkan::VulkanBuffer q_r = create_temp(hidden_dim * 4);
    vulkan::VulkanBuffer k_r = create_temp(hidden_dim * 4);
    vulkan::VulkanBuffer attn_out = create_temp(hidden_dim * 4); 
    vulkan::VulkanBuffer o_out = create_temp(hidden_dim * 4); 
    vulkan::VulkanBuffer res_1 = create_temp(hidden_dim * 4); 
    vulkan::VulkanBuffer norm_2 = create_temp(hidden_dim * 4);
    
    // 1. Norm 1 (Input -> Norm1)
    if (layer_weights.norm1) {
         std::string n_name = layer_weights.norm1->get_name();
         if (gpu_buffers_.find(n_name) == gpu_buffers_.end()) {
             GPUBuffers nb;
             size_t bsz = layer_weights.norm1->get_size();
             nb.input_buffer = vulkan_memory_->create_buffer(bsz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
             upload_to_gpu((float*)layer_weights.norm1->get_cpu_data(), bsz, nb.input_buffer);
             gpu_buffers_[n_name] = nb;
         }
         run_norm(l_buf.input_buffer, gpu_buffers_[n_name].input_buffer, norm_1, hidden_dim);
    }

    // 2. Q, K, V
    if (layer_weights.q_proj) run_linear_q4k(layer_weights.q_proj, norm_1, q);
    if (layer_weights.k_proj) run_linear_q4k(layer_weights.k_proj, norm_1, k);
    if (layer_weights.v_proj) run_linear_q4k(layer_weights.v_proj, norm_1, v);

    // 3. RoPE
    struct RopeParams { uint32_t N, hd, pos, seq; } rp = {hidden_dim, head_dim, 0, 1};
    std::vector<vulkan::VulkanBuffer*> rope_q_bufs = {&q, &q_r};
    dispatch_simple("rope.glsl", rope_q_bufs, &rp, sizeof(rp), (hidden_dim+255)/256, 1, 1, "rope_layout");
    std::vector<vulkan::VulkanBuffer*> rope_k_bufs = {&k, &k_r};
    dispatch_simple("rope.glsl", rope_k_bufs, &rp, sizeof(rp), (hidden_dim+255)/256, 1, 1, "rope_layout");
    
    // 4. Attention
    struct AttnParams { uint32_t seq, hd, nh, pad; } ap = {1, head_dim, num_heads, 0};
    std::vector<vulkan::VulkanBuffer*> attn_bufs = {&q_r, &k_r, &v, &attn_out};
    dispatch_simple("attention/flash_attention.glsl", attn_bufs, &ap, sizeof(ap), 1, num_heads, 1, "attn_layout");

    // 5. O Proj
    if (layer_weights.o_proj) run_linear_q4k(layer_weights.o_proj, attn_out, o_out);

    // 6. Residual 1
    struct AddParams { uint32_t N; } add_p = {hidden_dim};
    std::vector<vulkan::VulkanBuffer*> add_bufs = {&l_buf.input_buffer, &o_out, &res_1};
    dispatch_simple("add.glsl", add_bufs, &add_p, sizeof(add_p), (hidden_dim+255)/256, 1, 1, "add_layout");

    // 7. FFN Norm
    if (layer_weights.norm2) {
         std::string n_name = layer_weights.norm2->get_name();
         if (gpu_buffers_.find(n_name) == gpu_buffers_.end()) {
             GPUBuffers nb;
             size_t bsz = layer_weights.norm2->get_size();
             nb.input_buffer = vulkan_memory_->create_buffer(bsz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
             upload_to_gpu((float*)layer_weights.norm2->get_cpu_data(), bsz, nb.input_buffer);
             gpu_buffers_[n_name] = nb;
         }
         run_norm(res_1, gpu_buffers_[n_name].input_buffer, norm_2, hidden_dim);
    }

    // 8. FFN Weights
    uint32_t ffn_dim = 0;
    if (layer_weights.gate_proj) ffn_dim = layer_weights.gate_proj->get_shape()[1]; // out dim
    if (ffn_dim == 0) ffn_dim = hidden_dim * 4; 
    
    vulkan::VulkanBuffer gate = create_temp(ffn_dim * 4);
    vulkan::VulkanBuffer up = create_temp(ffn_dim * 4);
    vulkan::VulkanBuffer silu_gate = create_temp(ffn_dim * 4);
    vulkan::VulkanBuffer mul_res = create_temp(ffn_dim * 4);
    vulkan::VulkanBuffer down = create_temp(hidden_dim * 4);

    if (layer_weights.gate_proj) run_linear_q4k(layer_weights.gate_proj, norm_2, gate);
    if (layer_weights.up_proj) run_linear_q4k(layer_weights.up_proj, norm_2, up);

    // 9. Silu
    struct SiluParams { uint32_t N; } sp = {ffn_dim};
    std::vector<vulkan::VulkanBuffer*> silu_bufs = {&gate, &silu_gate};
    dispatch_simple("activation/silu.glsl", silu_bufs, &sp, sizeof(sp), (ffn_dim+255)/256, 1, 1, "silu_layout");

    // 10. Mul
    struct MulParams { uint32_t N; } mp = {ffn_dim};
    std::vector<vulkan::VulkanBuffer*> mul_bufs = {&silu_gate, &up, &mul_res};
    dispatch_simple("mul.glsl", mul_bufs, &mp, sizeof(mp), (ffn_dim+255)/256, 1, 1, "mul_layout");

    // 11. Down Proj
    if (layer_weights.down_proj) run_linear_q4k(layer_weights.down_proj, mul_res, down);

    // 12. Residual 2
    std::vector<vulkan::VulkanBuffer*> res2_bufs = {&res_1, &down, &l_buf.output_buffer};
    dispatch_simple("add.glsl", res2_bufs, &add_p, sizeof(add_p), (hidden_dim+255)/256, 1, 1, "add_layout");

    // Cleanup Temps
    vulkan_memory_->destroy_buffer(norm_1); vulkan_memory_->destroy_buffer(q); vulkan_memory_->destroy_buffer(k); vulkan_memory_->destroy_buffer(v);
    vulkan_memory_->destroy_buffer(q_r); vulkan_memory_->destroy_buffer(k_r); vulkan_memory_->destroy_buffer(attn_out);
    vulkan_memory_->destroy_buffer(o_out); vulkan_memory_->destroy_buffer(res_1); vulkan_memory_->destroy_buffer(norm_2);
    vulkan_memory_->destroy_buffer(gate); vulkan_memory_->destroy_buffer(up); vulkan_memory_->destroy_buffer(silu_gate);
    vulkan_memory_->destroy_buffer(mul_res); vulkan_memory_->destroy_buffer(down);

    download_from_gpu(l_buf.tensor_size, l_buf.output_buffer, output);
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

std::string InferenceEngine::get_architecture_name() const {
    if (!model_) return "None";
    return model_->get_architecture_str();
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
}

std::string InferenceEngine::get_performance_report() const {
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

void InferenceEngine::repack_q4k_tensor(const ggml::Tensor* tensor, 
                                        std::vector<uint32_t>& quants, 
                                        std::vector<float>& scales, 
                                        std::vector<float>& mins) {
    if (!tensor || !tensor->get_cpu_data()) return;

    const uint8_t* data = static_cast<const uint8_t*>(tensor->get_cpu_data());
    size_t size_bytes = tensor->get_size();
    
    // Q4_K block size is 256 weights. Block size in bytes is 144.
    size_t num_blocks = size_bytes / 144; 
    
    quants.resize(num_blocks * 32); 
    scales.resize(num_blocks * 8);  
    mins.resize(num_blocks * 8);    

    for (size_t i = 0; i < num_blocks; ++i) {
        const uint8_t* block = data + i * 144;
        
        uint16_t d_fp16 = *reinterpret_cast<const uint16_t*>(block);
        uint16_t min_fp16 = *reinterpret_cast<const uint16_t*>(block + 2);
        
        float d = half_to_float(d_fp16);
        float min_val = half_to_float(min_fp16);
        
        for (int j = 0; j < 8; ++j) {
            scales[i * 8 + j] = d;
            mins[i * 8 + j] = min_val;
        }

        const uint8_t* qs = block + 16;
        for (int j = 0; j < 32; ++j) {
            uint32_t q_word = *reinterpret_cast<const uint32_t*>(qs + j * 4);
            quants[i * 32 + j] = q_word;
        }
    }
}

}
