#include "model.h"
#include "../core/gguf_parser.h"
#include "../vulkan_backend/context.h"
#include "../vulkan_backend/memory.h"
#include <algorithm>
#include <sstream>
#include <cstring>

namespace inference {

KVCache::KVCache(uint32_t layers, uint32_t hidden_dim, uint32_t context_len)
    : layers_(layers), hidden_dim_(hidden_dim), context_len_(context_len),
      allocated_len_(0), vulkan_memory_(nullptr) {
    k_cache_.resize(layers_);
    v_cache_.resize(layers_);
}

KVCache::~KVCache() {
    cleanup_gpu();

    for (auto& k : k_cache_) {
        if (k) {
            delete[] k;
        }
    }
    for (auto& v : v_cache_) {
        if (v) {
            delete[] v;
        }
    }
}

void KVCache::set_vulkan_memory(vulkan::VulkanMemory* vulkan_memory) {
    vulkan_memory_ = vulkan_memory;
}

void KVCache::allocate_gpu() {
    if (!vulkan_memory_) {
        return;
    }

    cleanup_gpu();

    size_t cache_size = context_len_ * hidden_dim_ * sizeof(float);

    for (uint32_t layer = 0; layer < layers_; ++layer) {
        vulkan::VulkanBuffer k_buffer = vulkan_memory_->create_buffer(
            cache_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        vulkan::VulkanBuffer v_buffer = vulkan_memory_->create_buffer(
            cache_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        k_gpu_buffers_.push_back(k_buffer);
        v_gpu_buffers_.push_back(v_buffer);
    }

    for (uint32_t layer = 0; layer < layers_; ++layer) {
        k_cache_[layer] = new float[context_len_ * hidden_dim_]();
        v_cache_[layer] = new float[context_len_ * hidden_dim_]();
    }

    allocated_len_ = context_len_;
}

void KVCache::cleanup_gpu() {
    if (vulkan_memory_) {
        for (auto& buffer : k_gpu_buffers_) {
            vulkan_memory_->destroy_buffer(buffer);
        }
        for (auto& buffer : v_gpu_buffers_) {
            vulkan_memory_->destroy_buffer(buffer);
        }
        k_gpu_buffers_.clear();
        v_gpu_buffers_.clear();
    }
}

void KVCache::allocate_gpu_up_to(uint32_t max_pos) {
    if (max_pos <= allocated_len_) return;
    if (max_pos > context_len_) max_pos = context_len_;

    for (uint32_t layer = 0; layer < layers_; ++layer) {
        for (uint32_t pos = allocated_len_; pos < max_pos; ++pos) {
            uint32_t offset = pos * hidden_dim_;
            k_cache_[layer][offset] = 0.0f;
            v_cache_[layer][offset] = 0.0f;
        }
    }

    allocated_len_ = max_pos;
}

void KVCache::store(uint32_t layer, uint32_t pos, const float* k, const float* v) {
    if (layer >= layers_ || pos >= context_len_) return;

    float* k_layer = k_cache_[layer];
    float* v_layer = v_cache_[layer];

    memcpy(k_layer + pos * hidden_dim_, k, hidden_dim_ * sizeof(float));
    memcpy(v_layer + pos * hidden_dim_, v, hidden_dim_ * sizeof(float));
}

void KVCache::retrieve(uint32_t layer, uint32_t pos, float* k, float* v) const {
    if (layer >= layers_ || pos >= context_len_) return;

    const float* k_layer = k_cache_[layer];
    const float* v_layer = v_cache_[layer];

    memcpy(k, k_layer + pos * hidden_dim_, hidden_dim_ * sizeof(float));
    memcpy(v, v_layer + pos * hidden_dim_, hidden_dim_ * sizeof(float));
}

VkBuffer KVCache::get_gpu_k_buffer(uint32_t layer) const {
    if (layer < k_gpu_buffers_.size()) {
        return k_gpu_buffers_[layer].buffer;
    }
    return VK_NULL_HANDLE;
}

VkBuffer KVCache::get_gpu_v_buffer(uint32_t layer) const {
    if (layer < v_gpu_buffers_.size()) {
        return v_gpu_buffers_[layer].buffer;
    }
    return VK_NULL_HANDLE;
}

Model::Model() 
    : kv_cache_(nullptr), num_layers_(0), hidden_dim_(0), 
      num_heads_(0), context_len_(0) {
}

Model::~Model() {
    free_tensors();
    if (kv_cache_) {
        delete kv_cache_;
    }
}

static uint32_t safe_stoul(const std::string& str, uint32_t default_val) {
    if (str.empty()) return default_val;
    try {
        return std::stoul(str);
    } catch (...) {
        return default_val;
    }
}

bool Model::load_from_gguf(const std::string& filepath) {
    ggml::GGUFParser parser;
    if (!parser.parse(filepath)) {
        return false;
    }
    
    const auto& tensors = parser.get_tensors();
    const auto& metadata = parser.get_metadata();
    
    auto get_meta = [&](const std::string& key) -> std::string {
        auto it = metadata.find(key);
        return it != metadata.end() ? it->second : "";
    };
    
    num_layers_ = safe_stoul(get_meta("llama.block_count"), 32);
    hidden_dim_ = safe_stoul(get_meta("llama.embedding_length"), 4096);
    num_heads_ = safe_stoul(get_meta("llama.attention.head_count"), 32);
    context_len_ = safe_stoul(get_meta("llama.context_length"), 2048);
    vocab_size_ = safe_stoul(get_meta("llama.vocab_size"), 32000);

    detect_architecture(metadata);
    
    layers_.resize(num_layers_);
    
    for (const auto& tensor_info : tensors) {
        std::string name = tensor_info.name;
        ggml::Tensor* tensor = new ggml::Tensor(name, tensor_info.type, tensor_info.shape);
        tensor->allocate_cpu();
        
        if (!parser.read_tensor_data_direct(tensor_info, reinterpret_cast<uint8_t*>(tensor->get_cpu_data()), tensor->get_size())) {
            delete tensor;
            return false;
        }
        
        weights_[name] = tensor;
        
        uint32_t layer_id = UINT32_MAX;
        std::string weight_name;
        
        size_t layer_start = 0;
        size_t layer_end = 0;
        
        // Try "layers.N."
        size_t pos = name.find(".layers.");
        if (pos != std::string::npos) {
            layer_start = pos + 8;
        } else {
            // Try "blk.N."
            pos = name.find("blk.");
            if (pos == 0 || (pos != std::string::npos && name[pos-1] == '.')) {
                layer_start = pos + 4;
            }
        }
        
        if (layer_start > 0) {
            size_t next_dot = name.find(".", layer_start);
            if (next_dot != std::string::npos) {
                try {
                    layer_id = std::stoul(name.substr(layer_start, next_dot - layer_start));
                    weight_name = name.substr(next_dot + 1);
                } catch (...) {
                    layer_id = UINT32_MAX;
                }
            }
        }
        
        if (layer_id < num_layers_) {
            LayerWeights& layer = layers_[layer_id];
            
            // Normalize name for matching
            // Common patterns:
            // blk.N.attn_q.weight
            // layers.N.attention.wq.weight
            // layers.N.attention.wq
            
            bool is_weight = (weight_name.length() > 7 && weight_name.substr(weight_name.length() - 7) == ".weight");
            std::string base_name = is_weight ? weight_name.substr(0, weight_name.length() - 7) : weight_name;

            // Attention
            if (base_name == "attention.wq" || base_name == "attn_q") layer.q_proj = tensor;
            else if (base_name == "attention.wk" || base_name == "attn_k") layer.k_proj = tensor;
            else if (base_name == "attention.wv" || base_name == "attn_v") layer.v_proj = tensor;
            else if (base_name == "attention.wo" || base_name == "attn_output") layer.o_proj = tensor;
            
            // FFN
            else if (base_name == "feed_forward.w1" || base_name == "mlp_gate") layer.gate_proj = tensor;
            else if (base_name == "feed_forward.w3" || base_name == "mlp_up") layer.up_proj = tensor;
            else if (base_name == "feed_forward.w2" || base_name == "mlp_down") layer.down_proj = tensor;
            
            // Norms
            else if (base_name == "attention_norm" || base_name == "attn_norm") layer.norm1 = tensor;
            else if (base_name == "ffn_norm") layer.norm2 = tensor;
        }
    }
    
    kv_cache_ = new KVCache(num_layers_, hidden_dim_, context_len_);
    kv_cache_->allocate_gpu();
    
    allocate_tensors();
    return true;
}

void Model::allocate_tensors() {
    for (auto& pair : weights_) {
        pair.second->allocate_cpu();
    }
}

void Model::free_tensors() {
    for (auto& pair : weights_) {
        pair.second->free_cpu();
        delete pair.second;
    }
    weights_.clear();
}

const LayerWeights& Model::get_layer(uint32_t idx) const {
    return layers_[idx];
}

bool Model::detect_architecture(const std::map<std::string, std::string>& metadata) {
    auto get_meta = [&](const std::string& key) -> std::string {
        auto it = metadata.find(key);
        return it != metadata.end() ? it->second : "";
    };

    std::string general_arch = get_meta("general.architecture");
    std::string rope_type = get_meta("llama.rope.type");

    if (general_arch.find("llama") != std::string::npos) {
        if (general_arch.find("llama-3") != std::string::npos) {
            architecture_ = ModelArchitecture::LLAMA3;
        } else if (general_arch.find("llama-2") != std::string::npos) {
            architecture_ = ModelArchitecture::LLAMA2;
        } else {
            architecture_ = ModelArchitecture::LLAMA;
        }
    } else if (general_arch.find("mistral") != std::string::npos) {
        if (general_arch.find("mixtral") != std::string::npos) {
            architecture_ = ModelArchitecture::MIXTRAL;
        } else {
            architecture_ = ModelArchitecture::MISTRAL;
        }
    } else if (general_arch.find("gemma") != std::string::npos) {
        if (general_arch.find("gemma-2") != std::string::npos) {
            architecture_ = ModelArchitecture::GEMMA2;
        } else {
            architecture_ = ModelArchitecture::GEMMA;
        }
    } else if (general_arch.find("qwen") != std::string::npos) {
        if (general_arch.find("qwen2") != std::string::npos) {
            architecture_ = ModelArchitecture::QWEN2;
        } else {
            architecture_ = ModelArchitecture::QWEN;
        }
    } else if (general_arch.find("phi") != std::string::npos) {
        std::string phi_version = get_meta("general.version");
        if (phi_version.find("2") != std::string::npos) {
            architecture_ = ModelArchitecture::PHI2;
        } else if (phi_version.find("3") != std::string::npos) {
            architecture_ = ModelArchitecture::PHI3;
        } else {
            architecture_ = ModelArchitecture::PHI;
        }
    } else {
        architecture_ = ModelArchitecture::UNKNOWN;
    }

    return true;
}

std::string Model::get_architecture_str() const {
    switch (architecture_) {
        case ModelArchitecture::LLAMA: return "LLaMA";
        case ModelArchitecture::LLAMA2: return "LLaMA-2";
        case ModelArchitecture::LLAMA3: return "LLaMA-3";
        case ModelArchitecture::MISTRAL: return "Mistral";
        case ModelArchitecture::MIXTRAL: return "Mixtral";
        case ModelArchitecture::GEMMA: return "Gemma";
        case ModelArchitecture::GEMMA2: return "Gemma-2";
        case ModelArchitecture::QWEN: return "Qwen";
        case ModelArchitecture::QWEN2: return "Qwen2";
        case ModelArchitecture::PHI: return "Phi";
        case ModelArchitecture::PHI2: return "Phi-2";
        case ModelArchitecture::PHI3: return "Phi-3";
        default: return "Unknown";
    }
}

}
