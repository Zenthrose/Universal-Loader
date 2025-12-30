#include "model_v2.h"
#include "../core/gguf_parser.h"
#include <algorithm>
#include <sstream>

namespace inference {

ModelV2::ModelV2()
    : kv_cache_(nullptr), num_layers_(0), hidden_dim_(0),
      num_heads_(0), num_kv_heads_(0), context_len_(0),
      vocab_size_(0), num_experts_(0),
      architecture_(ModelArchitecture::UNKNOWN), is_moe_(false), has_gqa_(false) {
}

ModelV2::~ModelV2() {
    free_tensors();
    if (kv_cache_) {
        delete kv_cache_;
    }
}

bool ModelV2::load_from_gguf(const std::string& filepath) {
    ggml::GGUFParser parser;
    if (!parser.parse(filepath)) {
        return false;
    }

    const auto& tensors = parser.get_tensors();
    const auto& metadata = parser.get_metadata();

    if (!detect_architecture(metadata)) {
        return false;
    }

    auto get_meta = [&](const std::string& key) -> std::string {
        auto it = metadata.find(key);
        return it != metadata.end() ? it->second : "";
    };

    auto get_meta_int = [&](const std::string& key, uint32_t default_val = 0) -> uint32_t {
        auto str = get_meta(key);
        if (str.empty()) return default_val;
        return std::stoi(str);
    };

    // Common hparams (override in architecture cases if needed)
    num_layers_ = get_meta_int("block_count", num_layers_);
    hidden_dim_ = get_meta_int("embedding_length", hidden_dim_);
    num_heads_ = get_meta_int("attention.head_count", num_heads_);
    num_kv_heads_ = get_meta_int("attention.head_count_kv", num_heads_);
    context_len_ = get_meta_int("context_length", 32768);
    vocab_size_ = get_meta_int("vocab_size", vocab_size_);
    has_gqa_ = (num_kv_heads_ != num_heads_);

    // Architecture-specific parsing
    switch (architecture_) {
        case ModelArchitecture::LLAMA:
            return parse_llama_weights(tensors);
        case ModelArchitecture::MISTRAL:
            return parse_mistral_weights(tensors);
        case ModelArchitecture::GEMMA:
            return parse_gemma_weights(tensors);
        case ModelArchitecture::PHI:
            return parse_phi_weights(tensors);
        case ModelArchitecture::QWEN2:  // NEW: Support Qwen2/Qwen2.5
            // Qwen2 uses same tensor layout as Llama
            return parse_llama_weights(tensors);
        default:
            return false;
    }

    return true;
}

bool ModelV2::detect_architecture(const std::map<std::string, std::string>& metadata) {
    auto it = metadata.find("general.architecture");
    if (it == metadata.end()) {
        return false;
    }

    std::string arch = it->second;

    if (arch == "llama") {
        architecture_ = ModelArchitecture::LLAMA;
    } else if (arch == "mistral") {
        architecture_ = ModelArchitecture::MISTRAL;
    } else if (arch == "gemma") {
        architecture_ = ModelArchitecture::GEMMA;
    } else if (arch == "phi") {
        architecture_ = ModelArchitecture::PHI;
    } else if (arch == "qwen2") {  // NEW: Detect Qwen2/Qwen2.5
        architecture_ = ModelArchitecture::QWEN2;
    } else {
        return false;
    }

    return true;
}

// Add to enum in model_v2.h (if not already there)
 // QWEN2

// ... rest of your file unchanged (parse_llama_weights, etc.) ...

bool ModelV2::parse_llama_weights(const std::vector<ggml::TensorInfo>& tensors) {
    // Placeholder - not implemented
    return false;
}

bool ModelV2::parse_mistral_weights(const std::vector<ggml::TensorInfo>& tensors) {
    // Placeholder - not implemented
    return false;
}

bool ModelV2::parse_gemma_weights(const std::vector<ggml::TensorInfo>& tensors) {
    // Placeholder - not implemented
    return false;
}

bool ModelV2::parse_phi_weights(const std::vector<ggml::TensorInfo>& tensors) {
    return parse_llama_weights(tensors);
}

const LayerWeights& ModelV2::get_layer(uint32_t idx) const {
    if (idx < layers_.size()) {
        return layers_[idx];
    }
    static LayerWeights empty;
    return empty;
}

const MoELayerWeights& ModelV2::get_moe_layer(uint32_t idx) const {
    if (idx < moe_layers_.size()) {
        return moe_layers_[idx];
    }
    static MoELayerWeights empty;
    return empty;
}

void ModelV2::allocate_tensors() {
    for (auto& pair : weights_) {
        ggml::Tensor* tensor = pair.second;
        if (tensor && !tensor->get_cpu_data()) {
            tensor->allocate_cpu();
        }
    }
}

void ModelV2::free_tensors() {
    for (auto& pair : weights_) {
        if (pair.second) {
            delete pair.second;
        }
    }
    weights_.clear();
    layers_.clear();
    moe_layers_.clear();
}

} // namespace inference