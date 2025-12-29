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

    num_layers_ = std::stoul(get_meta("llama.block_count"));
    hidden_dim_ = std::stoul(get_meta("llama.embedding_length"));
    num_heads_ = std::stoul(get_meta("llama.attention.head_count"));
    context_len_ = std::stoul(get_meta("llama.context_length"));
    vocab_size_ = std::stoul(get_meta("llama.vocab_size"));

    if (has_gqa_) {
        num_kv_heads_ = std::stoul(get_meta("llama.attention.head_count_kv"));
    } else {
        num_kv_heads_ = num_heads_;
    }

    if (is_moe_) {
        num_experts_ = std::stoul(get_meta("llama.expert_count"));
    }

    layers_.resize(num_layers_);
    moe_layers_.resize(num_layers_);

    switch (architecture_) {
        case ModelArchitecture::LLAMA:
        case ModelArchitecture::LLAMA2:
        case ModelArchitecture::LLAMA3:
            return parse_llama_weights(tensors);
        case ModelArchitecture::MISTRAL:
            return parse_mistral_weights(tensors);
        case ModelArchitecture::MIXTRAL:
            return parse_mixtral_weights(tensors);
        case ModelArchitecture::GEMMA:
        case ModelArchitecture::GEMMA2:
            return parse_gemma_weights(tensors);
        case ModelArchitecture::QWEN:
        case ModelArchitecture::QWEN2:
            return parse_qwen_weights(tensors);
        case ModelArchitecture::PHI:
        case ModelArchitecture::PHI2:
        case ModelArchitecture::PHI3:
            return parse_phi_weights(tensors);
        default:
            return parse_llama_weights(tensors);
    }
}

bool ModelV2::detect_architecture(const std::map<std::string, std::string>& metadata) {
    auto get_meta = [&](const std::string& key) -> std::string {
        auto it = metadata.find(key);
        return it != metadata.end() ? it->second : "";
    };

    std::string arch = get_meta("general.architecture");

    if (arch == "llama") {
        std::string version = get_meta("general.version");
        if (version.find("llama-3") != std::string::npos ||
            get_meta("llama.rope_scaling") != "") {
            architecture_ = ModelArchitecture::LLAMA3;
            has_gqa_ = true;
        } else if (get_meta("llama.rope_scaling") != "") {
            architecture_ = ModelArchitecture::LLAMA2;
        } else {
            architecture_ = ModelArchitecture::LLAMA;
        }
        is_moe_ = false;
        return true;
    }

    if (arch == "mistral") {
        if (get_meta("llama.expert_count") != "") {
            architecture_ = ModelArchitecture::MIXTRAL;
            is_moe_ = true;
            has_gqa_ = true;
        } else {
            architecture_ = ModelArchitecture::MISTRAL;
            is_moe_ = false;
            has_gqa_ = true;
        }
        return true;
    }

    if (arch == "gemma") {
        architecture_ = ModelArchitecture::GEMMA;
        is_moe_ = false;
        has_gqa_ = false;
        return true;
    }

    if (arch == "qwen") {
        architecture_ = ModelArchitecture::QWEN;
        is_moe_ = false;
        has_gqa_ = true;
        return true;
    }

    if (arch == "phi") {
        architecture_ = ModelArchitecture::PHI;
        is_moe_ = false;
        has_gqa_ = false;
        return true;
    }

    architecture_ = ModelArchitecture::UNKNOWN;
    return false;
}

bool ModelV2::parse_llama_weights(const std::vector<ggml::TensorInfo>& tensors) {
    for (const auto& tensor_info : tensors) {
        std::string name = tensor_info.name;
        ggml::Tensor* tensor = new ggml::Tensor(name, tensor_info.type, tensor_info.shape);
        tensor->allocate_cpu();

        ggml::GGUFParser parser;
        std::vector<uint8_t> data = parser.read_tensor_data(tensor_info);
        memcpy(tensor->get_cpu_data(), data.data(), data.size());

        weights_[name] = tensor;

        size_t layer_pos = name.find(".layers.");
        if (layer_pos != std::string::npos) {
            size_t next_dot = name.find(".", layer_pos + 8);
            if (next_dot != std::string::npos) {
                uint32_t layer_id = std::stoul(name.substr(layer_pos + 8, next_dot - (layer_pos + 8)));

                if (layer_id < layers_.size()) {
                    LayerWeights& layer = layers_[layer_id];

                    std::string weight_name = name.substr(next_dot + 1);
                    if (weight_name == "attention.wq.weight") layer.q_proj = tensor;
                    else if (weight_name == "attention.wk.weight") layer.k_proj = tensor;
                    else if (weight_name == "attention.wv.weight") layer.v_proj = tensor;
                    else if (weight_name == "attention.wo.weight") layer.o_proj = tensor;
                    else if (weight_name == "feed_forward.w1.weight") layer.gate_proj = tensor;
                    else if (weight_name == "feed_forward.w2.weight") layer.up_proj = tensor;
                    else if (weight_name == "feed_forward.w3.weight") layer.down_proj = tensor;
                    else if (weight_name == "attention_norm.weight") layer.norm1 = tensor;
                    else if (weight_name == "ffn_norm.weight") layer.norm2 = tensor;

                    layer.num_kv_heads = has_gqa_ ? num_kv_heads_ : num_heads_;
                    layer.num_experts = 0;
                }
            }
        }
    }

    return true;
}

bool ModelV2::parse_mistral_weights(const std::vector<ggml::TensorInfo>& tensors) {
    return parse_llama_weights(tensors);
}

bool ModelV2::parse_mixtral_weights(const std::vector<ggml::TensorInfo>& tensors) {
    for (const auto& tensor_info : tensors) {
        std::string name = tensor_info.name;
        ggml::Tensor* tensor = new ggml::Tensor(name, tensor_info.type, tensor_info.shape);
        tensor->allocate_cpu();

        ggml::GGUFParser parser;
        std::vector<uint8_t> data = parser.read_tensor_data(tensor_info);
        memcpy(tensor->get_cpu_data(), data.data(), data.size());

        weights_[name] = tensor;

        size_t layer_pos = name.find(".layers.");
        if (layer_pos != std::string::npos) {
            size_t next_dot = name.find(".", layer_pos + 8);
            if (next_dot != std::string::npos) {
                uint32_t layer_id = std::stoul(name.substr(layer_pos + 8, next_dot - (layer_pos + 8)));

                if (layer_id < layers_.size()) {
                    LayerWeights& layer = layers_[layer_id];

                    std::string weight_name = name.substr(next_dot + 1);
                    if (weight_name == "attention.wq.weight") layer.q_proj = tensor;
                    else if (weight_name == "attention.wk.weight") layer.k_proj = tensor;
                    else if (weight_name == "attention.wv.weight") layer.v_proj = tensor;
                    else if (weight_name == "attention.wo.weight") layer.o_proj = tensor;
                    else if (weight_name == "attention_norm.weight") layer.norm1 = tensor;

                    layer.num_kv_heads = num_kv_heads_;
                    layer.num_experts = num_experts_;
                }
            }
        }

        size_t moe_pos = name.find(".block_sparse_moe.");
        if (moe_pos != std::string::npos) {
            size_t layer_pos = name.find(".layers.");
            if (layer_pos != std::string::npos) {
                uint32_t layer_id = std::stoul(name.substr(layer_pos + 8, moe_pos - (layer_pos + 8)));

                if (layer_id < moe_layers_.size()) {
                    MoELayerWeights& moe_layer = moe_layers_[layer_id];

                    std::string weight_name = name.substr(moe_pos + 17);
                    if (weight_name == "gate.weight") moe_layer.gate = tensor;
                }
            }
        }
    }

    return true;
}

bool ModelV2::parse_gemma_weights(const std::vector<ggml::TensorInfo>& tensors) {
    return parse_llama_weights(tensors);
}

bool ModelV2::parse_qwen_weights(const std::vector<ggml::TensorInfo>& tensors) {
    return parse_llama_weights(tensors);
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

}