#include "offload_manager.h"
#include <algorithm>

namespace inference {

OffloadManager::OffloadManager() : gpu_cache_size_(0) {
}

OffloadManager::~OffloadManager() {
}

void OffloadManager::set_model(const std::unordered_map<std::string, ggml::Tensor*>& weights) {
    weights_ = weights;
    calculate_layer_priorities();
}

void OffloadManager::set_gpu_cache_size(size_t size_mb) {
    gpu_cache_size_ = size_mb * 1024 * 1024;
}

bool OffloadManager::should_offload(ggml::Tensor* tensor) const {
    size_t tensor_size = estimate_size(tensor);
    
    if (tensor_size > gpu_cache_size_ / 2) {
        return false;
    }
    
    std::string name = tensor->get_name();
    if (name.find("layers.") != std::string::npos) {
        size_t layer_pos = name.find(".layers.");
        size_t next_dot = name.find(".", layer_pos + 8);
        if (next_dot != std::string::npos) {
            uint32_t layer_id = std::stoul(name.substr(layer_pos + 8, next_dot - (layer_pos + 8)));
            
            if (layer_id < layer_priorities_.size()) {
                float priority = layer_priorities_[layer_id].second;
                return priority > 0.5f;
            }
        }
    }
    
    return true;
}

void OffloadManager::offload_tensor(ggml::Tensor* tensor, bool to_gpu) {
    if (to_gpu && tensor->get_cpu_data()) {
        tensor->set_location(ggml::TensorLocation::GPU);
    } else if (!to_gpu && tensor->get_cpu_data()) {
        tensor->set_location(ggml::TensorLocation::CPU);
    }
}

void OffloadManager::update_layer_access(uint32_t layer_id) {
    if (layer_id < layer_priorities_.size()) {
        layer_priorities_[layer_id].second += 0.1f;
    }
    
    for (auto& layer : layer_priorities_) {
        layer.second *= 0.95f;
    }
}

size_t OffloadManager::estimate_size(ggml::Tensor* tensor) const {
    const auto& shape = tensor->get_shape();
    size_t total_elements = 1;
    for (uint32_t dim : shape) {
        total_elements *= dim;
    }
    
    ggml::GGMLType type = tensor->get_type();
    size_t type_size = ggml::ggml_type_size(type);
    size_t block_size = ggml::ggml_blck_size(type);
    
    size_t size_bytes = ((total_elements + block_size - 1) / block_size) * type_size;
    return size_bytes;
}

void OffloadManager::calculate_layer_priorities() {
    layer_priorities_.clear();
    
    std::unordered_map<uint32_t, std::vector<ggml::Tensor*>> layer_tensors;
    
    for (const auto& pair : weights_) {
        const std::string& name = pair.first;
        size_t layer_pos = name.find(".layers.");
        if (layer_pos != std::string::npos) {
            size_t next_dot = name.find(".", layer_pos + 8);
            if (next_dot != std::string::npos) {
                uint32_t layer_id = std::stoul(name.substr(layer_pos + 8, next_dot - (layer_pos + 8)));
                layer_tensors[layer_id].push_back(pair.second);
            }
        }
    }
    
    for (const auto& pair : layer_tensors) {
        size_t total_size = 0;
        for (ggml::Tensor* tensor : pair.second) {
            total_size += estimate_size(tensor);
        }
        
        float priority = std::min(1.0f, (float)total_size / (float)(gpu_cache_size_ / 8));
        layer_priorities_.push_back({pair.first, priority});
    }
    
    std::sort(layer_priorities_.begin(), layer_priorities_.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
}

}
