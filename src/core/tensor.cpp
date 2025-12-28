#include "tensor.h"
#include <cstring>
#include <cstdlib>
#include <cstdint>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace ggml {

Tensor::Tensor(const std::string& name, GGMLType type, const std::vector<uint32_t>& shape)
    : name_(name), type_(type), shape_(shape), location_(TensorLocation::CPU),
      cpu_data_(nullptr), gpu_buffer_(0) {
    
    num_elements_ = 1;
    for (uint32_t dim : shape_) {
        num_elements_ *= dim;
    }
    
    size_t block_size = ggml_blck_size(type_);
    size_t type_size = ggml_type_size(type_);
    
    size_ = ((num_elements_ + block_size - 1) / block_size) * type_size;
}

Tensor::~Tensor() {
    free_cpu();
}

void Tensor::allocate_cpu() {
    if (!cpu_data_) {
#ifdef _WIN32
        cpu_data_ = _aligned_malloc(size_, 32);
#else
        cpu_data_ = aligned_alloc(32, size_);
#endif
        memset(cpu_data_, 0, size_);
    }
}

void Tensor::allocate_gpu(uint64_t buffer) {
    gpu_buffer_ = buffer;
}

void Tensor::free_cpu() {
    if (cpu_data_) {
#ifdef _WIN32
        _aligned_free(cpu_data_);
#else
        free(cpu_data_);
#endif
        cpu_data_ = nullptr;
    }
}

void Tensor::free_gpu() {
    gpu_buffer_ = 0;
}

void Tensor::copy_to_gpu(void* cmd_buffer) {
    if (location_ == TensorLocation::CPU && cpu_data_ && gpu_buffer_) {
        memcpy(reinterpret_cast<void*>(gpu_buffer_), cpu_data_, size_);
        location_ = TensorLocation::GPU;
    }
}

void Tensor::copy_to_cpu(void* cmd_buffer) {
    if (location_ == TensorLocation::GPU && gpu_buffer_ && cpu_data_) {
        memcpy(cpu_data_, reinterpret_cast<const void*>(gpu_buffer_), size_);
        location_ = TensorLocation::CPU;
    }
}

}
