#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

#include "gguf_types.h"

namespace ggml {

enum class TensorLocation {
    CPU,
    GPU,
    Hybrid
};

class Tensor {
public:
    Tensor(const std::string& name, GGMLType type, const std::vector<uint32_t>& shape);
    ~Tensor();
    
    const std::string& get_name() const { return name_; }
    GGMLType get_type() const { return type_; }
    const std::vector<uint32_t>& get_shape() const { return shape_; }
    
    TensorLocation get_location() const { return location_; }
    void set_location(TensorLocation loc) { location_ = loc; }
    
    void* get_cpu_data() const { return cpu_data_; }
    void set_cpu_data(void* data, size_t size) { cpu_data_ = data; }

    void* get_quantized_data() const { return quantized_data_; }
    void set_quantized_data(void* data, size_t size) { quantized_data_ = data; size_ = size; }
    void set_quantized(bool quantized) { is_quantized_ = quantized; }
    void set_quantized_type(GGMLType type) { quantized_type_ = type; }
    bool is_quantized() const { return is_quantized_; }
    GGMLType get_quantized_type() const { return quantized_type_; }
    
    size_t get_size() const { return size_; }
    
    uint64_t get_gpu_buffer() const { return gpu_buffer_; }
    
    void allocate_cpu();
    void allocate_gpu(uint64_t buffer);
    void free_cpu();
    void free_gpu();
    
    void copy_to_gpu(void* cmd_buffer);
    void copy_to_cpu(void* cmd_buffer);

private:
    std::string name_;
    GGMLType type_;
    std::vector<uint32_t> shape_;
    
    TensorLocation location_;
    void* cpu_data_;
    void* quantized_data_;
    bool is_quantized_;
    GGMLType quantized_type_;
    uint64_t gpu_buffer_;
    
    size_t size_;
    size_t num_elements_;
};

}
