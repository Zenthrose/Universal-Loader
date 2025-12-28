#include <iostream>
#include <cassert>
#include "../src/core/gguf_types.h"
#include "../src/core/gguf_parser.h"
#include "../src/core/tensor.h"

void test_gguf_types() {
    std::cout << "Testing GGUF types..." << std::endl;
    
    assert(ggml::ggml_type_size(ggml::GGMLType::F32) == 4);
    assert(ggml::ggml_type_size(ggml::GGMLType::F16) == 2);
    assert(ggml::ggml_type_size(ggml::GGMLType::Q4_0) == sizeof(float) + sizeof(uint16_t));
    assert(ggml::ggml_blck_size(ggml::GGMLType::Q4_0) == 32);
    assert(ggml::ggml_blck_size(ggml::GGMLType::F32) == 1);
    
    std::cout << "GGUF types tests passed!" << std::endl;
}

void test_tensor() {
    std::cout << "Testing Tensor..." << std::endl;
    
    std::vector<uint32_t> shape = {512, 512};
    ggml::Tensor tensor("test_tensor", ggml::GGMLType::F32, shape);
    
    assert(tensor.get_name() == "test_tensor");
    assert(tensor.get_type() == ggml::GGMLType::F32);
    assert(tensor.get_shape() == shape);
    assert(tensor.get_location() == ggml::TensorLocation::CPU);
    
    tensor.allocate_cpu();
    assert(tensor.get_cpu_data() != nullptr);
    
    tensor.free_cpu();
    
    std::cout << "Tensor tests passed!" << std::endl;
}

void test_gguf_parser() {
    std::cout << "Testing GGUFParser..." << std::endl;
    
    ggml::GGUFParser parser;
    assert(!parser.is_valid());
    assert(parser.get_version() == 0);
    assert(parser.get_tensor_count() == 0);
    
    std::cout << "GGUFParser tests passed!" << std::endl;
}

int main() {
    std::cout << "Running Core Tests..." << std::endl;
    
    test_gguf_types();
    test_tensor();
    test_gguf_parser();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
