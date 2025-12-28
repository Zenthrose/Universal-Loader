#include "gguf_parser.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace ggml {

GGUFParser::GGUFParser() : valid_(false), version_(0), tensor_count_(0), kv_count_(0), total_size_(0), data_offset_(0) {}

GGUFParser::~GGUFParser() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool GGUFParser::parse(const std::string& filepath) {
    filepath_ = filepath;
    file_.open(filepath, std::ios::binary);
    
    if (!file_.is_open()) {
        return false;
    }
    
    if (!read_header(file_)) {
        file_.close();
        return false;
    }
    
    if (!read_metadata(file_)) {
        file_.close();
        return false;
    }
    
    if (!read_tensor_infos(file_)) {
        file_.close();
        return false;
    }
    
    data_offset_ = file_.tellg();
    valid_ = true;
    return true;
}

bool GGUFParser::read_header(std::ifstream& file) {
    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);
    
    if (memcmp(magic, "GGUF", 4) != 0) {
        return false;
    }
    
    version_ = read_u32(file);
    
    if (version_ > 3) {
        return false;
    }
    
    tensor_count_ = read_u32(file);
    kv_count_ = read_u32(file);
    
    return true;
}

bool GGUFParser::read_metadata(std::ifstream& file) {
    for (uint32_t i = 0; i < kv_count_; ++i) {
        uint32_t key_len = read_u32(file);
        std::string key;
        key.resize(key_len);
        file.read(&key[0], key_len);
        
        uint32_t value_type_val = read_u32(file);
        GGUFValueType value_type = static_cast<GGUFValueType>(value_type_val);
        
        if (value_type == GGUFValueType::STRING) {
            uint32_t value_len = read_u32(file);
            std::string value;
            value.resize(value_len);
            file.read(&value[0], value_len);
            metadata_[key] = value;
        }
        else if (value_type == GGUFValueType::UINT8) {
            uint8_t value = 0;
            file.read(reinterpret_cast<char*>(&value), 1);
            metadata_[key] = std::to_string((int)value);
        }
        else if (value_type == GGUFValueType::UINT16) {
            uint16_t value = read_u16(file);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::UINT32) {
            uint32_t value = read_u32(file);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::UINT64) {
            uint64_t value = read_u64(file);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::INT8) {
            int8_t value = 0;
            file.read(reinterpret_cast<char*>(&value), 1);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::INT16) {
            uint16_t v16 = read_u16(file);
            int16_t value = *reinterpret_cast<int16_t*>(&v16);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::INT32) {
            uint32_t v32 = read_u32(file);
            int32_t value = *reinterpret_cast<int32_t*>(&v32);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::INT64) {
            uint64_t v64 = read_u64(file);
            int64_t value = *reinterpret_cast<int64_t*>(&v64);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::FLOAT32) {
            float value = read_f32(file);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::FLOAT64) {
            double value = read_f64(file);
            metadata_[key] = std::to_string(value);
        }
        else if (value_type == GGUFValueType::BOOL) {
            uint8_t value = 0;
            file.read(reinterpret_cast<char*>(&value), 1);
            metadata_[key] = (value != 0) ? "true" : "false";
        }
        else if (value_type == GGUFValueType::ARRAY) {
            uint32_t array_type_val = read_u32(file);
            uint32_t array_count = read_u32(file);
            
            GGUFValueType array_type = static_cast<GGUFValueType>(array_type_val);
            
            if (array_type == GGUFValueType::UINT32) {
                std::vector<uint32_t> values(array_count);
                for (uint32_t j = 0; j < array_count; ++j) {
                    values[j] = read_u32(file);
                }
                
                std::string result = "[";
                for (uint32_t j = 0; j < array_count; ++j) {
                    result += std::to_string(values[j]);
                    if (j < array_count - 1) result += ", ";
                }
                result += "]";
                metadata_[key] = result;
            }
            else if (array_type == GGUFValueType::STRING) {
                std::string result = "[";
                for (uint32_t j = 0; j < array_count; ++j) {
                    uint32_t str_len = read_u32(file);
                    std::string str;
                    str.resize(str_len);
                    file.read(&str[0], str_len);
                    result += "\"" + str + "\"";
                    if (j < array_count - 1) result += ", ";
                }
                result += "]";
                metadata_[key] = result;
            }
            else {
                for (uint32_t j = 0; j < array_count; ++j) {
                    for (uint32_t k = 0; k < 8; ++k) {
                        uint8_t byte = 0;
                        file.read(reinterpret_cast<char*>(&byte), 1);
                    }
                }
                metadata_[key] = "<array_data>";
            }
        }
        else {
            return false;
        }
    }
    
    return true;
}

bool GGUFParser::read_tensor_infos(std::ifstream& file) {
    for (uint32_t i = 0; i < tensor_count_; ++i) {
        TensorInfo tensor;
        
        uint32_t name_len = read_u32(file);
        tensor.name.resize(name_len);
        file.read(&tensor.name[0], name_len);
        
        uint32_t num_dims = read_u32(file);
        tensor.shape.resize(num_dims);
        for (uint32_t j = 0; j < num_dims; ++j) {
            uint64_t dim = read_u64(file);
            tensor.shape[j] = static_cast<uint32_t>(dim);
        }
        
        uint32_t type_val = read_u32(file);
        tensor.type = static_cast<GGMLType>(type_val);
        
        tensor.offset = read_u64(file);
        
        size_t total_elements = 1;
        for (uint32_t dim : tensor.shape) {
            total_elements *= dim;
        }
        
        size_t block_size = ggml_blck_size(tensor.type);
        size_t type_size = ggml_type_size(tensor.type);
        tensor.size_bytes = ((total_elements + block_size - 1) / block_size) * type_size;
        
        tensors_.push_back(tensor);
        total_size_ += tensor.size_bytes;
    }
    
    return true;
}

std::vector<uint8_t> GGUFParser::read_tensor_data(const TensorInfo& tensor) {
    if (!valid_ || !file_.is_open()) {
        return {};
    }
    
    file_.seekg(data_offset_ + tensor.offset);
    std::vector<uint8_t> data(tensor.size_bytes);
    file_.read(reinterpret_cast<char*>(data.data()), tensor.size_bytes);
    
    return data;
}

std::string GGUFParser::read_string(std::ifstream& file) {
    uint32_t len = read_u32(file);
    std::string str;
    str.resize(len);
    file.read(&str[0], len);
    return str;
}

uint32_t GGUFParser::read_u32(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), 4);
    return __builtin_bswap32(value);
}

uint64_t GGUFParser::read_u64(std::ifstream& file) {
    uint64_t value;
    file.read(reinterpret_cast<char*>(&value), 8);
    return __builtin_bswap64(value);
}

uint16_t GGUFParser::read_u16(std::ifstream& file) {
    uint16_t value;
    file.read(reinterpret_cast<char*>(&value), 2);
    return __builtin_bswap16(value);
}

float GGUFParser::read_f32(std::ifstream& file) {
    uint32_t value = read_u32(file);
    return *reinterpret_cast<float*>(&value);
}

double GGUFParser::read_f64(std::ifstream& file) {
    uint64_t value = read_u64(file);
    return *reinterpret_cast<double*>(&value);
}

}
