#include "gguf_parser.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

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
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    // These will throw on error now
    read_header(file_);
    read_metadata(file_);
    read_tensor_infos(file_);

    // CRITICAL: The padding to alignment happens AFTER all tensor infos, before the data block.
    // Check for general.alignment
    uint32_t alignment = 32;
    if (metadata_.find("general.alignment") != metadata_.end()) {
        try {
             alignment = std::stoul(metadata_["general.alignment"]);
        } catch (...) { alignment = 32; }
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) alignment = 32;

    std::streampos current_pos = file_.tellg();
    std::streamoff align_mask = alignment - 1;
    std::streampos aligned_pos = (current_pos + align_mask) & ~align_mask;

    // Only align if we are not already aligned (although math handles this)
    // AND check if we need to verify file size or seek success?
    if (aligned_pos != current_pos) {
         // We don't necessarily strictly seek here unless we want to validate,
         // but data_offset_ MUST be the aligned position.
         // file_.seekg(aligned_pos); // Optional validation
    }

    data_offset_ = aligned_pos;
    valid_ = true;
    return true;
}

bool GGUFParser::read_header(std::ifstream& file) {
    uint8_t magic[4];
    if (!file.read(reinterpret_cast<char*>(magic), 4)) throw std::runtime_error("Failed to read header magic");

    if (memcmp(magic, "GGUF", 4) != 0) {
        throw std::runtime_error("Invalid GGUF magic bytes");
    }

    version_ = read_u32(file);

    if (version_ > 3) {
         throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version_));
    }

    tensor_count_ = read_u64(file);
    kv_count_ = read_u64(file);

    return true;
}

bool GGUFParser::read_metadata(std::ifstream& file) {
    for (uint64_t i = 0; i < kv_count_; ++i) {
        std::string key = read_string(file);

        uint32_t value_type_val = read_u32(file);
        GGUFValueType value_type = static_cast<GGUFValueType>(value_type_val);

        if (value_type == GGUFValueType::STRING) {
            metadata_[key] = read_string(file);
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
            uint64_t array_count = read_u64(file);

            if (array_count > 10000000) { // Limit to 10M elements to prevent bad_alloc
                 throw std::runtime_error("Array too large: " + std::to_string(array_count));
            }

            GGUFValueType array_type = static_cast<GGUFValueType>(array_type_val);

            if (array_type == GGUFValueType::UINT32) {
                std::vector<uint32_t> values(array_count);
                for (uint64_t j = 0; j < array_count; ++j) {
                    values[j] = read_u32(file);
                }

                std::string result = "[";
                for (uint64_t j = 0; j < array_count; ++j) {
                    result += std::to_string(values[j]);
                    if (j < array_count - 1) result += ", ";
                }
                result += "]";
                metadata_[key] = result;
            }
            else if (array_type == GGUFValueType::STRING) {
                std::string result = "[";
                for (uint64_t j = 0; j < array_count; ++j) {
                    std::string str = read_string(file);
                    result += "\"" + str + "\"";
                    if (j < array_count - 1) result += ", ";
                }
                result += "]";
                metadata_[key] = result;
            }
            else {
                size_t type_size = 0;
                switch (array_type) {
                    case GGUFValueType::UINT8: 
                    case GGUFValueType::INT8: 
                    case GGUFValueType::BOOL:
                        type_size = 1; break;
                    case GGUFValueType::UINT16: 
                    case GGUFValueType::INT16: 
                        type_size = 2; break;
                    case GGUFValueType::UINT32: 
                    case GGUFValueType::INT32: 
                    case GGUFValueType::FLOAT32: 
                        type_size = 4; break;
                    case GGUFValueType::UINT64: 
                    case GGUFValueType::INT64: 
                    case GGUFValueType::FLOAT64: 
                        type_size = 8; break;
                    default: 
                        // If complex type or unhandled, warn and maybe fail?
                        // For now assume 1 byte to minimize damage, or we cannot proceed.
                        type_size = 0; 
                }

                if (type_size > 0) {
                    // Skip data
                    size_t total_bytes = type_size * array_count;
                    file.ignore(total_bytes);
                    metadata_[key] = "<array_data>";
                } else {
                     throw std::runtime_error("Unsupported GGUF array type: " + std::to_string((int)array_type));
                }
            }
        }
        else {
            throw std::runtime_error("Unsupported GGUF value type: " + std::to_string((int)value_type));
        }
    }

    // Check for general.alignment in metadata
    uint32_t alignment = 32;
    if (metadata_.find("general.alignment") != metadata_.end()) {
        try {
             alignment = std::stoul(metadata_["general.alignment"]);
        } catch (...) {
            alignment = 32;
        }
    }
    
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        alignment = 32;
    }
    
    // Store for later use (or we can re-read it in parse)
    // For now, CRITICAL FIX: Do NOT seek here.
    return true;
}

bool GGUFParser::read_tensor_infos(std::ifstream& file) {
    for (uint64_t i = 0; i < tensor_count_; ++i) {
        TensorInfo tensor;

        uint64_t name_len = read_u64(file);
        if (file.fail() || name_len > 256) {
            throw std::runtime_error("Invalid tensor name length: " + std::to_string(name_len));
        }
        
        tensor.name.resize(name_len);
        file.read(&tensor.name[0], name_len);
        
        if (file.fail()) {
            throw std::runtime_error("Failed to read tensor name");
        }

        // NO alignment seek here. Tensor infos are packed.

        uint32_t num_dims = read_u32(file);
        if (file.fail() || num_dims > 4) {
             throw std::runtime_error("Invalid tensor dimensions: " + std::to_string(num_dims));
        }
        
        tensor.shape.resize(num_dims);
        for (uint32_t j = 0; j < num_dims; ++j) {
            uint64_t dim = read_u64(file);
            if (file.fail() || dim == 0) {
                 throw std::runtime_error("Invalid tensor dimension value (0 or read fail)");
            }
            tensor.shape[j] = static_cast<uint32_t>(dim);
        }

        uint32_t type_val = read_u32(file);
        if (file.fail()) {
            throw std::runtime_error("Failed to read tensor type");
        }
        tensor.type = static_cast<GGMLType>(type_val);

        tensor.offset = read_u64(file);
        if (file.fail()) {
            throw std::runtime_error("Failed to read tensor offset");
        }

        size_t total_elements = 1;
        for (uint32_t dim : tensor.shape) {
            total_elements *= dim;
        }

        size_t block_size = ggml_blck_size(tensor.type);
        size_t type_size = ggml_type_size(tensor.type);
        if (block_size == 0 || type_size == 0) {
            throw std::runtime_error("Unknown/Unsupported tensor type: " + std::to_string(type_val));
        }
        
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
    uint64_t len = read_u64(file);
    if (len > 10 * 1024 * 1024) { // Cap at 10MB
         throw std::runtime_error("String too long: " + std::to_string(len));
    }
    std::string str;
    str.resize(len);
    if (len > 0) {
        file.read(&str[0], len);
        if (file.fail()) throw std::runtime_error("Failed to read string data");
    }
    return str;
}

uint32_t GGUFParser::read_u32(std::ifstream& file) {
    uint32_t value = 0;
    if (!file.read(reinterpret_cast<char*>(&value), 4)) throw std::runtime_error("Failed to read u32");
    return value;
}

uint64_t GGUFParser::read_u64(std::ifstream& file) {
    uint64_t value = 0;
    if (!file.read(reinterpret_cast<char*>(&value), 8)) throw std::runtime_error("Failed to read u64");
    return value;
}

uint16_t GGUFParser::read_u16(std::ifstream& file) {
    uint16_t value = 0;
    if (!file.read(reinterpret_cast<char*>(&value), 2)) throw std::runtime_error("Failed to read u16");
    return value;
}

float GGUFParser::read_f32(std::ifstream& file) {
    uint32_t value = 0;
    if (!file.read(reinterpret_cast<char*>(&value), 4)) throw std::runtime_error("Failed to read f32");
    return *reinterpret_cast<float*>(&value);
}

double GGUFParser::read_f64(std::ifstream& file) {
    uint64_t value = 0;
    if (!file.read(reinterpret_cast<char*>(&value), 8)) throw std::runtime_error("Failed to read f64");
    return *reinterpret_cast<double*>(&value);
}

bool GGUFParser::read_tensor_data_direct(const TensorInfo& tensor, uint8_t* target_buffer, size_t buffer_size) {
    if (!valid_ || !file_.is_open()) {
        return false;
    }

    if (!target_buffer || buffer_size < tensor.size_bytes) {
        return false;
    }

    file_.seekg(data_offset_ + tensor.offset);
    file_.read(reinterpret_cast<char*>(target_buffer), tensor.size_bytes);

    return true;
}

} // namespace ggml