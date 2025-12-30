#pragma once
#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <fstream>
#include <memory>
#include "gguf_types.h"

namespace ggml {

enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

struct TensorInfo {
    std::string name;
    GGMLType type;
    std::vector<uint32_t> shape;
    size_t offset;
    size_t size_bytes;
};

class GGUFParser {
public:
    GGUFParser();
    ~GGUFParser();

    bool parse(const std::string& filepath);
    const std::vector<TensorInfo>& get_tensors() const { return tensors_; }
    const std::map<std::string, std::string>& get_metadata() const { return metadata_; }
    
    std::vector<uint8_t> read_tensor_data(const TensorInfo& tensor);
    bool read_tensor_data_direct(const TensorInfo& tensor, uint8_t* target_buffer, size_t buffer_size);
    
    size_t get_tensor_count() const { return tensors_.size(); }
    size_t get_total_size() const { return total_size_; }
    
    uint32_t get_version() const { return version_; }
    size_t get_tensor_count_in_file() const { return (size_t)tensor_count_; }
    
    bool is_valid() const { return valid_; }

private:
    bool read_header(std::ifstream& file);
    bool read_metadata(std::ifstream& file);
    bool read_tensor_infos(std::ifstream& file);
    
    std::string read_string(std::ifstream& file);
    std::vector<uint8_t> read_array_data(std::ifstream& file, GGUFValueType type, uint32_t count);
    std::string value_to_string(GGUFValueType type, const uint8_t* data);
    
    uint32_t read_u32(std::ifstream& file);
    uint64_t read_u64(std::ifstream& file);
    uint16_t read_u16(std::ifstream& file);
    float read_f32(std::ifstream& file);
    double read_f64(std::ifstream& file);
    int32_t read_i32(std::ifstream& file);
    int64_t read_i64(std::ifstream& file);
    int16_t read_i16(std::ifstream& file);
    
    std::string filepath_;
    std::ifstream file_;
    
    uint32_t version_;
    uint64_t tensor_count_;
    uint64_t kv_count_;
    
    std::vector<TensorInfo> tensors_;
    std::map<std::string, std::string> metadata_;
    
    size_t total_size_;
    size_t data_offset_;
    
    bool valid_;
};

}
