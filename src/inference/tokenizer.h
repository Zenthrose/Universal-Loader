#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <climits>

namespace inference {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    bool load_vocab(const std::vector<uint32_t>& tokens);
    void encode(const std::string& text, std::vector<uint32_t>& tokens);
    std::string decode(const std::vector<uint32_t>& tokens);

    void set_vocab_size(uint32_t size) { vocab_size_ = size; }
    uint32_t get_vocab_size() const { return vocab_size_; }

private:
    uint32_t vocab_size_;
    std::unordered_map<uint32_t, std::string> vocab_;
    std::unordered_map<std::string, uint32_t> vocab_reverse_;

    uint32_t match_token(const std::string& word);
};

}