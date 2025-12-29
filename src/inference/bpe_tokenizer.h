#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace inference {

struct TokenizerConfig {
    std::string vocab_file;
    std::string merges_file;
    std::string added_tokens_file;
    std::string type;
    uint32_t vocab_size;
};

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    bool load_from_file(const std::string& tokenizer_file);
    bool load_config(const TokenizerConfig& config);

    void encode(const std::string& text, std::vector<uint32_t>& tokens);
    void decode(const std::vector<uint32_t>& tokens, std::string& text);

    uint32_t get_vocab_size() const { return vocab_size_; }
    std::string get_token(uint32_t token_id) const;

private:
    void load_vocab(const std::string& vocab_file);
    void load_merges(const std::string& merges_file);
    void load_added_tokens(const std::string& added_tokens_file);

    std::vector<std::string> encode_word(const std::string& word);

    uint32_t vocab_size_;
    std::unordered_map<std::string, uint32_t> vocab_;
    std::unordered_map<uint32_t, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, std::string> added_tokens_;
    TokenizerConfig config_;
};

}
