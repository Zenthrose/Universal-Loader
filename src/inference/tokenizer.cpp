#include "tokenizer.h"
#include <algorithm>
#include <sstream>

namespace inference {

Tokenizer::Tokenizer() : vocab_size_(0) {
}

Tokenizer::~Tokenizer() {
}

bool Tokenizer::load_vocab(const std::vector<uint32_t>& tokens) {
    vocab_size_ = tokens.size();

    for (uint32_t i = 0; i < vocab_size_; ++i) {
        std::string token_str = std::to_string(i);
        vocab_[i] = token_str;
        vocab_reverse_[token_str] = i;
    }

    return true;
}

void Tokenizer::encode(const std::string& text, std::vector<uint32_t>& tokens) {
    tokens.clear();

    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        uint32_t token_id = match_token(word);

        if (token_id != UINT32_MAX) {
            tokens.push_back(token_id);
        } else {
            for (char c : word) {
                tokens.push_back(static_cast<uint32_t>(c));
            }
        }
    }
}

std::string Tokenizer::decode(const std::vector<uint32_t>& tokens) {
    std::string decoded;

    for (uint32_t token_id : tokens) {
        auto it = vocab_.find(token_id);
        if (it != vocab_.end()) {
            decoded += it->second;
        }
    }

    return decoded;
}

uint32_t Tokenizer::match_token(const std::string& word) {
    auto it = vocab_reverse_.find(word);
    if (it != vocab_reverse_.end()) {
        return it->second;
    }
    return UINT32_MAX;
}

}