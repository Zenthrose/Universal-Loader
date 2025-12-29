#include "bpe_tokenizer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <fstream>
#include <cmath>
#include <ctime>

namespace inference {

BPETokenizer::BPETokenizer()
    : vocab_size_(0) {
}

BPETokenizer::~BPETokenizer() {
}

bool BPETokenizer::load_from_file(const std::string& tokenizer_file) {
    std::ifstream file(tokenizer_file);
    if (!file.is_open()) {
        std::cerr << "[BPETokenizer] Failed to open tokenizer file: " << tokenizer_file << std::endl;
        return false;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line.find("[vocab]") != std::string::npos) {
            section = "vocab";
            continue;
        } else if (line.find("[merges]") != std::string::npos) {
            section = "merges";
            continue;
        } else if (line.find("[added_tokens]") != std::string::npos) {
            section = "added_tokens";
            continue;
        }

        if (section == "vocab") {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string token_str = line.substr(0, space_pos);
                uint32_t token_id = std::stoul(line.substr(space_pos + 1));
                vocab_[token_str] = token_id;
                id_to_token_[token_id] = token_str;
                vocab_size_ = std::max(vocab_size_, token_id + 1);
            }
        } else if (section == "merges") {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string first = line.substr(0, space_pos);
                std::string second = line.substr(space_pos + 1);
                merges_.push_back({first, second});
            }
        } else if (section == "added_tokens") {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string token_str = line.substr(0, space_pos);
                std::string replacement = line.substr(space_pos + 1);
                added_tokens_[token_str] = replacement;
            }
        }
    }

    file.close();
    std::cout << "[BPETokenizer] Loaded " << vocab_size_ << " tokens and " << merges_.size() << " merges" << std::endl;
    return true;
}

bool BPETokenizer::load_config(const TokenizerConfig& config) {
    config_ = config;

    load_vocab(config.vocab_file);
    load_merges(config.merges_file);
    load_added_tokens(config.added_tokens_file);

    return vocab_size_ > 0;
}

void BPETokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        return;
    }

    std::string line;
    uint32_t token_id = 0;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            vocab_[line] = token_id;
            id_to_token_[token_id] = line;
            token_id++;
        }
    }

    vocab_size_ = token_id;
    file.close();
}

void BPETokenizer::load_merges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        return;
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string first = line.substr(0, space_pos);
            std::string second = line.substr(space_pos + 1);

            first = std::regex_replace(first, std::regex("</w>"), "");
            second = std::regex_replace(second, std::regex("</w>"), "");

            merges_.push_back({first, second});
        }
    }

    file.close();
}

void BPETokenizer::load_added_tokens(const std::string& added_tokens_file) {
    std::ifstream file(added_tokens_file);
    if (!file.is_open()) {
        return;
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string token_str = line.substr(0, space_pos);
            std::string replacement = line.substr(space_pos + 1);
            added_tokens_[token_str] = replacement;
        }
    }

    file.close();
}

std::vector<std::string> BPETokenizer::encode_word(const std::string& word) {
    std::vector<std::string> chars;

    for (char c : word) {
        chars.push_back(std::string(1, c));
    }

    for (const auto& [first, second] : merges_) {
        for (size_t i = 0; i + 1 < chars.size(); ++i) {
            if (chars[i] == first && chars[i + 1] == second) {
                chars[i] = first + second;
                chars.erase(chars.begin() + i + 1);
                i--;
            }
        }
    }

    return chars;
}

void BPETokenizer::encode(const std::string& text, std::vector<uint32_t>& tokens) {
    tokens.clear();

    std::vector<std::string> words;
    std::string current_word;

    for (char c : text) {
        if (std::isspace(c)) {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            current_word += c;
        } else {
            current_word += c;
        }
    }

    if (!current_word.empty()) {
        words.push_back(current_word);
    }

    for (const auto& word : words) {
        std::vector<std::string> word_tokens = encode_word(word);

        for (const auto& token_str : word_tokens) {
            auto it = vocab_.find(token_str);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                auto added_it = added_tokens_.find(token_str);
                if (added_it != added_tokens_.end()) {
                    std::vector<std::string> added_tokens = encode_word(added_it->second);
                    for (const auto& added : added_tokens) {
                        auto added_vocab_it = vocab_.find(added);
                        if (added_vocab_it != vocab_.end()) {
                            tokens.push_back(added_vocab_it->second);
                        }
                    }
                }
            }
        }
    }
}

void BPETokenizer::decode(const std::vector<uint32_t>& tokens, std::string& text) {
    text.clear();

    for (uint32_t token_id : tokens) {
        std::string token = get_token(token_id);

        token = std::regex_replace(token, std::regex("</w>"), " ");

        text += token;
    }

    text = std::regex_replace(text, std::regex(" +"), " ");
    if (!text.empty() && text[0] == ' ') {
        text = text.substr(1);
    }
}

std::string BPETokenizer::get_token(uint32_t token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    return "<unk>";
}

}
