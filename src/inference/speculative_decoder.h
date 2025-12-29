#pragma once
#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <memory>
#include <functional>

namespace inference {

struct SpeculativeToken {
    uint32_t token_id;
    float probability;
    uint32_t position;
};

struct SpeculationResult {
    std::vector<uint32_t> accepted_tokens;
    std::vector<uint32_t> rejected_tokens;
    uint32_t num_accepted;
    float acceptance_rate;
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder(uint32_t vocab_size, uint32_t max_spec_len, float acceptance_threshold);
    ~SpeculativeDecoder();

    void set_draft_model(std::function<uint32_t(uint32_t*)> draft_callback);
    void set_verification_model(std::function<uint32_t(uint32_t*)> verify_callback);

    SpeculationResult generate_with_speculation(
        uint32_t context_token,
        uint32_t num_speculative_tokens = 4
    );

    void update_acceptance_stats(uint32_t accepted, uint32_t total);
    float get_acceptance_rate() const { return acceptance_rate_; }

    void set_speculation_length(uint32_t len) { max_spec_len_ = len; }

private:
    std::vector<uint32_t> generate_draft_tokens(uint32_t context);
    bool verify_speculation(const std::vector<uint32_t>& draft_tokens,
                          uint32_t verification_token,
                          std::vector<uint32_t>& accepted_tokens);

    uint32_t vocab_size_;
    uint32_t max_spec_len_;
    float acceptance_threshold_;

    std::function<uint32_t(uint32_t*)> draft_callback_;
    std::function<uint32_t(uint32_t*)> verify_callback_;

    float acceptance_rate_;
    uint32_t total_tokens_;
    uint32_t accepted_tokens_;

    std::map<uint32_t, float> token_probabilities_;
};

class TreeSpeculation {
public:
    TreeSpeculation(uint32_t vocab_size, uint32_t beam_width, uint32_t max_depth);
    ~TreeSpeculation();

    std::vector<uint32_t> generate_tree_speculation(
        uint32_t context_token,
        uint32_t& final_accepted
    );

private:
    struct TreeNode {
        uint32_t token_id;
        float probability;
        std::vector<TreeNode*> children;
    };

    void build_speculation_tree(uint32_t context);
    void verify_tree(TreeNode* root, std::vector<uint32_t>& accepted);

    uint32_t vocab_size_;
    uint32_t beam_width_;
    uint32_t max_depth_;

    std::vector<TreeNode> nodes_;
};

}
