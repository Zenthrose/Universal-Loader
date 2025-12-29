#include "speculative_decoder.h"
#include <algorithm>
#include <cmath>

namespace inference {

SpeculativeDecoder::SpeculativeDecoder(uint32_t vocab_size, uint32_t max_spec_len,
                                       float acceptance_threshold)
    : vocab_size_(vocab_size), max_spec_len_(max_spec_len),
      acceptance_threshold_(acceptance_threshold), acceptance_rate_(0.5f),
      total_tokens_(0), accepted_tokens_(0) {
}

SpeculativeDecoder::~SpeculativeDecoder() {
}

void SpeculativeDecoder::set_draft_model(std::function<uint32_t(uint32_t*)> draft_callback) {
    draft_callback_ = draft_callback;
}

void SpeculativeDecoder::set_verification_model(std::function<uint32_t(uint32_t*)> verify_callback) {
    verify_callback_ = verify_callback;
}

SpeculationResult SpeculativeDecoder::generate_with_speculation(
    uint32_t context_token,
    uint32_t num_speculative_tokens) {

    SpeculationResult result;
    result.num_accepted = 0;
    result.acceptance_rate = 0.0f;

    num_speculative_tokens = std::min(num_speculative_tokens, max_spec_len_);

    std::vector<uint32_t> draft_tokens = generate_draft_tokens(context_token);

    if (draft_tokens.empty() || !verify_callback_) {
        result.accepted_tokens.push_back(context_token);
        return result;
    }

    uint32_t* context_ptr = &draft_tokens[0];
    uint32_t verification_token = verify_callback_(context_ptr);

    bool success = verify_speculation(draft_tokens, verification_token, result.accepted_tokens);

    if (success) {
        result.num_accepted = static_cast<uint32_t>(result.accepted_tokens.size());
        total_tokens_ += num_speculative_tokens;
        accepted_tokens_ += result.num_accepted;
        acceptance_rate_ = static_cast<float>(accepted_tokens_) / total_tokens_;

        for (size_t i = 0; i < draft_tokens.size(); ++i) {
            bool accepted = std::find(result.accepted_tokens.begin(),
                                    result.accepted_tokens.end(),
                                    draft_tokens[i]) != result.accepted_tokens.end();
            if (!accepted) {
                result.rejected_tokens.push_back(draft_tokens[i]);
            }
        }

        result.acceptance_rate = static_cast<float>(result.num_accepted) / num_speculative_tokens;
    } else {
        result.accepted_tokens.push_back(context_token);
        result.rejected_tokens = draft_tokens;
        result.num_accepted = 1;
        result.acceptance_rate = 1.0f / (num_speculative_tokens + 1);
    }

    return result;
}

std::vector<uint32_t> SpeculativeDecoder::generate_draft_tokens(uint32_t context) {
    std::vector<uint32_t> draft_tokens;

    if (!draft_callback_) {
        return draft_tokens;
    }

    for (uint32_t i = 0; i < max_spec_len_; ++i) {
        uint32_t token = draft_callback_(&context);
        draft_tokens.push_back(token);
        context = token;
    }

    return draft_tokens;
}

bool SpeculativeDecoder::verify_speculation(const std::vector<uint32_t>& draft_tokens,
                                         uint32_t verification_token,
                                         std::vector<uint32_t>& accepted_tokens) {
    accepted_tokens.clear();

    uint32_t match_pos = 0;
    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        if (draft_tokens[i] == verification_token) {
            match_pos = static_cast<uint32_t>(i);
            break;
        }
    }

    if (match_pos > 0) {
        for (uint32_t i = 0; i < match_pos; ++i) {
            accepted_tokens.push_back(draft_tokens[i]);
        }
        return true;
    }

    return false;
}

void SpeculativeDecoder::update_acceptance_stats(uint32_t accepted, uint32_t total) {
    total_tokens_ += total;
    accepted_tokens_ += accepted;
    acceptance_rate_ = static_cast<float>(accepted_tokens_) / total_tokens_;
}

TreeSpeculation::TreeSpeculation(uint32_t vocab_size, uint32_t beam_width, uint32_t max_depth)
    : vocab_size_(vocab_size), beam_width_(beam_width), max_depth_(max_depth) {
}

TreeSpeculation::~TreeSpeculation() {
}

std::vector<uint32_t> TreeSpeculation::generate_tree_speculation(
    uint32_t context_token,
    uint32_t& final_accepted) {

    std::vector<uint32_t> accepted_tokens;
    build_speculation_tree(context_token);

    for (auto& node : nodes_) {
        accepted_tokens.push_back(node.token_id);
    }

    final_accepted = context_token;
    return accepted_tokens;
}

void TreeSpeculation::build_speculation_tree(uint32_t context) {
    nodes_.clear();

    TreeNode root;
    root.token_id = context;
    root.probability = 1.0f;

    nodes_.push_back(root);

    for (uint32_t depth = 0; depth < max_depth_; ++depth) {
        std::vector<TreeNode*> current_level;

        for (auto& node : nodes_) {
            if (node.children.empty() && depth > 0) {
                for (uint32_t b = 0; b < beam_width_; ++b) {
                    TreeNode child;
                    child.token_id = (node.token_id + b + 1) % vocab_size_;
                    child.probability = node.probability / (beam_width_ * (depth + 1));
                    nodes_.push_back(child);
                    node.children.push_back(&nodes_.back());
                }
            }
        }
    }
}

void TreeSpeculation::verify_tree(TreeNode* root, std::vector<uint32_t>& accepted) {
    accepted.clear();

    if (!root) return;

    accepted.push_back(root->token_id);

    for (auto* child : root->children) {
        verify_tree(child, accepted);
    }
}

}
