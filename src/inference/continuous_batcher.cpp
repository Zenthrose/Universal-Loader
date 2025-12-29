#include "continuous_batcher.h"
#include <algorithm>
#include <chrono>

namespace inference {

ContinuousBatcher::ContinuousBatcher(uint32_t max_batch_size, uint32_t context_len,
                                     uint32_t vocab_size)
    : max_batch_size_(max_batch_size), context_len_(context_len), vocab_size_(vocab_size),
      current_batch_size_(0) {

    padded_tokens_.resize(max_batch_size_ * context_len_);
    batch_input_.resize(max_batch_size_ * context_len_);
    batch_output_.resize(max_batch_size_ * vocab_size_);
}

ContinuousBatcher::~ContinuousBatcher() {
}

void ContinuousBatcher::submit_request(const BatchRequest& request) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    request_queue_.push(request);
}

BatchedInferenceResult ContinuousBatcher::get_result(uint32_t request_id) {
    std::lock_guard<std::mutex> lock(result_mutex_);

    auto it = pending_results_.find(request_id);
    if (it != pending_results_.end()) {
        BatchedInferenceResult result;
        result.request_id = request_id;
        result.generated_tokens = it->second;
        result.processing_time_ms = 0.0f;

        pending_results_.erase(it);
        return result;
    }

    BatchedInferenceResult empty_result;
    empty_result.request_id = request_id;
    return empty_result;
}

void ContinuousBatcher::process_batch() {
    build_batch();

    if (current_batch_.empty()) {
        return;
    }

    distribute_batch_to_devices();

    collect_results();

    current_batch_.clear();
}

void ContinuousBatcher::build_batch() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    uint32_t num_requests = std::min(static_cast<uint32_t>(request_queue_.size()),
                                  max_batch_size_);

    current_batch_.resize(num_requests);

    for (uint32_t i = 0; i < num_requests; ++i) {
        current_batch_[i] = request_queue_.front();
        request_queue_.pop();
    }

    uint32_t max_tokens = 0;
    for (const auto& req : current_batch_) {
        max_tokens = std::max(max_tokens, static_cast<uint32_t>(req.tokens.size()));
    }

    for (uint32_t i = 0; i < num_requests; ++i) {
        uint32_t num_tokens = static_cast<uint32_t>(current_batch_[i].tokens.size());

        for (uint32_t j = 0; j < num_tokens; ++j) {
            padded_tokens_[i * context_len_ + j] = current_batch_[i].tokens[j];
        }

        for (uint32_t j = num_tokens; j < context_len_; ++j) {
            padded_tokens_[i * context_len_ + j] = 0;
        }
    }

    current_batch_size_ = num_requests;
}

void ContinuousBatcher::distribute_batch_to_devices() {
    for (uint32_t i = 0; i < current_batch_size_; ++i) {
        for (uint32_t j = 0; j < context_len_; ++j) {
            batch_input_[i * context_len_ + j] =
                static_cast<float>(padded_tokens_[i * context_len_ + j]);
        }
    }
}

void ContinuousBatcher::collect_results() {
    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t i = 0; i < current_batch_size_; ++i) {
        if (current_batch_[i].callback) {
            std::vector<uint32_t> generated_tokens;

            for (uint32_t j = 0; j < vocab_size_; ++j) {
                if (batch_output_[i * vocab_size_ + j] > 0.5f) {
                    generated_tokens.push_back(j);
                }
            }

            std::lock_guard<std::mutex> lock(result_mutex_);
            pending_results_[current_batch_[i].request_id] = generated_tokens;

            current_batch_[i].callback(current_batch_[i].request_id, generated_tokens);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
}

DynamicBatchSizing::DynamicBatchSizing(uint32_t min_batch, uint32_t max_batch)
    : min_batch_size_(min_batch), max_batch_size_(max_batch),
      optimal_batch_size_(min_batch) {
}

uint32_t DynamicBatchSizing::calculate_optimal_batch_size(uint32_t num_requests) {
    find_optimal_batch_size();

    uint32_t batch_size = optimal_batch_size_;

    while (batch_size < num_requests && batch_size < max_batch_size_) {
        batch_size = std::min(batch_size * 2, max_batch_size_);
    }

    if (batch_size > max_batch_size_) {
        batch_size = max_batch_size_;
    }

    return batch_size;
}

void DynamicBatchSizing::update_throughput(uint32_t batch_size, float throughput_tok_per_s) {
    ThroughputEntry entry;
    entry.batch_size = batch_size;
    entry.throughput = throughput_tok_per_s;
    entry.timestamp = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    throughput_history_.push_back(entry);

    if (throughput_history_.size() > 100) {
        throughput_history_.erase(throughput_history_.begin());
    }
}

float DynamicBatchSizing::get_throughput(uint32_t batch_size) const {
    float sum = 0.0f;
    uint32_t count = 0;

    for (const auto& entry : throughput_history_) {
        if (entry.batch_size == batch_size) {
            sum += entry.throughput;
            count++;
        }
    }

    return count > 0 ? sum / count : 0.0f;
}

void DynamicBatchSizing::find_optimal_batch_size() {
    if (throughput_history_.empty()) {
        optimal_batch_size_ = min_batch_size_;
        return;
    }

    float best_throughput = 0.0f;
    uint32_t best_batch_size = min_batch_size_;

    for (const auto& entry : throughput_history_) {
        if (entry.throughput > best_throughput) {
            best_throughput = entry.throughput;
            best_batch_size = entry.batch_size;
        }
    }

    optimal_batch_size_ = best_batch_size;
}

}
