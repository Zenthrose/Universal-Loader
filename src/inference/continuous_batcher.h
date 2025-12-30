#pragma once
#include <vector>
#include <queue>
#include <mutex>
#include <memory>
#include <functional>
#include <cstdint>
#include <map>

namespace inference {

struct BatchRequest {
    uint32_t request_id;
    std::vector<uint32_t> tokens;
    uint32_t max_tokens;
    std::function<void(uint32_t, std::vector<uint32_t>)> callback;
};

struct BatchedInferenceResult {
    uint32_t request_id;
    std::vector<uint32_t> generated_tokens;
    float processing_time_ms;
};

class ContinuousBatcher {
public:
    ContinuousBatcher(uint32_t max_batch_size, uint32_t context_len, uint32_t vocab_size);
    ~ContinuousBatcher();

    void submit_request(const BatchRequest& request);
    BatchedInferenceResult get_result(uint32_t request_id);

    bool has_pending_requests() const { return !request_queue_.empty(); }
    uint32_t get_batch_size() const { return current_batch_size_; }

    void process_batch();

private:
    void build_batch();
    void distribute_batch_to_devices();
    void collect_results();

    uint32_t max_batch_size_;
    uint32_t context_len_;
    uint32_t vocab_size_;
    uint32_t current_batch_size_;

    std::queue<BatchRequest> request_queue_;
    std::vector<BatchRequest> current_batch_;
    std::vector<uint32_t> padded_tokens_;
    std::vector<float> batch_input_;
    std::vector<float> batch_output_;

    std::map<uint32_t, std::vector<uint32_t>> pending_results_;
    std::mutex queue_mutex_;
    std::mutex result_mutex_;
};

class DynamicBatchSizing {
public:
    DynamicBatchSizing(uint32_t min_batch, uint32_t max_batch);

    uint32_t calculate_optimal_batch_size(uint32_t num_requests);
    void update_throughput(uint32_t batch_size, float throughput_tok_per_s);

    float get_throughput(uint32_t batch_size) const;

private:
    struct ThroughputEntry {
        uint32_t batch_size;
        float throughput;
        uint32_t timestamp;
    };

    uint32_t min_batch_size_;
    uint32_t max_batch_size_;

    std::vector<ThroughputEntry> throughput_history_;
    uint32_t optimal_batch_size_;

    void find_optimal_batch_size();
};

}
