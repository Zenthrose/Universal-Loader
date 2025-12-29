#include "profiler.h"
#include <iostream>
#include <fstream>
#include <iomanip>

namespace vulkan {

Profiler::Profiler(VkDevice device, VkPhysicalDevice physical_device)
    : device_(device), physical_device_(physical_device), timestamp_query_pool_(VK_NULL_HANDLE),
      profiling_enabled_(false), timestamps_enabled_(false), timestamp_period_(1),
      peak_memory_bytes_(0), tokens_generated_(0), current_query_id_(0) {

    start_time_ = std::chrono::steady_clock::now();
}

Profiler::~Profiler() {
    destroy_query_pool();
}

void Profiler::begin_profiling() {
    profiling_enabled_ = true;
    start_time_ = std::chrono::steady_clock::now();
    tokens_generated_ = 0;
    peak_memory_bytes_ = 0;
    metrics_.clear();
    gpu_start_queries_.clear();
    gpu_end_queries_.clear();
    current_query_id_ = 0;

    std::cout << "[Profiler] Profiling started" << std::endl;
}

void Profiler::end_profiling() {
    profiling_enabled_ = false;
    end_time_ = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
    double total_seconds = duration.count() / 1000.0;

    if (total_seconds > 0) {
        profiling_data_.tokens_per_second = static_cast<double>(tokens_generated_) / total_seconds;
    }

    collect_gpu_timings();

    profiling_data_.start_time = start_time_;
    profiling_data_.end_time = end_time_;
    profiling_data_.total_tokens = tokens_generated_;
    profiling_data_.peak_memory_bytes = peak_memory_bytes_;
    profiling_data_.metrics = metrics_;

    std::cout << "[Profiler] Profiling ended. Tokens: " << tokens_generated_
              << ", Time: " << total_seconds << "s, tok/s: " << profiling_data_.tokens_per_second << std::endl;
}

void Profiler::start_timing(const std::string& operation, uint32_t layer_id) {
    if (!profiling_enabled_) return;

    std::string key = operation + "_" + std::to_string(layer_id);
    timing_start_[key] = std::chrono::steady_clock::now();
}

void Profiler::end_timing(const std::string& operation, uint32_t layer_id) {
    if (!profiling_enabled_) return;

    std::string key = operation + "_" + std::to_string(layer_id);

    auto it = timing_start_.find(key);
    if (it == timing_start_.end()) {
        std::cerr << "[Profiler] No timing start found for: " << key << std::endl;
        return;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - it->second);
    double duration_ms = duration.count() / 1000.0;

    PerformanceMetrics metric;
    metric.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
    metric.layer_id = layer_id;
    metric.operation = operation;
    metric.duration_ms = duration_ms;
    metric.memory_used_bytes = peak_memory_bytes_;

    metrics_.push_back(metric);
}

void Profiler::record_memory_usage(size_t bytes_used) {
    peak_memory_bytes_ = std::max(peak_memory_bytes_, bytes_used);
}

void Profiler::record_tokens_generated(uint32_t tokens) {
    tokens_generated_ += tokens;
}

void Profiler::export_to_chrome_trace(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Profiler] Failed to open trace file: " << filename << std::endl;
        return;
    }

    file << "[\n";

    for (size_t i = 0; i < metrics_.size(); ++i) {
        const auto& metric = metrics_[i];

        file << "  {\n";
        file << "    \"name\": \"" << metric.operation << "\",\n";
        file << "    \"ph\": \"X\",\n";
        file << "    \"pid\": 0,\n";
        file << "    \"tid\": 0,\n";
        file << "    \"ts\": " << metric.timestamp_ns / 1000 << ",\n";
        file << "    \"dur\": " << static_cast<uint64_t>(metric.duration_ms * 1000) << ",\n";
        file << "    \"args\": {\n";
        file << "      \"layer_id\": " << metric.layer_id << ",\n";
        file << "      \"memory_bytes\": " << metric.memory_used_bytes << "\n";
        file << "    }\n";

        if (i < metrics_.size() - 1) {
            file << "  },\n";
        } else {
            file << "  }\n";
        }
    }

    file << "]\n";
    file.close();

    std::cout << "[Profiler] Chrome trace exported to: " << filename << std::endl;
}

void Profiler::export_to_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Profiler] Failed to open JSON file: " << filename << std::endl;
        return;
    }

    file << "{\n";
    file << "  \"profiling_summary\": {\n";
    file << "    \"total_tokens\": " << profiling_data_.total_tokens << ",\n";
    file << "    \"tokens_per_second\": " << std::fixed << std::setprecision(2) << profiling_data_.tokens_per_second << ",\n";
    file << "    \"peak_memory_bytes\": " << profiling_data_.peak_memory_bytes << ",\n";
    file << "    \"duration_ms\": " << std::chrono::duration_cast<std::chrono::milliseconds>(profiling_data_.end_time - profiling_data_.start_time).count() << "\n";
    file << "  },\n";
    file << "  \"metrics\": [\n";

    for (size_t i = 0; i < metrics_.size(); ++i) {
        const auto& metric = metrics_[i];

        file << "    {\n";
        file << "      \"operation\": \"" << metric.operation << "\",\n";
        file << "      \"layer_id\": " << metric.layer_id << ",\n";
        file << "      \"duration_ms\": " << std::fixed << std::setprecision(3) << metric.duration_ms << ",\n";
        file << "      \"timestamp_ns\": " << metric.timestamp_ns << ",\n";
        file << "      \"memory_used_bytes\": " << metric.memory_used_bytes << "\n";

        if (i < metrics_.size() - 1) {
            file << "    },\n";
        } else {
            file << "    }\n";
        }
    }

    file << "  ]\n";
    file << "}\n";
    file.close();

    std::cout << "[Profiler] JSON profile exported to: " << filename << std::endl;
}

ProfilingData Profiler::get_profiling_data() const {
    return profiling_data_;
}

void Profiler::enable_vulkan_timestamps() {
    create_query_pool();
    timestamps_enabled_ = true;

    std::cout << "[Profiler] Vulkan timestamps enabled" << std::endl;
}

void Profiler::disable_vulkan_timestamps() {
    timestamps_enabled_ = false;
}

void Profiler::create_query_pool() {
    if (timestamp_query_pool_ != VK_NULL_HANDLE) {
        return;
    }

    VkQueryPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    pool_info.queryCount = 1000;

    VkResult result = vkCreateQueryPool(device_, &pool_info, nullptr, &timestamp_query_pool_);
    if (result != VK_SUCCESS) {
        std::cerr << "[Profiler] Failed to create query pool: " << result << std::endl;
        timestamp_query_pool_ = VK_NULL_HANDLE;
        return;
    }

    VkPhysicalDeviceProperties device_properties;
    vkGetPhysicalDeviceProperties(physical_device_, &device_properties);
    timestamp_period_ = device_properties.limits.timestampPeriod;

    std::cout << "[Profiler] Created timestamp query pool. Period: " << timestamp_period_ << "ns" << std::endl;
}

void Profiler::destroy_query_pool() {
    if (timestamp_query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, timestamp_query_pool_, nullptr);
        timestamp_query_pool_ = VK_NULL_HANDLE;
    }
}

uint64_t Profiler::get_timestamp_ns(uint32_t query_id) {
    if (!timestamps_enabled_ || timestamp_query_pool_ == VK_NULL_HANDLE) {
        return 0;
    }

    uint64_t timestamp;
    VkResult result = vkGetQueryPoolResults(device_, timestamp_query_pool_, query_id, 1,
                                         sizeof(timestamp), &timestamp, sizeof(timestamp),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result != VK_SUCCESS) {
        std::cerr << "[Profiler] Failed to get timestamp result: " << result << std::endl;
        return 0;
    }

    return timestamp * timestamp_period_;
}

void Profiler::start_gpu_timing(VkCommandBuffer cmd, const std::string& operation, uint32_t layer_id) {
    if (!timestamps_enabled_ || !profiling_enabled_ || timestamp_query_pool_ == VK_NULL_HANDLE) return;

    std::string key = operation + "_" + std::to_string(layer_id);
    uint32_t start_query = current_query_id_++;
    gpu_start_queries_[key] = start_query;

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamp_query_pool_, start_query);
}

void Profiler::end_gpu_timing(VkCommandBuffer cmd, const std::string& operation, uint32_t layer_id) {
    if (!timestamps_enabled_ || !profiling_enabled_ || timestamp_query_pool_ == VK_NULL_HANDLE) return;

    std::string key = operation + "_" + std::to_string(layer_id);
    uint32_t end_query = current_query_id_++;
    gpu_end_queries_[key] = end_query;

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamp_query_pool_, end_query);
}

void Profiler::collect_gpu_timings() {
    if (!timestamps_enabled_ || timestamp_query_pool_ == VK_NULL_HANDLE) return;

    for (const auto& pair : gpu_start_queries_) {
        const std::string& key = pair.first;
        uint32_t start_q = pair.second;
        auto it = gpu_end_queries_.find(key);
        if (it == gpu_end_queries_.end()) continue;

        uint32_t end_q = it->second;
        uint64_t start_ts = get_timestamp_ns(start_q);
        uint64_t end_ts = get_timestamp_ns(end_q);

        if (start_ts == 0 || end_ts == 0 || end_ts < start_ts) continue;

        double duration_ms = (end_ts - start_ts) / 1e6; // ns to ms

        size_t underscore = key.find_last_of('_');
        if (underscore == std::string::npos) continue;

        std::string op = key.substr(0, underscore);
        uint32_t layer = 0;
        try {
            layer = std::stoi(key.substr(underscore + 1));
        } catch (...) {
            continue;
        }

        PerformanceMetrics metric;
        metric.timestamp_ns = start_ts;
        metric.layer_id = layer;
        metric.operation = op;
        metric.duration_ms = duration_ms;
        metric.memory_used_bytes = peak_memory_bytes_;

        metrics_.push_back(metric);
    }
}

}
