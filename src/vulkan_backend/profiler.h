#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <atomic>

namespace vulkan {

struct PerformanceMetrics {
    uint64_t timestamp_ns;
    uint32_t layer_id;
    std::string operation;
    double duration_ms;
    size_t memory_used_bytes;
};

struct ProfilingData {
    std::vector<PerformanceMetrics> metrics;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    uint32_t total_tokens;
    double tokens_per_second;
    double gpu_utilization;
    size_t peak_memory_bytes;
};

class Profiler {
public:
    Profiler(VkDevice device, VkPhysicalDevice physical_device);
    ~Profiler();

    void begin_profiling();
    void end_profiling();

    void start_timing(const std::string& operation, uint32_t layer_id = 0);
    void end_timing(const std::string& operation, uint32_t layer_id = 0);

    void record_memory_usage(size_t bytes_used);
    void record_tokens_generated(uint32_t tokens);

    void export_to_chrome_trace(const std::string& filename) const;
    void export_to_json(const std::string& filename) const;

    ProfilingData get_profiling_data() const;

    void enable_vulkan_timestamps();
    void disable_vulkan_timestamps();

    uint64_t get_gpu_timestamp_period() const { return timestamp_period_; }

private:
    void create_query_pool();
    void destroy_query_pool();
    uint64_t get_timestamp_ns(uint32_t query_id);

    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkQueryPool timestamp_query_pool_;

    bool profiling_enabled_;
    bool timestamps_enabled_;
    uint64_t timestamp_period_;

    std::map<std::string, std::chrono::steady_clock::time_point> timing_start_;
    std::vector<PerformanceMetrics> metrics_;
    ProfilingData profiling_data_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;

    std::atomic<uint64_t> peak_memory_bytes_;
    std::atomic<uint32_t> tokens_generated_;
};

}
