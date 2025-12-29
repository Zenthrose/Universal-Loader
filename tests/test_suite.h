#pragma once
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <chrono>
#include <cassert>

namespace test {

struct TestResult {
    std::string test_name;
    bool passed;
    std::string error_message;
    double duration_ms;
};

class TestSuite {
public:
    TestSuite();
    ~TestSuite();

    void add_test(const std::string& name, std::function<bool()> test_func);
    void run_all_tests();
    void run_test(const std::string& test_name);

    std::vector<TestResult> get_results() const { return results_; }
    size_t get_passed_count() const;
    size_t get_total_count() const;
    double get_total_duration_ms() const;

private:
    std::map<std::string, std::function<bool()>> tests_;
    std::vector<TestResult> results_;
};

class UnitTests {
public:
    static bool test_gguf_parser(const std::string& test_model_path);
    static bool test_vulkan_initialization();
    static bool test_model_loading(const std::string& model_path);
    static bool test_inference_generation(const std::string& model_path, const std::string& prompt);
    static bool test_kv_cache();
    static bool test_pipeline_cache();
    static bool test_multi_gpu_detection();
    static bool test_speculative_decoding();
    static bool test_adapter_loading();
    static bool test_batching();

private:
    static float tolerance_for_comparison(float a, float b);
};

class IntegrationTests {
public:
    static bool test_end_to_end_inference(const std::string& model_path,
                                      const std::string& prompt,
                                      uint32_t expected_min_tokens);
    static bool test_multi_batch_inference(const std::string& model_path,
                                      const std::vector<std::string>& prompts,
                                      uint32_t min_tokens_per_prompt);
    static bool test_adapter_inference(const std::string& model_path,
                                   const std::string& adapter_path,
                                   const std::string& prompt);
    static bool test_stress_test(const std::string& model_path,
                               uint32_t iterations,
                               uint32_t tokens_per_iteration);
    static bool test_memory_leak_detection(const std::string& model_path,
                                       uint32_t iterations);
    static bool test_gpu_fallback_to_cpu(const std::string& model_path);
    static bool test_cross_platform_models(const std::vector<std::string>& model_paths);
};

class BenchmarkSuite {
public:
    static BenchmarkSuite& get_instance();

    void run_benchmark(const std::string& name, std::function<void()> benchmark_func,
                        uint32_t warmup_iterations, uint32_t benchmark_iterations);

    struct BenchmarkResult {
        std::string name;
        double mean_time_ms;
        double std_deviation_ms;
        double min_time_ms;
        double max_time_ms;
        uint32_t iterations;
        double ops_per_second;
        std::string unit;
    };

    BenchmarkResult benchmark_inference(const std::string& model_path,
                                    const std::string& prompt,
                                    uint32_t num_tokens,
                                    uint32_t iterations = 10);
    BenchmarkResult benchmark_layer_forward(const std::string& model_path,
                                       uint32_t layer_id,
                                       uint32_t hidden_dim,
                                       uint32_t iterations = 100);
    BenchmarkResult benchmark_attention(const std::string& model_path,
                                   uint32_t seq_len,
                                   uint32_t num_heads,
                                   uint32_t head_dim,
                                   uint32_t iterations = 50);
    BenchmarkResult benchmark_kv_cache_operations(const std::string& model_path,
                                          uint32_t cache_size,
                                          uint32_t iterations = 1000);
    BenchmarkResult benchmark_adapter_operations(const std::string& model_path,
                                        const std::string& adapter_path,
                                        uint32_t iterations = 100);
    BenchmarkResult benchmark_memory_allocation(const std::string& model_path,
                                          uint32_t iterations = 1000);

    std::vector<BenchmarkResult> get_results() const { return results_; }
    void clear_results();

private:
    std::vector<BenchmarkResult> results_;
    BenchmarkSuite();
};

class PerformanceProfiler {
public:
    struct LayerTiming {
        std::string layer_name;
        double cpu_time_ms;
        double gpu_time_ms;
        uint32_t num_invocations;
        double speedup_factor;
    };

    struct MemoryUsage {
        size_t total_allocated_bytes;
        size_t peak_usage_bytes;
        uint32_t buffer_count;
        uint32_t pipeline_cache_size;
    };

    void start_profiling(const std::string& test_name);
    void end_profiling(const std::string& test_name);
    void record_layer_timing(const std::string& layer_name,
                             double cpu_time_ms,
                             double gpu_time_ms,
                             uint32_t invocations);
    void record_memory_usage(const std::string& operation,
                           const MemoryUsage& usage);

    std::vector<LayerTiming> get_layer_timings() const;
    MemoryUsage get_peak_memory_usage() const;

    void export_to_chrome_trace(const std::string& filename) const;
    void export_to_json(const std::string& filename) const;
    void print_summary() const;

private:
    std::map<std::string, std::chrono::steady_clock::time_point> test_timings_;
    std::vector<LayerTiming> layer_timings_;
    MemoryUsage peak_memory_usage_;
    std::chrono::steady_clock::time_point profile_start_;
};

}
