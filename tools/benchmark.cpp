#include "../src/api/gguf_loader.h"
#include <iostream>
#include <chrono>
#include <iomanip>

void print_banner() {
    std::cout << "============================================" << std::endl;
    std::cout << "  VulkanGGUF Benchmark Tool v1.0" << std::endl;
    std::cout << "============================================" << std::endl << std::endl;
}

void print_separator() {
    std::cout << "--------------------------------------------" << std::endl;
}

void benchmark_inference_speed(api::InferenceSession& session, const std::string& prompt, uint32_t num_runs = 3) {
    std::cout << "Running inference speed benchmark (" << num_runs << " runs)..." << std::endl;
    print_separator();

    std::vector<double> times;
    double total_tokens = 0.0;

    for (uint32_t i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::string result = session.generate(prompt, 50);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();

        times.push_back(duration);

        std::cout << "Run " << (i + 1) << ": " << std::fixed << std::setprecision(2) << duration << " ms" << std::endl;
    }

    double avg_time = 0.0;
    for (double t : times) {
        avg_time += t;
    }
    avg_time /= times.size();

    double min_time = *std::min_element(times.begin(), times.end());
    double max_time = *std::max_element(times.begin(), times.end());

    std::cout << std::endl;
    std::cout << "Average: " << std::fixed << std::setprecision(2) << avg_time << " ms" << std::endl;
    std::cout << "Min:     " << std::fixed << std::setprecision(2) << min_time << " ms" << std::endl;
    std::cout << "Max:     " << std::fixed << std::setprecision(2) << max_time << " ms" << std::endl;
    std::cout << "Speed:   " << std::fixed << std::setprecision(2) << (50.0 / avg_time * 1000.0) << " tokens/sec" << std::endl;
    print_separator();
}

void benchmark_memory_usage(api::InferenceSession& session) {
    std::cout << "Memory Usage:" << std::endl;
    print_separator();
    std::cout << session.get_model_info() << std::endl;
    print_separator();
}

void benchmark_temperature_scaling(api::InferenceSession& session, const std::string& prompt) {
    std::cout << "Temperature Scaling Benchmark:" << std::endl;
    print_separator();

    std::vector<float> temperatures = {0.1f, 0.5f, 0.7f, 1.0f, 1.5f, 2.0f};

    for (float temp : temperatures) {
        auto start = std::chrono::high_resolution_clock::now();
        api::GenerationConfig config;
        config.temperature = temp;

        std::string result = session.generate(prompt, 20, config);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Temperature " << std::fixed << std::setprecision(1) << temp << ": " << duration << " ms" << std::endl;
    }

    print_separator();
}

void benchmark_batch_sizes(api::InferenceSession& session, const std::string& prompt) {
    std::cout << "Batch Size Benchmark:" << std::endl;
    print_separator();

    std::vector<uint32_t> batch_sizes = {10, 20, 50, 100};

    for (uint32_t batch_size : batch_sizes) {
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = session.generate(prompt, batch_size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Batch Size " << batch_size << ": " << std::fixed << std::setprecision(2) << duration << " ms" << std::endl;
    }

    print_separator();
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --speed              Run inference speed benchmark" << std::endl;
        std::cerr << "  --memory             Run memory usage benchmark" << std::endl;
        std::cerr << "  --temperature         Run temperature scaling benchmark" << std::endl;
        std::cerr << "  --batch              Run batch size benchmark" << std::endl;
        std::cerr << "  --all                Run all benchmarks" << std::endl;
        std::cerr << "  --backend <cpu|gpu>  Set backend (default: cpu)" << std::endl;
        std::cerr << "  --threads <n>        Set number of threads (default: 4)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example: " << argv[0] << " model.gguf --all --backend cpu --threads 8" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    inference::InferenceConfig config;
    config.num_threads = 4;
    config.gpu_memory_pool_mb = 2048;
    config.gpu_cache_mb = 1024;
    config.context_len = 2048;
    config.prefetch_layers = 2;
    config.backend = inference::BackendType::CPU;
    config.enable_validation = true;

    bool run_speed = false;
    bool run_memory = false;
    bool run_temperature = false;
    bool run_batch = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--speed") run_speed = true;
        else if (arg == "--memory") run_memory = true;
        else if (arg == "--temperature") run_temperature = true;
        else if (arg == "--batch") run_batch = true;
        else if (arg == "--all") {
            run_speed = true;
            run_memory = true;
            run_temperature = true;
            run_batch = true;
        }
        else if (arg == "--backend" && i + 1 < argc) {
            std::string backend = argv[++i];
            if (backend == "gpu") {
                config.backend = inference::BackendType::GPU;
            } else if (backend == "hybrid") {
                config.backend = inference::BackendType::HYBRID;
            } else {
                config.backend = inference::BackendType::CPU;
            }
        }
        else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoul(argv[++i]);
        }
    }

    std::cout << "Loading model: " << model_path << std::endl;
    print_separator();

    api::InferenceSession session;
    if (!session.initialize(config)) {
        std::cerr << "Failed to initialize inference session" << std::endl;
        return 1;
    }

    api::LoadResult result = session.load_model(model_path);
    if (!result.success) {
        std::cerr << "Failed to load model: " << result.error_message << std::endl;
        return 1;
    }

    std::cout << "Model loaded successfully!" << std::endl;
    std::cout << result.architecture << " - " << result.num_layers << " layers" << std::endl;
    print_separator();

    std::string prompt = "The quick brown fox jumps over the lazy dog.";

    if (run_memory) benchmark_memory_usage(session);
    if (run_speed) benchmark_inference_speed(session, prompt);
    if (run_temperature) benchmark_temperature_scaling(session, prompt);
    if (run_batch) benchmark_batch_sizes(session, prompt);

    std::cout << std::endl;
    std::cout << "Benchmark complete!" << std::endl;
    print_separator();

    return 0;
}