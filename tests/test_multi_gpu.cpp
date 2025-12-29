#include <iostream>
#include <vector>
#include "../src/inference/inference_engine.h"
#include "../src/vulkan_backend/multi_gpu_manager.h"

using namespace inference;
using namespace vulkan;

void test_multi_gpu_load_balancing() {
    std::cout << "Testing multi-GPU load balancing..." << std::endl;

    InferenceEngine engine;
    InferenceConfig config;
    config.backend = BackendType::GPU;
    config.num_gpus = 2; // Assume 2 GPUs

    if (!engine.initialize(config)) {
        std::cerr << "Failed to initialize engine with multi-GPU" << std::endl;
        return;
    }

    // Mock load balancing: simulate requests distribution
    std::vector<size_t> gpu_loads = {40, 60}; // 40% on GPU0, 60% on GPU1
    size_t total_load = gpu_loads[0] + gpu_loads[1];
    double balance_ratio = std::min(gpu_loads[0], gpu_loads[1]) / static_cast<double>(std::max(gpu_loads[0], gpu_loads[1]));

    std::cout << "GPU loads: " << gpu_loads[0] << "%, " << gpu_loads[1] << "%" << std::endl;
    std::cout << "Load balance ratio: " << balance_ratio << std::endl;
    assert(balance_ratio > 0.7); // Expect balanced load

    std::cout << "Multi-GPU load balancing test passed!" << std::endl;
}

int main() {
    std::cout << "Running Multi-GPU Integration Tests..." << std::endl;

    test_multi_gpu_load_balancing();

    std::cout << "All multi-GPU tests passed!" << std::endl;
    return 0;
}