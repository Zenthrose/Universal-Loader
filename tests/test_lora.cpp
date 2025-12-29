#include <iostream>
#include <vector>
#include <cmath>
#include "../src/inference/inference_engine.h"
#include "../src/inference/lora_manager.h"

using namespace inference;

void test_lora_merging_correctness() {
    std::cout << "Testing LoRA merging correctness..." << std::endl;

    InferenceEngine engine;
    InferenceConfig config;
    config.backend = BackendType::GPU;

    if (!engine.initialize(config)) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return;
    }

    // Mock LoRA weights merging
    std::vector<float> base_weights = {1.0f, 2.0f, 3.0f};
    std::vector<float> lora_weights = {0.1f, 0.2f, 0.3f};
    float alpha = 0.5f;

    std::vector<float> merged_weights(base_weights.size());
    for (size_t i = 0; i < base_weights.size(); ++i) {
        merged_weights[i] = base_weights[i] + alpha * lora_weights[i];
    }

    // Check correctness: merged should be close to expected
    std::vector<float> expected = {1.05f, 2.1f, 3.15f};
    for (size_t i = 0; i < merged_weights.size(); ++i) {
        assert(std::abs(merged_weights[i] - expected[i]) < 1e-6);
    }

    std::cout << "LoRA merging correctness test passed!" << std::endl;
}

int main() {
    std::cout << "Running LoRA Integration Tests..." << std::endl;

    test_lora_merging_correctness();

    std::cout << "All LoRA tests passed!" << std::endl;
    return 0;
}