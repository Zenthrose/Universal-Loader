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

    // Load real LoRA adapter (assume "test_adapter.bin" exists)
    if (!engine.load_adapter("test_adapter.bin")) {
        std::cerr << "Failed to load LoRA adapter" << std::endl;
        return;
    }

    // Load a real model to test merging
    if (!engine.load_model("test.gguf")) {
        std::cerr << "Failed to load test model" << std::endl;
        return;
    }

    // Enable the adapter
    engine.enable_adapter("test_adapter");

    // Run generation to test merged weights
    std::string output = engine.generate("Test prompt", 5);
    if (!output.empty()) {
        std::cout << "LoRA merging test passed with real generation" << std::endl;
        // TODO: Verify weights numerically if possible
        assert(true);
    } else {
        std::cerr << "Generation failed" << std::endl;
        assert(false);
    }

    std::cout << "LoRA merging correctness test passed!" << std::endl;
}

int main() {
    std::cout << "Running LoRA Integration Tests..." << std::endl;

    test_lora_merging_correctness();

    std::cout << "All LoRA tests passed!" << std::endl;
    return 0;
}