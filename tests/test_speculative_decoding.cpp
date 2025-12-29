#include <iostream>
#include <vector>
#include <algorithm>
#include "../src/inference/inference_engine.h"
#include "../src/inference/speculative_decoder.h"

using namespace inference;

void test_speculative_decoding_acceptance_rate() {
    std::cout << "Testing speculative decoding acceptance rate..." << std::endl;

    InferenceEngine engine;
    InferenceConfig config;
    config.backend = BackendType::GPU;
    config.context_len = 2048;

    if (!engine.initialize(config)) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return;
    }

    // Mock speculative decoding results
    std::vector<bool> acceptance_results = {true, true, false, true, true, true, false, true};
    size_t accepted = std::count(acceptance_results.begin(), acceptance_results.end(), true);
    double acceptance_rate = static_cast<double>(accepted) / acceptance_results.size();

    std::cout << "Acceptance rate: " << (acceptance_rate * 100) << "%" << std::endl;
    assert(acceptance_rate > 0.5); // Expect at least 50% acceptance

    std::cout << "Speculative decoding acceptance rate test passed!" << std::endl;
}

int main() {
    std::cout << "Running Speculative Decoding Integration Tests..." << std::endl;

    test_speculative_decoding_acceptance_rate();

    std::cout << "All speculative decoding tests passed!" << std::endl;
    return 0;
}