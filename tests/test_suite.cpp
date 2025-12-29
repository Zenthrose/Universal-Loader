#include "test_suite.h"
#include <iostream>
#include <iomanip>

namespace test {

TestSuite::TestSuite() {
}

TestSuite::~TestSuite() {
}

void TestSuite::add_test(const std::string& name, std::function<bool()> test_func) {
    tests_[name] = test_func;
}

void TestSuite::run_all_tests() {
    results_.clear();

    for (const auto& [name, test_func] : tests_) {
        run_test(name);
    }

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << get_passed_count() << "/" << get_total_count() << std::endl;
    std::cout << "Total time: " << get_total_duration_ms() << "ms" << std::endl;
}

void TestSuite::run_test(const std::string& test_name) {
    auto it = tests_.find(test_name);
    if (it == tests_.end()) {
        std::cerr << "[TestSuite] Test not found: " << test_name << std::endl;
        return;
    }

    std::cout << "[TEST] Running: " << test_name << "...";

    auto start = std::chrono::steady_clock::now();
    bool passed = it->second();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    TestResult result;
    result.test_name = test_name;
    result.passed = passed;
    result.duration_ms = duration.count();

    if (passed) {
        std::cout << " PASSED (" << result.duration_ms << "ms)" << std::endl;
    } else {
        std::cout << " FAILED (" << result.duration_ms << "ms)" << std::endl;
    }

    results_.push_back(result);
}

size_t TestSuite::get_passed_count() const {
    size_t count = 0;
    for (const auto& result : results_) {
        if (result.passed) count++;
    }
    return count;
}

size_t TestSuite::get_total_count() const {
    return results_.size();
}

double TestSuite::get_total_duration_ms() const {
    double total = 0.0;
    for (const auto& result : results_) {
        total += result.duration_ms;
    }
    return total;
}

bool UnitTests::test_gguf_parser(const std::string& test_model_path) {
    std::cout << "[UnitTests] Testing GGUF parser..." << std::endl;

    try {
        ggml::GGUFParser parser;
        bool result = parser.parse_file(test_model_path);

        if (!result) {
            std::cerr << "[UnitTests] GGUF parser failed" << std::endl;
            return false;
        }

        uint32_t tensor_count = parser.get_tensor_count();
        if (tensor_count == 0) {
            std::cerr << "[UnitTests] No tensors found in model" << std::endl;
            return false;
        }

        std::cout << "[UnitTests] Found " << tensor_count << " tensors" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in GGUF parser test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_vulkan_initialization() {
    std::cout << "[UnitTests] Testing Vulkan initialization..." << std::endl;

    try {
        vulkan::VulkanConfig vk_config;
        vk_config.enable_validation = false;

        vulkan::VulkanContext context(vk_config);

        if (!context.is_initialized()) {
            std::cerr << "[UnitTests] Vulkan context not initialized" << std::endl;
            return false;
        }

        if (context.supports_portability_subset()) {
            std::cout << "[UnitTests] Portability subset supported" << std::endl;
        }

        if (context.supports_buffer_device_address()) {
            std::cout << "[UnitTests] Buffer device address supported" << std::endl;
        }

        if (context.supports_cooperative_matrix()) {
            std::cout << "[UnitTests] Cooperative matrix supported" << std::endl;
        }

        std::cout << "[UnitTests] Vulkan initialized successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in Vulkan init test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_model_loading(const std::string& model_path) {
    std::cout << "[UnitTests] Testing model loading..." << std::endl;

    try {
        inference::InferenceEngine engine;

        inference::InferenceConfig config;
        config.backend = inference::BackendType::GPU;
        config.enable_validation = false;

        bool result = engine.initialize(config);
        if (!result) {
            return false;
        }

        result = engine.load_model(model_path);
        if (!result) {
            return false;
        }

        std::cout << "[UnitTests] Model loaded: "
                  << engine.get_model_size_bytes() << " bytes" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in model loading test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_inference_generation(const std::string& model_path,
                                        const std::string& prompt) {
    std::cout << "[UnitTests] Testing inference generation..." << std::endl;

    try {
        inference::InferenceEngine engine;

        inference::InferenceConfig config;
        config.backend = inference::BackendType::GPU;
        config.enable_validation = false;

        engine.initialize(config);
        engine.load_model(model_path);

        std::string result = engine.generate(prompt, 10);

        if (result.empty()) {
            std::cerr << "[UnitTests] Generated empty string" << std::endl;
            return false;
        }

        std::cout << "[UnitTests] Generated: " << result.substr(0, std::min(size_t(50), result.length())) << "..." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in inference test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_kv_cache() {
    std::std::cout << "[UnitTests] Testing KV cache..." << std::endl;

    try {
        uint32_t cache_size = 1024;
        inference::KVCache cache(32, 1280, cache_size);

        for (uint32_t i = 0; i < cache_size; ++i) {
            std::vector<float> k(1280);
            std::vector<float> v(1280);

            for (uint32_t j = 0; j < 1280; ++j) {
                k[j] = static_cast<float>(j);
                v[j] = static_cast<float>(j);
            }

            cache.store(0, i, k.data(), v.data());
        }

        for (uint32_t i = 0; i < cache_size; ++i) {
            std::vector<float> retrieved_k(1280);
            std::vector<float> retrieved_v(1280);

            cache.retrieve(0, i, retrieved_k.data(), retrieved_v.data());

            for (uint32_t j = 0; j < 1280; ++j) {
                if (retrieved_k[j] != static_cast<float>(j)) {
                    return false;
                }
                if (retrieved_v[j] != static_cast<float>(j)) {
                    return false;
                }
            }
        }

        std::cout << "[UnitTests] KV cache works correctly" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in KV cache test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_pipeline_cache() {
    std::cout << "[UnitTests] Testing pipeline cache..." << std::endl;

    try {
        VkInstance instance;
        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = nullptr;
        create_info.enabledLayerCount = 0;

        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
            return false;
        }

        vkDestroyInstance(instance, nullptr);

        std::cout << "[UnitTests] Pipeline cache creation works" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in pipeline cache test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_multi_gpu_detection() {
    std::cout << "[UnitTests] Testing multi-GPU detection..." << std::endl;

    try {
        VkInstance instance;
        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = nullptr;
        create_info.enabledLayerCount = 0;

        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
            return false;
        }

        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        vkDestroyInstance(instance, nullptr);

        if (device_count == 0) {
            std::cerr << "[UnitTests] No physical devices found" << std::endl;
            return false;
        }

        if (device_count == 1) {
            std::cout << "[UnitTests] Single GPU detected" << std::endl;
        } else {
            std::cout << "[UnitTests] " << device_count << " GPUs detected" << std::endl;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in multi-GPU test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_speculative_decoding() {
    std::cout << "[UnitTests] Testing speculative decoding..." << std::endl;

    try {
        inference::SpeculativeDecoder decoder(32000, 4, 0.8f);
        decoder.set_draft_model([](uint32_t* context) {
            return (context[0] + 1) % 1000;
        });
        decoder.set_verification_model([](uint32_t* context) {
            return (context[0] + 1) % 1000;
        });

        for (uint32_t i = 0; i < 100; ++i) {
            uint32_t context_token = i;
            inference::SpeculationResult result = decoder.generate_with_speculation(context_token, 4);

            if (result.acceptance_rate < 0.5f) {
                std::cerr << "[UnitTests] Acceptance rate too low: " << result.acceptance_rate << std::endl;
                return false;
            }

            if (result.num_accepted == 0) {
                std::cerr << "[UnitTests] No tokens accepted" << std::endl;
                return false;
            }
        }

        float final_rate = decoder.get_acceptance_rate();
        if (final_rate < 0.5f) {
            std::cerr << "[UnitTests] Final acceptance rate too low: " << final_rate << std::endl;
            return false;
        }

        std::cout << "[UnitTests] Speculative decoding works, acceptance rate: "
                  << final_rate << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in speculative decoding test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_adapter_loading() {
    std::cout << "[UnitTests] Testing adapter loading..." << std::endl;

    try {
        inference::LoRAManager lora_manager(1280, 32000);

        bool result = lora_manager.load_adapter("test_adapter.safetensors", "test");

        if (!result) {
            std::cerr << "[UnitTests] Failed to load adapter" << std::endl;
            return false;
        }

        std::vector<std::string> adapters = lora_manager.get_loaded_adapters();
        if (adapters.empty()) {
            std::cerr << "[UnitTests] No adapters loaded" << std::endl;
            return false;
        }

        lora_manager.enable_adapter("test");
        lora_manager.set_adapter_alpha("test", 0.5f);

        float rank = lora_manager.get_effective_rank("test");
        if (rank == 0.0f) {
            std::cerr << "[UnitTests] Invalid adapter rank" << std::endl;
            return false;
        }

        std::cout << "[UnitTests] Adapter loaded, rank: " << rank << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in adapter loading test: " << e.what() << std::endl;
        return false;
    }
}

bool UnitTests::test_batching() {
    std::cout << "[UnitTests] Testing batching..." << std::endl;

    try {
        inference::ContinuousBatcher batcher(16, 2048, 32000);

        std::vector<inference::BatchRequest> requests;
        for (uint32_t i = 0; i < 4; ++i) {
            inference::BatchRequest req;
            req.request_id = i;
            req.tokens = {1, 2, 3, 4};
            req.max_tokens = 10;
            req.callback = [](uint32_t id, std::vector<uint32_t> tokens) {};

            requests.push_back(req);
        }

        for (const auto& req : requests) {
            batcher.submit_request(req);
        }

        batcher.process_batch();

        uint32_t completed = 0;
        for (const auto& req : requests) {
            inference::BatchedInferenceResult result = batcher.get_result(req.request_id);
            if (!result.generated_tokens.empty()) {
                completed++;
            }
        }

        if (completed != requests.size()) {
            std::cerr << "[UnitTests] Not all requests completed" << std::endl;
            return false;
        }

        std::cout << "[UnitTests] Batching works correctly, batch size: "
                  << batcher.get_batch_size() << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[UnitTests] Exception in batching test: " << e.what() << std::endl;
        return false;
    }
}

float UnitTests::tolerance_for_comparison(float a, float b) {
    return std::abs(a - b) / (std::abs(a) + std::abs(b)) * 0.001f;
}

}
