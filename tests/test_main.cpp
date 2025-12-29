#include "test_suite.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "=== VulkanGGUF Test Suite ===" << std::endl;

    test::TestSuite suite;

    suite.add_test("GGUF Parser", &test::UnitTests::test_gguf_parser);
    suite.add_test("Vulkan Initialization", &test::UnitTests::test_vulkan_initialization);
    suite.add_test("Model Loading", &test::UnitTests::test_model_loading);
    suite.add_test("Inference Generation", &test::UnitTests::test_inference_generation);
    suite.add_test("KV Cache", &test::UnitTests::test_kv_cache);
    suite.add_test("Pipeline Cache", &test::UnitTests::test_pipeline_cache);
    suite.add_test("Multi-GPU Detection", &test::UnitTests::test_multi_gpu_detection);
    suite.add_test("Speculative Decoding", &test::UnitTests::test_speculative_decoding);
    suite.add_test("Adapter Loading", &test::UnitTests::test_adapter_loading);
    suite.add_test("Batching", &test::UnitTests::test_batching);

    suite.run_all_tests();

    return 0;
}
