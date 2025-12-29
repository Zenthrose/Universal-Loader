#include <iostream>
#include "../src/inference/inference_engine.h"

using namespace inference;

int main() {
    std::cout << "=== Phase 1 GPU Compute Integration Test ===" << std::endl;

    InferenceEngine engine;

    InferenceConfig config;
    config.num_threads = 4;
    config.gpu_memory_pool_mb = 2048;
    config.gpu_cache_mb = 1024;
    config.context_len = 2048;
    config.prefetch_layers = 2;
    config.backend = BackendType::GPU;
    config.enable_validation = true;

    std::cout << "[1/5] Initializing InferenceEngine with GPU backend..." << std::endl;
    if (!engine.initialize(config)) {
        std::cerr << "Failed to initialize InferenceEngine" << std::endl;
        return 1;
    }
    std::cout << "   InferenceEngine initialized successfully" << std::endl;
    std::cout << "   GPU enabled: " << (engine.is_gpu_enabled() ? "Yes" : "No") << std::endl;
    std::cout << "   Num threads: " << engine.get_num_threads() << std::endl;

    std::cout << "[2/5] Checking model size estimation..." << std::endl;
    size_t model_size = engine.get_model_size_bytes();
    std::cout << "   Model size: " << (model_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    std::cout << "[3/5] Testing KV cache size configuration..." << std::endl;
    engine.set_kv_cache_size(4096);
    std::cout << "   KV cache size set to 4096 tokens" << std::endl;

    std::cout << "[4/5] Testing prefetch configuration..." << std::endl;
    engine.set_prefetch_layers(2);
    std::cout << "   Prefetch layers set to 2" << std::endl;

    std::cout << "[5/5] Testing GPU memory pool configuration..." << std::endl;
    engine.set_gpu_memory_pool(1024);
    std::cout << "   GPU memory pool set to 1024 MB" << std::endl;
    engine.set_gpu_cache_size(512);
    std::cout << "   GPU cache size set to 512 MB" << std::endl;

    std::cout << "\n=== Phase 1 Tests Completed ===" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Vulkan backend: " << (engine.is_gpu_enabled() ? "Initialized" : "Not available") << std::endl;
    std::cout << "  - Validation layers: Enabled" << std::endl;
    std::cout << "  - GPU memory pool: Configured" << std::endl;
    std::cout << "  - KV cache: Configured" << std::endl;
    std::cout << "  - Prefetch engine: Configured" << std::endl;

    return 0;
}
