#pragma once
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include "../core/tensor.h"
#include "model.h"

namespace inference {

class PrefetchEngine {
public:
    explicit PrefetchEngine(Model* model);
    ~PrefetchEngine();

    void prefetch_layer(uint32_t layer_id);
    void prefetch_next_layers(uint32_t current_layer, uint32_t count);
    void wait_for_prefetch();

private:
    Model* model_;
    std::queue<std::future<void>> pending_prefetches_;
    std::mutex prefetch_mutex_;
};

}
