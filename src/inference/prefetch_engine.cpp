#include "prefetch_engine.h"

namespace inference {

PrefetchEngine::PrefetchEngine(Model* model) : model_(model) {
}

PrefetchEngine::~PrefetchEngine() {
    wait_for_prefetch();
}

void PrefetchEngine::prefetch_layer(uint32_t layer_id) {
    if (!model_ || layer_id >= model_->get_num_layers()) {
        return;
    }
    
    auto future = std::async(std::launch::async, [this, layer_id]() {
        const LayerWeights& layer = model_->get_layer(layer_id);
        
        if (layer.q_proj && layer.q_proj->get_cpu_data()) {
            layer.q_proj->allocate_cpu();
        }
        if (layer.k_proj && layer.k_proj->get_cpu_data()) {
            layer.k_proj->allocate_cpu();
        }
        if (layer.v_proj && layer.v_proj->get_cpu_data()) {
            layer.v_proj->allocate_cpu();
        }
        if (layer.o_proj && layer.o_proj->get_cpu_data()) {
            layer.o_proj->allocate_cpu();
        }
        if (layer.gate_proj && layer.gate_proj->get_cpu_data()) {
            layer.gate_proj->allocate_cpu();
        }
        if (layer.up_proj && layer.up_proj->get_cpu_data()) {
            layer.up_proj->allocate_cpu();
        }
        if (layer.down_proj && layer.down_proj->get_cpu_data()) {
            layer.down_proj->allocate_cpu();
        }
    });
    
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    pending_prefetches_.push(std::move(future));
}

void PrefetchEngine::prefetch_next_layers(uint32_t current_layer, uint32_t count) {
    for (uint32_t i = 1; i <= count; ++i) {
        uint32_t layer_id = current_layer + i;
        if (layer_id < model_->get_num_layers()) {
            prefetch_layer(layer_id);
        }
    }
}

void PrefetchEngine::wait_for_prefetch() {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    while (!pending_prefetches_.empty()) {
        auto& future = pending_prefetches_.front();
        future.wait();
        pending_prefetches_.pop();
    }
}

}
