#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>

namespace vulkan {

struct AsyncPipelineTask {
    uint32_t task_id;
    uint32_t layer_id;
    std::function<void()> compute_fn;
    std::function<void()> transfer_fn;
    VkSemaphore compute_semaphore;
    VkSemaphore transfer_semaphore;
    uint64_t timeline_value;
};

class AsyncPipelineManager {
public:
    AsyncPipelineManager(VkDevice device, VkQueue compute_queue, VkQueue transfer_queue,
                       uint32_t compute_family, uint32_t transfer_family);
    ~AsyncPipelineManager();

    bool initialize();
    void cleanup();

    void submit_compute_task(const AsyncPipelineTask& task);
    void submit_transfer_task(const AsyncPipelineTask& task);
    void wait_for_task(uint32_t task_id, uint64_t timeout_ns = UINT64_MAX);

    void update_binding(uint32_t binding_index, VkBuffer buffer, VkDeviceSize size);

    uint32_t get_next_task_id() { return next_task_id_.fetch_add(1); }

    bool is_idle() const { return active_tasks_ == 0; }

private:
    void task_thread_func();
    void execute_task(const AsyncPipelineTask& task);

    VkDevice device_;
    VkQueue compute_queue_;
    VkQueue transfer_queue_;
    uint32_t compute_family_;
    uint32_t transfer_family_;

    VkCommandPool compute_pool_;
    VkCommandPool transfer_pool_;

    std::vector<AsyncPipelineTask> pending_tasks_;
    std::vector<AsyncPipelineTask> active_tasks_;
    std::mutex task_mutex_;

    std::thread task_thread_;
    std::atomic<bool> running_;
    std::atomic<uint32_t> active_tasks_;
    std::atomic<uint32_t> next_task_id_;

    static constexpr uint32_t NUM_TRIPLE_BUFFERS = 3;
    VkCommandBuffer triple_compute_buffers_[NUM_TRIPLE_BUFFERS];
    VkCommandBuffer triple_transfer_buffers_[NUM_TRIPLE_BUFFERS];
    VkFence triple_fences_[NUM_TRIPLE_BUFFERS];
    std::atomic<uint32_t> triple_index_;
};

}
