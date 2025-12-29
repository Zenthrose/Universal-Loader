#pragma once
#include <vulkan/vulkan.h>
#include <functional>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

namespace vulkan {

class TransferEngine {
 public:
    explicit TransferEngine(VkDevice device, VkQueue queue, uint32_t queue_family);
    ~TransferEngine();

    void async_copy(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                   std::function<void()> callback);
    void wait_all();

    void triple_buffer_submit(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                            uint32_t buffer_index);
    bool triple_buffer_ready(uint32_t buffer_index) const;
    void triple_buffer_wait_all();

 private:
    VkDevice device_;
    VkQueue queue_;
    uint32_t queue_family_;

    VkCommandPool command_pool_;
    std::vector<VkCommandBuffer> active_buffers_;
    std::vector<VkFence> active_fences_;

    static constexpr uint32_t NUM_TRIPLE_BUFFERS = 3;
    VkCommandBuffer triple_cmd_buffers_[NUM_TRIPLE_BUFFERS];
    VkFence triple_fences_[NUM_TRIPLE_BUFFERS];
    std::atomic<uint32_t> triple_index_;
};

}

