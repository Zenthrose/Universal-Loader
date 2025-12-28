#pragma once
#include <vulkan/vulkan.h>
#include <functional>

namespace vulkan {

class TransferEngine {
public:
    explicit TransferEngine(VkDevice device, VkQueue queue, uint32_t queue_family);
    ~TransferEngine();

    void async_copy(VkBuffer src, VkBuffer dst, VkDeviceSize size, 
                   std::function<void()> callback);
    void wait_all();

private:
    VkDevice device_;
    VkQueue queue_;
    uint32_t queue_family_;
    VkCommandPool command_pool_;
    std::vector<VkCommandBuffer> active_buffers_;
    std::vector<VkFence> active_fences_;
};

}
