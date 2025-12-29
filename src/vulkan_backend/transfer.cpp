#include "transfer.h"

namespace vulkan {

TransferEngine::TransferEngine(VkDevice device, VkQueue queue, uint32_t queue_family)
    : device_(device), queue_(queue), queue_family_(queue_family), triple_index_(0) {

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = NUM_TRIPLE_BUFFERS;

    vkAllocateCommandBuffers(device_, &alloc_info, triple_cmd_buffers_);

    for (uint32_t i = 0; i < NUM_TRIPLE_BUFFERS; ++i) {
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device_, &fence_info, nullptr, &triple_fences_[i]);
    }
}

TransferEngine::~TransferEngine() {
    for (auto fence : active_fences_) {
        vkDestroyFence(device_, fence, nullptr);
    }
    for (auto fence : triple_fences_) {
        vkDestroyFence(device_, fence, nullptr);
    }
    if (command_pool_) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
}

void TransferEngine::async_copy(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                               std::function<void()> callback) {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy copy_region{};
    copy_region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy_region);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFence fence;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device_, &fence_info, nullptr, &fence);

    vkQueueSubmit(queue_, 1, &submit_info, fence);

    active_buffers_.push_back(cmd);
    active_fences_.push_back(fence);
}

void TransferEngine::wait_all() {
    vkWaitForFences(device_, active_fences_.size(), active_fences_.data(), VK_TRUE, UINT64_MAX);
    for (auto fence : active_fences_) {
        vkDestroyFence(device_, fence, nullptr);
    }
    for (auto cmd : active_buffers_) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
    }
    active_buffers_.clear();
    active_fences_.clear();
}

void TransferEngine::triple_buffer_submit(VkBuffer src, VkBuffer dst, VkDeviceSize size,
                                         uint32_t buffer_index) {
    uint32_t index = buffer_index % NUM_TRIPLE_BUFFERS;

    vkWaitForFences(device_, 1, &triple_fences_[index], VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &triple_fences_[index]);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(triple_cmd_buffers_[index], &begin_info);

    VkBufferCopy copy_region{};
    copy_region.size = size;
    vkCmdCopyBuffer(triple_cmd_buffers_[index], src, dst, 1, &copy_region);

    vkEndCommandBuffer(triple_cmd_buffers_[index]);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &triple_cmd_buffers_[index];

    vkQueueSubmit(queue_, 1, &submit_info, triple_fences_[index]);

    triple_index_.store((triple_index_.load() + 1) % NUM_TRIPLE_BUFFERS);
}

bool TransferEngine::triple_buffer_ready(uint32_t buffer_index) const {
    uint32_t index = buffer_index % NUM_TRIPLE_BUFFERS;
    return vkGetFenceStatus(device_, triple_fences_[index]) == VK_SUCCESS;
}

void TransferEngine::triple_buffer_wait_all() {
    for (uint32_t i = 0; i < NUM_TRIPLE_BUFFERS; ++i) {
        vkWaitForFences(device_, 1, &triple_fences_[i], VK_TRUE, UINT64_MAX);
    }
}

}
