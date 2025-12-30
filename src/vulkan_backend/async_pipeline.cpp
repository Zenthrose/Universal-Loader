#include "async_pipeline.h"
#include <iostream>

namespace vulkan {

AsyncPipelineManager::AsyncPipelineManager(VkDevice device, VkQueue compute_queue,
                                          VkQueue transfer_queue, uint32_t compute_family,
                                          uint32_t transfer_family)
    : device_(device), compute_queue_(compute_queue), transfer_queue_(transfer_queue),
      compute_family_(compute_family), transfer_family_(transfer_family),
      compute_pool_(VK_NULL_HANDLE), transfer_pool_(VK_NULL_HANDLE),
      running_(false), active_tasks_(0), next_task_id_(1), triple_index_(0) {

    for (uint32_t i = 0; i < NUM_TRIPLE_BUFFERS; ++i) {
        triple_compute_buffers_[i] = VK_NULL_HANDLE;
        triple_transfer_buffers_[i] = VK_NULL_HANDLE;
        triple_fences_[i] = VK_NULL_HANDLE;
    }
}

AsyncPipelineManager::~AsyncPipelineManager() {
    cleanup();
}

bool AsyncPipelineManager::initialize() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &pool_info, nullptr, &compute_pool_) != VK_SUCCESS) {
        std::cerr << "[AsyncPipeline] Failed to create compute command pool" << std::endl;
        return false;
    }

    pool_info.queueFamilyIndex = transfer_family_;
    if (vkCreateCommandPool(device_, &pool_info, nullptr, &transfer_pool_) != VK_SUCCESS) {
        std::cerr << "[AsyncPipeline] Failed to create transfer command pool" << std::endl;
        return false;
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = NUM_TRIPLE_BUFFERS;

    alloc_info.commandPool = compute_pool_;
    vkAllocateCommandBuffers(device_, &alloc_info, triple_compute_buffers_);

    alloc_info.commandPool = transfer_pool_;
    vkAllocateCommandBuffers(device_, &alloc_info, triple_transfer_buffers_);

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < NUM_TRIPLE_BUFFERS; ++i) {
        vkCreateFence(device_, &fence_info, nullptr, &triple_fences_[i]);
    }

    running_.store(true);
    task_thread_ = std::thread(&AsyncPipelineManager::task_thread_func, this);

    std::cout << "[AsyncPipeline] Initialized with triple buffering" << std::endl;
    return true;
}

void AsyncPipelineManager::cleanup() {
    running_.store(false);
    if (task_thread_.joinable()) {
        task_thread_.join();
    }

    for (auto fence : triple_fences_) {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence, nullptr);
        }
    }

    if (compute_pool_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, compute_pool_, NUM_TRIPLE_BUFFERS, triple_compute_buffers_);
        vkDestroyCommandPool(device_, compute_pool_, nullptr);
    }

    if (transfer_pool_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, transfer_pool_, NUM_TRIPLE_BUFFERS, triple_transfer_buffers_);
        vkDestroyCommandPool(device_, transfer_pool_, nullptr);
    }
}

void AsyncPipelineManager::submit_compute_task(const AsyncPipelineTask& task) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    pending_tasks_.push_back(task);
}

void AsyncPipelineManager::submit_transfer_task(const AsyncPipelineTask& task) {
    std::lock_guard<std::mutex> lock(task_mutex_);
    pending_tasks_.push_back(task);
}

void AsyncPipelineManager::wait_for_task(uint32_t task_id, uint64_t timeout_ns) {
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        {
            std::lock_guard<std::mutex> lock(task_mutex_);

            bool found = false;
            for (const auto& task : active_tasks_) {
                if (task.task_id == task_id) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                return;
            }
        }

        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();

        if (elapsed >= timeout_ns) {
            std::cerr << "[AsyncPipeline] Task " << task_id << " timeout" << std::endl;
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AsyncPipelineManager::task_thread_func() {
    while (running_.load()) {
        AsyncPipelineTask task;

        {
            std::lock_guard<std::mutex> lock(task_mutex_);
            if (pending_tasks_.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            task = pending_tasks_.back();
            pending_tasks_.pop_back();
        }

        execute_task(task);
    }
}

void AsyncPipelineManager::execute_task(const AsyncPipelineTask& task) {
    active_task_count_.fetch_add(1);

    try {
        if (task.compute_fn) {
            task.compute_fn();
        }

        if (task.transfer_fn) {
            task.transfer_fn();
        }
    } catch (const std::exception& e) {
        std::cerr << "[AsyncPipeline] Task " << task.task_id << " failed: " << e.what() << std::endl;
    }

    active_task_count_.fetch_sub(1);
}

void AsyncPipelineManager::update_binding(uint32_t binding_index, VkBuffer buffer, VkDeviceSize size) {
}

}
