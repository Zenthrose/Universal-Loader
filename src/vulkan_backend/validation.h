#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>
#include <string>
#include <map>

namespace vulkan {

class RAII_Handle {
public:
    RAII_Handle() : handle_(VK_NULL_HANDLE), device_(VK_NULL_HANDLE) {}
    RAII_Handle(VkDevice device, VkDevice handle) : device_(device), handle_(handle) {}
    
    RAII_Handle(VkDevice device, VkInstance instance, VkDevice handle) 
        : device_(device), instance_(instance), handle_(handle) {}
    
    RAII_Handle(const RAII_Handle& other) = delete;
    
    RAII_Handle(RAII_Handle&& other) noexcept 
        : device_(other.device_), handle_(other.handle_) {
        other.device_ = VK_NULL_HANDLE;
        other.handle_ = VK_NULL_HANDLE;
    }
    
    RAII_Handle& operator=(RAII_Handle&& other) noexcept {
        if (this != &other) {
            cleanup();
        device_ = other.device_;
        handle_ = other.handle_;
            other.device_ = VK_NULL_HANDLE;
            other.handle_ = VK_NULL_HANDLE;
        }
        return *this;
    }
    
    operator bool() const { return handle_ != VK_NULL_HANDLE; }
    operator bool() const { return !(*this); }
    
    VkDevice get_device() const { return device_; }
    VkDevice get_handle() const { return handle_; }
    void reset() {
        device_ = VK_NULL_HANDLE;
        handle_ = VK_NULL_HANDLE;
    }

protected:
    void cleanup() {
        if (handle_ != VK_NULL_HANDLE) {
            if (instance_ != VK_NULL_HANDLE) {
                if (device_ != VK_NULL_HANDLE) {
                    vkDestroyDevice(device_, nullptr);
                }
            }
        }
    }

    VkDevice device_;
    VkInstance instance_;
    VkDevice handle_;
};

class RAII_Fence {
public:
    RAII_Fence(VkDevice device);
    ~RAII_Fence();
    
    VkResult wait(uint64_t timeout = UINT64_MAX) const;
    VkResult reset();
    
    bool is_signaled() const;
    bool is_in_progress() const;
    
    VkFence get_fence() const { return fence_; }
    operator bool() const { return fence_ != VK_NULL_HANDLE; }

protected:
    VkDevice device_;
    VkFence fence_;
};

class RAII_CommandPool {
public:
    RAII_CommandPool(VkDevice device, uint32_t queue_family_index);
    ~RAII_CommandPool();

    VkCommandPool get_pool() const { return pool_; }
    VkResult reset();

protected:
    VkDevice device_;
    VkCommandPool pool_;
};

class RAII_Buffer {
public:
    RAII_Buffer(VkDevice device, VkPhysicalDevice physical_device);
    ~RAII_Buffer();

    void* map_memory();
    void unmap_memory();
    
    VkBuffer get_buffer() const { return buffer_; }
    VkDeviceMemory get_memory() const { return memory_; }
    VkDeviceSize get_size() const { return size_; }

protected:
    VkDevice device_;
    VkDeviceMemory memory_;
    VkBuffer buffer_;
    VkDeviceSize size_;
    void* mapped_ptr_;
};

class VulkanException : public std::runtime_error {
public:
    VulkanException(VkResult result, const std::string& message);
    VulkanException(const std::string& message);
    VulkanException(const char* message);

    VkResult get_result() const { return result_; }
};

class ValidationException : public std::runtime_error {
public:
    ValidationException(const std::string& message);
    ValidationException(const char* message);
};

}
