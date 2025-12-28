#pragma once
#include <vulkan/vulkan.hpp>
#include <memory>
#include <mutex>
#include "validation.h"

namespace vulkan {

struct VulkanConfig {
    uint32_t gpu_memory_pool_size;
    uint32_t max_compute_queues;
    bool enable_validation;
    bool enable_robust_error_handling;
};

class VulkanContext {
public:
    explicit VulkanContext(const VulkanConfig& config);
    ~VulkanContext();

    vk::Instance get_instance() const { return instance_; }
    vk::PhysicalDevice get_physical_device() const { return physical_device_; }
    vk::Device get_device() const { return device_; }
    vk::Queue get_compute_queue() const { return compute_queue_; }
    uint32_t get_compute_queue_family() const { return compute_queue_family_; }

    bool is_initialized() const { return initialized_; }
    bool has_validation() const { return validation_layer_ != nullptr; }

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

private:
    bool create_instance();
    bool select_physical_device();
    bool create_logical_device();
    bool find_queue_families();
    void destroy();

    std::mutex mutex_;
    
    VulkanConfig config_;
    bool initialized_;
    
    std::unique_ptr<vk::raii::Instance> instance_;
    std::unique_ptr<vk::raii::Device> device_;
    vk::Queue compute_queue_;
    uint32_t compute_queue_family_;
    
    std::unique_ptr<ValidationLayer> validation_layer_;
};

}
