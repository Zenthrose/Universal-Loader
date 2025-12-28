#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

namespace vulkan {

struct VulkanConfig {
    uint32_t gpu_memory_pool_size;
    uint32_t max_compute_queues;
    bool enable_validation;
};

class VulkanContext {
public:
    explicit VulkanContext(const VulkanConfig& config);
    ~VulkanContext();

    VkInstance get_instance() const { return instance_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    VkDevice get_device() const { return device_; }
    VkQueue get_compute_queue() const { return compute_queue_; }
    uint32_t get_compute_queue_family() const { return compute_queue_family_; }
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

    bool is_initialized() const { return initialized_; }

private:
    bool create_instance();
    bool select_physical_device();
    bool create_logical_device();
    bool find_queue_families();

    VulkanConfig config_;
    bool initialized_;

    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;
    uint32_t compute_queue_family_;
};

}
