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

    bool is_initialized() const { return initialized_; }
    bool has_validation() const { return validation_layer_ != nullptr; }

    VkInstance get_instance() const { return instance_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    VkDevice get_device() const { return device_; }
    VkQueue get_compute_queue() const { return compute_queue_; }
    uint32_t get_compute_queue_family() const { return compute_queue_family_; }

    bool handle_error(VkResult result, const std::string& operation);
    bool handle_robust_errors(VkResult result, const std::string& operation);

private:
    bool create_instance();
    bool select_physical_device();
    bool create_logical_device();
    bool find_queue_families();
    void destroy();

    std::mutex mutex_;
    
    VulkanConfig config_;
    bool initialized_;
    
    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;
    uint32_t compute_queue_family_;
    
    std::unique_ptr<ValidationLayer> validation_layer_;
};

}
