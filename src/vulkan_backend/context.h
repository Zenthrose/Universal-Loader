#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include "validation.h"

namespace vulkan {

struct VulkanConfig {
    uint32_t gpu_memory_pool_size;
    uint32_t max_compute_queues;
    bool enable_validation;
    bool enable_robust_error_handling;
};

struct WorkgroupSize {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

class VulkanContext {
public:
    explicit VulkanContext(const VulkanConfig& config);
    ~VulkanContext();

    bool is_initialized() const { return initialized_; }
    bool has_validation() const { return validation_layer_ != nullptr; }
    bool supports_portability_subset() const { return supports_portability_subset_; }
    bool supports_buffer_device_address() const { return supports_buffer_device_address_; }
    bool supports_cooperative_matrix() const { return supports_cooperative_matrix_; }

    VkInstance get_instance() const { return instance_; }
    VkPhysicalDevice get_physical_device() const { return physical_device_; }
    VkDevice get_device() const { return device_; }
    VkQueue get_compute_queue() const { return compute_queue_; }
    uint32_t get_compute_queue_family() const { return compute_queue_family_; }
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

    bool handle_error(VkResult result, const std::string& operation);
    bool handle_robust_errors(VkResult result, const std::string& operation);

    WorkgroupSize calculate_optimal_workgroup_size(uint32_t total_work);
    WorkgroupSize calculate_workgroup_for_gemm(uint32_t m, uint32_t n, uint32_t k);

    uint32_t get_subgroup_size() const { return subgroup_size_; }
    uint32_t get_max_compute_workgroup_size() const { return max_compute_workgroup_size_; }

private:
    bool create_instance();
    bool select_physical_device();
    bool create_logical_device();
    bool find_queue_families();
    void query_device_properties();

    void destroy();

    std::mutex mutex_;

    VulkanConfig config_;
    bool initialized_;

    VkInstance instance_;
    VkPhysicalDevice physical_device_;
    VkDevice device_;
    VkQueue compute_queue_;
    uint32_t compute_queue_family_;

    VkPhysicalDeviceProperties device_properties_;
    VkPhysicalDeviceSubgroupProperties subgroup_properties_;

    uint32_t subgroup_size_;
    uint32_t max_compute_workgroup_size_;
    bool supports_portability_subset_;
    bool supports_buffer_device_address_;
    bool supports_cooperative_matrix_;

    std::unique_ptr<ValidationLayer> validation_layer_;
};

}
