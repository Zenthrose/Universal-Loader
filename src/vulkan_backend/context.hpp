#pragma once

#include <vulkan/vulkan.hpp>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <iostream>

namespace vulkan {

struct VulkanConfig {
    bool enable_validation = true;
    bool enable_portability = false;  // For macOS/MoltenVK
    std::vector<std::string> required_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };
    std::vector<std::string> validation_layers = {
        "VK_LAYER_KHRONOS_validation"
    };
};

class VulkanContext {
public:
    VulkanContext(const VulkanConfig& config = VulkanConfig{});
    ~VulkanContext();

    // Initialization and cleanup
    bool initialize();
    void shutdown();

    // Core accessors
    const vk::Instance& get_instance() const { return *instance_; }
    const vk::Device& get_device() const { return *device_; }
    vk::PhysicalDevice get_physical_device() const { return physical_device_; }
    
    // Queue access
    vk::Queue get_compute_queue() const { return compute_queue_; }
    uint32_t get_compute_queue_family() const { return compute_queue_family_; }
    
    // Device properties
    const vk::PhysicalDeviceProperties& get_properties() const { return properties_; }
    const vk::PhysicalDeviceFeatures& get_features() const { return features_; }
    
    // Status checks
    bool is_initialized() const { return initialized_; }
    bool supports_timeline_semaphores() const { return supports_timeline_semaphores_; }
    bool supports_bindless() const { return supports_bindless_; }
    bool supports_cooperative_matrix() const { return supports_cooperative_matrix_; }

private:
    // Core objects - RAII managed by Vulkan-Hpp
    std::unique_ptr<vk::Instance> instance_;
    std::unique_ptr<vk::Device> device_;
    vk::PhysicalDevice physical_device_;
    std::unique_ptr<vk::DebugUtilsMessengerEXT> debug_messenger_;

    // Queue objects
    vk::Queue compute_queue_;
    uint32_t compute_queue_family_;

    // Device properties
    vk::PhysicalDeviceProperties properties_;
    vk::PhysicalDeviceFeatures features_;
    vk::PhysicalDeviceMemoryProperties memory_properties_;

    // Feature support flags
    bool initialized_ = false;
    bool supports_timeline_semaphores_ = false;
    bool supports_bindless_ = false;
    bool supports_cooperative_matrix_ = false;

    // Configuration
    VulkanConfig config_;

    // Helper methods
    bool create_instance();
    bool setup_debug_messenger();
    bool select_physical_device();
    bool create_device();
    bool query_device_features();
    
    // Extension and layer management
    std::vector<const char*> get_required_extensions();
    std::vector<const char*> get_validation_layers();
    bool check_extensions_support(const std::vector<const char*>& extensions);
    bool check_validation_layer_support(const std::vector<const char*>& layers);
    
    // Queue family detection
    bool find_compute_queue_family();
    
    // Debug callback
    VKAPI_ATTR static VKAPI_CALL VkBool32 debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void* user_data
    );
    
    // Utility methods
    void log_device_info();
    std::string api_version_to_string(uint32_t version);
};

} // namespace vulkan