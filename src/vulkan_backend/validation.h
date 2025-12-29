#pragma once
#include <vulkan/vulkan.h>
#include <cstring>
#include <iostream>
#include <sstream>

namespace vulkan {

enum class ValidationError : uint32_t {
    NO_ERROR = 0,
    INITIALIZATION_FAILED = 1,
    VALIDATION_LAYER_MISSING = 2,
    DEVICE_LOST = 3,
    OUT_OF_MEMORY = 4,
    SHADER_COMPILATION_FAILED = 5,
    PIPELINE_CACHE_MISS = 6,
    BINDING_INCONSISTENT = 7,
    DEVICE_MEMORY_LIMIT_EXCEEDED = 8,
    UNKNOWN_ERROR = 9,
    PORTABILITY_NOT_SUPPORTED = 10,
    TIMELINE_SEMAPHORE_NOT_SUPPORTED = 11
};

class VulkanException : public std::runtime_error {
public:
    VulkanException(uint32_t error_code, const std::string& message)
        : std::runtime_error(message), error_code_(error_code) {}

    uint32_t get_result() const { return error_code_; }

private:
    uint32_t error_code_;
};

class ValidationLayer {
public:
    ValidationLayer(VkDevice device, VkPhysicalDevice physical_device, VkInstance instance, bool enable_validation);
    ~ValidationLayer();

    bool is_validation_enabled() const { return enable_validation_; }

    void log_validation_error(uint32_t error_code, const char* operation, const char* details);
    void log_validation_info(const char* message);
    uint32_t check_validation_layers();
    uint32_t initialize_validation_layers();
    void destroy();

private:
    bool enable_validation_;
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkInstance instance_;
    VkDebugUtilsMessengerEXT debug_messenger_;

    static VkBool32 VKAPI_CALL debug_utils_messenger_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_types,
        const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
        void* p_user_data);

    void destroy_debug_messenger();
};

}
