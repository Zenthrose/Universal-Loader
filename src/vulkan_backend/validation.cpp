#include "validation.h"

namespace vulkan {

void ValidationLayer::log_validation_error(VkResult result, const char* operation, const char* details) {
    std::cerr << "[VALIDATION ERROR] " << operation << ": " << "Code " << result << std::endl;
    
    if (details) {
        std::cerr << "[VALIDATION ERROR] " << details << std::endl;
    }
}

void ValidationLayer::log_validation_info(const char* message) {
    std::cout << "[VALIDATION] " << message << std::endl;
}

VkResult ValidationLayer::check_validation_layers() {
    std::cout << "[VALIDATION] Checking Vulkan validation layer setup..." << std::endl;
    return VK_SUCCESS;
}

VkResult ValidationLayer::initialize_validation_layers(VkPhysicalDevice physical_device, VkDevice device, VkInstance instance) {
    std::cout << "[VALIDATION] Initializing validation layers (disabled)" << std::endl;
    return VK_SUCCESS;
}

void ValidationLayer::destroy() {
}

void VulkanException::log_exception(const char* message) {
    std::cerr << "[FATAL] " << message << std::endl;
}

void VulkanException::throw_if_error(const char* message) {
    throw std::runtime_error(message);
}
