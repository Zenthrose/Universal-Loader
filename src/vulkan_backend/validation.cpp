#include "validation.h"
#include <iostream>
#include <cstring>
#include <vector>

namespace vulkan {

ValidationLayer::ValidationLayer(VkDevice device, VkPhysicalDevice physical_device, VkInstance instance, bool enable_validation)
    : enable_validation_(enable_validation), device_(device), physical_device_(physical_device), instance_(instance), debug_messenger_(VK_NULL_HANDLE) {

    if (enable_validation_) {
        initialize_validation_layers();
    }
}

ValidationLayer::~ValidationLayer() {
    destroy();
}

uint32_t ValidationLayer::check_validation_layers() {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    const char* validation_layer_name = "VK_LAYER_KHRONOS_validation";
    for (const auto& layer_properties : available_layers) {
        if (strcmp(layer_properties.layerName, validation_layer_name) == 0) {
            return static_cast<uint32_t>(ValidationError::NO_ERROR);
        }
    }

    return static_cast<uint32_t>(ValidationError::VALIDATION_LAYER_MISSING);
}

uint32_t ValidationLayer::initialize_validation_layers() {
    if (!check_validation_layers()) {
        return static_cast<uint32_t>(ValidationError::VALIDATION_LAYER_MISSING);
    }

    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debug_utils_messenger_callback;
    create_info.pUserData = this;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        if (func(instance_, &create_info, nullptr, &debug_messenger_) != VK_SUCCESS) {
            return static_cast<uint32_t>(ValidationError::INITIALIZATION_FAILED);
        }
    } else {
        return static_cast<uint32_t>(ValidationError::VALIDATION_LAYER_MISSING);
    }

    log_validation_info("Validation layers initialized successfully");
    return static_cast<uint32_t>(ValidationError::NO_ERROR);
}

void ValidationLayer::destroy_debug_messenger() {
    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance_, debug_messenger_, nullptr);
        }
        debug_messenger_ = VK_NULL_HANDLE;
    }
}

void ValidationLayer::destroy() {
    destroy_debug_messenger();
}

void ValidationLayer::log_validation_error(uint32_t error_code, const char* operation, const char* details) {
    std::cerr << "[VULKAN ERROR] Code: " << error_code
              << " | Operation: " << operation
              << " | Details: " << (details ? details : "N/A") << std::endl;
}

void ValidationLayer::log_validation_info(const char* message) {
    if (enable_validation_) {
        std::cout << "[VULKAN INFO] " << message << std::endl;
    }
}

VkBool32 VKAPI_CALL ValidationLayer::debug_utils_messenger_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_types,
    const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
    void* p_user_data) {

    auto* validation_layer = static_cast<ValidationLayer*>(p_user_data);

    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        validation_layer->log_validation_error(
            static_cast<uint32_t>(p_callback_data->messageIdNumber),
            p_callback_data->pMessageIdName ? p_callback_data->pMessageIdName : "unknown",
            p_callback_data->pMessage
        );
    } else if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cout << "[VULKAN WARNING] " << p_callback_data->pMessage << std::endl;
    } else {
        std::cout << "[VULKAN DEBUG] " << p_callback_data->pMessage << std::endl;
    }

    return VK_FALSE;
}

}