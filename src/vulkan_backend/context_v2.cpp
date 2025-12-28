#include "context_v2.h"
#include <iostream>
#include <stdexcept>

namespace vulkan {

VulkanContext::VulkanContext(const VulkanConfig& config) 
    : config_(config), initialized_(false) {
    std::cout << "[VULKAN] Initializing Vulkan context..." << std::endl;
    
    if (!create_instance()) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }
    
    if (!select_physical_device()) {
        throw std::runtime_error("No suitable Vulkan physical device found");
    }
    
    if (!find_queue_families()) {
        throw std::runtime_error("No suitable queue family found");
    }
    
    if (!create_logical_device()) {
        throw std::runtime_error("Failed to create logical device");
    }
    
    validation_layer_ = std::make_unique<ValidationLayer>(device_, physical_device_, 
                                                         config_.enable_validation);
    
    if (validation_layer_->is_validation_enabled()) {
        auto result = validation_layer_->check_validation_layers();
        if (result != VK_SUCCESS) {
            std::cerr << "[VULKAN] Validation layer check failed: " << result << std::endl;
            std::cerr << "[VULKAN] Continuing without validation..." << std::endl;
        }
    }
    
    initialized_ = true;
    std::cout << "[VULKAN] Initialization complete" << std::endl;
}

VulkanContext::~VulkanContext() {
    std::cout << "[VULKAN] Destroying Vulkan context..." << std::endl;
    destroy();
}

bool VulkanContext::create_instance() {
    try {
        vk::ApplicationInfo app_info(
            "VulkanGGUF",
            vk::make_version(1, 0, 0),
            VK_API_VERSION_1_3,
            "VulkanGGUF LLM Inference Engine",
            VK_MAKE_VERSION(1, 3, 296)
        );
        
        vk::InstanceCreateInfo create_info({}, &app_info);
        instance_ = vk::createInstance(create_info);
        
        std::cout << "[VULKAN] Instance created successfully" << std::endl;
        return true;
    } catch (const vk::SystemError& e) {
        std::cerr << "[VULKAN ERROR] Failed to create instance: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanContext::select_physical_device() {
    try {
        std::vector<vk::PhysicalDevice> devices = instance_.enumeratePhysicalDevices();
        
        if (devices.empty()) {
            std::cerr << "[VULKAN ERROR] No physical devices found" << std::endl;
            return false;
        }
        
        std::cout << "[VULKAN] Found " << devices.size() << " physical device(s)" << std::endl;
        
        for (size_t i = 0; i < devices.size(); ++i) {
            auto props = devices[i].getProperties();
            std::cout << "[VULKAN]   Device " << i << ": " << props.deviceName 
                      << " (Vulkan " << props.apiVersion << ")" << std::endl;
            
            if (i == 0) {
                physical_device_ = devices[0];
            }
        }
        
        std::cout << "[VULKAN] Selected device: " 
                  << physical_device_.getProperties().deviceName << std::endl;
        return true;
    } catch (const vk::SystemError& e) {
        std::cerr << "[VULKAN ERROR] Failed to enumerate physical devices: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanContext::find_queue_families() {
    try {
        auto queue_families = physical_device_.getQueueFamilyProperties();
        
        for (size_t i = 0; i < queue_families.size(); ++i) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_family_ = i;
                std::cout << "[VULKAN] Selected queue family " << i 
                          << " for compute operations" << std::endl;
                return true;
            }
        }
        
        std::cerr << "[VULKAN ERROR] No compute queue family found" << std::endl;
        return false;
    } catch (const vk::SystemError& e) {
        std::cerr << "[VULKAN ERROR] Failed to get queue families: " << e.what() << std::endl;
        return false;
    }
}

bool VulkanContext::create_logical_device() {
    try {
        vk::PhysicalDeviceFeatures features = physical_device_.getFeatures();
        
        vk::DeviceCreateInfo create_info(
            {},
            physical_device_,
            1
        );
        
        create_info.setQueueCreateInfos(vk::DeviceQueueCreateInfo(
            compute_queue_family_, { 1.0f }
        ));
        
        device_ = physical_device_.createDevice(create_info);
        
        compute_queue_ = device_.getQueue(compute_queue_family_, 0);
        
        std::cout << "[VULKAN] Logical device created successfully" << std::endl;
        return true;
    } catch (const vk::SystemError& e) {
        std::cerr << "[VULKAN ERROR] Failed to create logical device: " << e.what() << std::endl;
        return false;
    }
}

void VulkanContext::destroy() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (validation_layer_) {
        validation_layer_->destroy();
        validation_layer_.reset();
    }
    
    compute_queue_.waitIdle();
    
    if (device_) {
        device_.waitIdle();
        device_.destroy();
    }
    
    if (instance_) {
        instance_.destroy();
    }
    
    instance_.reset();
    device_.reset();
    compute_queue_.reset();
    
    initialized_ = false;
}

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    try {
        auto memory_properties = physical_device_.getMemoryProperties();
        
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((type_filter & (1 << i)) && 
                (memory_properties.memoryTypes[i] & properties) == properties) {
                return i;
            }
        }
        
        return UINT32_MAX;
    } catch (const vk::SystemError& e) {
        std::cerr << "[VULKAN ERROR] Failed to query memory properties: " << e.what() << std::endl;
        return UINT32_MAX;
    }
}

}
