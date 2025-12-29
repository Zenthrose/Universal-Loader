#include "context.h"
#include <iostream>

namespace vulkan {

VulkanContext::VulkanContext(const VulkanConfig& config)
    : config_(config), initialized_(false), instance_(VK_NULL_HANDLE), physical_device_(VK_NULL_HANDLE),
      device_(VK_NULL_HANDLE), compute_queue_(VK_NULL_HANDLE), compute_queue_family_(0),
      subgroup_size_(32), max_compute_workgroup_size_(1024), supports_portability_subset_(false) {

    if (!create_instance()) {
        return;
    }
    if (!select_physical_device()) {
        return;
    }
    query_device_properties();

    // Check for required Vulkan extensions
    bool has_required_extensions = supports_buffer_device_address_ && supports_timeline_semaphore_ && supports_device_group_ && supports_cooperative_matrix_;
    if (!has_required_extensions) {
        std::cerr << "[VulkanContext] ERROR: Required Vulkan extensions not supported on this device:" << std::endl;
        if (!supports_buffer_device_address_) std::cerr << "  - VK_KHR_buffer_device_address" << std::endl;
        if (!supports_timeline_semaphore_) std::cerr << "  - VK_KHR_timeline_semaphore" << std::endl;
        if (!supports_device_group_) std::cerr << "  - VK_KHR_device_group" << std::endl;
        if (!supports_cooperative_matrix_) std::cerr << "  - VK_KHR_cooperative_matrix" << std::endl;
        std::cerr << "Please ensure your GPU and drivers support Vulkan 1.3+ and the required extensions." << std::endl;
        std::cerr << "Falling back to CPU-only mode if available." << std::endl;
        initialized_ = false;
        return;
    }

    if (!find_queue_families()) {
        return;
    }
    if (!create_logical_device()) {
        return;
    }
    initialized_ = true;
}

VulkanContext::~VulkanContext() {
    destroy();
}

void VulkanContext::destroy() {
    if (validation_layer_) {
        validation_layer_->destroy();
        validation_layer_.reset();
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
}

bool VulkanContext::create_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "VulkanGGUF";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "NoEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    std::vector<const char*> extensions;
    std::vector<const char*> layers;

#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
    if (config_.enable_validation) {
        bool validation_available = false;
        uint32_t layer_count;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> available_layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

        for (const auto& layer : available_layers) {
            if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                validation_available = true;
                break;
            }
        }

        if (validation_available) {
            layers.push_back(validation_layers[0]);
        } else {
            std::cerr << "[VulkanContext] Validation layers not available" << std::endl;
        }
    }

    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_extensions.data());

    for (const auto& ext : available_extensions) {
        if (strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0 && config_.enable_validation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }

    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();

#ifdef __APPLE__
    VkInstanceCreateInfoKHR create_info_khr{};
    create_info_khr.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info_khr.pApplicationInfo = &app_info;
    create_info_khr.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info_khr.ppEnabledExtensionNames = extensions.data();
    create_info_khr.enabledLayerCount = static_cast<uint32_t>(layers.size());
    create_info_khr.ppEnabledLayerNames = layers.data();
    create_info_khr.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    VkResult result = vkCreateInstance(&create_info_khr, nullptr, &instance_);
#else
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
#endif

    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance: " << result << std::endl;
        return false;
    }

    return true;
}

bool VulkanContext::select_physical_device() {
    uint32_t device_count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (result != VK_SUCCESS || device_count == 0) {
        std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
        return false;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    physical_device_ = devices[0];

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);

    std::cout << "Selected physical device: " << properties2.properties.deviceName << std::endl;

    return true;
}

void VulkanContext::query_device_properties() {
    vkGetPhysicalDeviceProperties(physical_device_, &device_properties_);

    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroup_properties_;

    subgroup_properties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup_properties_.pNext = nullptr;

    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);

    subgroup_size_ = subgroup_properties_.subgroupSize;
    max_compute_workgroup_size_ = device_properties_.limits.maxComputeWorkGroupSize[0];

    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, extensions.data());

    supports_portability_subset_ = false;
    supports_buffer_device_address_ = false;
    supports_cooperative_matrix_ = false;
    supports_timeline_semaphore_ = false;
    supports_device_group_ = false;

    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0) {
            supports_portability_subset_ = true;
            std::cout << "[VulkanContext] Device supports VK_KHR_portability_subset" << std::endl;
        }
        if (strcmp(ext.extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
            supports_buffer_device_address_ = true;
            std::cout << "[VulkanContext] Device supports VK_KHR_buffer_device_address" << std::endl;
        }
        if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            supports_cooperative_matrix_ = true;
            std::cout << "[VulkanContext] Device supports VK_KHR_cooperative_matrix" << std::endl;
        }
        if (strcmp(ext.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) {
            supports_timeline_semaphore_ = true;
            std::cout << "[VulkanContext] Device supports VK_KHR_timeline_semaphore" << std::endl;
        }
        if (strcmp(ext.extensionName, VK_KHR_DEVICE_GROUP_EXTENSION_NAME) == 0) {
            supports_device_group_ = true;
            std::cout << "[VulkanContext] Device supports VK_KHR_device_group" << std::endl;
        }
    }

    std::cout << "[VulkanContext] Device: " << device_properties_.deviceName << std::endl;
    std::cout << "[VulkanContext] Subgroup size: " << subgroup_size_ << std::endl;
    std::cout << "[VulkanContext] Max workgroup size: " << max_compute_workgroup_size_ << std::endl;
}

bool VulkanContext::find_queue_families() {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_ = i;
            return true;
        }
    }
    std::cerr << "Failed to find compute queue family" << std::endl;
    return false;
}

bool VulkanContext::create_logical_device() {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_;
    queue_create_info.queueCount = 1;
    float queue_priority = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features{};
    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT buffer_address_features{};
    buffer_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT;
    buffer_address_features.pNext = nullptr;
    buffer_address_features.bufferDeviceAddress = VK_TRUE;

    std::vector<const char*> device_extensions;
    if (supports_timeline_semaphore_) {
        device_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }
    if (supports_device_group_) {
        device_extensions.push_back(VK_KHR_DEVICE_GROUP_EXTENSION_NAME);
    }

    void* p_next = nullptr;
    if (supports_buffer_device_address_) {
        p_next = &buffer_address_features;
        device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }

#ifdef __APPLE__
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portability_features{};
    portability_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR;
    portability_features.pNext = p_next;
    portability_features.mutableComparisonSamplers = VK_TRUE;
    portability_features.triangleFans = VK_TRUE;

    if (supports_portability_subset_) {
        device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }

    p_next = &portability_features;
    VkDeviceCreateInfoKHR create_info_khr{};
    create_info_khr.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info_khr.pQueueCreateInfos = &queue_create_info;
    create_info_khr.queueCreateInfoCount = 1;
    create_info_khr.pEnabledFeatures = &device_features;
    create_info_khr.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info_khr.ppEnabledExtensionNames = device_extensions.data();
    create_info_khr.pNext = p_next;

    VkResult result = vkCreateDevice(physical_device_, &create_info_khr, nullptr, &device_);
#else
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();
    create_info.pNext = p_next;

    VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
#endif

    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanContext] Failed to create logical device: " << result << std::endl;
        return false;
    }

    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

    if (config_.enable_validation) {
        validation_layer_ = std::make_unique<ValidationLayer>(
            device_, physical_device_, instance_, true
        );
    }

    std::cout << "[VulkanContext] Logical device created successfully" << std::endl;
    return true;
}

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool VulkanContext::handle_error(VkResult result, const std::string& operation) {
    if (result != VK_SUCCESS) {
        if (validation_layer_) {
            validation_layer_->log_validation_error(result, operation.c_str(), nullptr);
        }
        return false;
    }
    return true;
}

bool VulkanContext::handle_robust_errors(VkResult result, const std::string& operation) {
    if (result != VK_SUCCESS) {
        switch (result) {
            case VK_ERROR_OUT_OF_HOST_MEMORY:
            case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                std::cerr << "[FATAL] Out of memory during " << operation << std::endl;
                break;
            case VK_ERROR_DEVICE_LOST:
                std::cerr << "[FATAL] Device lost during " << operation << std::endl;
                break;
            default:
                std::cerr << "[ERROR] Vulkan error " << result << " during " << operation << std::endl;
        }
        return false;
    }
    return true;
}

WorkgroupSize VulkanContext::calculate_optimal_workgroup_size(uint32_t total_work) {
    uint32_t optimal_x = std::min(subgroup_size_, max_compute_workgroup_size_);

    uint32_t num_groups_x = (total_work + optimal_x - 1) / optimal_x;

    WorkgroupSize wg;
    wg.x = optimal_x;
    wg.y = 1;
    wg.z = 1;

    return wg;
}

WorkgroupSize VulkanContext::calculate_workgroup_for_gemm(uint32_t m, uint32_t n, uint32_t k) {
    WorkgroupSize wg;

    uint32_t tile_size = 8;
    if (max_compute_workgroup_size_ >= 256) {
        tile_size = 16;
    }

    wg.x = tile_size;
    wg.y = tile_size;
    wg.z = 1;

    uint32_t workgroup_limit = device_properties_.limits.maxComputeWorkGroupInvocations;
    uint32_t total_threads = wg.x * wg.y * wg.z;

    if (total_threads > workgroup_limit) {
        wg.x = std::min(wg.x, workgroup_limit);
        wg.y = 1;
        wg.z = 1;
    }

    return wg;
}

}
