#include "multi_gpu_manager.h"
#include <iostream>

namespace vulkan {

MultiGPUManager::MultiGPUManager(VkInstance instance)
    : instance_(instance), supports_device_groups_(false) {

    memset(&device_group_props_, 0, sizeof(device_group_props_));
}

MultiGPUManager::~MultiGPUManager() {
    cleanup();
}

bool MultiGPUManager::initialize() {
    if (!enumerate_physical_devices()) {
        return false;
    }

    if (!query_device_group_properties()) {
        return false;
    }

    if (!create_logical_devices()) {
        return false;
    }

    std::cout << "[MultiGPU] Initialized with " << devices_.size() << " devices" << std::endl;
    return true;
}

void MultiGPUManager::cleanup() {
    for (auto device : devices_) {
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
        }
    }
    devices_.clear();
    queues_.clear();
    queue_families_.clear();
}

void MultiGPUManager::enable_device_groups() {
    if (!supports_device_groups_) {
        std::cerr << "[MultiGPU] Device groups not supported" << std::endl;
        return;
    }

    if (!create_device_group()) {
        std::cerr << "[MultiGPU] Failed to create device group" << std::endl;
    }
}

VkDevice MultiGPUManager::get_device(uint32_t index) const {
    if (index >= devices_.size()) {
        return VK_NULL_HANDLE;
    }
    return devices_[index];
}

bool MultiGPUManager::enumerate_physical_devices() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

    if (device_count == 0) {
        std::cerr << "[MultiGPU] No physical devices found" << std::endl;
        return false;
    }

    config_.physical_devices.resize(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, config_.physical_devices.data());

    std::cout << "[MultiGPU] Found " << device_count << " physical devices" << std::endl;
    return true;
}

bool MultiGPUManager::query_device_group_properties() {
    uint32_t group_count = 0;
    vkEnumeratePhysicalDeviceGroups(instance_, &group_count, nullptr);

    if (group_count > 1) {
        supports_device_groups_ = true;

        std::vector<VkPhysicalDeviceGroupProperties> groups(group_count);
        vkEnumeratePhysicalDeviceGroups(instance_, &group_count, groups.data());

        device_group_props_ = groups[0];

        std::cout << "[MultiGPU] Device groups supported, " << group_count << " groups" << std::endl;
    } else {
        std::cout << "[MultiGPU] Device groups not available" << std::endl;
    }

    return true;
}

bool MultiGPUManager::create_device_group() {
    if (!supports_device_groups_ || device_group_props_.physicalDeviceCount < 2) {
        return false;
    }

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::vector<float> queue_priorities(device_group_props_.physicalDeviceCount, 1.0f);

    for (uint32_t i = 0; i < device_group_props_.physicalDeviceCount; ++i) {
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = config_.queue_families[i];
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priorities[i];

        queue_create_infos.push_back(queue_info);
    }

    VkDeviceGroupDeviceCreateInfo group_info{};
    group_info.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO;
    group_info.physicalDeviceCount = device_group_props_.physicalDeviceCount;
    group_info.pPhysicalDevices = config_.physical_devices.data();

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();
    create_info.pNext = &group_info;

    VkDeviceGroupDeviceCreateInfo group_create_info{};
    group_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO;
    group_create_info.physicalDeviceCount = device_group_props_.physicalDeviceCount;
    group_create_info.pPhysicalDevices = config_.physical_devices.data();
    group_create_info.pPhysicalDeviceMasks = device_group_props_.subsetAllocationMasks;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &group_create_info;
    device_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    device_info.pQueueCreateInfos = queue_create_infos.data();

    VkDevice device;
    if (vkCreateDevice(config_.physical_devices[0], &device_info, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "[MultiGPU] Failed to create device group" << std::endl;
        return false;
    }

    devices_.push_back(device);
    std::cout << "[MultiGPU] Created device group successfully" << std::endl;
    return true;
}

bool MultiGPUManager::create_logical_devices() {
    VkPhysicalDeviceFeatures device_features{};

    for (uint32_t i = 0; i < config_.physical_devices.size(); ++i) {
        uint32_t queue_family = 0;

        VkQueueFamilyProperties queue_props;
        vkGetPhysicalDeviceQueueFamilyProperties(config_.physical_devices[i], &queue_family_count, &queue_props);

        VkPhysicalDeviceQueueFamilyProperties queue_props{};
        vkGetPhysicalDeviceQueueFamilyProperties(config_.physical_devices[i], &queue_family_count, &queue_props);

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(config_.physical_devices[i], &queue_family_count, &queue_props);

        for (uint32_t j = 0; j < queue_family_count; ++j) {
            VkPhysicalDeviceQueueFamilyProperties props{};
            vkGetPhysicalDeviceQueueFamilyProperties(config_.physical_devices[i], &j, &props);

            if (props.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family = j;
                break;
            }
        }

        config_.queue_families.push_back(queue_family);

        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1;
        float priority = 1.0f;
        queue_info.pQueuePriorities = &priority;

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.pQueueCreateInfos = &queue_info;
        create_info.queueCreateInfoCount = 1;
        create_info.pEnabledFeatures = &device_features;

        VkDevice device;
        if (vkCreateDevice(config_.physical_devices[i], &create_info, nullptr, &device) != VK_SUCCESS) {
            std::cerr << "[MultiGPU] Failed to create logical device " << i << std::endl;
            return false;
        }

        VkQueue queue;
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        devices_.push_back(device);
        queues_.push_back(queue);

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(config_.physical_devices[i], &props);

        std::cout << "[MultiGPU] Created logical device " << i << ": " << props.deviceName << std::endl;
    }

    return true;
}

uint32_t MultiGPUManager::get_device_memory(uint32_t device_index) const {
    if (device_index >= config_.physical_devices.size()) {
        return 0;
    }

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(config_.physical_devices[device_index], &mem_props);

    uint32_t total_memory = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        total_memory += mem_props.memoryHeaps[i].size;
    }

    return total_memory;
}

TensorPartition::TensorPartition(uint32_t num_devices, uint32_t tensor_size)
    : num_devices_(num_devices), tensor_size_(tensor_size) {

    partition_size_ = (tensor_size + num_devices - 1) / num_devices;
    partition_sizes_.resize(num_devices_, partition_size_);

    for (uint32_t i = 0; i < num_devices; ++i) {
        if (i < num_devices - 1) {
            partition_sizes_[i] = partition_size_;
        } else {
            partition_sizes_[i] = tensor_size - (num_devices - 1) * partition_size_;
        }
    }
}

TensorPartition::~TensorPartition() {
}

void TensorPartition::partition_tensor(const float* input, std::vector<float*>& partitions) {
    partitions.resize(num_devices_);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_devices_; ++i) {
        partitions[i] = new float[partition_sizes_[i]];

        for (uint32_t j = 0; j < partition_sizes_[i]; ++j) {
            partitions[i][j] = input[offset + j];
        }

        offset += partition_sizes_[i];
    }
}

void TensorPartition::merge_partitions(const std::vector<const float*>& partitions,
                                     float* output) {
    uint32_t offset = 0;

    for (uint32_t i = 0; i < num_devices_; ++i) {
        for (uint32_t j = 0; j < partition_sizes_[i]; ++j) {
            output[offset + j] = partitions[i][j];
        }

        offset += partition_sizes_[i];
    }
}

uint32_t TensorPartition::get_partition_size(uint32_t device_index) const {
    if (device_index >= num_devices_) {
        return 0;
    }
    return partition_sizes_[device_index];
}

}
