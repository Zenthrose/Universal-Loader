#include "device_address_memory.h"
#include <iostream>

namespace vulkan {

DeviceAddressMemory::DeviceAddressMemory(VkDevice device, VkPhysicalDevice physical_device)
    : device_(device), physical_device_(physical_device), supports_device_address_(false),
      vkGetBufferDeviceAddressKHR(nullptr) {

    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkGetPhysicalDeviceProperties2(physical_device_, &props);

    for (uint32_t i = 0; i < props.properties2Count; ++i) {
        VkPhysicalDeviceProperties2* pProps = reinterpret_cast<VkPhysicalDeviceProperties2*>(reinterpret_cast<uint8_t*>(&props) + props.pNext);
        
        if (pProps->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT) {
            VkPhysicalDeviceBufferDeviceAddressFeaturesKHR* addrFeatures = 
                reinterpret_cast<VkPhysicalDeviceBufferDeviceAddressFeaturesKHR*>(pProps);
            
            if (addrFeatures->bufferDeviceAddress) {
                supports_device_address_ = true;
                std::cout << "[DeviceAddressMemory] VK_KHR_buffer_device_address supported" << std::endl;
            }
            break;
        }
    }

    if (supports_device_address_) {
        vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
            vkGetDeviceProcAddr(device_, "vkGetBufferDeviceAddressKHR"));
    }
}

DeviceAddressMemory::~DeviceAddressMemory() {
    for (auto& buf : buffers_) {
        destroy_buffer(buf);
    }
}

DeviceAddressBuffer DeviceAddressMemory::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                                     VkMemoryPropertyFlags properties,
                                                     bool enable_device_address) {
    DeviceAddressBuffer buf{};
    buf.size = size;
    buf.device_address = 0;
    buf.is_device_address = false;
    buf.mapped_ptr = nullptr;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    
    if (enable_device_address && supports_device_address_) {
        buffer_info.flags |= VK_BUFFER_CREATE_DEVICE_ADDRESS_BIT_EXT;
    }

    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buf.buffer) != VK_SUCCESS) {
        std::cerr << "[DeviceAddressMemory] Failed to create buffer" << std::endl;
        return buf;
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_, buf.buffer, &mem_reqs);

    uint32_t memory_type_index = 0;
    bool found = false;
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_reqs.memoryTypeBits & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            memory_type_index = i;
            found = true;
            break;
        }
    }

    if (!found) {
        vkDestroyBuffer(device_, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        return buf;
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = memory_type_index;

    VkMemoryAllocateFlagsInfoKHR alloc_flags_info{};
    alloc_flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
    alloc_flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
    alloc_flags_info.pNext = nullptr;

    if (enable_device_address && supports_device_address_) {
        alloc_info.pNext = &alloc_flags_info;
    }

    if (vkAllocateMemory(device_, &alloc_info, nullptr, &buf.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        return buf;
    }

    vkBindBufferMemory(device_, buf.buffer, buf.memory, 0);

    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vkMapMemory(device_, buf.memory, 0, size, 0, &buf.mapped_ptr);
    }

    if (supports_device_address_ && vkGetBufferDeviceAddressKHR) {
        VkBufferDeviceAddressInfoKHR addr_info{};
        addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
        addr_info.buffer = buf.buffer;
        buf.device_address = vkGetBufferDeviceAddressKHR(&addr_info);
        buf.is_device_address = true;
    }

    buffers_.push_back(buf);
    return buf;
}

void DeviceAddressMemory::destroy_buffer(DeviceAddressBuffer& buffer) {
    if (buffer.mapped_ptr) {
        vkUnmapMemory(device_, buffer.memory);
    }

    if (buffer.memory) {
        vkFreeMemory(device_, buffer.memory, nullptr);
    }

    if (buffer.buffer) {
        vkDestroyBuffer(device_, buffer.buffer, nullptr);
    }

    buffer = {};
}

VkDeviceAddress DeviceAddressMemory::get_buffer_address(VkBuffer buffer) const {
    for (const auto& buf : buffers_) {
        if (buf.buffer == buffer) {
            return buf.device_address;
        }
    }
    return 0;
}

bool DeviceAddressMemory::get_buffer_device_address(VkBuffer buffer, VkDeviceAddress* out_address) {
    for (const auto& buf : buffers_) {
        if (buf.buffer == buffer && buf.is_device_address) {
            *out_address = buf.device_address;
            return true;
        }
    }
    *out_address = 0;
    return false;
}

}
