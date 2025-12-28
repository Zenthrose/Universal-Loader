#include "memory.h"

namespace vulkan {

VulkanMemory::VulkanMemory(VkDevice device) : device_(device) {}

VulkanMemory::~VulkanMemory() {
}

VulkanBuffer VulkanMemory::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                                         VkMemoryPropertyFlags properties) {
    VulkanBuffer buffer{};
    buffer.size = size;
    buffer.mapped_ptr = nullptr;
    
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device_, &buffer_info, nullptr, &buffer.buffer) != VK_SUCCESS) {
        return buffer;
    }
    
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_, buffer.buffer, &mem_requirements);
    
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, properties);
    
    if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer.buffer, nullptr);
        return buffer;
    }
    
    vkBindBufferMemory(device_, buffer.buffer, buffer.memory, 0);
    
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vkMapMemory(device_, buffer.memory, 0, size, 0, &buffer.mapped_ptr);
    }
    
    return buffer;
}

void VulkanMemory::destroy_buffer(VulkanBuffer& buffer) {
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

void VulkanMemory::copy_buffer(VkCommandBuffer cmd, VulkanBuffer src, VulkanBuffer dst, VkDeviceSize size) {
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &copy_region);
}

uint32_t VulkanMemory::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
    return 0;
}

}
