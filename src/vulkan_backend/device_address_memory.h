#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace vulkan {

struct DeviceAddressBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    VkDeviceAddress device_address;
    void* mapped_ptr;
    bool is_device_address;
};

class DeviceAddressMemory {
public:
    DeviceAddressMemory(VkDevice device, VkPhysicalDevice physical_device);
    ~DeviceAddressMemory();

    bool supports_device_address() const { return supports_device_address_; }

    DeviceAddressBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags properties,
                                       bool enable_device_address = true);

    void destroy_buffer(DeviceAddressBuffer& buffer);

    VkDevice get_device() const { return device_; }

    VkDeviceAddress get_buffer_address(VkBuffer buffer) const;

    bool get_buffer_device_address(VkBuffer buffer, VkDeviceAddress* out_address);

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    bool supports_device_address_;

    std::vector<DeviceAddressBuffer> buffers_;

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
};

}
