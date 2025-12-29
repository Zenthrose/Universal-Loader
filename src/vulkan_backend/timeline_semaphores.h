#pragma once
#include <vulkan/vulkan.h>
#include <vector>

namespace vulkan {

class TimelineSemaphores {
public:
    TimelineSemaphores(VkDevice device, uint32_t num_semaphores);
    ~TimelineSemaphores();

    bool initialize();
    void cleanup();

    VkSemaphore get_semaphore(uint32_t index) const {
        return semaphores_[index];
    }

    bool wait(uint32_t index, uint64_t value, uint64_t timeout = UINT64_MAX);
    bool signal(uint32_t index, uint64_t value);

    uint64_t get_value(uint32_t index) const;

private:
    VkDevice device_;
    uint32_t num_semaphores_;
    std::vector<VkSemaphore> semaphores_;
    VkPhysicalDeviceTimelineSemaphoreProperties timeline_properties_;
};

}