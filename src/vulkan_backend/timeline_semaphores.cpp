#include "timeline_semaphores.h"
#include <iostream>
#include <cstring>

namespace vulkan {

TimelineSemaphores::TimelineSemaphores(VkDevice device, uint32_t num_semaphores)
    : device_(device), num_semaphores_(num_semaphores) {
    semaphores_.resize(num_semaphores_, VK_NULL_HANDLE);
    memset(&timeline_properties_, 0, sizeof(timeline_properties_));
}

TimelineSemaphores::~TimelineSemaphores() {
    cleanup();
}

bool TimelineSemaphores::initialize() {
    VkSemaphoreTypeCreateInfoKHR type_info{};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;

    VkSemaphoreCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create_info.pNext = &type_info;

    for (uint32_t i = 0; i < num_semaphores_; ++i) {
        VkResult result = vkCreateSemaphore(device_, &create_info, nullptr, &semaphores_[i]);
        if (result != VK_SUCCESS) {
            std::cerr << "Failed to create timeline semaphore " << i << ": " << result << std::endl;
            return false;
        }
    }

    std::cout << "Created " << num_semaphores_ << " timeline semaphores" << std::endl;
    return true;
}

void TimelineSemaphores::cleanup() {
    for (auto semaphore : semaphores_) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
    }
    semaphores_.clear();
}

bool TimelineSemaphores::wait(uint32_t index, uint64_t value, uint64_t timeout) {
    if (index >= num_semaphores_) {
        std::cerr << "Invalid semaphore index: " << index << std::endl;
        return false;
    }

    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &semaphores_[index];
    wait_info.pValues = &value;

    VkResult result = vkWaitSemaphores(device_, &wait_info, timeout);
    return result == VK_SUCCESS;
}

bool TimelineSemaphores::signal(uint32_t index, uint64_t value) {
    if (index >= num_semaphores_) {
        std::cerr << "Invalid semaphore index: " << index << std::endl;
        return false;
    }

    VkSemaphoreSignalInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = semaphores_[index];
    signal_info.value = value;

    VkResult result = vkSignalSemaphore(device_, &signal_info);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to signal semaphore: " << result << std::endl;
        return false;
    }

    return true;
}

uint64_t TimelineSemaphores::get_value(uint32_t index) const {
    if (index >= num_semaphores_) {
        std::cerr << "Invalid semaphore index: " << index << std::endl;
        return 0;
    }

    uint64_t value = 0;
    VkResult result = vkGetSemaphoreCounterValue(device_, semaphores_[index], &value);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to get semaphore value: " << result << std::endl;
        return 0;
    }

    return value;
}

}