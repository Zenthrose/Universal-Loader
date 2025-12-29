#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include "timeline_semaphores.h"

namespace vulkan {

struct ComputeWork {
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
};

class ComputeDispatcher {
public:
    explicit ComputeDispatcher(VkDevice device, VkQueue queue, uint32_t queue_family,
                              TimelineSemaphores* timeline_semaphores = nullptr);
    ~ComputeDispatcher();

    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  const ComputeWork& work);
    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet descriptor_set,
                  const ComputeWork& work);
    void wait_for_completion();

    void set_timeline_semaphores(TimelineSemaphores* semaphores) {
        timeline_semaphores_ = semaphores;
    }

private:
    VkDevice device_;
    VkQueue queue_;
    uint32_t queue_family_;
    VkCommandPool command_pool_;
    VkFence fence_;
    TimelineSemaphores* timeline_semaphores_;
    uint64_t next_timeline_value_;
};

}
