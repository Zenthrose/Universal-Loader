#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include "timeline_semaphores.h"
#include "profiler.h"

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
                  const ComputeWork& work, const std::string& operation = "", uint32_t layer_id = 0);
    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet descriptor_set,
                  const ComputeWork& work, const std::string& operation = "", uint32_t layer_id = 0);
    void wait_for_completion();

    void set_timeline_semaphores(TimelineSemaphores* semaphores) {
        timeline_semaphores_ = semaphores;
    }
    void set_profiler(Profiler* profiler) {
        profiler_ = profiler;
    }

private:
    VkDevice device_;
    VkQueue queue_;
    uint32_t queue_family_;
    VkCommandPool command_pool_;
    VkFence fence_;
    TimelineSemaphores* timeline_semaphores_;
    Profiler* profiler_;
    uint64_t next_timeline_value_;
};

}
