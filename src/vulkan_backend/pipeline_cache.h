#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>

namespace vulkan {

class PipelineCache {
public:
    explicit PipelineCache(VkDevice device);
    ~PipelineCache();

    VkPipeline get_compute_pipeline(const std::string& name, 
                                    const VkShaderModuleCreateInfo& shader_info,
                                    const VkPipelineLayoutCreateInfo& layout_info);

    VkPipelineLayout get_pipeline_layout(const std::string& name,
                                          const VkPipelineLayoutCreateInfo& info);

private:
    VkDevice device_;
    VkPipelineCache cache_;
    std::unordered_map<std::string, VkPipeline> pipelines_;
    std::unordered_map<std::string, VkPipelineLayout> layouts_;
};

}
