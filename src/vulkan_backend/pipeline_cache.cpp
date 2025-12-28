#include "pipeline_cache.h"

namespace vulkan {

PipelineCache::PipelineCache(VkDevice device) : device_(device) {
    VkPipelineCacheCreateInfo cache_info{};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(device_, &cache_info, nullptr, &cache_);
}

PipelineCache::~PipelineCache() {
    for (auto& pair : pipelines_) {
        vkDestroyPipeline(device_, pair.second, nullptr);
    }
    for (auto& pair : layouts_) {
        vkDestroyPipelineLayout(device_, pair.second, nullptr);
    }
    if (cache_) {
        vkDestroyPipelineCache(device_, cache_, nullptr);
    }
}

VkPipeline PipelineCache::get_compute_pipeline(const std::string& name, 
                                                const VkShaderModuleCreateInfo& shader_info,
                                                const VkPipelineLayoutCreateInfo& layout_info) {
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) {
        return it->second;
    }

    VkShaderModule shader_module;
    vkCreateShaderModule(device_, &shader_info, nullptr, &shader_module);

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";

    VkPipelineLayout layout = get_pipeline_layout(name + "_layout", layout_info);

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = layout;

    VkPipeline pipeline;
    vkCreateComputePipelines(device_, cache_, 1, &pipeline_info, nullptr, &pipeline);
    vkDestroyShaderModule(device_, shader_module, nullptr);

    pipelines_[name] = pipeline;
    return pipeline;
}

VkPipelineLayout PipelineCache::get_pipeline_layout(const std::string& name,
                                                     const VkPipelineLayoutCreateInfo& info) {
    auto it = layouts_.find(name);
    if (it != layouts_.end()) {
        return it->second;
    }

    VkPipelineLayout layout;
    vkCreatePipelineLayout(device_, &info, nullptr, &layout);
    layouts_[name] = layout;
    return layout;
}

}
