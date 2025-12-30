#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>

namespace vulkan {

class ShaderCompiler;

class PipelineCache {
public:
    explicit PipelineCache(VkDevice device, ShaderCompiler* compiler, const std::string& cache_dir = ".");
    ~PipelineCache();

    VkPipeline get_compute_pipeline(const std::string& shader_filename,
                                    const VkPipelineLayoutCreateInfo& layout_info);

    VkPipelineLayout get_pipeline_layout(const std::string& name,
                                          const VkPipelineLayoutCreateInfo& info);

    void clear_cache();
    void set_shader_directory(const std::string& dir);

    bool save_to_disk(const std::string& path);
    bool load_from_disk(const std::string& path);
    std::string get_cache_path() const { return cache_path_; }

private:
    VkDevice device_;
    VkPipelineCache cache_;
    ShaderCompiler* compiler_;
    std::string shader_directory_;
    std::string cache_dir_;
    std::string cache_path_;
    std::unordered_map<std::string, VkPipeline> pipelines_;
    std::unordered_map<std::string, VkPipelineLayout> layouts_;
    std::unordered_map<std::string, VkShaderModule> shader_modules_;

    VkShaderModule load_shader(const std::string& filename);
    std::string get_shader_hash(const std::string& filename);
};

}
