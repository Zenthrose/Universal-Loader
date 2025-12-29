#include "pipeline_cache.h"
#include "shader_compiler.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace vulkan {

PipelineCache::PipelineCache(VkDevice device, ShaderCompiler* compiler, const std::string& cache_dir)
    : device_(device), compiler_(compiler), shader_directory_("src/shaders/"),
      cache_dir_(cache_dir), cache_path_(cache_dir + "/pipeline_cache.bin") {

    if (!load_from_disk()) {
        VkPipelineCacheCreateInfo cache_info{};
        cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        VkResult result = vkCreatePipelineCache(device_, &cache_info, nullptr, &cache_);
        if (result != VK_SUCCESS) {
            std::cerr << "[PipelineCache] Failed to create pipeline cache: " << result << std::endl;
        }
    }
}

PipelineCache::~PipelineCache() {
    for (auto& pair : pipelines_) {
        vkDestroyPipeline(device_, pair.second, nullptr);
    }
    for (auto& pair : layouts_) {
        vkDestroyPipelineLayout(device_, pair.second, nullptr);
    }
    for (auto& pair : shader_modules_) {
        vkDestroyShaderModule(device_, pair.second, nullptr);
    }
    if (cache_) {
        vkDestroyPipelineCache(device_, cache_, nullptr);
    }
}

VkShaderModule PipelineCache::load_shader(const std::string& filename) {
    std::string full_path = shader_directory_ + filename;

    auto it = shader_modules_.find(filename);
    if (it != shader_modules_.end()) {
        return it->second;
    }

    std::ifstream file(full_path);
    if (!file.is_open()) {
        std::cerr << "[PipelineCache] Failed to open shader file: " << full_path << std::endl;
        return VK_NULL_HANDLE;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string glsl_code = buffer.str();
    file.close();

    auto compile_result = compiler_->compile_glsl(glsl_code, filename);
    if (!compile_result.success) {
        std::cerr << "[PipelineCache] Shader compilation failed: " << filename << std::endl;
        std::cerr << "Error: " << compile_result.error_message << std::endl;
        return VK_NULL_HANDLE;
    }

    VkShaderModule module = compiler_->create_shader_module(compile_result.spirv, filename);
    if (module != VK_NULL_HANDLE) {
        shader_modules_[filename] = module;
    }

    return module;
}

VkPipeline PipelineCache::get_compute_pipeline(const std::string& shader_filename,
                                                const VkPipelineLayoutCreateInfo& layout_info) {
    auto it = pipelines_.find(shader_filename);
    if (it != pipelines_.end()) {
        return it->second;
    }

    VkShaderModule shader_module = load_shader(shader_filename);
    if (shader_module == VK_NULL_HANDLE) {
        std::cerr << "[PipelineCache] Failed to load shader: " << shader_filename << std::endl;
        return VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";

    VkPipelineLayout layout = get_pipeline_layout(shader_filename + "_layout", layout_info);

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = layout;

    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device_, cache_, 1, &pipeline_info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        std::cerr << "[PipelineCache] Failed to create pipeline: " << shader_filename
                  << " (VkResult: " << result << ")" << std::endl;
        return VK_NULL_HANDLE;
    }

    pipelines_[shader_filename] = pipeline;
    return pipeline;
}

VkPipelineLayout PipelineCache::get_pipeline_layout(const std::string& name,
                                                     const VkPipelineLayoutCreateInfo& info) {
    auto it = layouts_.find(name);
    if (it != layouts_.end()) {
        return it->second;
    }

    VkPipelineLayout layout;
    VkResult result = vkCreatePipelineLayout(device_, &info, nullptr, &layout);
    if (result != VK_SUCCESS) {
        std::cerr << "[PipelineCache] Failed to create pipeline layout: " << name
                  << " (VkResult: " << result << ")" << std::endl;
        return VK_NULL_HANDLE;
    }

    layouts_[name] = layout;
    return layout;
}

void PipelineCache::clear_cache() {
    for (auto& pair : pipelines_) {
        vkDestroyPipeline(device_, pair.second, nullptr);
    }
    for (auto& pair : layouts_) {
        vkDestroyPipelineLayout(device_, pair.second, nullptr);
    }
    for (auto& pair : shader_modules_) {
        vkDestroyShaderModule(device_, pair.second, nullptr);
    }

    pipelines_.clear();
    layouts_.clear();
    shader_modules_.clear();
}

void PipelineCache::set_shader_directory(const std::string& dir) {
    shader_directory_ = dir;
    clear_cache();
}

bool PipelineCache::save_to_disk() {
    size_t cache_size = 0;
    VkResult result = vkGetPipelineCacheData(device_, cache_, &cache_size, nullptr);

    if (result != VK_SUCCESS || cache_size == 0) {
        return false;
    }

    std::vector<uint8_t> cache_data(cache_size);
    result = vkGetPipelineCacheData(device_, cache_, &cache_size, cache_data.data());

    if (result != VK_SUCCESS) {
        std::cerr << "[PipelineCache] Failed to retrieve cache data" << std::endl;
        return false;
    }

    std::ofstream out_file(cache_path_, std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "[PipelineCache] Failed to open cache file for writing: " << cache_path_ << std::endl;
        return false;
    }

    out_file.write(reinterpret_cast<const char*>(cache_data.data()), cache_size);
    out_file.close();

    std::cout << "[PipelineCache] Saved pipeline cache to disk: " << cache_path_ << std::endl;
    return true;
}

bool PipelineCache::load_from_disk() {
    std::ifstream in_file(cache_path_, std::ios::binary | std::ios::ate);

    if (!in_file.is_open()) {
        std::cout << "[PipelineCache] No cached pipeline found, creating new cache" << std::endl;
        return false;
    }

    size_t cache_size = in_file.tellg();
    in_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> cache_data(cache_size);
    in_file.read(reinterpret_cast<char*>(cache_data.data()), cache_size);
    in_file.close();

    VkPipelineCacheCreateInfo cache_info{};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cache_info.initialDataSize = cache_size;
    cache_info.pInitialData = cache_data.data();

    VkResult result = vkCreatePipelineCache(device_, &cache_info, nullptr, &cache_);

    if (result != VK_SUCCESS) {
        std::cerr << "[PipelineCache] Failed to load cache from disk, creating new cache" << std::endl;
        VkPipelineCacheCreateInfo new_cache_info{};
        new_cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        vkCreatePipelineCache(device_, &new_cache_info, nullptr, &cache_);
        return false;
    }

    std::cout << "[PipelineCache] Loaded pipeline cache from disk: " << cache_path_ << std::endl;
    return true;
}

std::string PipelineCache::get_shader_hash(const std::string& filename) {
    std::string full_path = shader_directory_ + filename;

    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    std::hash<std::string> hasher;
    return std::to_string(hasher(buffer.str()));
}

}
