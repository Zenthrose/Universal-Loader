#pragma once
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <functional>

namespace vulkan {

class ShaderCompiler {
public:
    struct CompilationResult {
        std::vector<uint32_t> spirv;
        std::string error_message;
        bool success;
    };

    explicit ShaderCompiler(VkDevice device);
    ~ShaderCompiler();

    CompilationResult compile_glsl(const std::string& glsl_code,
                                 const std::string& shader_name = "shader");

    VkShaderModule create_shader_module(const std::vector<uint32_t>& spirv,
                                      const std::string& name);

    void clear_cache();

private:
    VkDevice device_;
    std::mutex cache_mutex_;
    std::unordered_map<std::string, std::vector<uint32_t>> compilation_cache_;

    std::string preprocess_glsl(const std::string& glsl_code);
    CompilationResult compile_with_glslang(const std::string& glsl_code,
                                         const std::string& shader_name);
};

class ShaderHotReloader {
public:
    using ReloadCallback = std::function<void(const std::string&, VkShaderModule)>;

    explicit ShaderHotReloader(VkDevice device,
                               ShaderCompiler* compiler);
    ~ShaderHotReloader();

    void watch_directory(const std::string& shader_dir,
                       ReloadCallback callback);
    void start_watching();
    void stop_watching();

private:
    VkDevice device_;
    ShaderCompiler* compiler_;
    std::string watch_directory_;
    ReloadCallback reload_callback_;
    std::atomic<bool> watching_;
    std::thread watch_thread_;

    void watch_thread_func();
    std::unordered_map<std::string, uint64_t> file_mod_times_;
};

}