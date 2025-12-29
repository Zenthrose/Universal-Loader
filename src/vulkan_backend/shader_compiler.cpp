#include "shader_compiler.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vulkan {

ShaderCompiler::ShaderCompiler(VkDevice device) : device_(device) {}

ShaderCompiler::~ShaderCompiler() {
    clear_cache();
}

std::string ShaderCompiler::preprocess_glsl(const std::string& glsl_code) {
    std::string processed = glsl_code;

    size_t pos = 0;
    while ((pos = processed.find("#include", pos)) != std::string::npos) {
        size_t start = processed.find("\"", pos) + 1;
        size_t end = processed.find("\"", start);

        if (start != std::string::npos && end != std::string::npos) {
            std::string include_file = processed.substr(start, end - start);

            std::ifstream include_stream(include_file);
            if (include_stream.is_open()) {
                std::stringstream buffer;
                buffer << include_stream.rdbuf();
                std::string included_code = buffer.str();

                processed.replace(pos, end - pos + 1, included_code);
            } else {
                std::cerr << "[ShaderCompiler] Failed to include file: " << include_file << std::endl;
            }
        }

        pos++;
    }

    return processed;
}

ShaderCompiler::CompilationResult ShaderCompiler::compile_with_glslang(
    const std::string& glsl_code,
    const std::string& shader_name) {

    CompilationResult result;
    result.success = false;

#ifdef _WIN32
    char temp_path[MAX_PATH];
    GetTempPathA(MAX_PATH, temp_path);
    std::string temp_dir(temp_path);

    std::string glsl_file = temp_dir + shader_name + ".glsl";
    std::string spv_file = temp_dir + shader_name + ".spv";

    std::ofstream out_file(glsl_file);
    out_file << glsl_code;
    out_file.close();

    std::string command = "glslangValidator -V \"" + glsl_file + "\" -o \"" + spv_file + "\"";

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    char cmd[1024];
    strcpy_s(cmd, command.c_str());

    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW,
                        NULL, NULL, &si, &pi)) {
        result.error_message = "Failed to launch glslangValidator: " + shader_name;
        DeleteFileA(glsl_file.c_str());
        return result;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exit_code;
    GetExitCodeProcess(pi.hProcess, &exit_code);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    if (exit_code != 0) {
        result.error_message = "glslangValidator compilation failed for: " + shader_name;
        DeleteFileA(glsl_file.c_str());
        return result;
    }

    std::ifstream spv_stream(spv_file, std::ios::binary);
    if (!spv_stream.is_open()) {
        result.error_message = "Failed to open compiled SPIR-V file: " + spv_file;
        DeleteFileA(glsl_file.c_str());
        return result;
    }

    spv_stream.seekg(0, std::ios::end);
    size_t spv_size = spv_stream.tellg();
    spv_stream.seekg(0, std::ios::beg);

    result.spirv.resize(spv_size / sizeof(uint32_t));
    spv_stream.read(reinterpret_cast<char*>(result.spirv.data()), spv_size);
    spv_stream.close();

    DeleteFileA(glsl_file.c_str());
    DeleteFileA(spv_file.c_str());

    result.success = true;
    return result;
#else
    std::cerr << "[ShaderCompiler] Runtime compilation only supported on Windows currently" << std::endl;
    result.error_message = "Runtime compilation not supported on this platform";
    return result;
#endif
}

ShaderCompiler::CompilationResult ShaderCompiler::compile_glsl(
    const std::string& glsl_code,
    const std::string& shader_name) {

    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto cache_key = shader_name + "_" + std::to_string(std::hash<std::string>{}(glsl_code));
    auto it = compilation_cache_.find(cache_key);
    if (it != compilation_cache_.end()) {
        CompilationResult result;
        result.spirv = it->second;
        result.success = true;
        return result;
    }

    std::string processed = preprocess_glsl(glsl_code);
    auto result = compile_with_glslang(processed, shader_name);

    if (result.success) {
        compilation_cache_[cache_key] = result.spirv;
    }

    return result;
}

VkShaderModule ShaderCompiler::create_shader_module(const std::vector<uint32_t>& spirv,
                                                   const std::string& name) {
    if (spirv.empty()) {
        std::cerr << "[ShaderCompiler] Cannot create shader module from empty SPIR-V: " << name << std::endl;
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv.size() * sizeof(uint32_t);
    create_info.pCode = spirv.data();

    VkShaderModule shader_module;
    VkResult result = vkCreateShaderModule(device_, &create_info, nullptr, &shader_module);

    if (result != VK_SUCCESS) {
        std::cerr << "[ShaderCompiler] Failed to create shader module: " << name
                  << " (VkResult: " << result << ")" << std::endl;
        return VK_NULL_HANDLE;
    }

    return shader_module;
}

void ShaderCompiler::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    compilation_cache_.clear();
}

ShaderHotReloader::ShaderHotReloader(VkDevice device,
                                     ShaderCompiler* compiler)
    : device_(device), compiler_(compiler), watching_(false) {}

ShaderHotReloader::~ShaderHotReloader() {
    stop_watching();
}

void ShaderHotReloader::watch_directory(const std::string& shader_dir,
                                       ReloadCallback callback) {
    watch_directory_ = shader_dir;
    reload_callback_ = callback;
}

void ShaderHotReloader::start_watching() {
    if (watching_.load()) {
        return;
    }

    watching_.store(true);
    watch_thread_ = std::thread(&ShaderHotReloader::watch_thread_func, this);
}

void ShaderHotReloader::stop_watching() {
    if (!watching_.load()) {
        return;
    }

    watching_.store(false);
    if (watch_thread_.joinable()) {
        watch_thread_.join();
    }
}

void ShaderHotReloader::watch_thread_func() {
    while (watching_.load()) {
#ifdef _WIN32
        WIN32_FILE_ATTRIBUTE_DATA file_data;
        std::string pattern = watch_directory_ + "/*.glsl";

        WIN32_FIND_DATAA find_data;
        HANDLE find_handle = FindFirstFileA(pattern.c_str(), &find_data);

        if (find_handle != INVALID_HANDLE_VALUE) {
            do {
                std::string filename = watch_directory_ + "/" + find_data.cFileName;

                GetFileAttributesExA(filename.c_str(), GetFileExInfoStandard, &file_data);
                uint64_t mod_time = (static_cast<uint64_t>(file_data.ftLastWriteTime.dwHighDateTime) << 32) |
                                   file_data.ftLastWriteTime.dwLowDateTime;

                auto it = file_mod_times_.find(filename);
                if (it != file_mod_times_.end()) {
                    if (it->second != mod_time) {
                        it->second = mod_time;

                        std::ifstream file(filename);
                        std::stringstream buffer;
                        buffer << file.rdbuf();
                        std::string glsl_code = buffer.str();

                        auto result = compiler_->compile_glsl(glsl_code, find_data.cFileName);
                        if (result.success) {
                            VkShaderModule module = compiler_->create_shader_module(result.spirv,
                                                                                   find_data.cFileName);
                            if (module != VK_NULL_HANDLE && reload_callback_) {
                                reload_callback_(filename, module);
                            }
                        }
                    }
                } else {
                    file_mod_times_[filename] = mod_time;
                }
            } while (FindNextFileA(find_handle, &find_data) != 0);

            FindClose(find_handle);
        }
#endif

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}