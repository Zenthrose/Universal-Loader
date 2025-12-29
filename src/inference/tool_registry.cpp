#include "tool_registry.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace inference {

ToolRegistry::ToolRegistry() {
    register_builtin_tools();
}

ToolRegistry::~ToolRegistry() {
}

void ToolRegistry::register_tool(const ToolMetadata& metadata, ToolFunction function) {
    std::lock_guard<std::mutex> lock(mutex_);

    tools_metadata_[metadata.name] = metadata;
    tool_functions_[metadata.name] = function;
    tool_enabled_[metadata.name] = metadata.enabled;

    std::cout << "[ToolRegistry] Registered tool: " << metadata.name << std::endl;
}

void ToolRegistry::unregister_tool(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    tools_metadata_.erase(name);
    tool_functions_.erase(name);
    tool_enabled_.erase(name);

    std::cout << "[ToolRegistry] Unregistered tool: " << name << std::endl;
}

ToolResult ToolRegistry::execute_tool(const ToolCall& call) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto func_it = tool_functions_.find(call.tool_name);
    if (func_it == tool_functions_.end()) {
        ToolResult result;
        result.success = false;
        result.error_message = "Tool not found: " + call.tool_name;
        return result;
    }

    auto enabled_it = tool_enabled_.find(call.tool_name);
    if (enabled_it == tool_enabled_.end() || !enabled_it->second) {
        ToolResult result;
        result.success = false;
        result.error_message = "Tool is disabled: " + call.tool_name;
        return result;
    }

    auto start = std::chrono::steady_clock::now();
    ToolResult result = func_it->second(call.arguments);
    auto end = std::chrono::steady_clock::now();

    result.tool_name = call.tool_name;
    result.execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return result;
}

std::vector<ToolMetadata> ToolRegistry::get_available_tools() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<ToolMetadata> tools;
    for (const auto& [name, metadata] : tools_metadata_) {
        auto enabled_it = tool_enabled_.find(name);
        if (enabled_it != tool_enabled_.end() && enabled_it->second) {
            tools.push_back(metadata);
        }
    }

    return tools;
}

ToolMetadata* ToolRegistry::find_tool(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tools_metadata_.find(name);
    if (it != tools_metadata_.end()) {
        return &it->second;
    }
    return nullptr;
}

void ToolRegistry::load_tools_from_parser(const ToolMetadataParser& parser) {
    const std::vector<ToolMetadata>& tools = parser.get_tools();

    for (const auto& metadata : tools) {
        if (metadata.source == "gguf") {
            std::cout << "[ToolRegistry] Loading tool from GGUF: " << metadata.name << std::endl;
            tool_enabled_[metadata.name] = metadata.enabled;
            tools_metadata_[metadata.name] = metadata;
        }
    }
}

void ToolRegistry::enable_tool(const std::string& name, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    tool_enabled_[name] = enabled;
}

bool ToolRegistry::is_tool_enabled(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tool_enabled_.find(name);
    if (it != tool_enabled_.end()) {
        return it->second;
    }
    return false;
}

void ToolRegistry::register_builtin_tools() {
    register_builtin_web_search();
    register_builtin_calculator();
    register_builtin_file_read();
    register_builtin_file_write();
    register_builtin_bash_execute();
    register_builtin_datetime();
}

void ToolRegistry::register_builtin_web_search() {
    ToolMetadata metadata;
    metadata.name = "web_search";
    metadata.description = "Search the web for information";
    metadata.type = ToolType::BUILTIN;
    metadata.enabled = true;
    metadata.source = "builtin";

    ToolParameter query_param;
    query_param.name = "query";
    query_param.type = "string";
    query_param.description = "Search query";
    query_param.required = true;
    metadata.parameters.push_back(query_param);

    ToolFunction func = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        ToolResult result;
        result.tool_name = "web_search";

        auto query_it = args.find("query");
        if (query_it == args.end()) {
            result.success = false;
            result.error_message = "Missing required parameter: query";
            return result;
        }

        result.output = "[Web Search Results for: " + query_it->second + "]\n"
                      "(Simulated results - web search not implemented)";
        result.success = true;

        return result;
    };

    register_tool(metadata, func);
}

void ToolRegistry::register_builtin_calculator() {
    ToolMetadata metadata;
    metadata.name = "calculator";
    metadata.description = "Perform mathematical calculations";
    metadata.type = ToolType::BUILTIN;
    metadata.enabled = true;
    metadata.source = "builtin";

    ToolParameter expression_param;
    expression_param.name = "expression";
    expression_param.type = "string";
    expression_param.description = "Mathematical expression to evaluate";
    expression_param.required = true;
    metadata.parameters.push_back(expression_param);

    ToolFunction func = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        ToolResult result;
        result.tool_name = "calculator";

        auto expr_it = args.find("expression");
        if (expr_it == args.end()) {
            result.success = false;
            result.error_message = "Missing required parameter: expression";
            return result;
        }

        std::string expr = expr_it->second;
        std::string output = "Calculator result for: " + expr + "\n"
                          "(Simulated - actual evaluation not implemented)";

        result.output = output;
        result.success = true;

        return result;
    };

    register_tool(metadata, func);
}

void ToolRegistry::register_builtin_file_read() {
    ToolMetadata metadata;
    metadata.name = "file_read";
    metadata.description = "Read contents of a file";
    metadata.type = ToolType::BUILTIN;
    metadata.enabled = true;
    metadata.source = "builtin";

    ToolParameter path_param;
    path_param.name = "path";
    path_param.type = "string";
    path_param.description = "File path to read";
    path_param.required = true;
    metadata.parameters.push_back(path_param);

    ToolFunction func = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        ToolResult result;
        result.tool_name = "file_read";

        auto path_it = args.find("path");
        if (path_it == args.end()) {
            result.success = false;
            result.error_message = "Missing required parameter: path";
            return result;
        }

        std::ifstream file(path_it->second);
        if (!file.is_open()) {
            result.success = false;
            result.error_message = "Failed to open file: " + path_it->second;
            return result;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        result.output = buffer.str();
        result.success = true;

        return result;
    };

    register_tool(metadata, func);
}

void ToolRegistry::register_builtin_file_write() {
    ToolMetadata metadata;
    metadata.name = "file_write";
    metadata.description = "Write content to a file";
    metadata.type = ToolType::BUILTIN;
    metadata.enabled = true;
    metadata.source = "builtin";

    ToolParameter path_param;
    path_param.name = "path";
    path_param.type = "string";
    path_param.description = "File path to write";
    path_param.required = true;
    metadata.parameters.push_back(path_param);

    ToolParameter content_param;
    content_param.name = "content";
    content_param.type = "string";
    content_param.description = "Content to write to file";
    content_param.required = true;
    metadata.parameters.push_back(content_param);

    ToolFunction func = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        ToolResult result;
        result.tool_name = "file_write";

        auto path_it = args.find("path");
        auto content_it = args.find("content");

        if (path_it == args.end() || content_it == args.end()) {
            result.success = false;
            result.error_message = "Missing required parameters: path and/or content";
            return result;
        }

        std::ofstream file(path_it->second);
        if (!file.is_open()) {
            result.success = false;
            result.error_message = "Failed to open file for writing: " + path_it->second;
            return result;
        }

        file << content_it->second;
        file.close();

        result.output = "Successfully wrote " + std::to_string(content_it->second.size()) + " bytes to " + path_it->second;
        result.success = true;

        return result;
    };

    register_tool(metadata, func);
}

void ToolRegistry::register_builtin_bash_execute() {
    ToolMetadata metadata;
    metadata.name = "bash_execute";
    metadata.description = "Execute a bash command";
    metadata.type = ToolType::BUILTIN;
    metadata.enabled = false;
    metadata.source = "builtin";

    ToolParameter command_param;
    command_param.name = "command";
    command_param.type = "string";
    command_param.description = "Command to execute";
    command_param.required = true;
    metadata.parameters.push_back(command_param);

    ToolFunction func = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        ToolResult result;
        result.tool_name = "bash_execute";

        auto cmd_it = args.find("command");
        if (cmd_it == args.end()) {
            result.success = false;
            result.error_message = "Missing required parameter: command";
            return result;
        }

        result.output = "[Bash Command: " + cmd_it->second + "]\n"
                      "(Simulated - actual execution disabled for safety)";
        result.success = true;

        return result;
    };

    register_tool(metadata, func);
}

void ToolRegistry::register_builtin_datetime() {
    ToolMetadata metadata;
    metadata.name = "datetime";
    metadata.description = "Get current date and time";
    metadata.type = ToolType::BUILTIN;
    metadata.enabled = true;
    metadata.source = "builtin";

    ToolFunction func = [](const std::unordered_map<std::string, std::string>& args) -> ToolResult {
        ToolResult result;
        result.tool_name = "datetime";

        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);

        std::string time_str = std::ctime(&time_t_now);
        time_str.pop_back();

        result.output = "Current date and time: " + time_str;
        result.success = true;

        return result;
    };

    register_tool(metadata, func);
}

}
