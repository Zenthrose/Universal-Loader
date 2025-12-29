#include "tool_calling_engine.h"
#include <regex>
#include <sstream>
#include <algorithm>

namespace inference {

ToolCallingEngine::ToolCallingEngine(ToolRegistry* registry)
    : registry_(registry) {

    config_.enabled = true;
    config_.max_concurrent_calls = 1;
    config_.max_retries = 3;
    config_.timeout_ms = 30000;
    config_.priority_order = "manual,metadata,generic";
}

ToolCallingEngine::~ToolCallingEngine() {
}

void ToolCallingEngine::set_config(const ToolCallingConfig& config) {
    config_ = config;
}

std::vector<ToolCall> ToolCallingEngine::detect_tool_calls(const std::string& text) {
    std::vector<ToolCall> calls;

    std::regex tool_call_pattern(R"(\[(\w+)\s*\(([^)]*)\)\])");
    std::sregex_iterator it(text.begin(), text.end(), tool_call_pattern);
    std::sregex_iterator end;

    for (; it != end; ++it) {
        std::smatch match = *it;

        ToolCall call;
        call.tool_name = match[1].str();

        std::string args_str = match[2].str();

        ToolMetadata* tool = registry_->find_tool(call.tool_name);
        if (tool) {
            call.arguments = extract_tool_arguments(args_str, *tool);
            calls.push_back(call);
        }
    }

    return calls;
}

ToolResult ToolCallingEngine::execute_tool_call(const ToolCall& call) {
    if (!config_.enabled) {
        ToolResult result;
        result.tool_name = call.tool_name;
        result.success = false;
        result.error_message = "Tool calling is disabled";
        return result;
    }

    if (!registry_->is_tool_enabled(call.tool_name)) {
        ToolResult result;
        result.tool_name = call.tool_name;
        result.success = false;
        result.error_message = "Tool is not enabled: " + call.tool_name;
        return result;
    }

    return registry_->execute_tool(call);
}

std::string ToolCallingEngine::format_tool_results(const std::vector<ToolResult>& results) {
    std::stringstream ss;

    for (size_t i = 0; i < results.size(); ++i) {
        const ToolResult& result = results[i];

        if (result.success) {
            ss << "[" << result.tool_name << " succeeded] " << result.output;
        } else {
            ss << "[" << result.tool_name << " failed] " << result.error_message;
        }

        if (i < results.size() - 1) {
            ss << "\n";
        }
    }

    return ss.str();
}

bool ToolCallingEngine::should_call_tool(const std::string& text) {
    std::vector<ToolCall> calls = detect_tool_calls(text);
    return !calls.empty();
}

std::string ToolCallingEngine::extract_tool_name(const std::string& text) {
    std::regex tool_name_pattern(R"(\[(\w+)\s*\()");
    std::smatch match;

    if (std::regex_search(text, match, tool_name_pattern)) {
        return match[1].str();
    }

    return "";
}

std::unordered_map<std::string, std::string> ToolCallingEngine::extract_tool_arguments(
    const std::string& text,
    const ToolMetadata& tool) {

    std::unordered_map<std::string, std::string> args;

    for (const auto& param : tool.parameters) {
        std::regex param_pattern(param.name + R"(=([^\s,\)]*))");
        std::smatch match;

        if (std::regex_search(text, match, param_pattern)) {
            args[param.name] = match[1].str();
        } else if (param.required) {
            std::cerr << "[ToolCallingEngine] Missing required parameter: " << param.name << std::endl;
        }
    }

    return args;
}

}
