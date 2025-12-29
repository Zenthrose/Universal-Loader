#pragma once
#include "tool_registry.h"
#include <vector>
#include <string>

namespace inference {

struct ToolCallingConfig {
    bool enabled;
    int max_concurrent_calls;
    int max_retries;
    int timeout_ms;
    std::string priority_order;
};

class ToolCallingEngine {
public:
    explicit ToolCallingEngine(ToolRegistry* registry);
    ~ToolCallingEngine();

    void set_config(const ToolCallingConfig& config);

    std::vector<ToolCall> detect_tool_calls(const std::string& text);
    ToolResult execute_tool_call(const ToolCall& call);
    std::string format_tool_results(const std::vector<ToolResult>& results);

    bool should_call_tool(const std::string& text);

private:
    ToolRegistry* registry_;
    ToolCallingConfig config_;

    std::string extract_tool_name(const std::string& text);
    std::unordered_map<std::string, std::string> extract_tool_arguments(const std::string& text, const ToolMetadata& tool);
};

}
