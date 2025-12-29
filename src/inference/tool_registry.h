#pragma once
#include "tool_metadata_parser.h"
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>

namespace inference {

class ToolRegistry {
public:
    ToolRegistry();
    ~ToolRegistry();

    void register_tool(const ToolMetadata& metadata, ToolFunction function);
    void unregister_tool(const std::string& name);

    ToolResult execute_tool(const ToolCall& call);

    std::vector<ToolMetadata> get_available_tools() const;
    ToolMetadata* find_tool(const std::string& name);

    void load_tools_from_parser(const ToolMetadataParser& parser);

    void enable_tool(const std::string& name, bool enabled);
    bool is_tool_enabled(const std::string& name) const;

private:
    void register_builtin_tools();
    void register_builtin_web_search();
    void register_builtin_calculator();
    void register_builtin_file_read();
    void register_builtin_file_write();
    void register_builtin_bash_execute();
    void register_builtin_datetime();

    std::unordered_map<std::string, ToolMetadata> tools_metadata_;
    std::unordered_map<std::string, ToolFunction> tool_functions_;
    std::unordered_map<std::string, bool> tool_enabled_;

    mutable std::mutex mutex_;
};

}
