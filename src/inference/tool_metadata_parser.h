#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace inference {

enum class ToolType {
    BUILTIN,
    PLUGIN,
    PYTHON
};

struct ToolParameter {
    std::string name;
    std::string type;
    std::string description;
    bool required;
    std::string default_value;
};

struct ToolMetadata {
    std::string name;
    std::string description;
    ToolType type;
    std::vector<ToolParameter> parameters;
    std::string function_schema;
    std::string source;
    bool enabled;
};

struct ToolCall {
    std::string tool_name;
    std::unordered_map<std::string, std::string> arguments;
    uint32_t call_id;
};

struct ToolResult {
    std::string tool_name;
    std::string output;
    bool success;
    std::string error_message;
    uint32_t execution_time_ms;
};

using ToolFunction = std::function<ToolResult(const std::unordered_map<std::string, std::string>&)>;

class ToolMetadataParser {
public:
    ToolMetadataParser();
    ~ToolMetadataParser();

    bool parse_from_gguf(const std::string& gguf_file);
    bool parse_from_json(const std::string& json_file);

    const std::vector<ToolMetadata>& get_tools() const { return tools_; }
    ToolMetadata* find_tool(const std::string& name);

    bool validate_tool_call(const ToolCall& call) const;

private:
    bool parse_tool_from_metadata(const std::string& key, const std::string& value);
    ToolMetadata create_builtin_tool_metadata(const std::string& name);
    std::string extract_string_value(const std::string& line, const std::string& key);

    std::vector<ToolMetadata> tools_;
    std::unordered_map<std::string, ToolMetadata> tools_by_name_;
};

}
