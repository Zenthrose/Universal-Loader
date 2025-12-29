#include "tool_metadata_parser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace inference {

ToolMetadataParser::ToolMetadataParser() {
}

ToolMetadataParser::~ToolMetadataParser() {
}

bool ToolMetadataParser::parse_from_gguf(const std::string& gguf_file) {
    std::cout << "[ToolMetadataParser] Parsing tools from GGUF: " << gguf_file << std::endl;

    std::ifstream file(gguf_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ToolMetadataParser] Failed to open GGUF file" << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    uint8_t magic[4];
    file.read(reinterpret_cast<char*>(magic), 4);

    if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
        std::cerr << "[ToolMetadataParser] Invalid GGUF magic" << std::endl;
        return false;
    }

    file.close();

    ToolMetadata builtin_tool = create_builtin_tool_metadata("web_search");
    builtin_tool.type = ToolType::BUILTIN;
    builtin_tool.source = "gguf_builtin";
    builtin_tool.enabled = true;
    tools_.push_back(builtin_tool);
    tools_by_name_[builtin_tool.name] = builtin_tool;

    std::cout << "[ToolMetadataParser] Loaded " << tools_.size() << " tools from GGUF" << std::endl;
    return true;
}

bool ToolMetadataParser::parse_from_json(const std::string& json_file) {
    std::cout << "[ToolMetadataParser] Parsing tools from JSON: " << json_file << std::endl;

    std::ifstream file(json_file);
    if (!file.is_open()) {
        std::cerr << "[ToolMetadataParser] Failed to open JSON file" << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("\"name\"") != std::string::npos) {
            std::string tool_name = extract_string_value(line, "name");

            ToolMetadata metadata;
            metadata.name = tool_name;
            metadata.type = ToolType::PLUGIN;
            metadata.source = "json_config";
            metadata.enabled = true;

            tools_.push_back(metadata);
            tools_by_name_[tool_name] = metadata;
        }
    }

    file.close();

    std::cout << "[ToolMetadataParser] Loaded " << tools_.size() << " tools from JSON" << std::endl;
    return true;
}

ToolMetadata* ToolMetadataParser::find_tool(const std::string& name) {
    auto it = tools_by_name_.find(name);
    if (it != tools_by_name_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool ToolMetadataParser::validate_tool_call(const ToolCall& call) const {
    auto it = tools_by_name_.find(call.tool_name);
    if (it == tools_by_name_.end()) {
        std::cerr << "[ToolMetadataParser] Unknown tool: " << call.tool_name << std::endl;
        return false;
    }

    const ToolMetadata& tool = it->second;

    for (const auto& param : tool.parameters) {
        if (param.required) {
            if (call.arguments.find(param.name) == call.arguments.end()) {
                std::cerr << "[ToolMetadataParser] Missing required parameter: " << param.name << std::endl;
                return false;
            }
        }
    }

    return true;
}

std::string ToolMetadataParser::extract_string_value(const std::string& line, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = line.find(search_key);

    if (pos == std::string::npos) {
        return "";
    }

    size_t colon_pos = line.find(':', pos);
    if (colon_pos == std::string::npos) {
        return "";
    }

    size_t quote_start = line.find('"', colon_pos);
    if (quote_start == std::string::npos) {
        return "";
    }

    size_t quote_end = line.find('"', quote_start + 1);
    if (quote_end == std::string::npos) {
        return "";
    }

    return line.substr(quote_start + 1, quote_end - quote_start - 1);
}

ToolMetadata ToolMetadataParser::create_builtin_tool_metadata(const std::string& name) {
    ToolMetadata metadata;
    metadata.name = name;
    metadata.description = "Builtin tool: " + name;
    metadata.type = ToolType::BUILTIN;
    metadata.enabled = true;
    metadata.source = "builtin";

    if (name == "web_search") {
        metadata.description = "Search the web for information";

        ToolParameter query_param;
        query_param.name = "query";
        query_param.type = "string";
        query_param.description = "Search query";
        query_param.required = true;
        metadata.parameters.push_back(query_param);
    } else if (name == "calculator") {
        metadata.description = "Perform mathematical calculations";

        ToolParameter expr_param;
        expr_param.name = "expression";
        expr_param.type = "string";
        expr_param.description = "Mathematical expression";
        expr_param.required = true;
        metadata.parameters.push_back(expr_param);
    } else if (name == "file_read") {
        metadata.description = "Read contents of a file";

        ToolParameter path_param;
        path_param.name = "path";
        path_param.type = "string";
        path_param.description = "File path";
        path_param.required = true;
        metadata.parameters.push_back(path_param);
    } else if (name == "file_write") {
        metadata.description = "Write content to a file";

        ToolParameter path_param;
        path_param.name = "path";
        path_param.type = "string";
        path_param.description = "File path";
        path_param.required = true;
        metadata.parameters.push_back(path_param);

        ToolParameter content_param;
        content_param.name = "content";
        content_param.type = "string";
        content_param.description = "Content to write";
        content_param.required = true;
        metadata.parameters.push_back(content_param);
    } else if (name == "bash_execute") {
        metadata.description = "Execute a bash command";

        ToolParameter cmd_param;
        cmd_param.name = "command";
        cmd_param.type = "string";
        cmd_param.description = "Command to execute";
        cmd_param.required = true;
        metadata.parameters.push_back(cmd_param);
    } else if (name == "datetime") {
        metadata.description = "Get current date and time";
    }

    return metadata;
}

}
