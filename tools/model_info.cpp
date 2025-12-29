#include "../src/core/gguf_parser.h"
#include <iostream>
#include <iomanip>

void print_banner() {
    std::cout << "============================================" << std::endl;
    std::cout << "  VulkanGGUF Model Info Tool v1.0" << std::endl;
    std::cout << "============================================" << std::endl << std::endl;
}

void print_separator() {
    std::cout << "--------------------------------------------" << std::endl;
}

void format_bytes(size_t bytes, std::ostream& os) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }

    os << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
}

void print_tensor_info(const ggml::TensorInfo& info) {
    std::cout << "  Name: " << info.name << std::endl;
    std::cout << "    Type: ";

    switch (info.type) {
        case ggml::GGMLType::F32: std::cout << "F32"; break;
        case ggml::GGMLType::F16: std::cout << "F16"; break;
        case ggml::GGMLType::Q4_0: std::cout << "Q4_0"; break;
        case ggml::GGMLType::Q4_1: std::cout << "Q4_1"; break;
        case ggml::GGMLType::Q4_K: std::cout << "Q4_K"; break;
        case ggml::GGMLType::Q5_K: std::cout << "Q5_K"; break;
        case ggml::GGMLType::Q6_K: std::cout << "Q6_K"; break;
        case ggml::GGMLType::Q8_0: std::cout << "Q8_0"; break;
        default: std::cout << "Unknown"; break;
    }

    std::cout << std::endl;
    std::cout << "    Shape: [";

    for (size_t i = 0; i < info.shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << info.shape[i];
    }

    std::cout << "]" << std::endl;

    size_t total_elements = 1;
    for (uint32_t dim : info.shape) {
        total_elements *= dim;
    }

    size_t type_size = ggml::ggml_type_size(info.type);
    size_t block_size = ggml::ggml_blck_size(info.type);
    size_t tensor_bytes = ((total_elements + block_size - 1) / block_size) * type_size;

    std::cout << "    Elements: " << total_elements << std::endl;
    std::cout << "    Size: ";
    format_bytes(tensor_bytes, std::cout);
    std::cout << std::endl;
}

void print_layer_statistics(const std::vector<ggml::TensorInfo>& tensors) {
    uint32_t layer_count = 0;
    size_t total_layer_size = 0;

    for (const auto& tensor : tensors) {
        if (tensor.name.find(".layers.") != std::string::npos) {
            layer_count++;
            size_t total_elements = 1;
            for (uint32_t dim : tensor.shape) {
                total_elements *= dim;
            }

            size_t type_size = ggml::ggml_type_size(tensor.type);
            size_t block_size = ggml::ggml_blck_size(tensor.type);
            size_t tensor_bytes = ((total_elements + block_size - 1) / block_size) * type_size;

            total_layer_size += tensor_bytes;
        }
    }

    std::cout << "  Total Layers: " << layer_count << std::endl;
    std::cout << "  Total Layer Size: ";
    format_bytes(total_layer_size, std::cout);
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [options]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --all               Show all tensors" << std::endl;
        std::cerr << "  --summary           Show model summary only" << std::endl;
        std::cerr << "  --metadata          Show metadata" << std::endl;
        std::cerr << "  --layers            Show layer statistics" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example: " << argv[0] << " model.gguf --summary --metadata" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    bool show_all = false;
    bool show_summary = false;
    bool show_metadata = false;
    bool show_layers = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--all") show_all = true;
        else if (arg == "--summary") show_summary = true;
        else if (arg == "--metadata") show_metadata = true;
        else if (arg == "--layers") show_layers = true;
    }

    if (!show_all && !show_summary && !show_metadata && !show_layers) {
        show_summary = true;
        show_metadata = true;
    }

    std::cout << "Loading GGUF file: " << model_path << std::endl;
    print_separator();

    ggml::GGUFParser parser;
    if (!parser.parse(model_path)) {
        std::cerr << "Failed to parse GGUF file" << std::endl;
        return 1;
    }

    const auto& tensors = parser.get_tensors();
    const auto& metadata = parser.get_metadata();

    std::cout << "File loaded successfully!" << std::endl;
    std::cout << "Number of tensors: " << tensors.size() << std::endl;
    std::cout << "Number of metadata entries: " << metadata.size() << std::endl;
    print_separator();

    if (show_summary) {
        std::cout << "Model Summary:" << std::endl;
        print_separator();

        auto get_meta = [&](const std::string& key) -> std::string {
            auto it = metadata.find(key);
            return it != metadata.end() ? it->second : "N/A";
        };

        std::cout << "  Architecture: " << get_meta("general.architecture") << std::endl;
        std::cout << "  Quantization: " << get_meta("general.quantization_version") << std::endl;
        std::cout << "  Version: " << get_meta("general.version") << std::endl;
        print_separator();

        std::cout << "Model Parameters:" << std::endl;
        print_separator();
        std::cout << "  Block Count: " << get_meta("llama.block_count") << std::endl;
        std::cout << "  Embedding Length: " << get_meta("llama.embedding_length") << std::endl;
        std::cout << "  Feed Forward Length: " << get_meta("llama.feed_forward_length") << std::endl;
        std::cout << "  Head Count: " << get_meta("llama.attention.head_count") << std::endl;
        std::cout << "  Head Count KV: " << get_meta("llama.attention.head_count_kv") << std::endl;
        std::cout << "  Context Length: " << get_meta("llama.context_length") << std::endl;
        std::cout << "  Vocab Size: " << get_meta("llama.vocab_size") << std::endl;
        print_separator();

        size_t total_size = 0;
        for (const auto& tensor : tensors) {
            size_t total_elements = 1;
            for (uint32_t dim : tensor.shape) {
                total_elements *= dim;
            }

            size_t type_size = ggml::ggml_type_size(tensor.type);
            size_t block_size = ggml::ggml_blck_size(tensor.type);
            size_t tensor_bytes = ((total_elements + block_size - 1) / block_size) * type_size;

            total_size += tensor_bytes;
        }

        std::cout << "Total Model Size: ";
        format_bytes(total_size, std::cout);
        std::cout << std::endl;
        print_separator();
    }

    if (show_metadata) {
        std::cout << "Metadata:" << std::endl;
        print_separator();

        for (const auto& pair : metadata) {
            std::cout << "  " << pair.first << ": " << pair.second << std::endl;
        }

        print_separator();
    }

    if (show_layers) {
        std::cout << "Layer Statistics:" << std::endl;
        print_separator();
        print_layer_statistics(tensors);
        print_separator();
    }

    if (show_all) {
        std::cout << "All Tensors:" << std::endl;
        print_separator();

        for (const auto& tensor : tensors) {
            print_tensor_info(tensor);
            std::cout << std::endl;
        }

        print_separator();
    }

    std::cout << "Analysis complete!" << std::endl;
    print_separator();

    return 0;
}