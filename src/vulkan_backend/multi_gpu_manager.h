#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <mutex>

namespace vulkan {

struct DeviceGroupConfig {
    std::vector<VkPhysicalDevice> physical_devices;
    std::vector<VkDevice> logical_devices;
    std::vector<VkQueue> compute_queues;
    std::vector<uint32_t> queue_families;
};

class MultiGPUManager {
public:
    MultiGPUManager(VkInstance instance);
    ~MultiGPUManager();

    bool initialize();
    void cleanup();

    void enable_device_groups();

    VkDevice get_primary_device() const { return devices_[0]; }
    VkDevice get_device(uint32_t index) const;

    uint32_t get_num_devices() const { return static_cast<uint32_t>(devices_.size()); }

    bool supports_device_groups() const { return supports_device_groups_; }

    uint32_t get_device_memory(uint32_t device_index) const;

private:
    bool enumerate_physical_devices();
    bool create_device_group();
    bool create_logical_devices();
    bool query_device_group_properties();

    VkInstance instance_;
    DeviceGroupConfig config_;

    std::vector<VkDevice> devices_;
    std::vector<VkQueue> queues_;
    std::vector<uint32_t> queue_families_;

    bool supports_device_groups_;
    VkPhysicalDeviceGroupProperties device_group_props_;

    std::mutex mutex_;
};

class TensorPartition {
public:
    TensorPartition(uint32_t num_devices, uint32_t tensor_size);
    ~TensorPartition();

    void partition_tensor(const float* input, std::vector<float*>& partitions);
    void merge_partitions(const std::vector<const float*>& partitions, float* output);

    uint32_t get_partition_size(uint32_t device_index) const;

private:
    uint32_t num_devices_;
    uint32_t tensor_size_;
    uint32_t partition_size_;

    std::vector<uint32_t> partition_sizes_;
};

}
