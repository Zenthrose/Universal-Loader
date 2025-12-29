#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace vulkan {

struct DescriptorSetLayoutBinding {
    uint32_t binding;
    VkDescriptorType descriptor_type;
    uint32_t descriptor_count;
    VkShaderStageFlags stage_flags;
};

class DescriptorPool {
public:
    DescriptorPool(VkDevice device);
    ~DescriptorPool();

    bool create_descriptor_pool(const std::vector<VkDescriptorPoolSize>& pool_sizes,
                              uint32_t max_sets);

    bool create_descriptor_set_layouts(const std::vector<DescriptorSetLayoutBinding>& bindings);

    bool allocate_descriptor_set(VkDescriptorSet& descriptor_set, VkDescriptorSetLayout layout);

    void update_buffer_descriptor(VkDescriptorSet descriptor_set, uint32_t binding,
                                VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset = 0);

    void cleanup();

    VkDescriptorSetLayout get_layout(uint32_t index = 0) const {
        return descriptor_set_layouts_.empty() ? VK_NULL_HANDLE : descriptor_set_layouts_[0];
    }

private:
    VkDevice device_;
    VkDescriptorPool descriptor_pool_;
    std::vector<VkDescriptorSetLayout> descriptor_set_layouts_;
};

class BindlessDescriptors {
public:
    BindlessDescriptors(VkDevice device, VkPhysicalDevice physical_device);
    ~BindlessDescriptors();

    bool initialize(uint32_t max_bindings);
    void cleanup();

    VkDescriptorPool get_pool() const { return pool_; }
    VkDescriptorSetLayout get_layout() const { return layout_; }
    VkDescriptorSet get_descriptor_set() const { return descriptor_set_; }

    bool update_buffer_binding(uint32_t binding_index, VkBuffer buffer, VkDeviceSize size);

private:
    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkDescriptorPool pool_;
    VkDescriptorSetLayout layout_;
    VkDescriptorSet descriptor_set_;
    uint32_t max_bindings_;
};

}