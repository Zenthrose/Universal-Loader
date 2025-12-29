#include "descriptors.h"
#include <iostream>

namespace vulkan {

DescriptorPool::DescriptorPool(VkDevice device)
    : device_(device), descriptor_pool_(VK_NULL_HANDLE) {
}

DescriptorPool::~DescriptorPool() {
    cleanup();
}

bool DescriptorPool::create_descriptor_pool(const std::vector<VkDescriptorPoolSize>& pool_sizes,
                                          uint32_t max_sets) {
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = max_sets;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VkResult result = vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool: " << result << std::endl;
        return false;
    }

    return true;
}

bool DescriptorPool::create_descriptor_set_layouts(const std::vector<DescriptorSetLayoutBinding>& bindings) {
    std::vector<VkDescriptorSetLayoutBinding> vk_bindings;
    for (const auto& binding : bindings) {
        VkDescriptorSetLayoutBinding vk_binding{};
        vk_binding.binding = binding.binding;
        vk_binding.descriptorType = binding.descriptor_type;
        vk_binding.descriptorCount = binding.descriptor_count;
        vk_binding.stageFlags = binding.stage_flags;
        vk_binding.pImmutableSamplers = nullptr;
        vk_bindings.push_back(vk_binding);
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(vk_bindings.size());
    layout_info.pBindings = vk_bindings.data();

    VkDescriptorSetLayout layout;
    VkResult result = vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout: " << result << std::endl;
        return false;
    }

    descriptor_set_layouts_.push_back(layout);
    return true;
}

bool DescriptorPool::allocate_descriptor_set(VkDescriptorSet& descriptor_set, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;

    VkResult result = vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set: " << result << std::endl;
        return false;
    }

    return true;
}

void DescriptorPool::update_buffer_descriptor(VkDescriptorSet descriptor_set, uint32_t binding,
                                             VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void DescriptorPool::cleanup() {
    for (auto layout : descriptor_set_layouts_) {
        vkDestroyDescriptorSetLayout(device_, layout, nullptr);
    }
    descriptor_set_layouts_.clear();

    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }
}

BindlessDescriptors::BindlessDescriptors(VkDevice device, VkPhysicalDevice physical_device)
    : device_(device), physical_device_(physical_device), pool_(VK_NULL_HANDLE),
      layout_(VK_NULL_HANDLE), descriptor_set_(VK_NULL_HANDLE), max_bindings_(0) {
}

BindlessDescriptors::~BindlessDescriptors() {
    cleanup();
}

bool BindlessDescriptors::initialize(uint32_t max_bindings) {
    max_bindings_ = max_bindings;

    std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_bindings}
    };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |
                     VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    VkResult result = vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create bindless descriptor pool: " << result << std::endl;
        return false;
    }

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = max_bindings;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.pImmutableSamplers = nullptr;

    VkDescriptorBindingFlagsEXT binding_flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT |
                                             VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT;

    VkDescriptorSetLayoutBindingFlagsCreateInfoEXT flags_info{};
    flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
    flags_info.bindingCount = 1;
    flags_info.pBindingFlags = &binding_flags;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.pNext = &flags_info;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;

    result = vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &layout_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create bindless layout: " << result << std::endl;
        return false;
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout_;

    result = vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to allocate bindless descriptor set: " << result << std::endl;
        return false;
    }

    std::cout << "Bindless descriptors initialized with " << max_bindings << " bindings" << std::endl;
    return true;
}

void BindlessDescriptors::cleanup() {
    if (descriptor_set_ != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device_, pool_, 1, &descriptor_set_);
        descriptor_set_ = VK_NULL_HANDLE;
    }

    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }

    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }
}

bool BindlessDescriptors::update_buffer_binding(uint32_t binding_index, VkBuffer buffer, VkDeviceSize size) {
    if (binding_index >= max_bindings_) {
        std::cerr << "Binding index " << binding_index << " exceeds max bindings " << max_bindings_ << std::endl;
        return false;
    }

    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = 0;
    buffer_info.range = size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_;
    write.dstBinding = 0;
    write.dstArrayElement = binding_index;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    return true;
}

}