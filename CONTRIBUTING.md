# Contributing to Universal GGUF Loader

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Coding Standards](#coding-standards)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Sprints](#sprints)
7. [Roadmap](#sprints)

---

## Getting Started

### Prerequisites

To contribute, you'll need:

**Required:**
- C++20 compatible compiler
- CMake 3.20+ or Ninja
- Git

**For GPU development:**
- Vulkan SDK 1.3+ installed
- GPU with Vulkan 1.3+ support

**Recommended:**
- Familiarity with Vulkan API or Vulkan-Hpp
- Understanding of LLM architectures (LLaMA, Mistral, etc.)
- Experience with compute shaders (GLSL) or CUDA

### First-Time Setup

1. Fork the repository:
```bash
gh repo fork Zenthrose/Universal-Loader
```

2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/Universal-Loader.git
cd Universal-Loader
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/Zenthrose/Universal-Loader.git
git fetch upstream
```

4. Create a branch for your work:
```bash
git checkout -b feature/my-cool-feature
```

---

## Development Setup

### Build Instructions

#### Initial Build
```bash
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j8
```

#### Incremental Build
```bash
cmake --build . --target gguf_core  # Build only core library
cmake --build . --target test_core  # Build and run tests
```

### Running Tests
```bash
cd build
ctest --output-on-failure --output-short-failure
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "Add my feature"
git push -u origin feature/my-feature
# Create pull request
gh pr create --title "Add my feature" --body "Description"
```

---

## Coding Standards

### C++ Guidelines

1. **Formatting**
   - Use 4-space indentation
   - Max line length: 120 characters
   - Place braces on new line for functions
   - Place braces on same line for everything else

2. **Naming Conventions**
   - Classes: `PascalCase` (e.g., `VulkanContext`)
   - Functions: `snake_case` (e.g., `load_model`, `get_num_threads`)
   - Member variables: `trailing_underscore_` (e.g., `gpu_enabled_`)
   - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_LAYERS`)

3. **Documentation**
   - All public APIs must have Doxygen comments
   - Add `@param`, `@return`, `@brief` tags
   - Document thread safety of each public method

4. **Error Handling**
   - Use exceptions sparingly (prefer error codes)
   - Always check return values from Vulkan functions
   - Log errors with descriptive messages
   - Handle edge cases (null pointers, out of bounds, etc.)

Example:
```cpp
bool VulkanContext::initialize(const VulkanConfig& config) {
    try {
        vk::raii::Context context(vk_create_instance);
        // ... initialization code ...
        initialized_ = true;
        return true;
    } catch (const vk::SystemError& e) {
        std::cerr << "Vulkan initialization failed: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }
}
```

5. **RAII (Resource Acquisition Is Initialization)**
   - Use Vulkan-Hpp RAII handles (vk::raii::Context, vk::raii::DeviceMemory, etc.)
   - Never call `vkDestroy*` manually
   - Resources are automatically cleaned up when going out of scope
   - Use `vk::raii::UniqueHandle` for Vulkan handles

Example:
```cpp
class VulkanContext {
    vk::raii::Context context_;
    vk::raii::Device device_;
    
public:
    bool initialize() {
        context_ = vk::raii::Context(vk::create_instance);
        device_ = vk::raii::Device(context_, physical_device_, ...);
        return true;
    }
};
// No manual cleanup needed - automatic in destructor
```

### GLSL Guidelines

1. **Versioning**
   - Use `#version 460` or higher
   - Add required extensions explicitly

2. **Layout Qualifiers**
   - Always use `binding` or `push_constant` for variables
   - Use `layout(local_size_x = X, local_size_y = Y, local_size_z = Z)` for workgroups
   - Add explicit `std140` or `std430` precision qualifiers for floats

3. **Performance**
   - Use shared memory where beneficial (shared memory size must match shader)
   - Avoid bank conflicts in shared memory
   - Use subgroup operations where available
   - Add `memoryBarrierShared()` before using shared memory
   - Use `barrier()` before using results

4. **Safety**
   - Always add bounds checking at start of shaders
   - Validate array access with `.length()` checks
   - Check for out-of-bounds reads/writes

Example:
```glsl
#version 460
layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Input { float x[]; };
layout(binding = 1) writeonly buffer Output { float y[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= x.length()) return; // Bounds check
    
    float x_val = x[i];
    y[i] = x_val * 2.0;
}
```

### Vulkan Guidelines

1. **Validation**
   - Test with validation layers in debug builds
   - Fix all validation errors before submitting PRs
   - Add validation layer configuration to PR description

2. **Error Handling**
   - Check all VkResult return codes
   - Use `vk::ResultValueExceptions` for error checking
   - Handle all error paths with fallbacks
   - Never ignore `VK_ERROR_OUT_OF_DEVICE_MEMORY`

3. **Resource Management**
   - Query device properties before allocating
   - Use proper descriptor pool management
   - Check available VRAM before large allocations
   - Implement graceful fallback to CPU on OOM

Example:
```cpp
VkPhysicalDeviceMemoryProperties props;
vkGetPhysicalDeviceMemoryProperties(physical_device_, &props);

if (required_memory > props.deviceTotalSize * 0.8) {
    std::cerr << "Insufficient VRAM, falling back to CPU" << std::endl;
    return false;
}
```

---

## Testing

### Unit Tests

We use GoogleTest framework for unit tests.

### Writing Tests

1. **Test File Structure**
   - Place test files in `tests/unit/` directory
   - Name test files: `test_<module>.cpp`
   - Organize by functionality

2. Test Categories:
   - Initialization tests
   - Error handling tests
   - Correctness tests
   - Performance tests

Example:
```cpp
#include <gtest/gtest.h>

TEST(VulkanContextTest, Initialization) {
    VulkanContext context;
    ASSERT_TRUE(context.initialize(config));
    EXPECT_TRUE(context.is_initialized());
}

TEST(VulkanContextTest, ErrorHandling) {
    VulkanContext context;
    context.initialize(config);
    EXPECT_FALSE(context.initialize(invalid_config));
}
```

3. Running Tests
```bash
cd build
ctest --output-on-failure --output-short-failure
```

### Integration Tests

1. Load real GGUF files and verify:
   - Model structure is parsed correctly
   - All tensors are loaded
   - Metadata is extracted

2. End-to-end generation:
   - Generate 100 tokens from prompt
   - Verify output reproducibility

3. Multi-architecture tests:
   - Test on LLaMA-2, LLaMA-3, Mistral, Mixtral
   - Verify quantization types

### Performance Tests

Run benchmarks before submitting performance-related PRs:
```bash
cd build
./tools/benchmark model.gguf
```

Include benchmark results in PR description:
```markdown
## Performance Results

Before:
- Model: LLaMA-2 7B Q4_K
- Tokens/sec: 15.2

After:
- Model: Lina-2 7B Q4_K
- Tokens/sec: 22.4 (+47%)
```

---

## Pull Request Process

### Before Submitting

1. **Code Review Your Own PR**
   - Ensure your code compiles without warnings
   - Run full test suite
   - Update documentation if needed
   - Self-review for issues

2. **Check Roadmap Alignment**
   - Reference [ROADMAP.md](ROADMAP.md)
   - Your PR should align with current sprint goals
   - Include roadmap phase in PR title (e.g., "Phase 1.2: Validation Layers")

3. **Test on Multiple Architectures** (if applicable)
   - AMD Polaris (your setup)
   - NVIDIA (if you have access)
   - Intel Arc (if you have access)

### Submitting Your PR

1. **Branch Name**
   - Feature branches: `feature/<feature-name>`
   - Fix branches: `fix/<description>`

2. **Commit Messages**
   - Use present tense and imperative mood
   - Describe what and why, not how
   - Reference issue numbers: `Resolves #123`

3. **Pull Request Title**
   - Clear and concise
   - Include issue number if applicable
   - Start with area: `[Phase X.Y]` to track roadmap progress

4. **Pull Request Body**
   - **Summary**: Brief description of changes
   - **Testing**: How you tested your changes
   - **Screenshots**: Include for UI changes
   - **Benchmarks**: For performance changes
   - **Breakdown**: List files changed

Example:
```markdown
## Summary
Adds Vulkan-Hpp RAII wrappers for automatic resource cleanup and exception safety.

## Testing
- Added unit tests for RAII resource management
- Tested on AMD RX 580 (Polaris) GPU
- Ran validation layers, 0 errors

## Performance
Before: No RAII, manual cleanup
After: Automatic RAII, no resource leaks

## Files Changed
- src/vulkan_backend/context.h/cpp
- src/vulkan_backend/memory.h/cpp
- src/vulkan_backend/compute.h/cpp
- tests/unit/test_vulkan_raii.cpp
```

### After Submitting

1. **Respond to Reviewer Feedback**
   - Address all comments
   - Make requested changes promptly
   - Re-run tests

2. **Keep PR Updated**
   - Add commits as needed
   - Update PR description with new test results

3. **Squash Commits When Needed**
   - Only if commit history is confusing
   - Keep related changes together

---

## Sprints

### Current Sprints

We organize work into sprints for easier review and merging:

| Sprint | Status | Focus Area | Time Frame |
|--------|--------|-----------|-----------|
| Sprint 1 | 🚧 In Progress | Safety & Portability | Weeks 1-2 |
| Sprint 2 | 📋 Planned | Performance Optimization | Weeks 3-4 |
| Sprint 3 | 📋 Planned | Modern Features | Weeks 5-6 |
| Sprint 4 | 📋 Planned | Advanced Optimizations | Weeks 7-8 |

### Choosing a Sprint

1. **Look at Current Sprint**: What's active?
2. **Review Roadmap**: What needs to be done?
3. **Check Issues**: What are the priorities?

**Good First PRs:**
- Bug fixes (always welcome)
- Documentation improvements
- Test additions
- Small feature additions

**Large PRs Require**:
- Discussion in issue first
- Approval from maintainers
- Design document

---

## Documentation

### Updating Documentation

When adding new features:
1. Update README.md with usage examples
2. Update ROADMAP.md with progress
3. Add API documentation to headers (Doxygen)
4. Update relevant sections in docs/

### When Updating Docs, Consider:
- Is the feature complete or in-progress?
- Are the examples clear and tested?
- Are limitations documented?
- Is performance impact measured?

---

## Architecture Decisions

When Making Changes:

1. **Before Starting Large Features**
   - Open issue for discussion
   - Document the design decision
   - Get feedback from maintainers

2. **Document Design Decisions**
   - Why this approach?
   - Alternatives considered
   - Trade-offs made

3. **Update Architecture Docs**
   - Modify `docs/architecture.md` if it exists
   - Update ROADMAP.md with decisions made

---

## Common Issues and Solutions

### Compilation Errors

**Issue**: "undefined reference to vk::raii"
**Solution**: Add Vulkan-Hpp as CPM dependency to CMakeLists.txt

**Issue**: "GLSL compilation failed"
**Solution**: Ensure glslangValidator is installed and in PATH

**Issue**: "Undefined reference to `std`"
**Solution**: Add proper `#include` directives

### Vulkan Errors

**Issue**: "VK_ERROR_INITIALIZATION_FAILED"
**Solution**: Check GPU driver version, update Vulkan SDK

**Issue**: "VK_ERROR_OUT_OF_DEVICE_MEMORY"
**Solution**: Reduce GPU memory pool or use smaller quantization

### Performance Issues

**Issue**: "GPU utilization is low (< 40%)"
**Solution**:
- Increase async overlap
- Reduce memory transfers
- Increase workgroup sizes

**Issue**: "Tokens/sec is slow (< 10 tok/s)"
**Solution**:
- Enable Flash Attention
- Enable prefetching
- Use faster quantization

---

## Git Workflow

### Branching Strategy

```bash
main              # Production releases
├── develop         # Development
│   ├── phase1-safety  # Phase 1 work
│   ├── phase2-perf    # Phase 2 work
│   └── phase3-modern   # Phase 3 work
```

### Commit Guidelines

1. **Commit Messages**
   - Use imperative present tense
   - First line: 50 chars or less
   - Describe WHAT and WHY, not HOW
   - Reference issue: `Resolves #123`

   Good:
   - `Add validation layer support with debug configuration`
   - `Fix OOM handling in Vulkan context`
   - `Implement bindless descriptors for 10-20% speedup`

   Bad:
   - `Add some validation stuff to context.h` (too vague)
   - `Fix error` (unhelpful)
   - `Update README.md` (too broad)

2. **Commit Frequency**
   - Commit frequently, small commits
   - Each commit should compile and pass tests
   - Atomic changes per commit

### Merging PRs

**For Small PRs:**
- Squash and merge to `develop` branch
- Test integration on `develop` before merging to `main`

**For Large PRs:**
- Merge to `develop` first
- Test thoroughly on `develop`
- Merge to `main` from `develop` after stability confirmed

---

## Questions?

Feel free to ask questions in:
- Issues: https://github.com/Zenthrose/Universal-Loader/issues
- Discussions: https://github.com/Zenthrose/Universal-Loader/discussions
- Email: [your email for maintainers]

### Getting Help

**Where to Start?**
- Choose a sprint item from ROADMAP.md
- Check for "good first issue" labels in issues
- Ask in Discussions: "I'd like to work on [Phase X.Y: Description]"

---

## Recognition

Contributors are recognized in:
- README.md (Contributors section)
- Release notes (Release history section)
- Documentation mentions

We appreciate all contributions, big and small!

---

## License

All contributions are licensed under MIT License. By contributing, you agree to license your contributions.
