# Vulkan Backend Modernization Roadmap

## Phase 1: Critical Safety & Portability (Week 1-2)

### 1.1 Vulkan-Hpp Migration
**Status**: ❌ Not started
**Priority**: CRITICAL

**Tasks**:
- [ ] Add Vulkan-Hpp as CPM dependency to CMakeLists.txt
- [ ] Update all `src/vulkan_backend/*.h` files to use vulkan:: RAII types
- [ ] Replace raw Vk* handles with `vk::UniqueHandle` wrappers
- [ ] Add RAII resource cleanup in destructors
- [ ] Remove manual `vkDestroy*` calls (handled by RAII)

**Benefits**: Automatic leak prevention, exception safety, better typing

**Files to Modify**:
- `CMakeLists.txt` - Add Vulkan-Hpp CPM
- `src/vulkan_backend/context.h/cpp` - Use vk::raii::Context
- `src/vulkan_backend/memory.h/cpp` - Use vk::raii::DeviceMemory
- `src/vulkan_backend/compute.h/cpp` - Use vk::raii::CommandBuffer

---

### 1.2 Validation Layers
**Status**: ❌ Not started
**Priority**: CRITICAL

**Tasks**:
- [ ] Add validation layers as optional dependency
- [ ] Create debug callback function with error logging
- [ ] Enable validation in Debug builds, disable in Release
- [ ] Add validation layer configuration to `VulkanConfig`
- [ ] Implement VK_EXT_validation_features support

**Files to Create**:
- `src/vulkan_backend/validation.h` - Validation layer management
- `src/vulkan_backend/validation.cpp` - Validation callbacks

**Files to Modify**:
- `src/vulkan_backend/context.h` - Add validation support
- `CMakeLists.txt` - Add validation layer package

---

### 1.3 Robust Error Handling
**Status**: ⚠️ Partial (basic error handling exists)
**Priority**: CRITICAL

**Tasks**:
- [ ] Create VulkanException class for error propagation
- [ ] Handle VK_ERROR_DEVICE_LOST gracefully (fallback to CPU)
- [ ] Detect and handle out-of-memory scenarios
- [ ] Add retry logic for transient errors
- [ ] Log all Vulkan errors with descriptive messages
- [ ] Implement per-error-type handling (OOM, device lost, timeout)

**Files to Modify**:
- `src/vulkan_backend/context.cpp` - Enhanced error handling
- `src/inference/inference_engine.cpp` - Fallback logic

---

### 1.4 Timeline Semaphores
**Status**: ⚠️ Partial (fences only)
**Priority**: HIGH

**Tasks**:
- [ ] Replace fences with timeline semaphores (Vulkan 1.2+)
- [ ] Implement semaphores for async transfer completion
- [ ] Add semaphores for compute completion
- [ ] Use semaphores for pipeline synchronization
- [ ] Remove legacy fence usage

**Benefits**: Better async overlap, reduced latency

**Files to Modify**:
- `src/vulkan_backend/transfer.h/cpp` - Timeline semaphores
- `src/vulkan_backend/compute.h/cpp` - Semaphore-based sync
- `src/inference/prefetch_engine.cpp` - Semaphore-based prefetch

---

## Phase 2: Performance Optimization (Week 3-4)

### 2.1 Bindless Descriptors
**Status**: ❌ Not started
**Priority**: HIGH

**Tasks**:
- [ ] Add VK_EXT_descriptor_indexing support
- [ ] Create descriptor index layout
- [ ] Remove descriptor set rebinds per layer
- [ ] Implement descriptor pool management
- [ ] Add bindless tensor descriptor sets

**Benefits**: No descriptor set changes, 10-20% speedup

**Files to Create**:
- `src/vulkan_backend/descriptors.h` - Bindless descriptor management

---

### 2.2 Dynamic Workgroup Sizes
**Status**: ❌ Not started
**Priority**: HIGH

**Tasks**:
- [ ] Query physical device properties
- [ ] Query subgroup size dynamically
- [ ] Calculate optimal workgroup size per kernel
- [ ] Add workgroup size calculation functions
- [ ] Update shaders to use push_constant workgroup sizes

**Benefits**: Optimal performance across all GPU architectures

**Files to Modify**:
- `src/vulkan_backend/context.h` - Device properties query
- `src/vulkan_backend/compute.cpp` - Dynamic workgroup calculation
- All shaders - Use push_constant workgroup sizes

---

### 2.3 Shared Memory Tiling
**Status**: ⚠️ Partial (basic tiling exists)
**Priority**: HIGH

**Tasks**:
- [ ] Implement tiled GEMM with shared memory
- [ ] Avoid bank conflicts in shared memory
- [ ] Add shared memory allocation in shaders
- [ ] Optimize for Polaris workgroup sizes
- [ ] Benchmark different tile sizes (8×8, 16×16, 32×32)

**Benefits**: 20-30% faster GEMM

**Files to Modify**:
- `src/shaders/gemm/gemm_tiled_polaris.glsl` - Shared memory tiling
- `src/shaders/gemm/gemm_shared_memory.glsl` - New shared memory version

---

### 2.4 Async Compute + Transfer Overlap
**Status**: ❌ Not started
**Priority**: HIGH

**Tasks**:
- [ ] Create triple-buffered system for async operations
- [ ] Pipeline prefetch transfers with compute
- [ ] Add command buffer recycling
- [ ] Implement async prefetching with timeline semaphores
- [ ] Add bandwidth utilization monitoring

**Benefits**: 30-40% better throughput

**Files to Modify**:
- `src/vulkan_backend/transfer.cpp` - Async pipeline
- `src/vulkan_backend/compute.cpp` - Overlap logic
- `src/inference/prefetch_engine.cpp` - Triple buffering

---

## Phase 3: Modern Features (Week 5-6)

### 3.1 Pipeline Caching to Disk
**Status**: ⚠️ Partial (in-memory only)
**Priority**: MEDIUM

**Tasks**:
- [ ] Implement vk::PipelineCache with disk persistence
- [ ] Add cache directory configuration
- [ ] Save compiled pipelines to disk on first run
- [ ] Load cached pipelines on subsequent runs
- [ ] Add cache invalidation on shader changes

**Benefits**: Instant startup, no shader compilation after first run

**Files to Modify**:
- `src/vulkan_backend/pipeline_cache.cpp` - Disk persistence

---

### 3.2 Device Address Buffers
**Status**: ❌ Not started
**Priority**: MEDIUM

**Tasks**:
- [ ] Add VK_KHR_buffer_device_address support
- [ ] Create device-local buffers
- [ ] Pass buffer addresses to shaders
- [ ] Update shaders to use device addresses
- [ ] Implement fallback for unsupported devices

**Benefits**: Faster irregular access patterns in attention

**Files to Modify**:
- `src/vulkan_backend/memory.cpp` - Device address support
- `src/shaders/attention/*.glsl` - Buffer address usage

---

### 3.3 Cooperative Matrices
**Status**: ❌ Not started
**Priority**: MEDIUM

**Tasks**:
- [ ] Add VK_KHR_cooperative_matrix support
- [ ] Create cooperative matrix operations
- [ ] Implement cooperative GEMM kernel
- [ ] Add device capability detection
- [ ] Implement fallback for unsupported devices

**Benefits**: Up to 2× faster matmul on supported hardware

**Files to Create**:
- `src/vulkan_backend/cooperative_matrix.h` - Cooperative matrix abstraction

---

### 3.4 Shader Pre-Compilation
**Status**: ❌ Not started
**Priority**: MEDIUM

**Tasks**:
- [ ] Add glslangValidator to CMake build process
- [ ] Pre-compile shaders during CMake configuration
- [ ] Embed SPIRV binaries in executable
- [ ] Add shader source hash for cache invalidation
- [ ] Add on-the-fly compilation for custom shaders

**Files to Create**:
- `src/vulkan_backend/shader_compiler.h` - Shader compilation pipeline

---

## Phase 4: Advanced Optimizations (Week 7-8)

### 4.1 Flash Attention 2.0
**Status**: ⚠️ Partial (Flash 1.0 implemented)
**Priority**: HIGH

**Tasks**:
- [ ] Implement causal masking in flash attention
- [ ] Add multi-query flash attention
- [ ] Implement attention kernel fusion
- [ ] Optimize shared memory usage
- [ ] Add dropout support for training

**Benefits**: 2-3× faster than Flash 1.0

**Files to Modify**:
- `src/shaders/attention/flash_attention.glsl` - Flash 2.0
- `src/shaders/attention/flash_causal.glsl` - Causal masking

---

### 4.2 Speculative Decoding
**Status**: ❌ Not started
**Priority**: MEDIUM

**Tasks**:
- [ ] Create draft model runner on CPU
- [ ] Run 2-3 draft generations in parallel
- - [ ] Verify draft outputs against main model
- [ ] Accept or reject drafts based on verification
- [ ] Estimate 1.8-2.2× speedup for batched generation

**Files to Create**:
- `src/inference/speculative_decode.h` - Speculative decoding engine
- `src/cpu_backend/draft_model.h` - Draft model runner

---

### 4.3 Multi-GPU Support
**Status**: ❌ Not started
**Priority**: LOW (future enhancement)

**Tasks**:
- [ ] Add VK_KHR_device_group support
- [ ] Implement layer splitting across GPUs
- [ ] Add inter-GPU synchronization
- [ ] Implement cross-GPU KV cache
- [ ] Add GPU selection policy

**Files to Create**:
- `src/vulkan_backend/multi_gpu.h` - Multi-GPU abstraction
- `src/inference/multi_gpu_scheduler.h` - Load balancing

---

### 4.4 LoRA Adapters
**Status**: ❌ Not started
**Priority**: LOW (nice to have)

**Tasks**:
- [ ] Implement LoRA adapter loading
- [ ] Add LoRA weight merging in shaders
- - [ ] Add multiple LoRA support per model
- - [ ] Implement dynamic LoRA enable/disable
- - [ ] Add LoRA adapter cache

**Files to Create**:
- `src/inference/lora_adapter.h` - LoRA adapter management
- `src/shaders/lora/lora_fusion.glsl` - LoRA weight fusion

---

## Phase 5: Portability & Testing (Week 9-10)

### 5.1 Cross-GPU Compatibility
**Status**: ❌ Not started
**Priority**: HIGH

**Tasks**:
- [ ] Add subgroup size queries
- [ ] Implement fallback workgroup sizes for NVIDIA/Intel
- [ ] Test on AMD (Polaris), NVIDIA (Pascal+), Intel (Arc)
- [ ] Add architecture-specific optimizations
- [ ] Implement performance detection

**Files to Modify**:
- `src/vulkan_backend/context.cpp` - Cross-GPU detection
- `src/shaders/*/*.glsl` - Portable workgroup sizes

---

### 5.2 Quantization Type Handling
**Status**: ⚠️ Partial (Q4_0 and Q4_K only)
**Priority**: CRITICAL

**Tasks**:
- [ ] Implement all quantization types (Q2_K, Q3_K, Q5_K, Q6_K, Q8_0)
- [ ] Handle block boundaries correctly
- [ ] Add per-type dequantization shaders
- [ ] Implement mixed-precision dequantization
- [ ] Add dequantization unit tests

**Files to Create**:
- `src/shaders/dequantize/*.glsl` - All dequantization kernels

---

### 5.3 Unit Tests
**Status**: ⚠️ Partial (basic tests only)
**Priority**: HIGH

**Tasks**:
- [ ] Create comprehensive test suite using GoogleTest/Catch2
- [ ] Test Vulkan initialization and cleanup
- [ ] Test all quantization types
- [ ] Test async operations
- [ ] Add CI/CD pipeline
- [ ] Add performance benchmarks

**Files to Create**:
- `tests/test_vulkan.h` - Vulkan-specific tests
- `tests/test_quantization.h` - Quantization tests
- `tests/test_flash_attention.h` - Flash attention tests
- `tests/benchmark.h` - Performance benchmarks

---

### 5.4 Profiling & Monitoring
**Status**: ❌ Not started
**Priority**: MEDIUM

**Tasks**:
- [ ] Add VK_KHR_performance_query support
- [ ] Implement timestamp queries for per-layer timing
- [ ] Add memory usage tracking
- [ ] Add GPU utilization metrics
- [ ] Export profiling data to Chrome tracing format

**Files to Create**:
- `src/vulkan_backend/profiler.h` - Profiling interface
- `src/inference/profiler.h` - High-level profiling

---

## Phase 6: Build System & API (Week 11-12)

### 6.1 Improved CMakeLists.txt
**Status**: ⚠️ Partial (basic CMake)
**Priority**: HIGH

**Tasks**:
- [ ] Add Vulkan-Hpp via CPM
- [ ] Add validation layers as optional dependency
- [ ] Separate CPU-only and Vulkan build targets
- [ ] Add shader pre-compilation step
- [ ] Add Python bindings configuration
- [ ] Add install targets

---

### 6.2 Clean Public API
**Status**: ⚠️ Partial (basic API exists)
**Priority**: MEDIUM

**Tasks**:
- [ ] Add strong types and enums
- [ ] Add Doxygen comments
- [ ] Create comprehensive API documentation
- [ ] Add usage examples
- [ ] Add error handling examples
- [ ] Create Python binding examples

---

## Phase 7: Out-of-the-Box Features (Week 13+)

### 7.1 Python Bindings
**Status**: ❌ Not started
**Priority**: LOW

**Tasks**:
- [ ] Add pybind11 as dependency
- - [ ] Create Python bindings for all public APIs
- [ ] Add NumPy array support
- [ ] Add Python examples
- [ ] Package as PyPI wheel

### 7.2 Ray Tracing Pipelines (Research)
**Status**: ❌ Not started
**Priority**: LOW (experimental)

**Tasks**:
- [ ] Research ray tracing for sparse attention
- [ ] Implement proof-of-concept
- - ] Benchmark vs standard attention
- - ] Document results

---

## Implementation Order

### Sprint 1 (Week 1-2): Phase 1 Complete
**Goal**: Safety, portability, robust error handling
**Deliverables**:
- Vulkan-Hpp integration
- Validation layers
- Proper error handling with fallback
- Timeline semaphores

### Sprint 2 (Week 3-4): Phase 2 Complete
**Goal**: Performance optimization
**Deliverables**:
- Bindless descriptors
- Dynamic workgroup sizes
- Shared memory tiling
- Async compute + transfer overlap

### Sprint 3 (Week 5-6): Phase 3 Complete
**Goal**: Modern features
**Deliverables**:
- Pipeline caching to disk
- Device address buffers
- Cooperative matrices
- Shader pre-compilation

### Sprint 4 (Week 7-8): Phase 4 Complete
**Goal**: Advanced optimizations
**Deliverables**:
- Flash Attention 2.0
- Speculative decoding
- Multi-GPU support
- LoRA adapters

### Sprint 5 (Week 9-10): Phase 5 Complete
**Goal**: Portability and testing
**Deliverables**:
- Cross-GPU compatibility
- All quantization types
- Comprehensive unit tests
- Profiling and monitoring

### Sprint 6 (Week 11-12): Phase 6 Complete
**Goal**: Build system and API
**Deliverables**:
- Improved CMakeLists.txt
- Clean public API
- Comprehensive documentation
- Python bindings

### Sprint 7 (Week 13+): Phase 7 Complete
**Goal**: Out-of-the-box features
**Deliverables**:
- Python package
- Research features
- Production deployment

---

## Success Criteria

### Sprint 1 Complete When:
- [x] All raw Vulkan handles replaced with RAII
- [x] Validation layers enabled in debug builds
- [x] All errors handled gracefully
- [x] Fallback to CPU on Vulkan failure
- [ ] Timeline semaphores replace all fences

### Sprint 2 Complete When:
- [ ] Descriptor set changes < 5% of time
- [ ] Workgroup sizes optimal for all GPUs
- [ ] Shared memory used in all kernels
- [ ] Async overlap achieves 30%+ speedup
- [ ] Bandwidth utilization > 70%

### Sprint 3 Complete When:
- [ ] Pipeline load time < 1ms
- [ ] Device addresses used in attention
- [ ] Cooperative matrices enabled where available
- [ ] Shader compilation time < 5s on first run

### Sprint 4 Complete When:
- [ ] Flash attention 2× faster than Flash 1.0
- [ ] Speculative decoding achieves 1.8× speedup
- [ ] Multi-GPU scales linearly (2× GPUs = 1.8× speed)
- [ ] LoRA adapter overhead < 5%

### Sprint 5 Complete When:
- [ ] Works on AMD, NVIDIA, Intel GPUs
- [ ] All quantization types supported
- [ ] Test coverage > 90%
- [ ] Profiling data exported to Chrome tracing

### Sprint 6 Complete When:
- [ ] CMake builds all configurations
- [ ] API is clean and intuitive
- [ ] Documentation is comprehensive
- [ ] Python package installable via pip

---

## Risk Mitigations

### Vulkan-Hpp Migration Risk
**Risk**: Breaks existing code, requires learning new API
**Mitigation**: 
- Implement in parallel with old API
- Gradual migration per module
- Extensive testing in each step

### Validation Layers Risk
**Risk**: Performance overhead, complexity
**Mitigation**:
- Disable in release builds by default
- Add configuration flag
- Only validate errors/warnings

### Timeline Semaphores Risk
**Risk**: New synchronization bugs, complexity
**Mitigation**:
- Start with simple use cases
- Add extensive logging
- Test thoroughly in isolation

### Bindless Descriptors Risk
**Risk**: Requires Vulkan 1.2+ support
**Mitigation**:
- Add device capability queries
- Fallback to descriptor sets on old devices
- Add feature flag to disable

### Cooperative Matrices Risk
**Risk**: Only supported on latest hardware
**Mitigation**:
- Device capability detection
- Graceful fallback
- Feature flag to disable

---

## Performance Targets

### Current Baseline (FX-4130 + RX 580)
- Tokens/sec: 15.2 (Q4_K)
- GPU utilization: ~40%
- Memory bandwidth: ~20 GB/s (of 150 GB/s theoretical)
- Pipeline change overhead: ~5%

### Target After Phase 2
- Tokens/sec: 25+ (65% improvement)
- GPU utilization: ~60%
- Memory bandwidth: ~45 GB/s
- Pipeline change overhead: < 1%

### Target After Phase 3
- Tokens/sec: 35+ (130% improvement)
- GPU utilization: ~75%
- Memory bandwidth: ~70 GB/s
- Startup time: < 1s

### Target After Phase 4
- Tokens/sec: 45+ (200% improvement)
- GPU utilization: ~90%
- Startup time: < 100ms
- Supports batch generation

---

## Dependencies

### Required
- Vulkan SDK 1.3+
- CPM for Vulkan-Hpp
- CMake 3.20+
- C++23 compiler

### Optional
- VK_KHR_device_group (multi-GPU)
- VK_KHR_cooperative_matrix (cooperative matmul)
- VK_KHR_buffer_device_address (device addresses)
- VK_EXT_descriptor_indexing (bindless descriptors)
- VK_KHR_performance_query (profiling)
- GoogleTest or Catch2
- pybind11 (Python bindings)
- glslangValidator (shader pre-compilation)

---

## Testing Strategy

### Unit Tests (GoogleTest)
```bash
tests/
├── unit/
│   ├── test_vulkan.cpp
│   ├── test_quantization.cpp
│   ├── test_flash_attention.cpp
│   └── test_api.cpp
├── integration/
│   ├── test_full_generation.cpp
│   ├── test_gpu_fallback.cpp
│   └── test_multigpu.cpp
└── benchmarks/
    ├── benchmark_throughput.cpp
    ├── benchmark_memory.cpp
    └── benchmark_kernels.cpp
```

### Continuous Integration
```yaml
.github/workflows/
├── build.yml
├── test.yml
└── benchmark.yml
```

### Performance Regression Testing
- Run benchmarks on every commit
- Compare against baseline
- Alert on >10% regression

---

## Timeline Summary

| Sprint | Duration | Key Features | Performance Goal |
|--------|----------|--------------|-------------------|
| 1 | 2 weeks | Safety, portability | Baseline |
| 2 | 2 weeks | Performance optimization | +65% |
| 3 | 2 weeks | Modern features | +130% |
| 4 | 2 weeks | Advanced optimizations | +200% |
| 5 | 2 weeks | Portability & testing | Cross-platform |
| 6 | 2 weeks | Build system & API | Production ready |
| 7 | Ongoing | Out-of-the-box features | Cutting edge |

**Total**: 3 months to production-ready, state-of-the-art Vulkan LLM engine
