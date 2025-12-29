cmake_minimum_required(VERSION 3.20)
project(VulkanGGUF VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2 /fp:fast")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4")  # Enable exceptions
elseif(MINGW)
    # MinGW64-specific optimizations for FX-4130 and general x86_64
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c -O3 -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -static-libgcc -static-libstdc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-variable")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c -O3 -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread") # Add threading support
endif()

if(WIN32)
    find_package(Vulkan REQUIRED)
    set(VULKAN_INCLUDE_DIRS "${VULKAN_SDK_DIR}/include")
    
    if(MSVC)
        # MSVC uses .lib files
        set(VULKAN_LIBRARIES "${VULKAN_SDK_DIR}/lib/volk/volk.lib")
    elseif(MINGW)
        # MinGW64 uses .a files, need to detect properly
        find_library(VOLK_LIBRARY volk PATHS "${VULKAN_SDK_DIR}/Lib" "${VULKAN_SDK_DIR}/lib")
        find_library(VULKAN_LIBRARY vulkan-1 PATHS "${VULKAN_SDK_DIR}/Lib" "${VULKAN_SDK_DIR}/lib")
        
        if(VOLK_LIBRARY AND VULKAN_LIBRARY)
            set(VULKAN_LIBRARIES ${VOLK_LIBRARY} ${VULKAN_LIBRARY})
        else()
            # Fallback to system packages (MSYS2)
            find_library(VOLK_LIBRARY volk)
            find_library(VULKAN_LIBRARY vulkan-1)
            if(VOLK_LIBRARY AND VULKAN_LIBRARY)
                set(VULKAN_LIBRARIES ${VOLK_LIBRARY} ${VULKAN_LIBRARY})
            else()
                message(WARNING "Vulkan libraries not found. Using default vulkan-1.lib")
                set(VULKAN_LIBRARIES "${VULKAN_SDK_DIR}/Lib/vulkan-1.lib")
            endif()
        endif()
        
        message(STATUS "MinGW64 Vulkan libraries: ${VULKAN_LIBRARIES}")
    endif()
    
    # Enable Vulkan-Hpp for RAII and type safety with dynamic loading
    set(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1)
    add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
    add_compile_definitions(VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1)
    
    # Enable exceptions for better error handling
    set(VULKAN_HPP_NO_EXCEPTIONS OFF)
    set(VULKAN_HPP_NO_TO_STRING OFF)
    add_compile_definitions(VULKAN_HPP_ASSERT_ON_ASSERT=1)
    
    # Enable all Vulkan extensions we need
    add_compile_definitions(VK_ENABLE_BETA_EXTENSIONS=1)
elseif(UNIX AND NOT APPLE)
    find_package(Vulkan REQUIRED)
    find_library(Volk REQUIRED)
    find_library(VulkanHeaders REQUIRED)
    
    # Enable Vulkan-Hpp for Linux
    set(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1)
    add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
    add_compile_definitions(VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1)
    add_compile_definitions(VULKAN_HPP_ASSERT_ON_ASSERT=1)
    add_compile_definitions(VK_ENABLE_BETA_EXTENSIONS=1)
elseif(UNIX AND APPLE)
    find_library(Vulkan REQUIRED)
    find_library(MoltenVK REQUIRED)
    
    # Enable Vulkan-Hpp for macOS (MoltenVK support)
    set(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1)
    add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
    add_compile_definitions(VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1)
    add_compile_definitions(VULKAN_HPP_ENABLE_PORTABILITY_SUBSET=1)
    add_compile_definitions(VULKAN_HPP_ASSERT_ON_ASSERT=1)
    add_compile_definitions(VK_ENABLE_BETA_EXTENSIONS=1)
endif()

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)

set(CORE_SOURCES
    src/core/gguf_types.h
    src/core/gguf_parser.cpp
    src/core/gguf_parser.h
    src/core/tensor.cpp
    src/core/tensor.h
)

set(CPU_BACKEND_SOURCES
    src/cpu_backend/cpu_context.h
    src/cpu_backend/cpu_context.cpp
    src/cpu_backend/gemm_avx2.h
    src/cpu_backend/gemm_avx2.cpp
    src/cpu_backend/activation_cpu.h
    src/cpu_backend/activation_cpu.cpp
    src/cpu_backend/dequantize_cpu.cpp
)

set(VULKAN_BACKEND_SOURCES
    src/vulkan_backend/validation.h
    src/vulkan_backend/validation.cpp
    src/vulkan_backend/context.h
    src/vulkan_backend/context.cpp
    src/vulkan_backend/memory.h
    src/vulkan_backend/memory.cpp
    src/vulkan_backend/transfer.h
    src/vulkan_backend/transfer.cpp
    src/vulkan_backend/pipeline_cache.h
    src/vulkan_backend/pipeline_cache.cpp
    src/vulkan_backend/shader_compiler.h
    src/vulkan_backend/shader_compiler.cpp
    src/vulkan_backend/compute.h
    src/vulkan_backend/compute.cpp
    src/vulkan_backend/descriptors.h
    src/vulkan_backend/descriptors.cpp
    src/vulkan_backend/timeline_semaphores.h
    src/vulkan_backend/timeline_semaphores.cpp
    src/vulkan_backend/async_pipeline.h
    src/vulkan_backend/async_pipeline.cpp
    src/vulkan_backend/device_address_memory.h
    src/vulkan_backend/device_address_memory.cpp
    src/vulkan_backend/multi_gpu_manager.h
    src/vulkan_backend/multi_gpu_manager.cpp
    src/vulkan_backend/profiler.h
    src/vulkan_backend/profiler.cpp
)

set(INFERENCE_SOURCES
    src/inference/model.h
    src/inference/model.cpp
    src/inference/inference_engine.h
    src/inference/inference_engine.cpp
    src/inference/tokenizer.h
    src/inference/tokenizer.cpp
    src/inference/lru_cache.h
    src/inference/lru_cache.cpp
    src/inference/offload_manager.h
    src/inference/offload_manager.cpp
    src/inference/prefetch_engine.h
    src/inference/prefetch_engine.cpp
    src/inference/speculative_decoder.h
    src/inference/speculative_decoder.cpp
    src/inference/continuous_batcher.h
    src/inference/continuous_batcher.cpp
    src/inference/lora_manager.h
    src/inference/lora_manager.cpp
)

set(API_SOURCES
    src/api/inference_api.h
    src/api/inference_api.cpp
)

set(TEST_SOURCES
    tests/test_suite.h
    tests/test_suite.cpp
    tests/test_main.cpp
)

file(GLOB SHADERS
    "src/shaders/*.glsl"
)

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPIRV ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND glslangValidator -V ${SHADER} -o ${SPIRV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SPIRV_FILES ${SPIRV})
endforeach()

add_library(gguf_core STATIC ${CORE_SOURCES})
target_link_libraries(gguf_core ${Vulkan_LIBRARIES})

add_library(cpu_backend STATIC ${CPU_BACKEND_SOURCES} ${CORE_SOURCES})
target_link_libraries(cpu_backend ${Vulkan_LIBRARIES})

add_library(vulkan_backend STATIC ${VULKAN_BACKEND_SOURCES} ${CORE_SOURCES})
target_link_libraries(vulkan_backend ${Vulkan_LIBRARIES})

add_library(inference STATIC ${INFERENCE_SOURCES} ${CORE_SOURCES} ${VULKAN_BACKEND_SOURCES} ${CPU_BACKEND_SOURCES})
target_link_libraries(inference ${Vulkan_LIBRARIES})

set(GGUF_LOADER_SOURCES
    src/api/gguf_loader.h
    src/api/gguf_loader.cpp
)

add_library(gguf_api STATIC ${GGUF_LOADER_SOURCES} ${INFERENCE_SOURCES} ${CORE_SOURCES})
target_link_libraries(gguf_api ${Vulkan_LIBRARIES})

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(vulkangguf ${API_SOURCES} ${INFERENCE_SOURCES} ${CORE_SOURCES} ${VULKAN_BACKEND_SOURCES} ${CPU_BACKEND_SOURCES})
target_link_libraries(vulkangguf PRIVATE ${Vulkan_LIBRARIES})

add_executable(test_core tests/test_core.cpp ${CORE_SOURCES})
target_link_libraries(test_core ${Vulkan_LIBRARIES})

add_executable(test_gpu_integration tests/test_gpu_integration.cpp ${API_SOURCES} ${INFERENCE_SOURCES} ${CORE_SOURCES} ${VULKAN_BACKEND_SOURCES} ${CPU_BACKEND_SOURCES})
target_link_libraries(test_gpu_integration ${Vulkan_LIBRARIES})

add_executable(benchmark tools/benchmark.cpp ${API_SOURCES} ${INFERENCE_SOURCES} ${CORE_SOURCES} ${VULKAN_BACKEND_SOURCES} ${CPU_BACKEND_SOURCES})
target_link_libraries(benchmark ${Vulkan_LIBRARIES} cpu_backend)

add_executable(model_info tools/model_info.cpp ${CORE_SOURCES})
target_link_libraries(model_info ${Vulkan_LIBRARIES})

# MinGW64-specific linker optimizations
if(MINGW)
    # Add static linking flags for standalone executables
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    # Suppress MinGW64-specific warnings
    add_compile_options(-Wno-format)
endif()

enable_testing()
add_test(NAME CoreTests COMMAND test_core)
