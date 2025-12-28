cmake_minimum_required(VERSION 3.20)
project(VulkanGGUF VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2 /fp:fast")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4")  # Enable exceptions
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c -O3 -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread") # Add threading support
endif()

find_package(Vulkan REQUIRED)

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
    src/cpu_backend/activation_cpu.cpp
    src/cpu_backend/dequantize_cpu.cpp
)

set(VULKAN_BACKEND_SOURCES
    src/vulkan_backend/context.h
    src/vulkan_backend/context.cpp
    src/vulkan_backend/memory.h
    src/vulkan_backend/memory.cpp
    src/vulkan_backend/transfer.h
    src/vulkan_backend/transfer.cpp
    src/vulkan_backend/pipeline_cache.h
    src/vulkan_backend/pipeline_cache.cpp
    src/vulkan_backend/compute.h
    src/vulkan_backend/compute.cpp
)

set(INFERENCE_SOURCES
    src/inference/model.h
    src/inference/model.cpp
    src/inference/lru_cache.h
    src/inference/lru_cache.cpp
    src/inference/offload_manager.h
    src/inference/offload_manager.cpp
    src/inference/prefetch_engine.h
    src/inference/prefetch_engine.cpp
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

add_executable(test_core tests/test_core.cpp ${CORE_SOURCES})
target_link_libraries(test_core ${Vulkan_LIBRARIES})

enable_testing()
add_test(NAME CoreTests COMMAND test_core)
