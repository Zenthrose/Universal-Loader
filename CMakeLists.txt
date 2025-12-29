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
elseif(UNIX AND NOT APPLE)
    find_package(Vulkan REQUIRED)
    find_library(Volk REQUIRED)
    find_library(VulkanHeaders REQUIRED)
elseif(UNIX AND APPLE)
    find_library(Vulkan REQUIRED)
    find_library(MoltenVK REQUIRED)
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
)

set(INFERENCE_SOURCES
    src/inference/model.h
    src/inference/model.cpp
    src/inference/model_v2.h
    src/inference/model_v2.cpp
    src/inference/tokenizer.h
    src/inference/tokenizer.cpp
    src/inference/bpe_tokenizer.h
    src/inference/bpe_tokenizer.cpp
    src/inference/inference_engine.h
    src/inference/inference_engine.cpp
    src/inference/lru_cache.h
    src/inference/lru_cache.cpp
    src/inference/offload_manager.h
    src/inference/offload_manager.cpp
    src/inference/prefetch_engine.h
    src/inference/prefetch_engine.cpp
    src/inference/tool_metadata_parser.h
    src/inference/tool_metadata_parser.cpp
    src/inference/tool_registry.h
    src/inference/tool_registry.cpp
    src/inference/tool_calling_engine.h
    src/inference/tool_calling_engine.cpp
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

set(API_SOURCES
    src/api/gguf_loader.h
    src/api/gguf_loader.cpp
)

add_library(gguf_api STATIC ${API_SOURCES} ${INFERENCE_SOURCES} ${CORE_SOURCES})
target_link_libraries(gguf_api ${Vulkan_LIBRARIES})

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
