cmake_minimum_required(VERSION 3.16)
project(VulkanGUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
endif()

find_package(Vulkan REQUIRED COMPONENTS VK_KHR_surface VK_KHR_win32_surface)
find_package(glfw3 REQUIRED)
find_package(imgui REQUIRED)

add_executable(VulkanGUI WIN32
    main.cpp
)

target_include_directories(VulkanGUI PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIR}
    ${IMGUI_INCLUDE_DIR}
)

target_link_libraries(VulkanGUI PRIVATE
    Vulkan::Vulkan
    Vulkan::VulkanWin32
    glfw
    imgui
)

if(WIN32)
    target_link_options(VulkanGUI PRIVATE -mwindows)
endif()

file(COPY ${Vulkan_LIBRARIES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
