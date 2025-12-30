cmake_minimum_required(VERSION 3.10)
project(VulkanGUI)

set(CMAKE_CXX_STANDARD 17)

find_package(Qt5 REQUIRED COMPONENTS Widgets Gui Core)

find_package(Vulkan REQUIRED)

if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mwindows")
endif()

if(WIN32 AND VULKAN_SDK_DIR)
    # Copy Vulkan dll
    file(COPY "${VULKAN_SDK_DIR}/Lib/vulkan-1.dll" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_executable(vulkan_gui main.cpp)

target_link_libraries(vulkan_gui Qt5::Widgets Qt5::Gui Qt5::Core Vulkan::Vulkan inference)