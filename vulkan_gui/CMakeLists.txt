cmake_minimum_required(VERSION 3.10)
project(VulkanGUI)

set(CMAKE_CXX_STANDARD 17)

find_package(Qt5 REQUIRED COMPONENTS Widgets Gui)

find_package(Vulkan REQUIRED)

if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mwindows")
endif()

add_executable(vulkan_gui main.cpp)

target_link_libraries(vulkan_gui Qt5::Widgets Qt5::Gui Vulkan::Vulkan)