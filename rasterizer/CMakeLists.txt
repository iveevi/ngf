cmake_minimum_required(VERSION 3.20)

project(nsc CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# MAx CUDA optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo -Xptxas -O3")

# Set architecture to 8.0+
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)

# ImGui object library
add_library(imgui OBJECT
	imgui/imgui.cpp
	imgui/imgui_demo.cpp
	imgui/imgui_draw.cpp
	imgui/imgui_widgets.cpp
	imgui/imgui_tables.cpp
	imgui/backends/imgui_impl_glfw.cpp
	imgui/backends/imgui_impl_vulkan.cpp)

add_executable(testbed
	main.cpp
	$<TARGET_OBJECTS:imgui>)

target_link_libraries(testbed
	assimp
	glfw
	SPIRV
	glslang::glslang
	glslang::glslang-default-resource-limits
	OpenMP::OpenMP_CXX
	Vulkan::Vulkan)

include_directories(. ../thirdparty/glm imgui)
