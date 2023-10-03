cmake_minimum_required(VERSION 3.20)

project(nsc CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glslang REQUIRED)

# ImGui object library
add_library(imgui OBJECT
	imgui/imgui.cpp
	imgui/imgui_demo.cpp
	imgui/imgui_draw.cpp
	imgui/imgui_widgets.cpp
	imgui/imgui_tables.cpp
	imgui/backends/imgui_impl_glfw.cpp
	imgui/backends/imgui_impl_vulkan.cpp
)

file(GLOB_RECURSE SOURCES "rasterizer/*.cu")

add_executable(rasterizer ${SOURCES} $<TARGET_OBJECTS:imgui>)

target_link_libraries(rasterizer
	assimp
	glfw
	SPIRV
	glslang::glslang-default-resource-limits
	OpenMP::OpenMP_CXX
	Vulkan::Vulkan
)

include_directories(. glm imgui)
