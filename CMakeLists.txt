cmake_minimum_required(VERSION 3.20)

project(srnm CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)

add_executable(cellulize
	cellulize.cu
	
	imgui/imgui.cpp
	imgui/imgui_draw.cpp
	imgui/imgui_widgets.cpp
	imgui/imgui_tables.cpp
	imgui/backends/imgui_impl_glfw.cpp
	imgui/backends/imgui_impl_vulkan.cpp
)

target_link_libraries(cellulize
	glfw
	assimp
	OpenMP::OpenMP_CXX
	SPIRV
	Vulkan::Vulkan
	glslang::glslang
	glslang::glslang-default-resource-limits
)

include_directories(.
	glm
	imgui
)
