cmake_minimum_required(VERSION 3.20)

project(ngf CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_definitions("-DSHADERS_DIRECTORY=\"${CMAKE_SOURCE_DIR}/rasterizer/shaders\"")

find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)

add_library(imgui OBJECT
	thirdparty/imgui/imgui.cpp
	thirdparty/imgui/imgui_demo.cpp
	thirdparty/imgui/imgui_draw.cpp
	thirdparty/imgui/imgui_widgets.cpp
	thirdparty/imgui/imgui_tables.cpp
	thirdparty/imgui/backends/imgui_impl_glfw.cpp
	thirdparty/imgui/backends/imgui_impl_vulkan.cpp
	thirdparty/implot/implot.cpp
	thirdparty/implot/implot_items.cpp)

add_executable(testbed
	rasterizer/main.cpp
	rasterizer/mesh.cpp
	rasterizer/projection.cpp
	$<TARGET_OBJECTS:imgui>)

target_link_libraries(testbed
	assimp
	glfw
	SPIRV
	glslang::glslang
	glslang::glslang-default-resource-limits
	Vulkan::Vulkan)

include_directories(.
	thirdparty
	thirdparty/glm
	thirdparty/imgui
	thirdparty/implot)
