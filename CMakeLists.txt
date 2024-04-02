cmake_minimum_required(VERSION 3.20)

project(nsc CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# Mxx CUDA optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo -Xptxas -O3")

add_definitions("-DSHADERS_DIRECTORY=\"${CMAKE_SOURCE_DIR}/rasterizer/shaders\"")

# Set architecture to 8.0+
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)
find_package(OpenCV REQUIRED)

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
	rasterizer/util.cpp
	rasterizer/pipeline.cpp
	$<TARGET_OBJECTS:imgui>)

target_link_libraries(testbed
	assimp
	glfw
	SPIRV
	glslang::glslang
	glslang::glslang-default-resource-limits
	OpenMP::OpenMP_CXX
	Vulkan::Vulkan
	avcodec
	swscale
	avdevice
	avformat
	avfilter
	avutil
	${OpenCV_LIBS})

include_directories(.
	thirdparty
	thirdparty/glm
	thirdparty/imgui
	thirdparty/implot)