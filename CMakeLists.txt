cmake_minimum_required(VERSION 3.20)

project(nsc CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "-fpermissive")

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(polyscope)

add_executable(nsc
	nsc.cu
	closest_point.cu
	viewer.cpp
	
	imgui/imgui.cpp
	imgui/imgui_draw.cpp
	imgui/imgui_widgets.cpp
	imgui/imgui_tables.cpp
	imgui/backends/imgui_impl_glfw.cpp
	imgui/backends/imgui_impl_vulkan.cpp
)

add_executable(decimate decimate.cpp)
add_executable(opt opt.cu casdf/casdf.cu)

add_executable(reshape reshape.cu casdf/casdf.cu)
add_executable(measure measure.cu casdf/casdf.cu)
add_executable(rasterizer rasterizer/rasterizer.cu)
add_executable(to_quads to_quads.cpp)

target_link_libraries(nsc
	glfw
	cusparse
	assimp
	OpenMP::OpenMP_CXX
	SPIRV
	Vulkan::Vulkan
	glslang::glslang
	glslang::glslang-default-resource-limits
)

target_link_libraries(decimate
	polyscope
	assimp
	OpenMP::OpenMP_CXX
)

target_link_libraries(opt
	polyscope
	assimp
	OpenMP::OpenMP_CXX
	SPIRV
	Vulkan::Vulkan
	glslang::glslang
	glslang::glslang-default-resource-limits
)

target_link_libraries(reshape
	polyscope
	assimp
	OpenMP::OpenMP_CXX
	SPIRV
	Vulkan::Vulkan
	glslang::glslang
	glslang::glslang-default-resource-limits
)

target_link_libraries(measure
	assimp
	OpenMP::OpenMP_CXX
)

target_link_libraries(rasterizer
	assimp
	polyscope
	OpenMP::OpenMP_CXX
)

target_link_libraries(to_quads assimp)

include_directories(.
	glm
	imgui
	libigl/include
	indicators/include
	${EIGEN3_INCLUDE_DIR}
)
