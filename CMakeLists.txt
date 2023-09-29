cmake_minimum_required(VERSION 3.20)

project(nsc CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glslang REQUIRED)

add_executable(rasterizer rasterizer/rasterizer.cu)

target_link_libraries(rasterizer
	assimp
	glfw
	SPIRV
	glslang::glslang-default-resource-limits
	OpenMP::OpenMP_CXX
	Vulkan::Vulkan
)

include_directories(. glm)
