cmake_minimum_required(VERSION 3.20)

project(nsc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_executable(simplify quads/main.cpp)
add_executable(quadrangulate quads/quadrangulate.cpp)
add_executable(convert quads/convert.cpp)

target_compile_options(quadrangulate PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(convert PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(quadrangulate assimp)
target_link_libraries(convert assimp)

include_directories(. glm)
