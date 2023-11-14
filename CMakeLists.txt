cmake_minimum_required(VERSION 3.20)

project(nsc CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 REQUIRED)

add_executable(simplify      tools/main.cpp)
add_executable(quadrangulate tools/quadrangulate.cpp)
add_executable(convert       tools/convert.cpp)
add_executable(headless      tools/headless.cpp)

target_compile_options(simplify      PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(quadrangulate PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(convert       PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(headless      PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(quadrangulate assimp)
target_link_libraries(convert       assimp)
target_link_libraries(headless      glfw)

include_directories(. glm)
