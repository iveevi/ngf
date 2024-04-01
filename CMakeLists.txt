cmake_minimum_required(VERSION 3.20)

project(nsc CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(simplify      tools/main.cpp)

target_compile_options(simplify      PRIVATE -Wall -Wextra -Wpedantic)

include_directories(. glm)
