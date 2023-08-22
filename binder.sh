#!/bin/bash

PYBIND11_INCLUDE=$(python3 -m pybind11 --includes)
PYBIND11_LIB=mesh$(python3-config --extension-suffix)

INCLUDE='-I glm'

echo $PYBIND11_INCLUDE
echo $PYBIND11_LIB

clang++ -O3 -Wall -shared -std=c++17 -fPIC $INCLUDE $PYBIND11_INCLUDE bindings/mesh.cpp -o $PYBIND11_LIB -lassimp
