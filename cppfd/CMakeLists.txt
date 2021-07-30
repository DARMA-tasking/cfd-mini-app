cmake_minimum_required(VERSION 3.10)

# set the project name
project(main)

add_library(mesh mesh.cpp)

# add the executable
add_executable(test test.cpp)
