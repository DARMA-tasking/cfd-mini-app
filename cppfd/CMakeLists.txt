cmake_minimum_required(VERSION 3.10)

# Set the project name
project(main)

# Find required dependencies
find_package(Kokkos)
include_directories(/usr/local/include)

# Add library targets
add_library(mesh mesh.cpp)

# Add executable target
add_executable(test test.cpp)

# Link targets to libraries
target_link_libraries(mesh Kokkos::kokkos)
