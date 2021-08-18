cmake_minimum_required(VERSION 3.10)

# Set the project name
project(main)

# Find required dependencies
find_package(Kokkos)
find_package(KokkosKernels REQUIRED)
include_directories(/usr/local/include)

# Add library targets
add_library(mesh mesh.cpp)
add_library(boundaryconditions boundaryconditions.cpp)
add_library(solver solver.cpp)

# Add executable target
add_executable(test test.cpp)

# Link targets to libraries
target_link_libraries(mesh Kokkos::kokkos)
target_link_libraries(boundaryconditions Kokkos::kokkos)
target_link_libraries(solver Kokkos::kokkos)
target_link_libraries(solver Kokkos::kokkoskernels)
target_link_libraries(test Kokkos::kokkos)
target_link_libraries(test Kokkos::kokkoskernels)
