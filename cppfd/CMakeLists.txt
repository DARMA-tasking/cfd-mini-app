cmake_minimum_required(VERSION 3.15)

# Set the project name
project(cfd-mini-app)

# Find required dependencies
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

find_package(VTK REQUIRED COMPONENTS vtkCommonCore vtkIOXML)

find_package(GTest CONFIG REQUIRED)

# Try to find ccache to speed up compilation
find_program(ccache_binary ccache)
if (ccache_binary)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${ccache_binary}")
endif()

# Add library targets
add_library(mesh_chunk mesh_chunk.cpp)
add_library(boundary_conditions boundary_conditions.cpp)
add_library(solver solver.cpp)

# Add executable target
add_executable(test test.cpp)
add_executable(allTests allTests.cpp)

# message(STATUS "libs: ${VTK_LIBRARIES}")

# Link targets to libraries
target_link_libraries(mesh_chunk Kokkos::kokkos ${VTK_LIBRARIES})
target_link_libraries(boundary_conditions Kokkos::kokkos)
target_link_libraries(solver Kokkos::kokkos Kokkos::kokkoskernels)
target_link_libraries(test Kokkos::kokkos Kokkos::kokkoskernels ${VTK_LIBRARIES})
target_link_libraries(allTests Kokkos::kokkos Kokkos::kokkoskernels ${VTK_LIBRARIES} GTest::gtest)
