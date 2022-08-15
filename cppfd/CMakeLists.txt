cmake_minimum_required(VERSION 3.15)

# Set the project name
project(cfd-mini-app)

# Setting configure file
configure_file(cfd_config.h.in cfd_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Setting options
option(OUTPUT_VTK_FILES "When OFF allows for a non-VTK build" OFF)
option(USE_MPI "When OFF allows for a non-MPI build" OFF)

# Find required dependencies
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)

if(OUTPUT_VTK_FILES)
  find_package(VTK REQUIRED COMPONENTS vtkCommonCore vtkIOXML)
endif()

if(USE_MPI)
  find_package(MPI REQUIRED)
endif()

find_package(GTest CONFIG REQUIRED)

# Try to find ccache to speed up compilation
find_program(ccache_binary ccache)
if (ccache_binary)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${ccache_binary}")
endif()

# Add library targets
add_library(mesh_chunk mesh_chunk.cpp)
add_library(parallel_mesh parallel_mesh.cpp)
add_library(boundary_conditions boundary_conditions.cpp)
add_library(solver solver.cpp)

# Add executable target
add_executable(test test.cpp)
add_executable(parallel_test parallel_test.cpp)
add_executable(allTests allTests.cpp)

message(STATUS "VTK libraries: ${VTK_LIBRARIES}")

# Link targets to libraries
if (${USE_MPI} AND MPI_FOUND)
    MESSAGE(STATUS "MPI found. Enabling MPI.")
    target_link_libraries(mesh_chunk MPI::MPI_CXX)
    target_link_libraries(parallel_mesh MPI::MPI_CXX)
    target_link_libraries(boundary_conditions MPI::MPI_CXX)
    target_link_libraries(solver MPI::MPI_CXX)
    target_link_libraries(test MPI::MPI_CXX)
    target_link_libraries(parallel_test MPI::MPI_CXX)
    target_link_libraries(allTests MPI::MPI_CXX)
elseif(${USE_MPI} AND NOT MPI_FOUND)
    MESSAGE(WARNING "MPI not found. Disabling MPI.")
endif()
target_link_libraries(mesh_chunk boundary_conditions Kokkos::kokkos ${VTK_LIBRARIES})
target_link_libraries(parallel_mesh mesh_chunk Kokkos::kokkos ${VTK_LIBRARIES})
target_link_libraries(boundary_conditions Kokkos::kokkos ${VTK_LIBRARIES})
target_link_libraries(solver parallel_mesh Kokkos::kokkos Kokkos::kokkoskernels ${VTK_LIBRARIES})
target_link_libraries(test solver mesh_chunk Kokkos::kokkos Kokkos::kokkoskernels ${VTK_LIBRARIES})
target_link_libraries(parallel_test solver Kokkos::kokkos Kokkos::kokkoskernels ${VTK_LIBRARIES})
target_link_libraries(allTests solver parallel_mesh Kokkos::kokkos Kokkos::kokkoskernels ${VTK_LIBRARIES} GTest::gtest)
