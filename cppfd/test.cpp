#include <iostream>
#include <array>
#include <map>
#include <cmath>

#include <Kokkos_Core.hpp>

#include "mesh_chunk.h"
#include "boundary_conditions.h"
#include "solver.h"

int main(int argc, char** argv) {
  // handle Kokkos boilerplate
  Kokkos::ScopeGuard kokkos(argc, argv);

  // define input parameters
  double density = 1.225;
  double dynamic_viscosity = 0.3;
  double delta_t = 0.001;
  double t_final = 0.1;
  double max_C = 0.5;
  uint64_t n_c_x = 32;
  uint64_t n_c_y = 32;
  uint64_t n_parallel_meshes_x = 1;
  uint64_t n_parallel_meshes_y = 1;
  uint64_t n_colors_x = 3;
  uint64_t n_colors_y = 3;
  std::cout << "Input parameters:"
	    << "\n  density: " << density
	    << "\n  dynamic viscosity: " << dynamic_viscosity
	    << "\n  initial time-step: " << delta_t
	    << "\n  final time: " << t_final
	    << "\n  max_C: " << max_C << "\n";
  uint64_t n_cells = n_c_x * n_c_y;
  double cell_size = 1. / sqrt(n_c_x * n_c_y);
  std::cout << "Derived parameters:"
	    << "\n  number of cells: " << n_c_x << "x" << n_c_y
	    << " = " << n_cells
	    << "\n  cell size: " << cell_size << "\n\n";

  // create mesh
  std::map<PointIndexEnum, PointTypeEnum> point_types = {
    {PointIndexEnum::CORNER_0, PointTypeEnum::BOUNDARY},
    {PointIndexEnum::CORNER_1, PointTypeEnum::BOUNDARY},
    {PointIndexEnum::CORNER_2, PointTypeEnum::BOUNDARY},
    {PointIndexEnum::CORNER_3, PointTypeEnum::BOUNDARY},
    {PointIndexEnum::EDGE_0, PointTypeEnum::BOUNDARY},
    {PointIndexEnum::EDGE_1, PointTypeEnum::BOUNDARY},
    {PointIndexEnum::EDGE_2, PointTypeEnum::BOUNDARY},
    {PointIndexEnum::EDGE_3, PointTypeEnum::BOUNDARY}
  };

  // define boundary conditions
  std::map<std::string, double> velocity_values = {
    {"v_x_t", 1.0},
    {"v_y_t", 0.0},
    {"v_x_b", 0.0},
    {"v_y_b", 0.0},
    {"v_x_l", 0.0},
    {"v_y_l", 0.0},
    {"v_x_r", 0.0},
    {"v_y_r", 0.0}
  };

  ParallelMesh pmesh(0, n_c_x, n_c_y,
                        cell_size, n_colors_x, n_colors_y, static_cast<int>(LocationIndexEnum::SINGLE),
                        n_colors_x,
                        n_colors_y,
                        velocity_values,
                        0, 0,
                        0, 0);

  auto mesh = std::make_shared<MeshChunk>
    (&pmesh, n_c_x, n_c_y, cell_size, point_types,
    n_parallel_meshes_x * n_colors_x, n_parallel_meshes_y * n_colors_y);


  BoundaryConditions b_c(mesh, velocity_values);

  // run numerical scheme
  Solver solver(mesh, velocity_values, b_c, delta_t, t_final, density, dynamic_viscosity, max_C, 1, n_c_x, n_c_y, cell_size, n_parallel_meshes_x, n_parallel_meshes_y, n_colors_x, n_colors_y);
  solver.solve(Solver::stopping_point::NONE, Solver::linear_solver::GAUSS_SEIDEL, Solver::adaptative_time_step::ON);

  #ifdef OUTPUT_VTK_FILES
  // save results
  std::string file_name = mesh->write_vti("test");
  std::cout << std::endl
	    << "Created VTK structured grid file: \""
	    << file_name<<"\""
	    << std::endl;
  #endif

  // terminate cleanly
  std::cout << std::endl;
  return 0;
}
