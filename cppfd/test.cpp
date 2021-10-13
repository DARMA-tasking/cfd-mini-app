#include <iostream>
#include <array>
#include <map>

#include <Kokkos_Core.hpp>

#include "mesh_chunk.cpp"
#include "parallel_mesh.cpp"
#include "boundary_conditions.cpp"
#include "solver.cpp"

int main(int argc, char** argv) {
  // handle Kokkos boilerplate
  Kokkos::ScopeGuard kokkos(argc, argv);

  // define input parameters
  double density = 1.225;
  double dynamic_viscosity = 0.3;
  double delta_t = 0.001;
  double t_final = 0.1;
  double max_C = 0.5;
  uint64_t n_cells = 35;
  std::cout << "Input parameters:"
	    << "\n  density: " << density
	    << "\n  dynamic_viscosity: " << dynamic_viscosity
	    << "\n  delta_t: " << delta_t
	    << "\n  t_final: " << t_final
	    << "\n  max_C: " << max_C
	    << "\n  n_cells:: " << n_cells
	    << "x" << n_cells << " = "
	    << n_cells * n_cells << "\n\n";

  // create parallel mesh
  ParallelMesh p_mesh(n_cells, n_cells, 1. / n_cells, 3, 2);

  // create mesh
  std::map<uint8_t, PointTypeEnum> point_types = {
    { 4, PointTypeEnum::BOUNDARY },
    { 6, PointTypeEnum::SHARED_OWNED },
    { 7, PointTypeEnum::GHOST },
    { 5, PointTypeEnum::SHARED_OWNED },
    { 0, PointTypeEnum::BOUNDARY },
    { 1, PointTypeEnum::BOUNDARY },
    { 3, PointTypeEnum::GHOST },
    { 2, PointTypeEnum::SHARED_OWNED }
  };
  MeshChunk mesh(n_cells, n_cells, 1. / n_cells, point_types);

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
  BoundaryConditions b_c(mesh, velocity_values);

  // run numerical scheme
  Solver solver(mesh, b_c, delta_t, t_final, density, dynamic_viscosity, max_C, 1);
  solver.solve(Solver::stopping_point::NONE, Solver::linear_solver::GAUSS_SEIDEL, Solver::adaptative_time_step::ON);

  // save results
  std::string file_name = mesh.write_vti("test");
  std::cout<< std::endl
	   << "Created VTK regular mesh file: \""
	   << file_name<<"\""
	   << std::endl;

  auto names = p_mesh.write_vti("p_test_");
  std::cout<< std::endl
	   << "Created VTK regular mesh chunk files:"
	   << std::endl;
  for (const auto& it_names : names){
    std::cout<< "  \""
	     << it_names.second
	     << '\"'
	     << std::endl;
  }

  // terminate cleanly
  std::cout<< std::endl;
  return 0;
}
