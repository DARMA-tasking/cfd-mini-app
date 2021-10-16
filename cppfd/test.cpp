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
  uint64_t n_p = 3;
  uint64_t n_q = 2;
  ParallelMesh p_mesh(n_cells, n_cells, 1. / n_cells, n_p, n_q);
  std::map<std::array<uint64_t,2>, uint64_t> block_counts = {};
  for (uint64_t n = 0; n < n_cells; n++)
    for (uint64_t m = 0; m < n_cells; m++){
      LocalCoordinates loc = p_mesh.GlobalToLocalCellIndices(m, n);
      block_counts[{loc.block[0], loc.block[1]}] ++;
      auto g = p_mesh.LocalToGlobalCellIndices(loc);
      if (m != g[0])
	std::cout << "** ERROR: "
		  << m << " != " << g[0] << "\n";
      if (n != g[1])
	std::cout << "** ERROR: "
		  << n << " != " << g[1] << "\n";
    }
  std::cout << "Mesh block counts in "
	    << n_p << " x " << n_q
	    << " partition of "
	    << n_cells << "x" << n_cells
	    << " parallel mesh:\n";
  uint64_t c_total = 0;
  for (const auto& it_block_counts : block_counts){
    auto n_in_block = it_block_counts.second;
    c_total += n_in_block;
    std:: cout << "  block ( "
	       << it_block_counts.first[0] << " ; "
	       << it_block_counts.first[1] << " ): "
	       << n_in_block << "\n";
  }
  std::cout << "  grand total: " << c_total << "\n\n";

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
	   << "Created VTK structured grid file: \""
	   << file_name<<"\""
	   << std::endl;

  file_name = p_mesh.write_vtm("p_test");
  std::cout<< std::endl
	   << "Created VTK multi-block mesh file: \""
	   << file_name<<"\""
	   << std::endl;

  // terminate cleanly
  std::cout<< std::endl;
  return 0;
}
