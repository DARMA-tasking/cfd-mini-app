#include <iostream>
#include <array>
#include <map>
#include <cmath>

#include <Kokkos_Core.hpp>

#include "mesh_chunk.h"
#include "parallel_mesh.h"
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
  uint64_t n_c_x = 35;
  uint64_t n_c_y = 35;
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

  // create parallel mesh
  uint64_t n_p = 3;
  uint64_t n_q = 2;
  ParallelMesh p_mesh(n_c_x, n_c_y, cell_size, 8, n_p, n_q);

  // check partition of cells
  std::map<std::array<uint64_t,2>, uint64_t> c_per_block = {};
  uint64_t c_mismatch = 0;
  for (uint64_t n = 0; n < n_c_y; n++)
    for (uint64_t m = 0; m < n_c_x; m++){
      LocalCoordinates loc = p_mesh.GlobalToLocalCellIndices(m, n);
      c_per_block[{loc.block[0], loc.block[1]}] ++;
      auto g = p_mesh.LocalToGlobalCellIndices(loc);
      if (m != g[0]){
	c_mismatch ++;
	std::cout << "** ERROR: "
		  << m << " != " << g[0] << "\n";
      }
      if (n != g[1]){
	c_mismatch ++;
	std::cout << "** ERROR: "
		  << n << " != " << g[1] << "\n";
      }
    }
  std::cout << "Mesh block cell counts in "
	    << n_p << "x" << n_q
	    << " partition of "
	    << n_c_x << "x" << n_c_y
	    << " = " << n_cells
	    << " cells:\n";
  uint64_t c_total = 0;
  uint64_t c_max = 0;
  for (const auto& it_c_per_block : c_per_block){
    auto c_in_block = it_c_per_block.second;
    c_total += c_in_block;
    if (c_in_block > c_max)
      c_max = c_in_block;
    std:: cout << "  block ( "
	       << it_c_per_block.first[0] << " ; "
	       << it_c_per_block.first[1] << " ): "
	       << c_in_block << "\n";
  }
  std::cout << "  grand total: " << c_total << "\n";
  std::cout << "  cell imbalance: "
	    << static_cast<double>(n_p * n_q * c_max) / c_total - 1.
	    << std::endl;
  std::cout << "  global -> local -> global cell coordinates mismatches: "
	    << c_mismatch << "\n\n";

  // check partition of points
  std::map<std::array<uint64_t,2>, uint64_t> p_per_block = {};
  uint64_t p_mismatch = 0;
  uint64_t n_p_x = p_mesh.get_n_points_x();
  uint64_t n_p_y = p_mesh.get_n_points_y();
  for (uint64_t n = 0; n < n_p_y; n++)
    for (uint64_t m = 0; m < n_p_x; m++){
      LocalCoordinates loc = p_mesh.GlobalToLocalPointIndices(m, n);
      p_per_block[{loc.block[0], loc.block[1]}] ++;
      auto g = p_mesh.LocalToGlobalPointIndices(loc);
      if (m != g[0]){
	p_mismatch ++;
	std::cout << "** ERROR: "
		  << m << " != " << g[0] << "\n";
      }
      if (n != g[1]){
	p_mismatch ++;
	std::cout << "** ERROR: "
		  << n << " != " << g[1] << "\n";
      }
    }
  std::cout << "Mesh block point counts in "
	    << n_p << "x" << n_q
	    << " partition of "
	    << n_p_x << "x" << n_p_y
	    << " = " << n_p_x * n_p_y
	    << " points:\n";
  uint64_t p_total = 0;
  uint64_t p_max = 0;
  for (const auto& it_p_per_block : p_per_block){
    auto p_in_block = it_p_per_block.second;
    p_total += p_in_block;
    if (p_in_block > p_max)
      p_max = p_in_block;
    std:: cout << "  block ( "
	       << it_p_per_block.first[0] << " ; "
	       << it_p_per_block.first[1] << " ): "
	       << p_in_block << "\n";
  }
  std::cout << "  grand total: " << p_total << "\n";
  std::cout << "  point imbalance: "
	    << static_cast<double>(n_p * n_q * p_max) / p_total - 1.
	    << std::endl;
  std::cout << "  global -> local -> global point coordinates mismatches: "
	    << p_mismatch << "\n\n";

  #ifdef OUTPUT_VTK_FILES
  // save results
  std::string file_name = p_mesh.write_vtm("p_test");
  std::cout << std::endl
	    << "Created VTK multi-block mesh file: \""
	    << file_name<<"\""
	    << std::endl;
  #endif

  // terminate cleanly
  std::cout << std::endl;
  return 0;
}
