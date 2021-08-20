#include <iostream>
#include <array>
#include <map>
#include <Kokkos_Core.hpp>

#include "mesh.cpp"
#include "boundaryconditions.cpp"
#include "solver.cpp"

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  double density = 1.225;
  double dynamic_viscosity = 0.3;
  double delta_t = 0.001;
  double t_final = 0.1;
  double max_C = 0.5;
  Mesh mesh(10, 10, 0.1);

  std::map<std::string, double> velocity_values = {
                                                  {"u_top", 1.0},
                                                  {"v_top", 0.0},
                                                  {"u_bot", 0.0},
                                                  {"v_bot", 0.0},
                                                  {"u_left", 0.0},
                                                  {"v_left", 0.0},
                                                  {"u_right", 0.0},
                                                  {"v_right", 0.0}
                                                };

  BoundaryConditions b_c(mesh, velocity_values);
  Solver solver(mesh, b_c, delta_t, t_final, density, dynamic_viscosity, max_C, 1);
  solver.solve(Solver::stopping_point::NONE, Solver::linear_solver::CONJUGATE_GRADIENT);

  mesh.write_vtk("test.vti");

  return 0;
}
