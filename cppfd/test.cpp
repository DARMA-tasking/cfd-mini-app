#include<iostream>
#include<array>
#include<map>
#include<Kokkos_Core.hpp>

#include"mesh.cpp"
#include"boundaryconditions.cpp"
#include"solver.cpp"

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  double density = 1.225;
  double dynamic_viscosity = 0.001;
  double delta_t = 0.001;
  double t_final = 0.02;
  double max_C = 0.5;
  Mesh mesh(10, 10, 0.1);


  auto o = mesh.get_origin();
  std::cout<<o[0]<<" "<<o[1]<<std::endl;
  auto c = mesh.index_to_cartesian(6, 5, 25);
  std::cout<<c[0]<<" "<<c[1]<<std::endl;
  auto i = mesh.cartesian_to_index(c[0], c[1], 5, 5);
  std::cout<<i<<std::endl;

  mesh.set_pressure(0, 0, 5);
  double p = mesh.get_pressure(0, 0);
  std::cout<<p<<std::endl;

  mesh.set_velocity_u(0, 0, 3.2);
  mesh.set_velocity_v(0, 0, 7.5);
  double u = mesh.get_velocity_u(0, 0);
  double v = mesh.get_velocity_v(0, 0);
  std::cout<<u<<", "<<v<<std::endl;

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
  solver.solve();
  std::cout<<mesh.get_velocity_u(5, 10)<<std::endl;

  return 0;
}
