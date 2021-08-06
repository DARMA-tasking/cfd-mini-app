#include<iostream>
#include<array>
#include<Kokkos_Core.hpp>
#include"mesh.cpp"

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  Mesh m(10, 10, 0.1);
  // m.set_origin(0.1, 0.5);
  // m.set_n_cells_x(10);
  // m.set_n_cells_y(10);
  // m.set_n_points_x(11);
  // m.set_n_points_y(11);

  auto o = m.get_origin();
  std::cout<<o[0]<<" "<<o[1]<<std::endl;
  auto c = m.index_to_cartesian(6, 5, 25);
  std::cout<<c[0]<<" "<<c[1]<<std::endl;
  auto i = m.cartesian_to_index(c[0], c[1], 5, 5);
  std::cout<<i<<std::endl;

  m.create_pressure_data_storage();
  m.set_pressure(0, 0, 5);
  double p = m.get_pressure(0, 0);
  std::cout<<p<<std::endl;

  m.create_velocity_data_storage();
  m.set_velocity_u(0, 0, 3.2);
  m.set_velocity_v(0, 0, 7.5);
  double u = m.get_velocity_u(0, 0);
  double v = m.get_velocity_v(0, 0);
  std::cout<<u<<", "<<v<<std::endl;
  return 0;
}
