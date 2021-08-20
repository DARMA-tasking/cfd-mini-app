#pragma once
#include <array>
#include <string>
#include <cmath>
#include <map>

#include <Kokkos_Core.hpp>

#include"mesh.h"

class BoundaryConditions
{
  public:
    BoundaryConditions(Mesh& m, std::map<std::string, double> velocity_values)
      : mesh(m)
      , u_top(velocity_values["u_top"])
      , v_top(velocity_values["v_top"])
      , u_bot(velocity_values["u_bot"])
      , v_bot(velocity_values["v_bot"])
      , u_left(velocity_values["u_left"])
      , v_left(velocity_values["v_left"])
      , u_right(velocity_values["u_right"])
      , v_right(velocity_values["v_right"])
      {}

    // set values at boundaries of fluid domain
    void apply_velocity_bc();

    // get maximum of velocity norms on boundary
    double get_velocity_bc_max_norm();

  private:
    // mesh object to be worked on
    Mesh mesh;

    // boundary condition velocity values
    double u_top = 0;
    double v_top = 0;
    double u_bot = 0;
    double v_bot = 0;
    double u_left = 0;
    double v_left = 0;
    double u_right = 0;
    double v_right = 0;
};
