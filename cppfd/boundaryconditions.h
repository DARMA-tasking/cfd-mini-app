#pragma once
#include<array>
#include<string>
#include<cmath>
#include<map>

#include<Kokkos_Core.hpp>

#include"mesh.h"

class BoundaryConditions
{
  public:
    // constructor
    BoundaryConditions(Mesh& m, std::map<std::string, double> velocity_values);

    // set values at boundaries of fluid domain
    void apply_velocity_bc();

    // get maximum of velocity norms on boundary
    double get_velocity_bc_max_norm();

  private:
    // mesh object to be worked on
    Mesh mesh;

    // boundary condition velocity values
    double u_top, v_top, u_bot, v_bot, u_left, v_left, u_right, v_right;
};
