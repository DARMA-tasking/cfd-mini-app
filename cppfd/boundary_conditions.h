#pragma once
#include <array>
#include <string>
#include <cmath>
#include <map>

#include"mesh_chunk.h"

class BoundaryConditions
{
  public:
  BoundaryConditions(std::shared_ptr<MeshChunk> m, std::map<std::string, double> velocity_values)
      : mesh_chunk(m)
      , v_x_t(velocity_values["v_x_t"])
      , v_y_t(velocity_values["v_y_t"])
      , v_x_b(velocity_values["v_x_b"])
      , v_y_b(velocity_values["v_y_b"])
      , v_x_l(velocity_values["v_x_l"])
      , v_y_l(velocity_values["v_y_l"])
      , v_x_r(velocity_values["v_x_r"])
      , v_y_r(velocity_values["v_y_r"])
      {}

    // set values at boundaries of fluid domain
    void apply_velocity_bc();

    // get maximum of velocity norms on boundary
    double get_velocity_bc_max_norm();

  private:
    // reference to mesh onto which boundary conditions apply
    std::shared_ptr<MeshChunk> mesh_chunk;

    // boundary condition velocity values
    double v_x_t = 0;
    double v_y_t = 0;
    double v_x_b = 0;
    double v_y_b = 0;
    double v_x_l = 0;
    double v_y_l = 0;
    double v_x_r = 0;
    double v_y_r = 0;
};
