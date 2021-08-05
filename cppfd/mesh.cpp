#include<iostream>
#include<array>
#include<cmath>
#include<Kokkos_Core.hpp>

#include "mesh.h"

////////////////////////////////////////////////////////////////
// BASIC
////////////////////////////////////////////////////////////////

void Mesh::set_origin(double x, double y)
{
  this->O[0] = x;
  this->O[1] = y;
}

std::array<int,2> Mesh::index_to_cartesian(int k, int n, int nmax)
{
  if(k<0 || k>=nmax)
  {
    // Return invalid values when index is out of bounds
    return  {-1, -1};
  }else
  {
    std::div_t dv = std::div(k, n);
    return {dv.rem, dv.quot};
  }
}

int Mesh::cartesian_to_index(int i, int j, int ni, int nj)
{
  if(i<0 || i>=ni || j<0 || j>=nj)
  {
    // Return invalid value when coordinates are out of bounds
    return -1;
  }else
  {
    return j * ni + i;
  }
}

////////////////////////////////////////////////////////////////
// CELLS (PRESSURE)
////////////////////////////////////////////////////////////////

void Mesh::create_pressure_data_storage()
{
  const size_t NC = this->get_n_cells();
  Kokkos::View<double*> pressure_data("pressure", NC);
  this->pressure = pressure_data;
}

void Mesh::set_pressure(int i, int j, double scalar)
{
  int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k != -1)
  {
    this->pressure(k) = scalar;
  }
}

double Mesh::get_pressure(int i, int j)
{
  int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k == -1)
  {
    return std::nan("");
  }else
  {
    return this->pressure(k);
  }
}

////////////////////////////////////////////////////////////////
// POINTS (VELOCITY)
////////////////////////////////////////////////////////////////

void Mesh::create_velocity_data_storage()
{
  const size_t NP = this->get_n_points();
  Kokkos::View<double*[2]> velocity_data("velocity", NP);
  this->velocity = velocity_data;
}

void Mesh::set_velocity_u(int i, int j, double u)
{
  int k = this->cartesian_to_index(i, j, this->n_points_x, this->n_points_y);
  if(k != -1)
  {
    this->velocity(k, 0) = u;
  }
}

void Mesh::set_velocity_v(int i, int j, double v)
{
  int k = this->cartesian_to_index(i, j, this->n_points_x, this->n_points_y);
  if(k != -1)
  {
    this->velocity(k, 1) = v;
  }
}

double Mesh::get_velocity_u(int i, int j)
{
  int k = this->cartesian_to_index(i, j, this->n_points_x, this->n_points_y);
  if(k != -1)
  {
    return this->velocity(k, 0);
  }
}

double Mesh::get_velocity_v(int i, int j)
{
  int k = this->cartesian_to_index(i, j, this->n_points_x, this->n_points_y);
  if(k != -1)
  {
    return this->velocity(k, 1);
  }
}
