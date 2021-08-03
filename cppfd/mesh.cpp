#include<iostream>
#include<array>
#include<cmath>
// #include <Kokkos_Core.hpp>

#include "mesh.h"

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

// int Mesh::get_cell_scalar(int i, int j)
// {
//   int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
//   if(k == -1)
//   {
//     return -1;
//   }else
//   {
//     return this->cell_scalars(k);
//   }
// }
//
// void Mesh::set_cell_scalar(int i, int j, double scalar)
// {
//   int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
//   if(k != -1)
//   {
//     this->cell_scalars(k) = scalar;
//   }
// }
