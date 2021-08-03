#pragma once
#include<array>
#include<string>

// #include <Kokkos_Core.hpp>

class Mesh
{
  private:
    // physical origin f the mesh
    std::array<double,2> O;
    // dimension characteristics of the mesh
    int n_points, n_cells, n_cells_x, n_cells_y, n_points_x, n_points_y;
    // physical cell size
    double h;
    // mesh data names
    std::string cell_scalars_name, point_vectors_name;
    // // mesh cell data
    // Kokkos::View<double*> cell_scalars("cell_scalars_data", NCELLS);
    // // mesh point data
    // Kokkos::View<double*[2]> point_vectors("point_vectors_data", NPOINTS);
  public:
    // setter/getter for physical origin member variable
    void set_origin(double x, double y);
    std::array<double,2> get_origin(){return this->O;};
    // getters for size characteristics of the mesh member variables
    int get_n_points(){return this->n_points;};
    int get_n_cells(){return this->n_cells;};
    // setters/getters for dimension characteristics of the mesh member variables
    void set_n_cells_x(int n_x){this->n_cells_x = n_x;};
    void set_n_cells_y(int n_y){this->n_cells_y = n_y;};
    int get_n_cells_x(){return this->n_cells_x;};
    int get_n_cells_y(){return this->n_cells_y;};
    // coordinate systems converters
    std::array<int,2> index_to_cartesian(int k, int n, int nmax);
    int cartesian_to_index(int i, int j, int ni, int nj);
    // // setters/getters for mesh data
    // int get_cell_scalar(int i, int j);
    // void set_cell_scalar(int i, int j, double scalar);
};
