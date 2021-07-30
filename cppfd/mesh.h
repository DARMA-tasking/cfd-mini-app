#pragma once
#include<array>

class Mesh
{
  private:
    // physical origin f the mesh
    std::array<double,2> O;
  public:
    // int size, n_points, n_cells, h, n_cells_x, n_cells_y, n_points_x, n_points_y;
    // string cell_scalars_name, point_vectors_name;
    // std::array<int,2> index_to_cartesian()
    // setter/getter for physical origin member variable
    void set_origin(double x, double y);
    std::array<double,2> get_origin(){return this->O;};
    // void set_cell_scalars_name(string name);
    // void set_cell_scalar(int i, int j, int scalar);
    // void set_point_vectors_name(string name);
    // void set_point_vector(int i, int j, int vector[2]);
    // void set_point_vector_u(int i, int j, int u);
    // void set_point_vector_v(int i, int j, int v);

};
