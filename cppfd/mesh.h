#pragma once
#include<array>
#include<string>
#include<cstdio>

#include<Kokkos_Core.hpp>

class Mesh
{
  public:
    // delete default constructor
    Mesh() = delete;

    // initialization constructor
    Mesh(int n_x, int n_y, double cell_size)
      : n_cells_x{ n_x },
      n_cells_y{ n_y },
      O{0, 0},
      h{cell_size}
    {}


    // setter/getter for physical cell size
    void set_cell_size(double size){this->h = size;};
    double get_cell_size(){return this->h;};

    // setter/getter for physical origin member variable
    void set_origin(double x, double y);
    std::array<double,2> get_origin(){return this->O;};

    // getters for size characteristics of the mesh member variables
    int get_n_points(){return this->n_points;};
    int get_n_cells(){return this->n_cells;};

    // setters/getters for number of mesh cells in each dimension
    void set_n_cells_x(int n_x){this->n_cells_x = n_x;};
    void set_n_cells_y(int n_y){this->n_cells_y = n_y;};
    int get_n_cells_x(){return this->n_cells_x;};
    int get_n_cells_y(){return this->n_cells_y;};

    // only getters for number of points which depend on number of cells
    int get_n_points_x(){return this->n_cells_x + 1;};
    int get_n_points_y(){return this->n_cells_y + 1;};

    // coordinate systems converters
    std::array<int,2> index_to_cartesian(int k, int n, int nmax);
    int cartesian_to_index(int i, int j, int ni, int nj);

    // setters/getters for mesh data
    void set_pressure(int i, int j, double scalar);
    double get_pressure(int i, int j);

    void set_velocity_u(int i, int j, double u);
    void set_velocity_v(int i, int j, double v);
    double get_velocity_u(int i, int j);
    double get_velocity_v(int i, int j);

    // create Views with the correct extent for pressure and velocity data
    void create_pressure_data_storage();
    void create_velocity_data_storage();

  private:
    // physical origin f the mesh
    std::array<double,2> O;

    // dimension characteristics of the mesh
    int n_points, n_cells, n_cells_x, n_cells_y;

    // physical cell size
    double h;

    // mesh cell data
    Kokkos::View<double*> pressure = {};

    // mesh point data
    Kokkos::View<double*[2]> velocity = {};


};
