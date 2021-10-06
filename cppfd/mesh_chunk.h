#pragma once
#include <array>
#include <string>
#include <cstdio>

#include <Kokkos_Core.hpp>

enum struct PointTypeEnum : int8_t {
  INTERIOR = 0,
  BOUNDARY = 1,
  SHARED_OWNED = 2,
  GHOST = 3,
  INVALID = 4
};

class MeshChunk
{
  public:
    MeshChunk(int n_x, int n_y, double cell_size)
      : n_cells_x(n_x)
      , n_cells_y(n_y)
      , origin{0, 0}
      , h(cell_size)
    {
      // instantiate internal containers
      this->point_type = Kokkos::
	View<PointTypeEnum**>("type", n_x + 1, n_y + 1);
      this->pressure = Kokkos::
	View<double*>("pressure", n_x * n_y);
      this->velocity = Kokkos::
	View<double**[2]>("velocity", n_x + 1, n_y + 1);
    }

    // setter/getter for physical cell size
    void set_cell_size(double size) { this->h = size; }
    double get_cell_size() { return this->h; }

    // setter/getter for physical origin member variable
    void set_origin(double x, double y);
    std::array<double,2> get_origin() {return this->origin; }

    // setters/getters for number of mesh cells in each dimension
    void set_n_cells_x(int n_x) { this->n_cells_x = n_x; }
    void set_n_cells_y(int n_y) { this->n_cells_y = n_y; }
    int get_n_cells_x() { return this->n_cells_x; }
    int get_n_cells_y() { return this->n_cells_y; }

    // only getters for number of points which depend on number of cells
    int get_n_points_x() {return this->n_cells_x + 1; }
    int get_n_points_y() { return this->n_cells_y + 1; }

    // coordinate systems converters
    std::array<int,2> index_to_cartesian(int k, int n, int nmax);
    int cartesian_to_index(int i, int j, int ni, int nj);

    // setters/getters for mesh data
    void set_point_type(int i, int j, PointTypeEnum t);
    PointTypeEnum get_point_type(int i, int j);
    void set_pressure(int i, int j, double scalar);
    double get_pressure(int i, int j);
    void set_velocity_x(int i, int j, double u);
    void set_velocity_y(int i, int j, double v);
    double get_velocity_x(int i, int j);
    double get_velocity_y(int i, int j);

    // setter to assign new mesh point data
    void set_velocity(Kokkos::View<double**[2]> v) { this->velocity = v; }

    // setter to assign new mesh cell data
    void set_pressure(Kokkos::View<double*> p) { this->pressure = p; }

    // VTK visualization file writer
    void write_vtk(std::string file_name);

  private:
    // physical origin f the mesh
    std::array<double,2> origin;

    // dimension characteristics of the mesh
    int n_cells_x = 3;
    int n_cells_y = 3;
    int n_points = 16;
    int n_cells = 9;

    // physical cell size
    double h = 1;

    // mesh point types
    Kokkos::View<PointTypeEnum**> point_type = {};

    // mesh point data
    Kokkos::View<double**[2]> velocity = {};

    // mesh cell data
    Kokkos::View<double*> pressure = {};
};
