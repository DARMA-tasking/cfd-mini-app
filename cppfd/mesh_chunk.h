#pragma once
#include <array>
#include <string>
#include <map>
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
    MeshChunk(int n_x, int n_y, double cell_size,
	      const std::map<std::string, PointTypeEnum>& point_types);

    // setter/getter for physical cell size
    void set_cell_size(double size) { this->h = size; }
    double get_cell_size() const { return this->h; }

    // setter/getter for physical origin member variable
    void set_origin(double x, double y);
    std::array<double,2> get_origin() const {return this->origin; }

    // setters/getters for number of mesh cells in each dimension
    void set_n_cells_x(int n_x) { this->n_cells_x = n_x; }
    void set_n_cells_y(int n_y) { this->n_cells_y = n_y; }
    int get_n_cells_x() const { return this->n_cells_x; }
    int get_n_cells_y() const { return this->n_cells_y; }

    // only getters for number of points which depend on number of cells
    int get_n_points_x() const {return this->n_cells_x + 1; } 
    int get_n_points_y() const { return this->n_cells_y + 1; }

    // coordinate systems converters
    std::array<uint64_t,2> index_to_Cartesian(uint64_t k, uint64_t n, uint64_t nmax) const;
    uint64_t Cartesian_to_index(uint64_t i, uint64_t j, uint64_t ni, uint64_t nj) const;

    // setters/getters for mesh data
    void set_point_type(int i, int j, PointTypeEnum t);
    PointTypeEnum get_point_type(int i, int j) const;
    void set_pressure(int i, int j, double scalar);
    double get_pressure(int i, int j) const;
    void set_velocity_x(int i, int j, double u);
    void set_velocity_y(int i, int j, double v);
    double get_velocity_x(int i, int j) const;
    double get_velocity_y(int i, int j) const;

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
