#pragma once
#include <cstdio>
#include <array>
#include <string>
#include <map>

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
    MeshChunk(uint64_t n_x, uint64_t n_y, double cell_size,
	      const std::map<uint8_t, PointTypeEnum>& point_types);

    // only getters for unchangeable mesh characteristics
    uint64_t get_n_cells_x() const { return this->n_cells_x; }
    uint64_t get_n_cells_y() const { return this->n_cells_y; }
    uint64_t get_n_points_x() const { return this->n_cells_x + 1; }
    uint64_t get_n_points_y() const { return this->n_cells_y + 1; }
    double get_cell_size() const { return this->h; }

    // setter/getter for physical origin member variable
    void set_origin(const double x, const double y);
    std::array<double,2> get_origin() const {return this->origin; }

    // coordinate systems converters
    std::array<uint64_t,2> index_to_Cartesian(uint64_t k, uint64_t n, uint64_t nmax) const;
    uint64_t Cartesian_to_index(uint64_t i, uint64_t j, uint64_t ni, uint64_t nj) const;

    // setters/getters for mesh data
    void set_point_type(uint64_t i, uint64_t j, PointTypeEnum t);
    PointTypeEnum get_point_type(uint64_t i, uint64_t j) const;
    void set_pressure(uint64_t i, uint64_t j, double scalar);
    double get_pressure(uint64_t i, uint64_t j) const;
    void set_velocity_x(uint64_t i, uint64_t j, double u);
    void set_velocity_y(uint64_t i, uint64_t j, double v);
    double get_velocity_x(uint64_t i, uint64_t j) const;
    double get_velocity_y(uint64_t i, uint64_t j) const;

    // setter to assign new mesh point data
    void set_velocity(Kokkos::View<double**[2]> v) { this->velocity = v; }

    // setter to assign new mesh cell data
    void set_pressure(Kokkos::View<double*> p) { this->pressure = p; }

    // writer to VTK file
    void write_VTK(const std::string& file_name);

  private:
    // physical origin of the mesh block
    std::array<double,2> origin;

    // characteristic dimensions of the mesh
    uint64_t n_cells_x;
    uint64_t n_cells_y;

    // physical cell size
    double h = 1.;

    // mesh point types
    Kokkos::View<PointTypeEnum**> point_type = {};

    // mesh point data
    Kokkos::View<double**[2]> velocity = {};

    // mesh cell data
    Kokkos::View<double*> pressure = {};
};
