#pragma once
#include "cfd_config.h"
#include <cstdio>
#include <array>
#include <string>
#include <map>

#include <Kokkos_Core.hpp>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef OUTPUT_VTK_FILES
#include <vtkSmartPointer.h>
#endif

enum struct PointIndexEnum : int8_t {
  CORNER_0 = 0, // Bottom Left
  CORNER_1 = 1, // Bottom Right
  CORNER_2 = 2, // Top Right
  CORNER_3 = 3, // Top Left
  EDGE_0 = 4, // Bottom
  EDGE_1 = 5, // Right
  EDGE_2 = 6, // Top
  EDGE_3 = 7, // Left
  INTERIOR = 8 // Interior
};

enum struct PointTypeEnum : int8_t {
  INTERIOR = 0,
  BOUNDARY = 1,
  SHARED_OWNED = 2,
  GHOST = 3,
  INVALID = 4
};

enum struct Border : int8_t {
  BOTTOM = 0,
  LEFT = 1,
  RIGHT = 2,
  TOP = 3
};

class ParallelMesh;
#ifdef OUTPUT_VTK_FILES
class vtkUniformGrid;
#endif

class MeshChunk
{
  public:
    MeshChunk(ParallelMesh* pp_mesh, uint64_t n_x, uint64_t n_y, double cell_size,
	      const std::map<PointIndexEnum, PointTypeEnum>& point_types,
        uint64_t n_ch_glob_x, uint64_t n_ch_glob_y,
        uint64_t chunk_position_global_x = 0, uint64_t chunk_position_global_y = 0,
	      double o_x = 0., double o_y = 0.);

    // only getters for unchangeable mesh characteristics
    uint64_t get_n_cells_x() const { return this->n_cells_x; }
    uint64_t get_n_cells_y() const { return this->n_cells_y; }
    uint64_t get_n_points_x() const { return this->n_cells_x + 1; }
    uint64_t get_n_points_y() const { return this->n_cells_y + 1; }
    double get_cell_size() const { return this->h; }
    std::array<double, 2> get_origin() const {return this->origin; }
    std::array<uint64_t, 2> get_global_position() const {return this->global_position; }

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
    double get_velocity_x(int64_t i, int64_t j);
    double get_velocity_y(int64_t i, int64_t j);

    // setter to assign new mesh point data
    void set_velocity(Kokkos::View<double**[2]> v) { this->velocity = v; }

    // setter to assign new mesh cell data
    void set_pressure(Kokkos::View<double*> p) { this->pressure = p; }

    // apply velocity boundary conditions to mesh chunk points
    void apply_velocity_bc(std::map<std::string, double> velocity_values);

    // predict velocity for points in mesh chunk
    void chunk_predict_velocity(double delta_t, double nu);

    #ifdef USE_MPI
    // mpi sends to other mesh chunk
    void mpi_send_border_velocity_x(double border_velocity_x, uint64_t pos_x, uint64_t pos_y, uint64_t dest_rank, uint8_t border);
    void mpi_send_border_velocity_y(double border_velocity_y, uint64_t pos_x, uint64_t pos_y, uint64_t dest_rank, uint8_t border);

    // mpi receive from other mesh chunk
    void mpi_receive_border_velocity_x(double border_velocity_x, uint64_t pos_x, uint64_t pos_y, uint64_t source_rank, uint8_t border, MPI_Status status);
    void mpi_receive_border_velocity_y(double border_velocity_y, uint64_t pos_x, uint64_t pos_y, uint64_t source_rank, uint8_t border, MPI_Status status);
    #endif

    // converter to VTK uniform grid
    #ifdef OUTPUT_VTK_FILES
    vtkSmartPointer<vtkUniformGrid> make_VTK_uniform_grid() const;
    #endif

    // writer to VTK file
    std::string write_vti(const std::string& file_name) const;

  private:
    // parent parallel mesh
    ParallelMesh* parent_parallel_mesh;

    // location of mesh chunk in global domain
    std::array<uint64_t, 2> global_position;

    // physical origin of the mesh block
    std::array<double,2> origin;

    // characteristic dimensions of the mesh
    uint64_t n_cells_x;
    uint64_t n_cells_y;

    // global characteristics
    uint64_t n_chunks_global_x;
    uint64_t n_chunks_global_y;

    // physical cell size
    double h = 1.;

    // mesh point types
    Kokkos::View<PointTypeEnum**> point_type = {};

    // mesh point data
    Kokkos::View<double**[2]> velocity = {};

    // mesh cell data
    Kokkos::View<double*> pressure = {};

    // storage for bordering velocity values
    std::map<Border, Kokkos::View<double*[2]>> border_velocities = {};
};
