#pragma once
#include <cstdio>
#include <array>
#include <map>

#include "cfd_config.h"
#include "boundary_conditions.h"
#include "mesh_chunk.h"

enum struct LocationIndexEnum : int8_t {
  BOTTOM = 0,
  RIGHT = 1,
  TOP = 2,
  LEFT = 3,
  BOTTOM_L = 4,
  BOTTOM_R = 5,
  TOP_R = 6,
  TOP_L = 7,
  // case where there are only two parallel meshes in domain along y axis -> (1, 2)
  VERT_BAR_TOP = 8,
  VERT_BAR_MID  = 9,
  VERT_BAR_BOT = 10,
  // case where there are only two parallel meshes in domain along x axis -> (2, 1)
  HORIZ_BAR_L = 11,
  HORIZ_BAR_MID = 12,
  HORIZ_BAR_R = 13,
  // normal case but interior
  INTERIOR = 14,
  // case where there is only one parallel mesh in domain -> (1, 1)
  SINGLE = 15
};

struct LocalCoordinates
{
  uint64_t block[2];
  uint64_t local[2];
};

class ParallelMesh
{
  public:
    ParallelMesh(uint64_t r, uint64_t n_x, uint64_t n_y, double cell_size,
		 uint16_t n_p, uint16_t n_q, int8_t border,
     uint64_t n_chunks_glob_x, uint64_t n_chunks_glob_y,
     std::map<std::string, double> velocity_values,
     uint64_t glob_position_x = 0, uint64_t glob_position_y = 0,
		 double o_x = 0., double o_y = 0.);

    // only getters for unchangeable parallel mesh characteristics
    uint64_t get_n_cells_x() const { return this->n_cells_x; }
    uint64_t get_n_cells_y() const { return this->n_cells_y; }
    uint64_t get_n_chunks_x() const { return this->n_chunks_x; }
    uint64_t get_n_chunks_y() const { return this->n_chunks_y; }
    uint64_t get_n_points_x() const { return this->n_cells_x + 1; }
    uint64_t get_n_points_y() const { return this->n_cells_y + 1; }
    double get_cell_size() const { return this->h; }
    std::array<double,2> get_origin() const {return this->origin; }
    std::array<uint64_t,2> get_global_position() const {return this->global_position; }
    uint8_t get_location_type() const { return this->location_type; }

    // getters and setters for attributes in specific mesh chunks
    double get_velocity_mesh_chunk_x(uint64_t, uint64_t, uint64_t, uint64_t);
    double get_velocity_mesh_chunk_y(uint64_t, uint64_t, uint64_t, uint64_t);
    void set_velocity_mesh_chunk_x(uint64_t, uint64_t, uint64_t, uint64_t, double);
    void set_velocity_mesh_chunk_y(uint64_t, uint64_t, uint64_t, uint64_t, double);
    uint64_t get_n_points_x_mesh_chunk(uint64_t, uint64_t);
    uint64_t get_n_points_y_mesh_chunk(uint64_t, uint64_t);
    PointTypeEnum get_chunk_point_type(uint64_t, uint64_t, uint64_t, uint64_t) const;

    // global to local indexing converters
    LocalCoordinates GlobalToLocalCellIndices(uint64_t, uint64_t) const;
    LocalCoordinates GlobalToLocalPointIndices(uint64_t, uint64_t) const;

    // local to global indexing converters
    std::array<uint64_t,2> LocalToGlobalCellIndices(const LocalCoordinates&) const;
    std::array<uint64_t,2> LocalToGlobalPointIndices(const LocalCoordinates&) const;

    // condition verifiers for mesh chunk assignements
    bool is_chunk_in_parent_p_mesh(uint64_t, uint64_t);

    // apply boundary conditions
    void apply_velocity_bc();

    // get Reynolds number across parallel mesh
    double get_reynolds_l();

    // compute max courant number across all owned chunks
    double compute_global_courant_number();

    // get specific boundary velocity value
    double get_boundary_velocity_value(std::string);

    // predict velocity in all owned mesh chunks
    void pmesh_predict_velocity(double, double);

    // synchronize chunk borders of velocities
    void receive_border_velocities();

    // writer to VTK files
    std::string write_vtm(const std::string&) const;

  private:
    // global position of the parallel mesh
    std::array<uint64_t, 2> global_position;

    // physical origin of the parallel mesh
    std::array<double,2> origin;

    // characteristic dimensions of the parallel mesh
    uint64_t n_cells_x;
    uint64_t n_cells_y;
    uint64_t n_chunks_x;
    uint64_t n_chunks_y;
    uint64_t q_x;
    uint64_t q_y;
    uint64_t r_x;
    uint64_t r_y;
    uint64_t cutoff_x;
    uint64_t cutoff_y;

    // physical cell size
    double h = 1.;

    // Storage for mesh chunks
    std::map<std::array<uint64_t,2>,MeshChunk> mesh_chunks = {};

    // border type
    int8_t location_type;

    // boundary velocity values
    std::map<std::string, double> boundary_velocity_values;

    // rank identifier of parallel mesh
    uint64_t rank;
};
