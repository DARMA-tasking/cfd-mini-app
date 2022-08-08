#pragma once
#include <cstdio>
#include <array>
#include <map>

#include "cfd_config.h"
#include "mesh_chunk.h"

enum struct LocationIndexEnum : int8_t {
  BOTTOM = 0,
  RIGHT = 1,
  LEFT = 2,
  TOP = 3,
  BOTTOM_L = 4,
  BOTTOM_R = 5,
  TOP_L = 6,
  TOP_R = 7,
  INTERIOR = 8
};

struct LocalCoordinates
{
  uint64_t block[2];
  uint64_t local[2];
};

class ParallelMesh
{
  public:
    ParallelMesh(uint64_t n_x, uint64_t n_y, double cell_size,
		 uint16_t n_p, uint16_t n_q, int8_t border,
		 double o_x = 0., double o_y = 0.);

    // only getters for unchangeable mesh characteristics
    uint64_t get_n_cells_x() const { return this->n_cells_x; }
    uint64_t get_n_cells_y() const { return this->n_cells_y; }
    uint64_t get_n_blocks_x() const { return this->n_blocks_x; }
    uint64_t get_n_blocks_y() const { return this->n_blocks_y; }
    uint64_t get_n_points_x() const { return this->n_cells_x + 1; }
    uint64_t get_n_points_y() const { return this->n_cells_y + 1; }
    double get_cell_size() const { return this->h; }
    std::array<double,2> get_origin() const {return this->origin; }

    // global to local indexing converters
    LocalCoordinates GlobalToLocalCellIndices(uint64_t, uint64_t) const;
    LocalCoordinates GlobalToLocalPointIndices(uint64_t, uint64_t) const;

    // local to global indexing converters
    std::array<uint64_t,2> LocalToGlobalCellIndices(const LocalCoordinates&) const;
    std::array<uint64_t,2> LocalToGlobalPointIndices(const LocalCoordinates&) const;

    // writer to VTK files
    std::string write_vtm(const std::string&) const;

  private:
    // physical origin of the mesh block
    std::array<double,2> origin;

    // characteristic dimensions of the mesh
    uint64_t n_cells_x;
    uint64_t n_cells_y;
    uint64_t n_blocks_x;
    uint64_t n_blocks_y;
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
    int8_t border_type;

    #ifdef USE_MPI
    // storage for bordering velocity values
    std::map<Border, Kokkos::View<double*[2]>> border_velocities = {};

    // MPI rank of parallel mesh
    int64_t mpi_rank;
    #endif
};
