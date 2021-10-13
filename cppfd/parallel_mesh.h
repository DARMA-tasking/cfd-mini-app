#pragma once
#include <cstdio>
#include <array>
#include <vector>
#include <map>

#include "mesh_chunk.h"

class ParallelMesh
{
  public:
    ParallelMesh(uint64_t n_x, uint64_t n_y, double cell_size,
		 uint16_t n_p, uint16_t n_q,
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

    // writer to VTK files
    std::map<uint16_t, std::string> write_vti(const std::string&) const;

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

    // physical cell size
    double h = 1.;

    // Storage for mesh chunks
    std::vector<MeshChunk> mesh_chunks = {};
};
