#pragma once
#include <cstdio>
#include <array>

class ParallelMesh
{
  public:
    ParallelMesh(uint64_t n_x, uint64_t n_y, double cell_size,
		 uint16_t n_p, uint16_t n_q);

    // only getters for unchangeable mesh characteristics
    uint64_t get_n_cells_x() const { return this->n_cells_x; }
    uint64_t get_n_cells_y() const { return this->n_cells_y; }
    uint64_t get_n_blocks_x() const { return this->n_blocks_x; }
    uint64_t get_n_blocks_y() const { return this->n_blocks_y; }
    uint64_t get_n_points_x() const { return this->n_cells_x + 1; }
    uint64_t get_n_points_y() const { return this->n_cells_y + 1; }
    double get_cell_size() const { return this->h; }

    // setter/getter for physical origin member variable
    void set_origin(double x, double y);
    std::array<double,2> get_origin() {return this->origin; }

  private:
    // physical origin of the mesh block
    std::array<double,2> origin;

    // characteristic dimensions of the mesh
    uint64_t n_cells_x;
    uint64_t n_cells_y;
    uint64_t n_blocks_x;
    uint64_t n_blocks_y;
    uint64_t n_x_per_block;
    uint64_t n_y_per_block;
    uint64_t n_x_rem;
    uint64_t n_y_rem;

    // physical cell size
    double h = 1.;

    // 2-D Cartesian array of mesh chunks
};
