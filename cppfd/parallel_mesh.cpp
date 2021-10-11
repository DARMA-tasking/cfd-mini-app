#include "parallel_mesh.h"

#include <iostream>
#include <array>


ParallelMesh::ParallelMesh(uint64_t n_x, uint64_t n_y, double cell_size,
			   uint16_t n_p, uint16_t n_q){

  // set instance variables
  this->n_cells_x = n_x;
  this->n_cells_y = n_y;
  this->n_blocks_x = n_p;
  this->n_blocks_y = n_q;
  this->h = cell_size;
  this->origin = {0., 0.};

  // create mesh chunks
  this->n_x_per_block = n_x / n_p;
  this->n_y_per_block = n_y / n_q;
  this->n_x_rem = n_x % n_p;
  this->n_y_rem = n_y % n_q;
  for (auto q = 0; q < n_q; q++){
    uint64_t n_b_y = (q < this->n_y_rem) ?
      this->n_y_per_block + 1 : this->n_y_per_block;
    for (auto p = 0; p < n_p; p++){
      std::cout << "Mesh block " << p << " , " << q << std::endl;
      uint64_t n_b_x = (p < this->n_x_rem) ?
	this->n_x_per_block + 1 : this->n_x_per_block;
      std::cout << "   size: " << n_b_x << " , " << n_b_y << std::endl;
    }
  }
}

void ParallelMesh::set_origin(double x, double y){
  this->origin[0] = x;
  this->origin[1] = y;
}
