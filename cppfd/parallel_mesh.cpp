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
  std::div_t dv = std::div(n_x, n_p);
  std::cout << dv.quot << " " << dv.rem << std::endl;
  dv = std::div(n_y, n_q);
  std::cout << dv.quot << " " << dv.rem << std::endl;
}

void ParallelMesh::set_origin(double x, double y){
  this->origin[0] = x;
  this->origin[1] = y;
}
