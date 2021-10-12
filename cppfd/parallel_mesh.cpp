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
  this->q_x = n_x / n_p;
  this->q_y = n_y / n_q;
  this->r_x = n_x % n_p;
  this->r_y = n_y % n_q;

  // create mesh chunks
  for (auto q = 0; q < n_q; q++){
    // determine row height
    auto n = (q < this->r_y) ? this->q_y + 1 : this->q_y;

    for (auto p = 0; p < n_p; p++){
      // determine column width
      auto m = (p < this->r_x) ? this->q_x + 1 : this->q_x;

      // create default mesh chunk boundary point types
      std::map<uint8_t, PointTypeEnum> pt;
      pt[0] = pt[1] = pt[3] = pt[4] = pt[7] = PointTypeEnum::GHOST;
      pt[2] = pt[5] = pt[5] = PointTypeEnum::SHARED_OWNED;

      // override outer boundary point types when applicable
      if (p == 0)
	pt[0] = pt[1] = pt[4] = PointTypeEnum::BOUNDARY;
      if (p == n_p - 1)
	pt[1] = pt[2] = pt[5] = PointTypeEnum::BOUNDARY;
      if (q == 0)
	pt[0] = pt[3] = pt[7] = PointTypeEnum::BOUNDARY;
      if (q == n_q - 1)
	pt[2] = pt[3] = pt[6] = PointTypeEnum::BOUNDARY;

      // instantiate and store new mesh block
      this->mesh_chunks.push_back(MeshChunk(m, n, this->h, pt));
    }
  }
}

void ParallelMesh::set_origin(const double x, const double y){
  this->origin[0] = x;
  this->origin[1] = y;
}

void ParallelMesh::write_VTK(const std::string& file_name){
  // iterate pover mesh chunks
  for (const auto& mesh : this->mesh_chunks){
    std::cout << mesh.get_n_cells_x() << " " << mesh.get_n_cells_y() << std::endl;
  }
  
}
