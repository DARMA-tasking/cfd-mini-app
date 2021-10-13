#include "parallel_mesh.h"

#include <iomanip>
#include <iostream>
#include <array>
#include <map>
#include <cmath> 

ParallelMesh::ParallelMesh(uint64_t n_x, uint64_t n_y, double cell_size,
			   uint16_t n_p, uint16_t n_q,
			   double o_x, double o_y)
  : n_cells_x(n_x)
  , n_cells_y(n_y)
  , n_blocks_x(n_p)
  , n_blocks_y(n_q)
  , h(cell_size)
  , origin({o_x, o_y})
  , q_x (n_x / n_p)
  , q_y (n_y / n_q)
  , r_x (n_x % n_p)
  , r_y (n_y % n_q){

  // iterate over row (Y) major over mesh chunks
  for (auto q = 0; q < n_q; q++){
    // determine row height
    auto n = (q < this->r_y) ? this->q_y + 1 : this->q_y;
    
    // initialze column (X) horizontal origin
    o_x = this->get_origin()[0];

    // iterate over column (X) minor over mesh chunks
    for (auto p = 0; p < n_p; p++){
      // determine column width
      auto m = (p < this->r_x) ? this->q_x + 1 : this->q_x;

      // create default mesh chunk boundary point types
      std::map<uint8_t, PointTypeEnum> pt;
      pt[0] = pt[1] = pt[3] = pt[4] = pt[7] = PointTypeEnum::GHOST;
      pt[2] = pt[5] = pt[6] = PointTypeEnum::SHARED_OWNED;

      // override outer boundary point types when applicable
      if (p == 0)
	pt[0] = pt[3] = pt[7] = PointTypeEnum::BOUNDARY;
      if (p == n_p - 1)
	pt[1] = pt[2] = pt[5] = PointTypeEnum::BOUNDARY;
      if (q == 0)
	pt[0] = pt[1] = pt[4] = PointTypeEnum::BOUNDARY;
      if (q == n_q - 1)
	pt[2] = pt[3] = pt[6] = PointTypeEnum::BOUNDARY;

      // instantiate and store new mesh block
      this->mesh_chunks.push_back(MeshChunk(m, n, this->h, pt, o_x, o_y));
      std::cout << o_x << " " << o_y << std::endl;
      // slide horizontal origin rightward
      o_x += m * this->h;
    } // p
    // slide vertical origin upward
    o_y += n * this->h;
  } // q
}

std::map<uint16_t, std::string> ParallelMesh::
write_vti(const std::string& file_stem) const{
  // determine 0-padding extent
  uint8_t z_width = static_cast<uint8_t>
    (ceil(log10(this->mesh_chunks.size())));

  // initialize mesh chunks counter and name container
  uint16_t i = 0;
  std::map<uint16_t, std::string> file_name_map;

  // iterate pover mesh chunks
  for (const auto& it_mesh_chunks : this->mesh_chunks){
    // assemble current chunk file name
    std::ostringstream ss;
    ss << file_stem
       << std::setfill('0')
       << std::setw(z_width)
       << i;

    // write mesh chunk file and increment counter
    file_name_map[i++] = it_mesh_chunks.write_vti(ss.str());
  }

  // return map of assembled file names
  return file_name_map;
}
