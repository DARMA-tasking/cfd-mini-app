#include "parallel_mesh.h"

#include <iostream>
#include <array>
#include <map>
#include <cmath>

#include <vtkSmartPointer.h>
#include <vtkUniformGrid.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkXMLMultiBlockDataWriter.h>

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
    o_x = this->origin[0];

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

      // append new mesh block to existing ones
      this->mesh_chunks.emplace_back(m, n, this->h, pt, o_x, o_y);

      // slide horizontal origin rightward
      o_x += m * this->h;
    } // p
    // slide vertical origin upward
    o_y += n * this->h;
  } // q
}

std::string ParallelMesh::write_vtm(const std::string& file_name) const{
  // assemble full file name with extension
  std::string full_file_name = file_name + ".vtm";

  // aggregate all mesh chunks as VTK multi-block data set
  vtkSmartPointer<vtkMultiBlockDataSet>
    mbs = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  mbs->SetNumberOfBlocks(this->mesh_chunks.size());
  uint16_t i = 0;
  for (const auto& it_mesh_chunks : this->mesh_chunks)
    mbs->SetBlock(i++, it_mesh_chunks.make_VTK_uniform_grid().GetPointer());

  // write VTK multi-block data set (vtm) file
  std::cout << full_file_name << std::endl;
  vtkSmartPointer<vtkXMLMultiBlockDataWriter>
    output_file = vtkSmartPointer<vtkXMLMultiBlockDataWriter>::New();
  output_file->SetFileName(full_file_name.c_str());
  output_file->SetInputData(mbs);
  output_file->Write();

  // return fill name with extension
  return full_file_name;
}
