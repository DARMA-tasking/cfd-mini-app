#include "parallel_mesh.h"

#include <iostream>
#include <array>
#include <map>
#include <cmath>

#include <vtkSmartPointer.h>
#include <vtkUniformGrid.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkXMLMultiBlockDataWriter.h>

ParallelMesh::
ParallelMesh(uint64_t n_x, uint64_t n_y, double cell_size,
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

  // Compute cutoff between wide and narrow blocks
  this->cutoff_x = this->r_x * (this->q_x + 1);
  this->cutoff_y = this->r_y * (this->q_y + 1);

  // iterate over row (Y) major over mesh chunks
  for (uint64_t q = 0; q < n_q; q++){
    // determine row height
    uint64_t n = (q < this->r_y) ? this->q_y + 1 : this->q_y;

    // initialze column (X) horizontal origin
    o_x = this->origin[0];

    // iterate over column (X) minor over mesh chunks
    for (uint64_t p = 0; p < n_p; p++){
      // determine column width
      uint64_t m = (p < this->r_x) ? this->q_x + 1 : this->q_x;

      // create default mesh chunk boundary point types
      std::map<PointIndexEnum, PointTypeEnum> pt = {
        {PointIndexEnum::CORNER_0, PointTypeEnum::GHOST},
	{PointIndexEnum::CORNER_2, PointTypeEnum::SHARED_OWNED},
	{PointIndexEnum::CORNER_1, PointTypeEnum::GHOST},
	{PointIndexEnum::CORNER_3, PointTypeEnum::GHOST},
	{PointIndexEnum::EDGE_0, PointTypeEnum::GHOST},
	{PointIndexEnum::EDGE_1, PointTypeEnum::SHARED_OWNED},
	{PointIndexEnum::EDGE_2, PointTypeEnum::SHARED_OWNED},
	{PointIndexEnum::EDGE_3, PointTypeEnum::GHOST},
	{PointIndexEnum::INTERIOR, PointTypeEnum::INTERIOR}
      };

      // override outer boundary point types when applicable
      if (q == 0)
	pt[PointIndexEnum::EDGE_0]
	  = pt[PointIndexEnum::CORNER_0]
	  = pt[PointIndexEnum::CORNER_1]
	  = PointTypeEnum::BOUNDARY;
      if (p == n_p - 1)
	pt[PointIndexEnum::EDGE_1]
	  = pt[PointIndexEnum::CORNER_1]
	  = pt[PointIndexEnum::CORNER_2]
	  = PointTypeEnum::BOUNDARY;
      if (q == n_q - 1)
	pt[PointIndexEnum::EDGE_2]
	  = pt[PointIndexEnum::CORNER_2]
	  = pt[PointIndexEnum::CORNER_3]
	  = PointTypeEnum::BOUNDARY;
      if (p == 0)
	pt[PointIndexEnum::EDGE_3]
	  = pt[PointIndexEnum::CORNER_3]
	  = pt[PointIndexEnum::CORNER_0]
	  = PointTypeEnum::BOUNDARY;

      // append new mesh block to existing ones
      this->mesh_chunks.emplace
	(std::piecewise_construct,
	 std::forward_as_tuple(std::array<uint64_t,2>{q,p}),
	 std::forward_as_tuple(m, n, this->h, pt, o_x, o_y));

      // slide horizontal origin rightward
      o_x += m * this->h;
    } // p
    // slide vertical origin upward
    o_y += n * this->h;
  } // q
}

LocalCoordinates ParallelMesh::
GlobalToLocalCellIndices(uint64_t m, uint64_t n) const{
  // return invalid values when global coordinates are out of bounds
  if (m < 0 || m >= this->n_cells_x
      || n < 0 || n >= this->n_cells_y)
    return {static_cast<uint64_t>(-1)};
  
  // compute X-axis local coordinates
  uint64_t p, i;
  if (m < this->cutoff_x){
    // coordinate falls in wider blocks
    auto d = ldiv(m, this->q_x + 1);
    p = d.quot;
    i = d.rem;
  } else{
    // coordinate falls in narrower blocks
    auto d = ldiv(m - this->cutoff_x, this->q_x);
    p = d.quot + this->r_x;
    i = d.rem;
  }

  // compute Y-axis local coordinates
  uint64_t q, j;
  if (n < this->cutoff_y){
    // coordinates fall in wider blocks
    auto d = ldiv(n, this->q_y + 1);
    q = d.quot;
    j = d.rem;
  } else{
    // coordinates fall in narrower blocks
    auto d = ldiv(n - this->cutoff_y, this->q_y);
    q = d.quot + this->r_y;
    j = d.rem;
  }

  // return valid indices
  return {p, q, i, j};
}

LocalCoordinates ParallelMesh::
GlobalToLocalPointIndices(uint64_t m, uint64_t n) const{
  // return invalid values when global coordinates are out of bounds
  if (m < 0 || m >= this->get_n_points_x()
      || n < 0 || n >= this->get_n_points_y())
    return {static_cast<uint64_t>(-1)};

  return {static_cast<uint64_t>(-1)};
}

std::array<uint64_t,2> ParallelMesh::
LocalToGlobalCellIndices(const LocalCoordinates& loc) const{
  // return invalid values when indices are out of bounds
  uint64_t p = loc.block[0];
  uint64_t q = loc.block[1];
  uint64_t i = loc.local[0];
  uint64_t j = loc.local[1];
  if (p < 0 || q < 0 || i < 0 || j < 0 
      || p >= this->n_blocks_x || q >= this->n_blocks_y)
    return {static_cast<uint64_t>(-1)};

  // compute X-axis global coordinate
  uint64_t m;
  if (p < this->r_x){
    // return invalid values when local index is out of bounds
    if (i > this->q_x)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in wider blocks
    m = (this->q_x + 1) * p + i;
  } else{
    // return invalid values when local index is out of bounds
    if (i >= this->q_x)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in narrower blocks
    m = this->cutoff_x + this->q_x * (p - this->r_x) + i;
  }
  
  // compute Y-axis global coordinate
  uint64_t n;
  if (q < this->r_y){
    // return invalid values when local index is out of bounds
    if (j > this->q_y)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in wider blocks
    n = (this->q_y + 1) * q + j;
  } else{
    // return invalid values when local index is out of bounds
    if (j >= this->q_y)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in narrower blocks
    n = this->cutoff_y + this->q_y * (q - this->r_y) + j;
  }
  
  // return valid indices
  return {m, n};
}

std::array<uint64_t,2> ParallelMesh::
LocalToGlobalPointIndices(const LocalCoordinates& loc) const{
  // return invalid values when indices are out of bounds
  uint64_t p = loc.block[0];
  uint64_t q = loc.block[1];
  uint64_t i = loc.local[0];
  uint64_t j = loc.local[1];
  if (p < 0 || q < 0 || i < 0 || j < 0 
      || p >= this->n_blocks_x || q >= this->n_blocks_y)
    return {static_cast<uint64_t>(-1)};

  // compute X-axis global coordinate
  uint64_t m;
  if (p < this->r_x){
    // return invalid values when local index is out of bounds
    if (i > this->q_x + 1)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in wider blocks
    m = (this->q_x + 1) * p + i;
  } else{
    // return invalid values when local index is out of bounds
    if (i > this->q_x)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in narrower blocks
    m = this->cutoff_x + this->q_x * (p - this->r_x) + i;
  }
  
  // compute Y-axis global coordinate
  uint64_t n;
  if (q < this->r_y){
    // return invalid values when local index is out of bounds
    if (j > this->q_y + 1)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in wider blocks
    n = (this->q_y + 1) * q + j;
  } else{
    // return invalid values when local index is out of bounds
    if (j > this->q_y)
      return {static_cast<uint64_t>(-1)};

    // coordinate falls in narrower blocks
    n = this->cutoff_y + this->q_y * (q - this->r_y) + j;
  }
  
  // return valid indices
  return {m, n};
}

std::string ParallelMesh::
write_vtm(const std::string& file_name) const{
  // assemble full file name with extension
  std::string full_file_name = file_name + ".vtm";

  // aggregate all mesh chunks as VTK multi-block data set
  vtkSmartPointer<vtkMultiBlockDataSet>
    mbs = vtkSmartPointer<vtkMultiBlockDataSet>::New();
  mbs->SetNumberOfBlocks(this->mesh_chunks.size());
  uint16_t i = 0;
  for (const auto& it_mesh_chunks : this->mesh_chunks)
    mbs->SetBlock(i++, it_mesh_chunks.second.make_VTK_uniform_grid().GetPointer());

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
