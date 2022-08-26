#include "parallel_mesh.h"

#include <iostream>
#include <array>
#include <map>
#include <cmath>

#ifdef OUTPUT_VTK_FILES
#include <vtkSmartPointer.h>
#include <vtkUniformGrid.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkXMLMultiBlockDataWriter.h>
#endif

const uint64_t uint64_nan = static_cast<uint64_t>(-1);

ParallelMesh::
ParallelMesh(uint64_t r, uint64_t n_x, uint64_t n_y, double cell_size,
     uint16_t n_p, uint16_t n_q, int8_t border,
     uint64_t n_chunks_glob_x, uint64_t n_chunks_glob_y,
     std::map<std::string, double> velocity_values,
     uint64_t glob_position_x, uint64_t glob_position_y,
     double o_x, double o_y)
    : n_cells_x(n_x)
    , n_cells_y(n_y)
    , n_chunks_x(n_p)
    , n_chunks_y(n_q)
    , h(cell_size)
    , boundary_velocity_values(velocity_values)
    , global_position({glob_position_x, glob_position_y})
    , origin({o_x, o_y})
    , q_x (n_x / n_p)
    , q_y (n_y / n_q)
    , r_x (n_x % n_p)
    , r_y (n_y % n_q)
    , location_type(border)
    , rank(r){

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
      switch (this->get_location_type()) {
        case static_cast<int>(LocationIndexEnum::BOTTOM):
          if (q == 0)
          pt[PointIndexEnum::EDGE_0]
          = pt[PointIndexEnum::CORNER_0]
          = pt[PointIndexEnum::CORNER_1]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::RIGHT):
          if (p == n_p - 1)
          pt[PointIndexEnum::EDGE_1]
          = pt[PointIndexEnum::CORNER_1]
          = pt[PointIndexEnum::CORNER_2]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::TOP):
          if (q == n_q - 1)
          pt[PointIndexEnum::EDGE_2]
          = pt[PointIndexEnum::CORNER_2]
          = pt[PointIndexEnum::CORNER_3]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::LEFT):
          if (p == 0)
          pt[PointIndexEnum::EDGE_3]
          = pt[PointIndexEnum::CORNER_3]
          = pt[PointIndexEnum::CORNER_0]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::BOTTOM_L):
          if (q == 0)
          pt[PointIndexEnum::EDGE_0]
          = pt[PointIndexEnum::CORNER_0]
          = pt[PointIndexEnum::CORNER_1]
          = PointTypeEnum::BOUNDARY;
          if (p == 0)
          pt[PointIndexEnum::EDGE_3]
          = pt[PointIndexEnum::CORNER_3]
          = pt[PointIndexEnum::CORNER_0]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::BOTTOM_R):
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
          break;

        case static_cast<int>(LocationIndexEnum::TOP_R):
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
          break;

        case static_cast<int>(LocationIndexEnum::TOP_L):
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
          break;

        case static_cast<int>(LocationIndexEnum::INTERIOR):
          // this case uses default mesh chunk boundary types
          break;

        case static_cast<int>(LocationIndexEnum::SINGLE):
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
          break;

        case static_cast<int>(LocationIndexEnum::VERT_BAR_TOP):
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
          break;

        case static_cast<int>(LocationIndexEnum::VERT_BAR_MID):
          if (p == n_p - 1)
          pt[PointIndexEnum::EDGE_1]
          = pt[PointIndexEnum::CORNER_1]
          = pt[PointIndexEnum::CORNER_2]
          = PointTypeEnum::BOUNDARY;
          if (p == 0)
          pt[PointIndexEnum::EDGE_3]
          = pt[PointIndexEnum::CORNER_3]
          = pt[PointIndexEnum::CORNER_0]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::VERT_BAR_BOT):
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
          if (p == 0)
          pt[PointIndexEnum::EDGE_3]
          = pt[PointIndexEnum::CORNER_3]
          = pt[PointIndexEnum::CORNER_0]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::HORIZ_BAR_L):
          if (q == 0)
          pt[PointIndexEnum::EDGE_0]
          = pt[PointIndexEnum::CORNER_0]
          = pt[PointIndexEnum::CORNER_1]
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
          break;

        case static_cast<int>(LocationIndexEnum::HORIZ_BAR_MID):
          if (q == 0)
          pt[PointIndexEnum::EDGE_0]
          = pt[PointIndexEnum::CORNER_0]
          = pt[PointIndexEnum::CORNER_1]
          = PointTypeEnum::BOUNDARY;
          if (q == n_q - 1)
          pt[PointIndexEnum::EDGE_2]
          = pt[PointIndexEnum::CORNER_2]
          = pt[PointIndexEnum::CORNER_3]
          = PointTypeEnum::BOUNDARY;
          break;

        case static_cast<int>(LocationIndexEnum::HORIZ_BAR_R):
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
          break;

        default:
          break;
      }

      // append new mesh chunk to existing ones
       this->mesh_chunks.emplace
          (std::piecewise_construct,
           std::forward_as_tuple(std::array<uint64_t,2>{q,p}),
           std::forward_as_tuple(this, m, n, this->h, pt, n_chunks_glob_x, n_chunks_glob_y,
                                p + this->n_chunks_x * this->global_position[0],
                                q + this->n_chunks_y * this->global_position[1],
                                o_x, o_y));

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
    return {uint64_nan, uint64_nan, uint64_nan, uint64_nan};

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
    return {uint64_nan, uint64_nan, uint64_nan, uint64_nan};

  // return early for global mesh origin case
  if (m == 0 && n == 0)
    return {0, 0, 0, 0};

  // bottom left cell ownership of point when available
  uint64_t m_c =  (m == 0) ? 0 : m - 1;
  uint64_t n_c =  (n == 0) ? 0 : n - 1;
  LocalCoordinates loc_c = this->GlobalToLocalCellIndices(m_c, n_c);

  // return valid indices
  return {
    loc_c.block[0],
    loc_c.block[1],
    (m == 0) ? 0 : loc_c.local[0] + 1,
    (n == 0) ? 0 : loc_c.local[1] + 1
  };
}

std::array<uint64_t,2> ParallelMesh::
LocalToGlobalCellIndices(const LocalCoordinates& loc) const{
  // return invalid values when indices are out of bounds
  uint64_t p = loc.block[0];
  uint64_t q = loc.block[1];
  uint64_t i = loc.local[0];
  uint64_t j = loc.local[1];
  if (p < 0 || q < 0 || i < 0 || j < 0
      || p >= this->n_chunks_x || q >= this->n_chunks_y)
    return {uint64_nan, uint64_nan};

  // compute X-axis global coordinate
  uint64_t m;
  if (p < this->r_x){
    // return invalid values when local index is out of bounds
    if (i > this->q_x)
      return {uint64_nan, uint64_nan};

    // coordinate falls in wider blocks
    m = (this->q_x + 1) * p + i;
  } else{
    // return invalid values when local index is out of bounds
    if (i >= this->q_x)
      return {uint64_nan, uint64_nan};

    // coordinate falls in narrower blocks
    m = this->cutoff_x + this->q_x * (p - this->r_x) + i;
  }

  // compute Y-axis global coordinate
  uint64_t n;
  if (q < this->r_y){
    // return invalid values when local index is out of bounds
    if (j > this->q_y)
      return {uint64_nan, uint64_nan};

    // coordinate falls in wider blocks
    n = (this->q_y + 1) * q + j;
  } else{
    // return invalid values when local index is out of bounds
    if (j >= this->q_y)
      return {uint64_nan, uint64_nan};

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
      || p >= this->n_chunks_x || q >= this->n_chunks_y)
    return {uint64_nan, uint64_nan};

  // compute X-axis global coordinate
  uint64_t m;
  if (p < this->r_x){
    // return invalid values when local index is out of bounds
    if (i > this->q_x + 1)
      return {uint64_nan, uint64_nan};

    // coordinate falls in wider blocks
    m = (this->q_x + 1) * p + i;
  } else{
    // return invalid values when local index is out of bounds
    if (i > this->q_x)
      return {uint64_nan, uint64_nan};

    // coordinate falls in narrower blocks
    m = this->cutoff_x + this->q_x * (p - this->r_x) + i;
  }

  // compute Y-axis global coordinate
  uint64_t n;
  if (q < this->r_y){
    // return invalid values when local index is out of bounds
    if (j > this->q_y + 1)
      return {uint64_nan, uint64_nan};

    // coordinate falls in wider blocks
    n = (this->q_y + 1) * q + j;
  } else{
    // return invalid values when local index is out of bounds
    if (j > this->q_y)
      return {uint64_nan, uint64_nan};

    // coordinate falls in narrower blocks
    n = this->cutoff_y + this->q_y * (q - this->r_y) + j;
  }

  // return valid indices
  return {m, n};
}

double ParallelMesh::get_velocity_mesh_chunk_x(uint64_t chunk_i, uint64_t chunk_j, uint64_t point_i, uint64_t point_j){
  bool owned_chunk = false;
  double result;
  for (auto& it_mesh_chunk : this->mesh_chunks){
    if(it_mesh_chunk.first[0] == chunk_i && it_mesh_chunk.first[1] == chunk_j){
      owned_chunk = true;
      result = it_mesh_chunk.second.get_velocity_x(point_i, point_j);
    }
  }
  if(!owned_chunk){
    std::cout << "/!\\ Warning : Chunk not owned, MPI exchange here" << '\n';
    result = 0;
  }
  return result;
}

double ParallelMesh::get_velocity_mesh_chunk_y(uint64_t chunk_i, uint64_t chunk_j, uint64_t point_i, uint64_t point_j){
  bool owned_chunk = false;
  double result;
  for (auto& it_mesh_chunk : this->mesh_chunks){
    if(it_mesh_chunk.first[0] == chunk_i && it_mesh_chunk.first[1] == chunk_j){
      owned_chunk = true;
      result = it_mesh_chunk.second.get_velocity_y(point_i, point_j);
    }
  }
  if(!owned_chunk){
    std::cout << "/!\\ Warning : Chunk not owned, MPI exchange here" << '\n';
    result = 0;
  }
  return result;
}

void ParallelMesh::set_velocity_mesh_chunk_x(uint64_t chunk_i, uint64_t chunk_j, uint64_t point_i, uint64_t point_j, double value){
  bool owned_chunk = false;
  double result;
  for (auto& it_mesh_chunk : this->mesh_chunks){
    if(it_mesh_chunk.first[0] == chunk_i && it_mesh_chunk.first[1] == chunk_j){
      owned_chunk = true;
      it_mesh_chunk.second.set_velocity_x(point_i, point_j, value);
    }
  }
  if(!owned_chunk){
    std::cout << "/!\\ Warning : Chunk not owned, MPI exchange here" << '\n';
  }
}

void ParallelMesh::set_velocity_mesh_chunk_y(uint64_t chunk_i, uint64_t chunk_j, uint64_t point_i, uint64_t point_j, double value){
  bool owned_chunk = false;
  double result;
  for (auto& it_mesh_chunk : this->mesh_chunks){
    if(it_mesh_chunk.first[0] == chunk_i && it_mesh_chunk.first[1] == chunk_j){
      owned_chunk = true;
      it_mesh_chunk.second.set_velocity_y(point_i, point_j, value);
    }
  }
  if(!owned_chunk){
    std::cout << "/!\\ Warning : Chunk not owned, MPI exchange here" << '\n';
  }
}

uint64_t ParallelMesh::get_n_points_x_mesh_chunk(uint64_t i, uint64_t j){
  bool owned_chunk = false;
  double result;
  for (auto& it_mesh_chunk : this->mesh_chunks){
    if(it_mesh_chunk.first[0] == i && it_mesh_chunk.first[1] == j){
      owned_chunk = true;
      result = it_mesh_chunk.second.get_n_points_x();
    }
  }
  if(!owned_chunk){
    std::cout << "/!\\ Warning : Chunk not owned, MPI exchange here" << '\n';
    result = 0;
  }
  return result;
}

uint64_t ParallelMesh::get_n_points_y_mesh_chunk(uint64_t i, uint64_t j){
  bool owned_chunk = false;
  double result;
  for (auto& it_mesh_chunk : this->mesh_chunks){
    if(it_mesh_chunk.first[0] == i && it_mesh_chunk.first[1] == j){
      owned_chunk = true;
      result = it_mesh_chunk.second.get_n_points_y();
    }
  }
  if(!owned_chunk){
    std::cout << "/!\\ Warning : Chunk not owned, MPI exchange here" << '\n';
    result = 0;
  }
  return result;
}

void ParallelMesh::apply_velocity_bc() {
  for (auto& it_mesh_chunk : this->mesh_chunks){
    it_mesh_chunk.second.apply_velocity_bc(this->boundary_velocity_values);
  }
}

double ParallelMesh::get_reynolds_l(){
  double max = 0;
  double l;
  for (auto& it_mesh_chunk : this->mesh_chunks){
    l = it_mesh_chunk.second.get_cell_size() * std::max(it_mesh_chunk.second.get_n_cells_x(), it_mesh_chunk.second.get_n_cells_y());
    if(l > max){
      max = l;
    }
  }
  return max;
}

double ParallelMesh::get_boundary_velocity_value(std::string boundary){
  return this->boundary_velocity_values.at(boundary);
}

void ParallelMesh::pmesh_predict_velocity(double delta_t, double nu){
  for(auto& it_mesh_chunk : this->mesh_chunks){
    it_mesh_chunk.second.chunk_predict_velocity(delta_t, nu);
  }
}

#ifdef OUTPUT_VTK_FILES
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
  vtkSmartPointer<vtkXMLMultiBlockDataWriter>
    output_file = vtkSmartPointer<vtkXMLMultiBlockDataWriter>::New();
  output_file->SetFileName(full_file_name.c_str());
  output_file->SetInputData(mbs);
  output_file->Write();

  // return fill name with extension
  return full_file_name;
}
#endif
